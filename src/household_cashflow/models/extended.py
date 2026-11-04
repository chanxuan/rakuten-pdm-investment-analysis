"""LightGBM and LSTM quantile models with validation-only selection."""

from __future__ import annotations

import time
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..features import sequence_inputs
from ..io import array_hash


class QuantileLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, head_dropout, head_width=32, n_features=16):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_features, head_width),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_width, 1),
        )

    def forward(self, sequence, known):
        _, (state, _) = self.recurrent(sequence)
        return self.head(torch.cat([state[-1], known], dim=1)).squeeze(-1)


def pinball_torch(prediction, target, alpha=0.9):
    residual = target - prediction
    return torch.maximum(alpha * residual, (alpha - 1) * residual).mean()


@dataclass
class LSTMBundle:
    model: QuantileLSTM
    sequence_scaler: StandardScaler
    head_scaler: StandardScaler
    target_mean: float
    target_scale: float
    provenance: dict

    def predict_unclipped(self, seq, x):
        s = self.sequence_scaler.transform(seq.reshape(-1, 1)).reshape(seq.shape).astype(np.float32)
        head = self.head_scaler.transform(x).astype(np.float32)
        self.model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(x), 2048):
                out.append(
                    self.model(
                        torch.from_numpy(s[start : start + 2048]),
                        torch.from_numpy(head[start : start + 2048]),
                    ).numpy()
                )
        return np.concatenate(out).astype(float) * self.target_scale + self.target_mean

    def predict(self, seq, x):
        return np.maximum(0, self.predict_unclipped(seq, x))


def derived_seed(seed, family, candidate):
    return int(np.random.SeedSequence([seed, family, candidate]).generate_state(1)[0])


def fit_extended_models(panel, config, seed, progress=print):
    train, val = panel.indices("train"), panel.indices("validation")
    st, xt, yt = sequence_inputs(panel, train, np.arange(12, 36))
    sv, xv, yv = sequence_inputs(panel, val, np.arange(24, 36))
    records, light_models = [], []
    for i, params in enumerate(config["lightgbm_candidates"]):
        started = time.perf_counter()
        rng_seed = derived_seed(seed, 10, i)
        model = LGBMRegressor(
            objective="quantile",
            alpha=0.9,
            random_state=rng_seed,
            n_jobs=config["threads"],
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **params,
        )
        model.fit(xt, yt)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pv = np.maximum(0, model.predict(xv))
        loss = float(mean_pinball_loss(yv, pv, alpha=0.9))
        records.append(
            {
                "seed": seed,
                "model": "lightgbm",
                "candidate": i,
                "validation_pinball": loss,
                "training_seconds": time.perf_counter() - started,
                "epochs": None,
                "best_epoch": None,
                "rng_seed": rng_seed,
                "parameters": params,
            }
        )
        light_models.append(model)
        progress(
            f"seed {seed} LightGBM candidate {i}: validation loss={loss:.1f}, seconds={records[-1]['training_seconds']:.1f}"
        )
    best_light = int(np.argmin([r["validation_pinball"] for r in records]))
    # Fit all normalizers using training households ONLY.
    seqsc = StandardScaler().fit(st.reshape(-1, 1))
    headsc = StandardScaler().fit(xt)
    ym, ys = float(yt.mean()), float(yt.std())
    ts = torch.from_numpy(seqsc.transform(st.reshape(-1, 1)).reshape(st.shape).astype(np.float32))
    tx = torch.from_numpy(headsc.transform(xt).astype(np.float32))
    ty = torch.from_numpy(((yt - ym) / ys).astype(np.float32))
    vs = torch.from_numpy(seqsc.transform(sv.reshape(-1, 1)).reshape(sv.shape).astype(np.float32))
    vx = torch.from_numpy(headsc.transform(xv).astype(np.float32))
    provenance = {
        "fit_split": "train",
        "household_ids_sha256": array_hash(panel.household_id[train]),
        "label_months": [13, 36],
        "head_train_sha256": array_hash(xt),
        "sequence_train_sha256": array_hash(st),
        "target_train_sha256": array_hash(yt),
        "sequence_mean": seqsc.mean_.tolist(),
        "sequence_scale": seqsc.scale_.tolist(),
        "head_mean": headsc.mean_.tolist(),
        "head_scale": headsc.scale_.tolist(),
        "target_mean": ym,
        "target_scale": ys,
    }
    lstm_bundles, epoch_records = [], []
    torch.use_deterministic_algorithms(config["lstm"]["deterministic"])
    for i, params in enumerate(config["lstm_candidates"]):
        started = time.perf_counter()
        rng_seed = derived_seed(seed, 20, i)
        torch.manual_seed(rng_seed)
        generator = torch.Generator().manual_seed(derived_seed(seed, 30, i))
        model = QuantileLSTM(
            **{k: v for k, v in params.items() if k != "lr"},
            head_width=config["lstm"]["head_width"],
        )
        with torch.no_grad():
            model.head[-1].bias.fill_(float((np.quantile(yt, 0.9) - ym) / ys))
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
        batch = config["lstm"]["batch_size"]
        for epoch in range(1, config["lstm"]["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(yt), generator=generator)
            total_loss = 0.0
            for start in range(0, len(yt), batch):
                ix = order[start : start + batch]
                optimizer.zero_grad(set_to_none=True)
                loss = pinball_torch(model(ts[ix], tx[ix]), ty[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["lstm"]["gradient_clip"])
                optimizer.step()
                total_loss += float(loss.detach()) * len(ix)
            model.eval()
            valp = []
            with torch.inference_mode():
                for start in range(0, len(yv), 2048):
                    valp.append(model(vs[start : start + 2048], vx[start : start + 2048]).numpy())
            pv = np.maximum(0, np.concatenate(valp).astype(float) * ys + ym)
            vl = float(mean_pinball_loss(yv, pv, alpha=0.9))
            epoch_records.append(
                {
                    "seed": seed,
                    "candidate": i,
                    "epoch": epoch,
                    "training_pinball": total_loss / len(yt) * ys,
                    "validation_pinball": vl,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if vl < best_loss:
                best_loss, best_epoch, best_state, stale = (
                    vl,
                    epoch,
                    deepcopy(model.state_dict()),
                    0,
                )
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0 or stale >= config["lstm"]["patience"]:
                progress(
                    f"seed {seed} LSTM {i} epoch {epoch}: val={vl:.1f}, best={best_loss:.1f}, elapsed={time.perf_counter() - started:.1f}s"
                )
            if stale >= config["lstm"]["patience"]:
                break
        model.load_state_dict(best_state)
        model.eval()
        lstm_bundles.append(LSTMBundle(model, seqsc, headsc, ym, ys, provenance))
        records.append(
            {
                "seed": seed,
                "model": "lstm",
                "candidate": i,
                "validation_pinball": best_loss,
                "training_seconds": time.perf_counter() - started,
                "epochs": epoch,
                "best_epoch": best_epoch,
                "rng_seed": rng_seed,
                "parameters": params,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
        )
    lstm_records = [r for r in records if r["model"] == "lstm"]
    best_lstm = int(np.argmin([r["validation_pinball"] for r in lstm_records]))
    for r in records:
        r["selected"] = r["candidate"] == (best_light if r["model"] == "lightgbm" else best_lstm)
    return light_models[best_light], lstm_bundles[best_lstm], records, epoch_records
