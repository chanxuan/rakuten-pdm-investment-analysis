import importlib
from dataclasses import replace

import numpy as np
import pytest

from household_cashflow.features import sequence_inputs
from household_cashflow.io import array_hash

pytestmark = pytest.mark.extended


@pytest.fixture
def extended():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    torch.set_num_threads(2)
    module = importlib.import_module("household_cashflow.models.extended")
    return torch, module


def test_lstm_uses_one_past_sequence_and_outputs_one_amount(panel, extended):
    torch, models = extended
    idx = panel.indices("test")
    seq, features, _ = sequence_inputs(panel, idx, np.array([24, 36]))
    model = models.QuantileLSTM(32, 1, 0.1)
    assert seq.shape == (len(idx) * 2, 12, 1) and features.shape == (len(idx) * 2, 16)
    assert model(
        torch.tensor(seq, dtype=torch.float32), torch.tensor(features, dtype=torch.float32)
    ).shape == (len(idx) * 2,)


def test_quantile_loss_has_asymmetric_gradients(extended):
    torch, models = extended
    prediction = torch.tensor([0.0, 2.0], requires_grad=True)
    loss = models.pinball_torch(prediction, torch.tensor([1.0, 1.0]))
    loss.backward()
    np.testing.assert_allclose(prediction.grad, [-0.45, 0.05])


def test_extended_fit_excludes_test_data_and_is_deterministic(panel, config, extended):
    torch, models = extended
    _, a, records_a, _ = models.fit_extended_models(panel, config, 811, lambda _: None)
    seq, features, target = sequence_inputs(panel, panel.indices("train"), np.arange(12, 36))
    np.testing.assert_allclose(a.head_scaler.mean_, features.mean(axis=0))
    np.testing.assert_allclose(a.sequence_scaler.mean_, seq.mean(axis=(0, 1)))
    assert a.provenance["target_train_sha256"] == array_hash(target)
    changed = panel.variable_expense.copy()
    changed[panel.indices("test")] *= 100
    _, b, records_b, _ = models.fit_extended_models(
        replace(panel, variable_expense=changed), config, 811, lambda _: None
    )
    assert a.provenance == b.provenance
    assert [r["validation_pinball"] for r in records_a] == [
        r["validation_pinball"] for r in records_b
    ]
    for name, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[name], rtol=0, atol=0)
