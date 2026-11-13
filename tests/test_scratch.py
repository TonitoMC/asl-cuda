"""Verificación de la CNN desde cero (NumPy). Correr: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asl import torch_models as tm        # noqa: E402
from scratch import numpy_cnn as nc      # noqa: E402

PAIRS = [("CNN-1 Baseline", nc.cnn1, tm.cnn1), ("CNN-2 Profunda", nc.cnn2, tm.cnn2),
         ("CNN-3 Regularizada", nc.cnn3, tm.cnn3)]


def copy_to_mine(tmodel, mine):
    """Copia pesos de torch a mi modelo (Dense: transponer; BN: buffers)."""
    tl = [m for m in tmodel if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
    ml = [l for l in mine.layers if l.params]
    assert len(tl) == len(ml)
    for t, m in zip(tl, ml):
        if isinstance(t, nn.Conv2d):
            m.params["W"][...] = t.weight.detach().numpy(); m.params["b"][...] = t.bias.detach().numpy()
        elif isinstance(t, nn.Linear):
            m.params["W"][...] = t.weight.detach().numpy().T; m.params["b"][...] = t.bias.detach().numpy()
        else:
            m.params["gamma"][...] = t.weight.detach().numpy(); m.params["beta"][...] = t.bias.detach().numpy()
    return tl, ml


def no_dropout(tmodel, mine):
    for m in tmodel:
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.p = 0.0
    for l in mine.layers:
        if isinstance(l, nc.Dropout):
            l.p = 0.0


@pytest.mark.parametrize("name,mk,tk", PAIRS)
def test_params(name, mk, tk):
    assert mk().count_params() == tm.KERAS_PARAMS[name]


@pytest.mark.parametrize("name,mk,tk", PAIRS)
def test_vs_torch(name, mk, tk):
    torch.manual_seed(0)
    tmodel, mine = tk(), mk()
    tl, ml = copy_to_mine(tmodel, mine)
    no_dropout(tmodel, mine)
    tmodel.train()
    rng = np.random.default_rng(1)
    x = rng.random((8, 3, 64, 64), dtype=np.float32)
    y = rng.integers(0, 29, 8)

    xt = torch.from_numpy(x)
    out_t = tmodel(xt)
    loss_t = nn.functional.cross_entropy(out_t, torch.from_numpy(y))
    loss_t.backward()

    logits = mine.forward(x, train=True)
    loss, d = nc.softmax_cross_entropy(logits, y)
    mine.backward(d)

    assert np.abs(logits - out_t.detach().numpy()).max() < 1e-4
    assert abs(loss - loss_t.item()) < 1e-4
    for t, m in zip(tl, ml):
        pairs = []
        if isinstance(t, nn.Conv2d):
            pairs = [(m.grads["W"], t.weight.grad), (m.grads["b"], t.bias.grad)]
        elif isinstance(t, nn.Linear):
            pairs = [(m.grads["W"], t.weight.grad.T), (m.grads["b"], t.bias.grad)]
        else:
            pairs = [(m.grads["gamma"], t.weight.grad), (m.grads["beta"], t.bias.grad)]
            # medias/varianzas móviles tras un paso de entrenamiento
            assert np.allclose(m.buffers["mean"], t.running_mean.numpy(), atol=1e-5)
            assert np.allclose(m.buffers["var"], t.running_var.numpy(), rtol=1e-4, atol=1e-5)
        for g, gt in pairs:
            gt = gt.numpy()
            rel = np.abs(g - gt).max() / (np.abs(gt).max() + 1e-12)
            if gt.shape == m.params.get("b", gt).shape and np.abs(gt).max() < 1e-5:
                continue  # bias antes de BatchNorm: gradiente teóricamente 0, solo ruido
            assert rel < 1e-3, (name, type(t).__name__, gt.shape, rel)


def test_finite_differences():
    """Conv + BN + ReLU + MaxPool + Dense chica, en float64 emulado con eps grande."""
    rng = np.random.default_rng(0)
    model = nc.Model([nc.Conv2D(2, 3, 3, rng), nc.BatchNorm2D(3), nc.ReLU(), nc.MaxPool2(),
                      nc.GlobalAvgPool(), nc.Dense(3, 4, rng)])
    x = rng.standard_normal((4, 2, 6, 6)).astype(np.float32)
    y = np.array([0, 1, 2, 3])

    def loss_fn():
        return float(nc.softmax_cross_entropy(model.forward(x, True), y)[0])

    loss, d = nc.softmax_cross_entropy(model.forward(x, True), y)
    model.backward(d)
    h = 1e-2
    for p, g in model.params():
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for i in rng.choice(flat.size, min(5, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + h; lp = loss_fn()
            flat[i] = old - h; lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * h)
            assert abs(num - gflat[i]) < 2e-2 * max(1.0, abs(num)), (p.shape, num, gflat[i])
