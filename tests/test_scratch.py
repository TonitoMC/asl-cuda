"""Verificación de la CNN desde cero (NumPy y CuPy). Correr: python -m pytest tests -q"""
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

PAIRS_NP = [
    ("CNN-1 Baseline", nc.cnn1, tm.cnn1),
    ("CNN-2 Profunda", nc.cnn2, tm.cnn2),
    ("CNN-3 Regularizada", nc.cnn3, tm.cnn3)
]


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
        if hasattr(l, "p"):
            l.p = 0.0


@pytest.mark.parametrize("name,mk,tk", PAIRS_NP)
def test_params(name, mk, tk):
    assert mk().count_params() == tm.KERAS_PARAMS[name]


@pytest.mark.parametrize("name,mk,tk", PAIRS_NP)
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
            assert np.allclose(m.buffers["mean"], t.running_mean.numpy(), atol=1e-5)
            assert np.allclose(m.buffers["var"], t.running_var.numpy(), rtol=1e-4, atol=1e-5)
        for g, gt in pairs:
            gt = gt.numpy()
            rel = np.abs(g - gt).max() / (np.abs(gt).max() + 1e-12)
            if gt.shape == m.params.get("b", gt).shape and np.abs(gt).max() < 1e-5:
                continue
            # 1e-2: la conv1 de CNN-1 acumula 16384 entradas de la Dense en float32 (medido 4.3e-3)
            assert rel < 1e-2, (name, type(t).__name__, gt.shape, rel)


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


# ----------------------------------------------------------------------------- CuPy tests

def test_cupy_available():
    try:
        import cupy as cp
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")
    assert cp.cuda.is_available()


def test_cupy_params():
    try:
        import cupy as cp
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")

    for name, mk in [("CNN-1 Baseline", cc.cnn1), ("CNN-2 Profunda", cc.cnn2), ("CNN-3 Regularizada", cc.cnn3)]:
        assert mk().count_params() == tm.KERAS_PARAMS[name]


def test_cupy_vs_numpy_adam_20_steps():
    """Verificación 3 del enunciado: mismos pesos y mismo lote, la pérdida coincide a 1e-4 durante 20 pasos de Adam."""
    try:
        import cupy as cp
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")

    np_model = nc.cnn1(seed=42)
    cp_model = cc.cnn1(seed=42)
    no_dropout([], np_model)
    no_dropout([], cp_model)

    for (np_p, _), (cp_p, _) in zip(np_model.params(), cp_model.params()):
        assert np.allclose(np_p, cp.asnumpy(cp_p), atol=1e-5)

    opt_np = nc.Adam(np_model.params(), lr=1e-3)
    opt_cp = cc.Adam(cp_model.params(), lr=1e-3)

    rng = np.random.default_rng(123)
    x = rng.random((32, 3, 64, 64), dtype=np.float32)
    y = rng.integers(0, 29, 32)
    xc = cp.asarray(x)
    yc = cp.asarray(y)

    for step in range(20):
        loss_np = float(nc.train_step(np_model, opt_np, x, y))
        loss_cp = float(cp.asnumpy(cc.train_step(cp_model, opt_cp, xc, yc)))
        assert abs(loss_np - loss_cp) < 1e-4, f"Paso {step}: np={loss_np}, cp={loss_cp}"


@pytest.mark.parametrize("name,mk,tk", [
    ("CNN-1 Baseline", "cnn1", tm.cnn1),
    ("CNN-2 Profunda", "cnn2", tm.cnn2),
    ("CNN-3 Regularizada", "cnn3", tm.cnn3)
])
def test_cupy_vs_torch(name, mk, tk):
    """Verificación de CuPy contra PyTorch."""
    try:
        import cupy as cp
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")

    maker = getattr(cc, mk)
    torch.manual_seed(0)
    tmodel = tk()
    mine = maker()

    tl = [m for m in tmodel if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
    ml = [l for l in mine.layers if l.params]
    assert len(tl) == len(ml)
    for t, m in zip(tl, ml):
        if isinstance(t, nn.Conv2d):
            m.params["W"][...] = cp.asarray(t.weight.detach().numpy())
            m.params["b"][...] = cp.asarray(t.bias.detach().numpy())
        elif isinstance(t, nn.Linear):
            m.params["W"][...] = cp.asarray(t.weight.detach().numpy().T)
            m.params["b"][...] = cp.asarray(t.bias.detach().numpy())
        else:
            m.params["gamma"][...] = cp.asarray(t.weight.detach().numpy())
            m.params["beta"][...] = cp.asarray(t.bias.detach().numpy())

    no_dropout(tmodel, mine)
    tmodel.train()
    rng = np.random.default_rng(1)
    x = rng.random((8, 3, 64, 64), dtype=np.float32)
    y = rng.integers(0, 29, 8)

    xt = torch.from_numpy(x)
    out_t = tmodel(xt)
    loss_t = nn.functional.cross_entropy(out_t, torch.from_numpy(y))
    loss_t.backward()

    xc = cp.asarray(x)
    yc = cp.asarray(y)
    logits = mine.forward(xc, train=True)
    loss, d = cc.softmax_cross_entropy(logits, yc)
    mine.backward(d)

    assert np.abs(cp.asnumpy(logits) - out_t.detach().numpy()).max() < 1e-4
    assert abs(float(cp.asnumpy(loss)) - loss_t.item()) < 1e-4
    for t, m in zip(tl, ml):
        if isinstance(t, nn.Conv2d):
            pairs = [(cp.asnumpy(m.grads["W"]), t.weight.grad.numpy()), (cp.asnumpy(m.grads["b"]), t.bias.grad.numpy())]
        elif isinstance(t, nn.Linear):
            pairs = [(cp.asnumpy(m.grads["W"]), t.weight.grad.T.numpy()), (cp.asnumpy(m.grads["b"]), t.bias.grad.numpy())]
        else:
            pairs = [(cp.asnumpy(m.grads["gamma"]), t.weight.grad.numpy()), (cp.asnumpy(m.grads["beta"]), t.bias.grad.numpy())]
            assert np.allclose(cp.asnumpy(m.buffers["mean"]), t.running_mean.numpy(), atol=1e-5)
            assert np.allclose(cp.asnumpy(m.buffers["var"]), t.running_var.numpy(), rtol=1e-4, atol=1e-5)
        for g, gt in pairs:
            rel = np.abs(g - gt).max() / (np.abs(gt).max() + 1e-12)
            if gt.shape == m.params.get("b", gt).shape and np.abs(gt).max() < 1e-5:
                continue
            assert rel < 1e-3, (name, type(t).__name__, gt.shape, rel)


@pytest.mark.skipif(not (ROOT / "data" / "cache" / "train_X.npy").exists(),
                    reason="falta data/cache (correr src/scratch/cache_data.py)")
def test_cupy_learns():
    """CNN-1 en CuPy supera 75 % de accuracy en test tras 5 épocas (<5 s en CuPy)."""
    try:
        import cupy as cp
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")

    from scratch.cache_data import load_cached
    X, y = load_cached("train")
    Xt, yt = load_cached("test")
    model = cc.cnn1(42)
    opt = cc.Adam(model.params())
    rng = np.random.RandomState(42)
    for _ in range(5):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb = cp.asarray(X[idx])
            yb = cp.asarray(y[idx])
            cc.train_step(model, opt, xb, yb)
    pred = cc.predict(model, Xt)
    acc = float((pred == yt).mean())
    assert acc > 0.75, f"Accuracy insuficiente: {acc}"


@pytest.mark.skipif(not (ROOT / "data" / "cache" / "train_X.npy").exists(),
                    reason="falta data/cache (correr src/scratch/cache_data.py)")
def test_learns():
    """CNN-1 supera 75 % de accuracy en test tras 5 épocas (~8 min en NumPy)."""
    from scratch.cache_data import load_cached
    X, y = load_cached("train")
    Xt, yt = load_cached("test")
    model = nc.cnn1(42)
    opt = nc.Adam(model.params())
    rng = np.random.RandomState(42)
    for _ in range(5):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            nc.train_step(model, opt, X[idx], y[idx])
    assert (nc.predict(model, Xt) == yt).mean() > 0.75


# ----------------------------------------------------------------------------- kernels CUDA propios

def _run_both(cc, make_layer, x, dout_fn):
    """Corre una capa con los kernels propios y con operaciones de CuPy; devuelve (y, dx) de cada una."""
    import cupy as cp
    out = []
    for flag in (True, False):
        cc.USE_KERNELS = flag
        try:
            layer = make_layer()
            y = layer.forward(cp.asarray(x), True)
            dx = layer.backward(cp.asarray(dout_fn(y.shape)))
            out.append((cp.asnumpy(y), cp.asnumpy(dx)))
        finally:
            cc.USE_KERNELS = True
    return out


@pytest.mark.parametrize("cin,cout,k,h", [(3, 8, 3, 16), (4, 6, 5, 12), (2, 5, 3, 7)])
def test_kernel_conv_vs_cupy_ops(cin, cout, k, h):
    try:
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, cin, h, h)).astype(np.float32)
    g = rng.standard_normal((5, cout, h, h)).astype(np.float32)
    (yk, dk), (yr, dr) = _run_both(cc, lambda: cc.Conv2D(cin, cout, k, np.random.default_rng(1)), x,
                                   lambda s: g)
    np.testing.assert_allclose(yk, yr, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(dk, dr, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("make", [lambda cc: cc.ReLU(), lambda cc: cc.MaxPool2()])
def test_kernel_relu_maxpool_vs_cupy_ops(make):
    try:
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 4, 10, 14)).astype(np.float32)
    x[0, 0, :2, :2] = 1.0                       # empate dentro de una ventana 2x2: gana la primera
    g = rng.standard_normal((3, 4, 10, 14)).astype(np.float32)
    (yk, dk), (yr, dr) = _run_both(cc, lambda: make(cc), x,
                                   lambda s: g if len(s) == 4 and s[2] == 10 else g[:, :, :5, :7])
    np.testing.assert_array_equal(yk, yr)
    np.testing.assert_array_equal(dk, dr)


def test_kernel_adam_vs_cupy_ops():
    try:
        import cupy as cp
        from scratch import cupy_cnn as cc
    except Exception as e:
        pytest.skip(f"CuPy no disponible: {e}")
    rng = np.random.default_rng(3)
    p0 = rng.standard_normal(1000).astype(np.float32)
    grads = [rng.standard_normal(1000).astype(np.float32) for _ in range(10)]
    res = []
    for flag in (True, False):
        cc.USE_KERNELS = flag
        try:
            p, g = cp.asarray(p0.copy()), cp.zeros(1000, dtype=cp.float32)
            opt = cc.Adam([(p, g)])
            for gi in grads:
                g[...] = cp.asarray(gi)
                opt.step()
            res.append(cp.asnumpy(p))
        finally:
            cc.USE_KERNELS = True
    np.testing.assert_allclose(res[0], res[1], atol=1e-6, rtol=1e-5)
