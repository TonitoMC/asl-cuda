"""
CNN desde cero en NumPy (CPU): forward y backward a mano, sin autograd ni framework.

Formato NCHW, float32, padding 'same', stride 1. La convolución usa im2col: cada parche
kxk pasa a una columna y la convolución se vuelve una multiplicación de matrices.

El mismo algoritmo se porta a CuPy cambiando `np` por `cp` (ver cupy_cnn.py).

Interfaz común:
    Layer.forward(x, train), Layer.backward(dout), Layer.params / Layer.grads / Layer.buffers
    Model.forward / backward / params() -> lista de (arreglo, gradiente)
    softmax_cross_entropy(logits, y) -> (loss, dlogits)
    Adam(params).step()
"""

import numpy as np

DT = np.float32
NUM_CLASSES = 29


# ----------------------------------------------------------------------------- capas

class Layer:
    def __init__(self):
        self.params = {}    # entrenables: nombre -> arreglo
        self.grads = {}     # mismo nombre -> gradiente (se escribe en sitio en backward)
        self.buffers = {}   # no entrenables (medias/varianzas móviles de BatchNorm)

    def forward(self, x, train):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def _alloc_grads(self):
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}


def _glorot(rng, shape, fan_in, fan_out):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=shape).astype(DT)


class Conv2D(Layer):
    """Conv kxk, stride 1, padding 'same' (k impar). W: (Cout, Cin, k, k)."""

    def __init__(self, cin, cout, k, rng):
        super().__init__()
        self.cin, self.cout, self.k = cin, cout, k
        self.params = {"W": _glorot(rng, (cout, cin, k, k), cin * k * k, cout * k * k),
                       "b": np.zeros(cout, DT)}
        self._alloc_grads()

    def _im2col(self, x):
        n, c, h, w = x.shape
        k, p = self.k, self.k // 2
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        cols = np.empty((c, k * k, n, h, w), DT)
        for i in range(k):
            for j in range(k):
                cols[:, i * k + j] = xp[:, :, i:i + h, j:j + w].transpose(1, 0, 2, 3)
        return cols.reshape(c * k * k, n * h * w)

    def forward(self, x, train):
        n, _, h, w = x.shape
        self.xshape = x.shape
        self.cols = self._im2col(x)
        wm = self.params["W"].reshape(self.cout, -1)
        out = wm @ self.cols + self.params["b"][:, None]          # (Cout, N*H*W)
        return np.ascontiguousarray(out.reshape(self.cout, n, h, w).transpose(1, 0, 2, 3))

    def backward(self, dout):
        n, _, h, w = self.xshape
        k, p, c = self.k, self.k // 2, self.cin
        d = np.ascontiguousarray(dout.transpose(1, 0, 2, 3)).reshape(self.cout, -1)
        self.grads["W"][...] = (d @ self.cols.T).reshape(self.params["W"].shape)
        self.grads["b"][...] = d.sum(axis=1)
        dcols = (self.params["W"].reshape(self.cout, -1).T @ d).reshape(c, k * k, n, h, w)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), DT)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + h, j:j + w] += dcols[:, i * k + j].transpose(1, 0, 2, 3)
        self.cols = None
        return dxp[:, :, p:p + h, p:p + w] if p else dxp


class ReLU(Layer):
    def forward(self, x, train):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2(Layer):
    """MaxPool 2x2. En empate gana el primero (orden de filas), como PyTorch."""

    def forward(self, x, train):
        n, c, h, w = x.shape
        win = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5)
        win = win.reshape(n, c, h // 2, w // 2, 4)
        self.idx = win.argmax(axis=-1)
        self.shape = x.shape
        return np.take_along_axis(win, self.idx[..., None], axis=-1)[..., 0]

    def backward(self, dout):
        n, c, h, w = self.shape
        dwin = np.zeros((n, c, h // 2, w // 2, 4), DT)
        np.put_along_axis(dwin, self.idx[..., None], dout[..., None], axis=-1)
        dx = dwin.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5)
        return np.ascontiguousarray(dx).reshape(n, c, h, w)


class Flatten(Layer):
    def forward(self, x, train):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class GlobalAvgPool(Layer):
    def forward(self, x, train):
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        n, c, h, w = self.shape
        return np.broadcast_to((dout / (h * w))[:, :, None, None], self.shape).astype(DT)


class Dense(Layer):
    """W: (in, out)."""

    def __init__(self, nin, nout, rng):
        super().__init__()
        self.params = {"W": _glorot(rng, (nin, nout), nin, nout), "b": np.zeros(nout, DT)}
        self._alloc_grads()

    def forward(self, x, train):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"][...] = self.x.T @ dout
        self.grads["b"][...] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class Dropout(Layer):
    """Invertido. spatial=True apaga canales completos (Dropout2d). Semilla propia."""

    def __init__(self, p, seed, spatial=False):
        super().__init__()
        self.p, self.spatial = p, spatial
        self.rng = np.random.default_rng(seed)

    def forward(self, x, train):
        if not train or self.p == 0:
            self.mask = None
            return x
        shape = x.shape[:2] + (1, 1) if self.spatial else x.shape
        self.mask = (self.rng.random(shape, dtype=DT) >= self.p).astype(DT) / DT(1 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


class BatchNorm2D(Layer):
    """eps=1e-3, momentum=0.01 (fracción de la media nueva), como torch_models.py."""

    def __init__(self, c, eps=1e-3, momentum=0.01):
        super().__init__()
        self.eps, self.momentum = DT(eps), momentum
        self.params = {"gamma": np.ones(c, DT), "beta": np.zeros(c, DT)}
        self.buffers = {"mean": np.zeros(c, DT), "var": np.ones(c, DT)}
        self._alloc_grads()

    def forward(self, x, train):
        ax = (0, 2, 3)
        if train:
            mu = x.mean(axis=ax)
            var = x.var(axis=ax)
            m = x.shape[0] * x.shape[2] * x.shape[3]
            mo = self.momentum
            self.buffers["mean"][...] = (1 - mo) * self.buffers["mean"] + mo * mu
            self.buffers["var"][...] = (1 - mo) * self.buffers["var"] + mo * var * m / max(m - 1, 1)
        else:
            mu, var = self.buffers["mean"], self.buffers["var"]
        self.istd = (1.0 / np.sqrt(var + self.eps)).astype(DT)
        self.xhat = (x - mu[None, :, None, None]) * self.istd[None, :, None, None]
        g, b = self.params["gamma"], self.params["beta"]
        return self.xhat * g[None, :, None, None] + b[None, :, None, None]

    def backward(self, dout):
        ax = (0, 2, 3)
        m = dout.shape[0] * dout.shape[2] * dout.shape[3]
        dgamma = (dout * self.xhat).sum(axis=ax)
        dbeta = dout.sum(axis=ax)
        self.grads["gamma"][...] = dgamma
        self.grads["beta"][...] = dbeta
        k = (self.params["gamma"] * self.istd / m)[None, :, None, None]
        return k * (m * dout - dbeta[None, :, None, None] - self.xhat * dgamma[None, :, None, None])


# ----------------------------------------------------------------------------- modelo

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, dlogits):
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def params(self):
        return [(l.params[k], l.grads[k]) for l in self.layers for k in l.params]

    def count_params(self):
        n = sum(a.size for a, _ in self.params())
        return n + sum(b.size for l in self.layers for b in l.buffers.values())


def softmax_cross_entropy(logits, y):
    """Devuelve (loss media, dlogits = (softmax - onehot) / N)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p
    d[np.arange(n), y] -= 1
    return DT(loss), (d / n).astype(DT)


class Adam:
    """Igual que torch.optim.Adam (sin weight decay). Actualiza los arreglos en sitio."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for (p, g), m, v in zip(self.params, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            denom = np.sqrt(v) / np.sqrt(bc2) + self.eps
            p -= (self.lr / bc1) * m / denom


# ----------------------------------------------------------------------------- arquitecturas

def _block(cin, cout, k, rng):
    return [Conv2D(cin, cout, k, rng), BatchNorm2D(cout), ReLU(), MaxPool2()]


def cnn1(seed=42):
    rng = np.random.default_rng(seed)
    return Model([
        Conv2D(3, 32, 3, rng), ReLU(), MaxPool2(),
        Conv2D(32, 64, 3, rng), ReLU(), MaxPool2(),
        Flatten(), Dense(64 * 16 * 16, 256, rng), ReLU(), Dropout(0.5, seed + 1),
        Dense(256, NUM_CLASSES, rng)])


def cnn2(seed=42):
    rng = np.random.default_rng(seed)
    return Model([
        *_block(3, 64, 3, rng), *_block(64, 128, 3, rng), *_block(128, 256, 3, rng),
        GlobalAvgPool(), Dense(256, 512, rng), ReLU(), Dropout(0.4, seed + 1),
        Dense(512, NUM_CLASSES, rng)])


def cnn3(seed=42):
    rng = np.random.default_rng(seed)
    return Model([
        *_block(3, 64, 5, rng), Dropout(0.1, seed + 1, spatial=True),
        *_block(64, 128, 3, rng), Dropout(0.1, seed + 2, spatial=True),
        *_block(128, 256, 3, rng), Dropout(0.1, seed + 3, spatial=True),
        GlobalAvgPool(), Dense(256, 256, rng), ReLU(), Dropout(0.3, seed + 4),
        Dense(256, NUM_CLASSES, rng)])


MODELS = {"CNN-1 Baseline": cnn1, "CNN-2 Profunda": cnn2, "CNN-3 Regularizada": cnn3}


def train_step(model, opt, xb, yb):
    logits = model.forward(xb, train=True)
    loss, dlogits = softmax_cross_entropy(logits, yb)
    model.backward(dlogits)
    opt.step()
    return loss


def predict(model, X, batch=256):
    out = [model.forward(X[i:i + batch], train=False).argmax(1) for i in range(0, len(X), batch)]
    return np.concatenate(out)
