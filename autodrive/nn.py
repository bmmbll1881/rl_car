"""A tiny numpy MLP with hand-written backprop and Adam.

Why not PyTorch?  The point of this project is *debuggability*: every gradient
that moves the policy is visible in ~100 lines of code, the whole training run
fits on a bare ``python + numpy`` install, and a numerical gradient check
(``tests/test_gradients.py``) can validate each analytic derivative.

The MLP is deliberately boring: ``Linear -> tanh`` blocks and a linear output
layer.  Actors add a softmax head on top, critics use a single output unit.
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class Adam:
    """Adam with bias correction; operates in-place on a list of arrays."""

    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, max_grad_norm: float | None = None):
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.max_grad_norm = max_grad_norm
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def _clip(self, grads):
        if self.max_grad_norm is None:
            return grads
        total = np.sqrt(sum(float((g * g).sum()) for g in grads))
        if total > self.max_grad_norm and total > 0:
            scale = self.max_grad_norm / (total + 1e-12)
            return [g * scale for g in grads]
        return grads

    def step(self, grads):
        grads = self._clip(grads)
        self.t += 1
        b1, b2 = self.b1, self.b2
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            mhat = m / (1.0 - b1 ** self.t)
            vhat = v / (1.0 - b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class MLP:
    """Fully connected network: [in, h1, ..., out] with tanh/relu hidden layers."""

    def __init__(self, sizes, hidden: str = "tanh", seed: int = 0, out_scale: float = 1.0):
        sizes = [int(s) for s in sizes]
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.hidden = hidden
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            n_in, n_out = sizes[i], sizes[i + 1]
            if hidden == "tanh":
                limit = np.sqrt(6.0 / (n_in + n_out))
            else:  # He init for relu
                limit = np.sqrt(6.0 / n_in)
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        if out_scale != 1.0:
            self.W[-1] *= out_scale
            self.b[-1] *= out_scale
        self.params = self.W + self.b   # Adam updates these in place
        self._acts = None

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        acts = [x]
        h = x
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < last:
                h = np.tanh(z) if self.hidden == "tanh" else np.maximum(z, 0.0)
            else:
                h = z
            acts.append(h)
        self._acts = acts
        return h[0] if single else h

    def __call__(self, x):
        return self.forward(x)

    # ----------------------------------------------------------------- backward
    def backward(self, dout):
        """Backprop ``dout`` = dLoss/d(output).  Returns grads ordered [dW..., db...]."""
        if self._acts is None:
            raise RuntimeError("backward() called before forward()")
        acts = self._acts
        n = len(self.W)
        dout = np.asarray(dout, dtype=float)
        if dout.ndim == 1:
            dout = dout[None, :]
        gW = [None] * n
        gb = [None] * n
        g = dout
        for i in reversed(range(n)):
            if i < n - 1:  # hidden activation derivative, using the *output* of that layer
                out = acts[i + 1]
                g = g * (1.0 - out * out) if self.hidden == "tanh" else g * (out > 0)
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = g @ self.W[i].T
        self._acts = None
        return gW + gb

    # ------------------------------------------------------------- serialisation
    def get_state(self) -> dict:
        st = {}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            st[f"W{i}"] = W.copy()
            st[f"b{i}"] = b.copy()
        st["n_layers"] = np.array(len(self.W))
        return st

    def set_state(self, state):
        n = int(state["n_layers"])
        for i in range(n):
            if self.W[i].shape != state[f"W{i}"].shape:
                raise ValueError(f"layer {i} shape mismatch: {self.W[i].shape} vs {state[f'W{i}'].shape}")
            self.W[i][...] = state[f"W{i}"]
            self.b[i][...] = state[f"b{i}"]
