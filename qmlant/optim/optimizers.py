from __future__ import annotations

import math
import numpy as np


# =============================================================================
# Optimizer (base class)
# =============================================================================
class Optimizer:
    def __init__(self):
        self.xp = np  # numpy, cupy, torch, ...
        self.hooks = []

    def setup(self, xp):
        self.xp = xp
        return self

    def update(self, params, grads) -> None:
        for f in self.hooks:
            f(params)

        self.update_one(params, grads)

    def update_one(self, params, grads) -> None:
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


# =============================================================================
# SGD / MomentumSGD / AdaGrad / AdaDelta / Adam
# =============================================================================
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, params, grads):
        params -= self.lr * grads


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, params, grads):
        v_key = "MomentumSGD"
        if v_key not in self.vs:
            self.vs[v_key] = self.xp.zeros_like(params)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * grads
        params += v


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, params, grads):
        h_key = "AdaGrad"
        if h_key not in self.hs:
            self.hs[h_key] = self.xp.zeros_like(params)

        lr = self.lr
        eps = self.eps
        grad = grads
        h = self.hs[h_key]

        h += grad * grad
        params -= lr * grad / (self.xp.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, params, grads):
        key = "AdaDelta"
        if key not in self.msg:
            self.msg[key] = self.xp.zeros_like(params)
            self.msdx[key] = self.xp.zeros_like(params)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = grads

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = self.xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        params -= dx


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1.0 - math.pow(self.beta1, self.t)
        fix2 = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, params, grads):
        key = "Adam"
        if key not in self.ms:
            self.ms[key] = self.xp.zeros_like(params)
            self.vs[key] = self.xp.zeros_like(params)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = grads

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        params -= self.lr * m / (self.xp.sqrt(v) + eps)
