"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if self.weight_decay:
                grad = grad + self.weight_decay * p.detach()
            key = id(p)
            if self.momentum:
                velocity = self.u.get(key, ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype))
                velocity = self.momentum * velocity + grad
                self.u[key] = velocity.detach()
                grad = velocity
            p.data = p.detach() - self.lr * grad

    def clip_grad_norm(self, max_norm=0.25):
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                g = p.grad.numpy()
                total += float((g * g).sum())
        norm = total ** 0.5
        if norm > max_norm and norm > 0:
            scale = max_norm / norm
            for p in self.params:
                if p.grad is not None:
                    p.grad = p.grad * scale
        return norm


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.u = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if self.weight_decay:
                grad = grad + self.weight_decay * p.detach()
            key = id(p)
            m = self.u.get(key, ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype))
            v = self.v.get(key, ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype))
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            self.u[key] = m.detach()
            self.v[key] = v.detach()
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data = p.detach() - self.lr * m_hat / ((v_hat ** 0.5) + self.eps)
