"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from functools import reduce
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _named_params(value: object, prefix: str = ""):
    if isinstance(value, Parameter):
        return [(prefix.rstrip("."), value)]
    if isinstance(value, Module):
        return _named_params(value.__dict__, prefix)
    if isinstance(value, dict):
        params = []
        for k, v in value.items():
            if k.startswith("_"):
                continue
            params += _named_params(v, f"{prefix}{k}.")
        return params
    if isinstance(value, (list, tuple)):
        params = []
        for i, v in enumerate(value):
            params += _named_params(v, f"{prefix}{i}.")
        return params
    return []

def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def named_parameters(self):
        """Return (name, parameter) pairs for serialization and debugging."""
        return _named_params(self.__dict__)

    def state_dict(self):
        """Return a NumPy-backed snapshot of all trainable parameters."""
        return {name: param.numpy().copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict):
        """Load parameters from a mapping produced by state_dict()."""
        for name, param in self.named_parameters():
            if name not in state_dict:
                raise KeyError(f"missing parameter {name}")
            param.data = Tensor(state_dict[name], device=param.device, dtype=param.dtype, requires_grad=False)

    def save(self, path):
        """Persist module parameters to an .npz file."""
        np.savez(path, **self.state_dict())

    def load(self, path):
        """Load module parameters from an .npz file."""
        with np.load(path) as state:
            self.load_state_dict({key: state[key] for key in state.files})

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            device=device, 
            dtype=dtype,
          )
        )
        self.bias = Parameter(
          init.kaiming_uniform(
            fan_in=out_features,
            fan_out=1,
            device=device, 
            dtype=dtype,
          ).reshape((1, out_features))
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight # (n, out_features)

        if self.bias is not None:
          y += ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # TODO
        ### BEGIN YOUR SOLUTION
        flattened_dim = reduce(lambda a, b: a * b, X.shape[1:])
        return X.reshape((X.shape[0], flattened_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: List["Module"]):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch_size, num_classes = logits.shape
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        logsumexp = ops.logsumexp(logits, axes=(1,))
        correct_logits = (logits * y_one_hot).sum(axes=(1,))
        return (logsumexp - correct_logits).sum() / batch_size
        
class CrossEntrophyLoss(Module):
    def forward(self,logits: Tensor, y: Tensor):
        return SoftmaxLoss()(logits, y)
        
class BinaryCrossEntrophyLoss(Module):
    def forward(self,logits: Tensor, y: Tensor):
        probs = 1 / (1 + ops.exp(-logits))
        loss = -(y * ops.log(probs) + (1 - y) * ops.log(1 - probs))
        return loss.sum() / reduce(lambda a, b: a * b, loss.shape)
        
class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor):
        diff = input - target
        return (diff * diff).sum() / reduce(lambda a, b: a * b, diff.shape)
        
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        shape = (1, self.dim)
        if self.training:
            mean = x.sum(axes=(0,)) / batch_size
            centered = x - mean.reshape(shape).broadcast_to(x.shape)
            var = (centered * centered).sum(axes=(0,)) / batch_size
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var).detach()
        else:
            mean = self.running_mean
            var = self.running_var
            centered = x - mean.reshape(shape).broadcast_to(x.shape)
        denom = ((var + self.eps) ** 0.5).reshape(shape).broadcast_to(x.shape)
        norm = centered / denom
        return norm * self.weight.reshape(shape).broadcast_to(x.shape) + self.bias.reshape(shape).broadcast_to(x.shape)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.sum(axes=(1,)).reshape((x.shape[0], 1)) / self.dim
        centered = x - mean.broadcast_to(x.shape)
        var = (centered * centered).sum(axes=(1,)).reshape((x.shape[0], 1)) / self.dim
        norm = centered / ((var + self.eps) ** 0.5).broadcast_to(x.shape)
        shape = (1, self.dim)
        return norm * self.weight.reshape(shape).broadcast_to(x.shape) + self.bias.reshape(shape).broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        if self.p >= 1:
            return init.zeros(*x.shape, device=x.device, dtype=x.dtype)
        mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
        return x * mask / (1 - self.p)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
