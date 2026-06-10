import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(6.0 / fan_in)
    shape = shape or (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2.0 / fan_in)
    shape = shape or (fan_in, fan_out)
    return randn(*shape, mean=0.0, std=std, **kwargs)
