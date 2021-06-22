import cupy as cp
from functools import wraps
import numpy as np

# TODO: improve performance
def np2xp(func):
    @wraps(func)
    def _inner(*args, **kwargs):
        _args = (cp.array(i) if isinstance(i, np.ndarray) else i for i in args)
        return func(*_args, **kwargs)

    return _inner


def __getattr__(name):
    return np2xp(getattr(cp.fft, name))
