from typing import Optional

from reikna import cluda
from reikna.cluda import api as api_base

_THREAD: Optional[api_base.Thread] = None


def get_thread() -> api_base.Thread:
    global _THREAD
    if _THREAD is None:
        api = cluda.ocl_api()
        _THREAD = api.Thread.create()
    return _THREAD


def to_device(ary):
    return ary if is_cluda_array(ary) else get_thread().to_device(ary)


def empty(shape, dtype):
    return get_thread().array(shape, dtype)


def empty_like(arr, dtype=None):
    return get_thread().array(
        arr.shape,
        arr.dtype if dtype is None else dtype,
        strides=getattr(arr, "strides", None) if dtype is None else None,
        offset=getattr(arr, "offset", 0) if dtype is None else 0,
        nbytes=getattr(arr, "nbytes", None) if dtype is None else None,
        allocator=getattr(arr, "allocator", None),
    )


def is_cluda_array(obj):
    return type(obj).__name__ == "Array" and type(obj).__module__.startswith("reikna")
