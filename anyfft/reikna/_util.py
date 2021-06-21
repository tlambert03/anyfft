import reikna.cluda as cluda

api = cluda.ocl_api()
# api = cluda.cuda_api()
THREAD = api.Thread.create()


def to_device(ary):
    return ary if is_cluda_array(ary) else THREAD.to_device(ary)


def empty(shape, dtype):
    return THREAD.array(shape, dtype)


def empty_like(arr, dtype=None):
    return THREAD.array(
        arr.shape,
        arr.dtype if dtype is None else dtype,
        strides=getattr(arr, "strides", None) if dtype is None else None,
        offset=getattr(arr, "offset", 0) if dtype is None else 0,
        nbytes=getattr(arr, "nbytes", None) if dtype is None else None,
        allocator=getattr(arr, "allocator", None),
    )


def is_cluda_array(obj):
    return type(obj).__name__ == "Array" and type(obj).__module__.startswith("reikna")
