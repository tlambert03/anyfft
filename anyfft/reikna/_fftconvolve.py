import numpy as np
from pyopencl.elementwise import ElementwiseKernel

from ._fft import fftn, ifftn
from ._util import THREAD, empty, is_cluda_array, to_device

_mult_complex = ElementwiseKernel(
    THREAD._context,
    "cfloat_t *a, cfloat_t * b",
    "a[i] = cfloat_mul(a[i], b[i])",
    "mult",
)


def _fix_shape(arr, shape):
    if arr.shape == shape:
        return arr
    if is_cluda_array(arr):
        result = empty(shape, arr.dtype)
        result[:] = 0
    if isinstance(arr, np.ndarray):
        result = np.zeros(shape, dtype=arr.dtype)
    result[tuple(slice(i) for i in arr.shape)] = arr
    return result


def fftconvolve(
    data,
    kernel,
    mode="full",
    axes=None,
    output_arr=None,
    inplace=False,
    kernel_is_fft=False,
):
    if mode not in {"valid", "same", "full"}:
        raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")
    if data.ndim != kernel.ndim:
        raise ValueError("data and kernel should have the same dimensionality")

    # expand arrays
    s1 = data.shape
    s2 = kernel.shape
    axes = tuple(range(len(s1))) if axes is None else tuple(axes)
    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(data.ndim)
    ]
    data = _fix_shape(data, shape)
    kernel = _fix_shape(kernel, shape)

    if data.shape != kernel.shape:
        raise ValueError("in1 and in2 must have the same shape")

    data_g = to_device(data.astype(np.complex64))
    kernel_g = to_device(kernel.astype(np.complex64))
    result_g = data_g if inplace else data_g.copy()

    if not kernel_is_fft:
        kern_g = kernel_g.copy()
        fftn(kern_g, inplace=True)
    else:
        kern_g = kernel_g

    fftn(result_g, inplace=True, axes=axes)
    _mult_complex(result_g, kern_g)
    ifftn(result_g, inplace=True, axes=axes)

    _out = result_g.real if np.isrealobj(data) else result_g

    if mode == "same":
        return _crop_centered(_out, s1)
    elif mode == "valid":
        shape_valid = [
            _out.shape[a] if a not in axes else s1[a] - s2[a] + 1
            for a in range(_out.ndim)
        ]
        return _crop_centered(_out, shape_valid)
    return _out


def _crop_centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2

    endind = startind + newshape
    myslice = (slice(startind[k], endind[k]) for k in range(len(endind)))
    return arr[tuple(myslice)]
