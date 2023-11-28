from __future__ import annotations

from typing import TYPE_CHECKING, Union
from warnings import filterwarnings

import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT

from ._util import get_context

filterwarnings("ignore", module="pyopencl")


if TYPE_CHECKING:
    from reikna.cluda.cuda import Array as cudaArray
    from reikna.cluda.ocl import Array as oclArray

    Array = Union[cudaArray, oclArray]

context = get_context()
queue = cl.CommandQueue(context)

#  plan cache
_PLAN_CACHE = {}


def _normalize_axes(dshape, axes):
    """Convert possibly negative axes to positive axes."""
    if axes is None:
        return None
    _axes = [axes] if np.isscalar(axes) else list(axes)
    try:
        return tuple(np.arange(len(dshape))[_axes])
    except Exception as e:
        raise TypeError(f"Cannot normalize axes {axes}: {e}")


def _get_fft_plan(arr, axes=None, fast_math=False):
    """Cache and return a reikna FFT plan suitable for `arr` type and shape."""
    axes = _normalize_axes(arr.shape, axes)
    plan_key = (arr.shape, arr.dtype, axes, fast_math)

    if plan_key not in _PLAN_CACHE:
        _PLAN_CACHE[plan_key] = FFT(context, queue, arr, axes=axes, fast_math=fast_math)

    return _PLAN_CACHE[plan_key]


def _fftn(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, ...] | None = None,
    inplace: bool = False,
    fast_math: bool = True,
    *,
    _inverse: bool = False,
) -> Array:
    """Perform fast Fourier transformation on `input_array`.

    Parameters
    ----------
    input_arr : numpy or OCL array
        A numpy or OCL array to transform.  If an OCL array is provided, it must already
        be of type `complex64`.  If a numpy array is provided, it will be converted
        to `float32` before the transformation is performed.
    output_arr : numpy or OCL array, optional
        An optional array/buffer to use for output, by default None
    axes : tuple of int, optional
        T tuple with axes over which to perform the transform.
        If not given, the transform is performed over all the axes., by default None
    inplace : bool, optional
        Whether to place output data in the `input_arr` buffer, by default False
    fast_math : bool, optional
        Whether to enable fast (less precise) mathematical operations during
        compilation, by default True
    _inverse : bool, optional
        Perform inverse FFT, by default False.  (prefer using `ifftn`)

    Returns
    -------
    OCLArray
        result of transformation (still on GPU). Use `.get()` or `cle.pull`
        to retrieve from GPU.
        If `inplace` or  `output_arr` where used, data will also be placed in
        the corresponding buffer as a side effect.

    Raises
    ------
    TypeError
        If OCL array is provided that is not of type complex64.  Or if an unrecognized
        array is provided.
    ValueError
        If inplace is used for numpy array, or both `output_arr` and `inplace` are used.
    """
    if output_arr is not None and inplace:
        raise ValueError("`output_arr` cannot be provided if `inplace` is True")
    assert input_arr.dtype in (np.float32, np.float64, np.complex64, np.complex128)

    if not np.iscomplexobj(input_arr):
        input_arr = input_arr.astype(np.complex64)  # TODO

    _input_array = (
        cla.to_device(queue, input_arr)
        if isinstance(input_arr, np.ndarray)
        else input_arr
    )
    transform = _get_fft_plan(_input_array, axes=axes, fast_math=fast_math)

    if not inplace:
        if output_arr is None:
            output_arr = cla.empty_like(_input_array)
        transform.result = output_arr

    (event,) = transform.enqueue(forward=not _inverse)
    event.wait()

    if not inplace:
        return output_arr
    return _input_array


def fft(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: int = -1,
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    return fftn(input_arr, output_arr, (axes,), inplace, fast_math)


def ifft(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: int = -1,
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    return ifftn(input_arr, output_arr, (axes,), inplace, fast_math)


def fft2(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, int] = (-2, -1),
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    return fftn(input_arr, output_arr, axes, inplace, fast_math)


def ifft2(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, int] = (-2, -1),
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    return ifftn(input_arr, output_arr, axes, inplace, fast_math)


def fftn(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, ...] | None = None,
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    return _fftn(input_arr, output_arr, axes, inplace, fast_math)


def ifftn(
    input_arr,
    output_arr=None,
    axes=None,
    inplace=False,
    fast_math=False,
):
    return _fftn(input_arr, output_arr, axes, inplace, fast_math, _inverse=True)


def rfft(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: int = -1,
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    x = _fftn(input_arr, output_arr, (axes,), inplace, fast_math)
    return x[:, : input_arr.shape[-1] // 2 + 1]


# FIXME
# def irfft(
#     input_arr: np.ndarray | Array,
#     output_arr: np.ndarray | Array = None,
#     axes: int = -1,
#     inplace: bool = False,
#     fast_math: bool = True,
# ) -> Array:
#     x = _fftn(input_arr, output_arr, axes, inplace, fast_math, _inverse=True)
#     shp = list(input_arr.shape)
#     n = shp[axes]
#     shp[axes] = 2 * n - 2
#     result = empty(shp, np.float32)
#     result[..., :n] = x.real
#     result[..., n - 1 :] = x.real[..., 1:][::-1]
#     return result.astype(np.float64)


def rfft2(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, int] = (-2, -1),
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    x = _fftn(input_arr, output_arr, axes, inplace, fast_math)
    return x[:, : input_arr.shape[1] // 2 + 1]


# FIXME
# def irfft2(
#     input_arr: np.ndarray | Array,
#     output_arr: np.ndarray | Array = None,
#     axes: Tuple[int, int] = (-2, -1),
#     inplace: bool = False,
#     fast_math: bool = True,
# ) -> Array:
#     x = _fftn(input_arr, output_arr, axes, inplace, fast_math)
#     return x[:, : input_arr.shape[1] // 2 + 1]


def rfftn(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, ...] | None = None,
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    x = _fftn(input_arr, output_arr, axes, inplace, fast_math)
    return x[:, : input_arr.shape[1] // 2 + 1]


# FIXME
# def irfftn(
#     input_arr: np.ndarray | Array,
#     output_arr: np.ndarray | Array = None,
#     axes: tuple[int, ...] | None = None,
#     inplace: bool = False,
#     fast_math: bool = True,
# ) -> Array:
#     x = _fftn(input_arr, output_arr, axes, inplace, fast_math)
#     return x[..., : input_arr.shape[1] // 2 + 1]
