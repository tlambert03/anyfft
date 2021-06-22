from __future__ import annotations

from typing import TYPE_CHECKING, Union
from warnings import filterwarnings

import numpy as np
import reikna.cluda as cluda
from reikna.cluda.api import Thread
from reikna.core import Type
import reikna.fft
from reikna.transformations import broadcast_const, combine_complex, split_complex

from ._util import THREAD, empty_like, is_cluda_array, to_device

filterwarnings("ignore", module="pyopencl")  # FIXME


if TYPE_CHECKING:
    from reikna.cluda.cuda import Array as cudaArray
    from reikna.cluda.ocl import Array as oclArray

    Array = Union[cudaArray, oclArray]

#  plan cache
_PLAN_CACHE = {}


def _get_fft_plan(
    arr, axes=None, fast_math=False, thread: Thread = THREAD, real_out=False
):
    """Cache and return a reikna FFT plan suitable for `arr` type and shape."""
    axes = _normalize_axes(arr.shape, axes)
    plan_key = (arr.shape, arr.dtype, axes, fast_math)

    if plan_key not in _PLAN_CACHE:
        if np.iscomplexobj(arr):
            plan = reikna.fft.FFT(arr, axes=axes)
        else:
            type_ = Type(cluda.dtypes.complex_for(arr.dtype), arr.shape)
            plan = reikna.fft.FFT(type_, axes=axes)

            # joins two real inputs into complex output
            cc = combine_complex(plan.parameter.input)
            # broadcasts 0 to the imaginary component
            bc = broadcast_const(cc.imag, 0)
            plan.parameter.input.connect(
                cc, cc.output, real_input=cc.real, imag_input=cc.imag
            )
            plan.parameter.imag_input.connect(bc, bc.output)

        # if real_out:
        #     # split complex into 2 outputs
        #     split = split_complex(plan.parameter.output)
        #     plan.parameter.output.connect(
        #         split, split.real, c_input=split.input, imag_out=split.imag
        #     )
        #     # plan.parameter.real_output.connect(split, split.real)

        _PLAN_CACHE[plan_key] = plan.compile(thread, fast_math=fast_math)

    return _PLAN_CACHE[plan_key]


def _fftn(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, ...] | None = None,
    inplace: bool = False,
    fast_math: bool = True,
    *,
    _inverse: bool = False,
    _real_out: bool = False,
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
    _isnumpy = isinstance(input_arr, np.ndarray)
    if is_cluda_array(input_arr):
        if input_arr.dtype != np.complex64:
            raise TypeError("OCLArray input_arr has to be of complex64 type")
    elif _isnumpy:
        if inplace:
            raise ValueError("inplace FFT not supported for numpy arrays")
        if input_arr.dtype not in (np.float32, np.complex64):
            input_arr = input_arr.astype(np.float32, copy=False)
    else:
        raise TypeError("input_arr must be either OCLArray or np.ndarray")
    if output_arr is not None and inplace:
        raise ValueError("`output_arr` cannot be provided if `inplace` is True")

    if _isnumpy:
        arr_dev = to_device(input_arr)
        res_dev = empty_like(input_arr, np.complex64)
    else:
        arr_dev = input_arr
        if inplace:
            res_dev = arr_dev
        else:
            res_dev = (
                empty_like(arr_dev, np.complex64) if output_arr is None else output_arr
            )

    plan = _get_fft_plan(input_arr, axes=axes, fast_math=fast_math, real_out=_real_out)
    plan(res_dev, arr_dev, inverse=_inverse)

    if _isnumpy and output_arr is not None:
        output_arr[:] = res_dev.get()
    return res_dev


def _normalize_axes(dshape, axes):
    """Convert possibly negative axes to positive axes."""
    if axes is None:
        return None
    try:
        return tuple(np.arange(len(dshape))[list(axes)])
    except Exception as e:
        raise TypeError(f"Cannot normalize axes {axes}: {e}")


def fftn(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, ...] | None = None,
    inplace: bool = False,
    fast_math: bool = True,
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
    return _fftn(input_arr, output_arr, axes, inplace, fast_math)


def rfftn(
    input_arr: np.ndarray | Array,
    output_arr: np.ndarray | Array = None,
    axes: tuple[int, ...] | None = None,
    inplace: bool = False,
    fast_math: bool = True,
) -> Array:
    x = _fftn(input_arr, output_arr, axes, inplace, fast_math, _real_out=True)
    return x[:, : input_arr.shape[1] // 2 + 1]


def ifftn(
    input_arr,
    output_arr=None,
    axes=None,
    inplace=False,
    fast_math=True,
):
    """Perform inverse Fourier transformation on `input_array`.

    Parameters
    ----------
    input_arr : numpy or OCL array
        A numpy or OCL array to transform.  If an OCL array is provided, it must already
        be of type `complex64`.  If a numpy array is provided, it will be converted
        to `float32` before the transformation is performed.
    output_arr : numpy or OCL array, optional
        An optional array/buffer to use for output, by default None.
    axes : tuple of int, optional
        T tuple with axes over which to perform the transform.
        If not given, the transform is performed over all the axes., by default None.
    inplace : bool, optional
        Whether to place output data in the `input_arr` buffer, by default False.
    fast_math : bool, optional
        Whether to enable fast (less precise) mathematical operations during
        compilation, by default True.

    Returns
    -------
    OCLArray
        result of transformation (still on GPU). Use `.get()` or `cle.pull`
        to retrieve from GPU.
        If `inplace` or  `output_arr` where used, data will also be placed in
        the corresponding buffer as a side effect.
    """
    return _fftn(input_arr, output_arr, axes, inplace, fast_math, _inverse=True)


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
