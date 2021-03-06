from reikna.fft import FFTShift

from ._util import empty_like, get_thread, to_device


def _fftshift(
    arr, axes=None, output_arr=None, inplace=False, *, thread=None, inverse=False
):
    shift = FFTShift(arr, axes=axes)
    shiftc = shift.compile(thread or get_thread(), fast_math=False)

    arr_dev = to_device(arr)
    if inplace:
        res_dev = arr_dev
    else:
        res_dev = empty_like(arr_dev) if output_arr is None else output_arr
    shiftc(res_dev, arr_dev, inverse=inverse)
    return res_dev


def fftshift(arr, axes=None, output_arr=None, inplace=False):
    return _fftshift(arr, axes, output_arr, inplace)


def ifftshift(arr, axes=None, output_arr=None, inplace=False):
    return _fftshift(arr, axes, output_arr, inplace, inverse=True)
