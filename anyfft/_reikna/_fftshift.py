from reikna.fft import FFTShift

from ._util import empty_like, to_device, THREAD


def fftshift(arr, axes=None, output_arr=None, inplace=False, thread=THREAD):
    shift = FFTShift(arr, axes=axes)
    shiftc = shift.compile(thread)

    arr_dev = to_device(arr)
    if inplace:
        res_dev = arr_dev
    else:
        res_dev = empty_like(arr_dev) if output_arr is None else output_arr
    shiftc(res_dev, arr_dev)
    return res_dev
