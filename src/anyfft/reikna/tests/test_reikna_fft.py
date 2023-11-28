import numpy as np
import numpy.testing as npt
import pytest
from scipy import fftpack, misc, signal

import anyfft.reikna
from anyfft.reikna._util import empty_like, to_device

FACE = misc.face(gray=True).astype("float32")
KERNEL = np.outer(
    signal.windows.gaussian(70, 8), signal.windows.gaussian(70, 8)
).astype("float32")


def test_scipy_fft():
    scipy_fface = fftpack.fftn(FACE)
    scipy_ifface = fftpack.ifftn(scipy_fface)

    assert not np.allclose(scipy_fface, FACE, atol=100)
    assert scipy_fface.dtype == np.complex64
    assert scipy_fface.shape == FACE.shape
    # inverse
    npt.assert_allclose(scipy_ifface.real, FACE, atol=0.0001)


def test_fft():
    gpu_fface = anyfft.reikna.fftn(FACE, fast_math=False)
    fface = gpu_fface.get()

    assert not np.allclose(fface, FACE, atol=100)
    npt.assert_allclose(fface, fftpack.fftn(FACE), rtol=4e-2)

    gpu_ifgrss = anyfft.ifftn(gpu_fface)
    ifgrss = gpu_ifgrss.get()
    npt.assert_allclose(ifgrss.real, FACE, atol=0.04)


def test_fft3d():
    img = np.random.rand(128, 128, 128)
    gpu_fimg = anyfft.reikna.fftn(img, fast_math=False)
    fimg = gpu_fimg.get()

    assert not np.allclose(fimg, img, atol=100)
    npt.assert_allclose(fimg, fftpack.fftn(img), rtol=4e-2)

    gpu_ifgrss = anyfft.ifftn(gpu_fimg)
    ifgrss = gpu_ifgrss.get()
    npt.assert_allclose(ifgrss.real, img, atol=0.001)


def test_fft_output_array():
    _input = to_device(FACE.astype(np.complex64))
    out = empty_like(_input)
    anyfft.reikna.fftn(_input, out, fast_math=False)
    npt.assert_allclose(out.get(), fftpack.fftn(FACE), rtol=4e-2)


def test_fft_inplace():
    _input = to_device(FACE.astype(np.complex64))
    anyfft.reikna.fftn(_input, inplace=True, fast_math=False)
    npt.assert_allclose(_input.get(), fftpack.fftn(FACE), rtol=4e-2)


def test_fft_errors():
    with pytest.raises(TypeError):
        # existing OCLArray must be of complex type
        anyfft.reikna.fftn(to_device(FACE))

    anyfft.reikna.fftn(to_device(FACE.astype(np.complex64)))
    anyfft.reikna.fftn(FACE)

    _input = to_device(FACE.astype(np.complex64))
    out = empty_like(_input)

    with pytest.raises(ValueError):
        # cannot provide both output and inplace
        anyfft.reikna.fftn(_input, out, inplace=True)

    with pytest.raises(ValueError):
        # cannot use inplace with numpy array
        anyfft.reikna.fftn(FACE, inplace=True)


def test_fftshift():
    scp_shift = fftpack.fftshift(FACE)
    shift = anyfft.reikna.fftshift(FACE)
    npt.assert_allclose(scp_shift, shift.get())

    scp_unshift = fftpack.ifftshift(scp_shift)
    unshift = anyfft.reikna.ifftshift(shift)
    npt.assert_allclose(scp_unshift, unshift.get())


def test_fftconvolve_same():
    out = anyfft.reikna.fftconvolve(FACE, KERNEL, mode="same", fast_math=False).get()
    scp_out = signal.fftconvolve(FACE, KERNEL, mode="same")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)


def test_fftconvolve_from_oclarray():
    out = anyfft.reikna.fftconvolve(FACE.astype("complex64"), KERNEL, mode="same")
    out2 = anyfft.reikna.fftconvolve(
        to_device(FACE.astype("complex64")), KERNEL, mode="same"
    )
    npt.assert_allclose(out.get(), out2.get(), atol=0.2)


def test_fftconvolve_full():
    out = anyfft.reikna.fftconvolve(FACE, KERNEL, mode="full", fast_math=False).get()
    scp_out = signal.fftconvolve(FACE, KERNEL, mode="full")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)


def test_fftconvolve_valid():
    out = anyfft.reikna.fftconvolve(FACE, KERNEL, mode="valid", fast_math=False).get()
    scp_out = signal.fftconvolve(FACE, KERNEL, mode="valid")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)
