import numpy as np
import numpy.testing as npt
import pytest
from scipy import fftpack, misc, signal

import anyfft
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
    npt.assert_allclose(scipy_ifface.real, FACE, atol=1e-4)


def test_fft():
    gpu_fface = anyfft.fftn(FACE)
    fface = gpu_fface.get()

    assert not np.allclose(fface, FACE, atol=100)
    npt.assert_allclose(fface, fftpack.fftn(FACE), atol=2e-1)

    gpu_ifgrss = anyfft.ifftn(gpu_fface)
    ifgrss = gpu_ifgrss.get()
    npt.assert_allclose(ifgrss.real, FACE, atol=1e-4)


def test_fft3d():
    img = np.random.rand(128, 128, 128)
    gpu_fimg = anyfft.fftn(img)
    fimg = gpu_fimg.get()

    assert not np.allclose(fimg, img, atol=100)
    npt.assert_allclose(fimg, fftpack.fftn(img), atol=2e-1)

    gpu_ifgrss = anyfft.ifftn(gpu_fimg)
    ifgrss = gpu_ifgrss.get()
    npt.assert_allclose(ifgrss.real, img, atol=1e-4)


def test_fft_output_array():
    input = to_device(FACE.astype(np.complex64))
    out = empty_like(input)
    anyfft.fftn(input, out)
    npt.assert_allclose(out.get(), fftpack.fftn(FACE), atol=2e-1)


def test_fft_inplace():
    input = to_device(FACE.astype(np.complex64))
    anyfft.fftn(input, inplace=True)
    npt.assert_allclose(input.get(), fftpack.fftn(FACE), atol=2e-1)


def test_fft_errors():

    with pytest.raises(TypeError):
        # existing OCLArray must be of complex type
        anyfft.fftn(to_device(FACE))

    anyfft.fftn(to_device(FACE.astype(np.complex64)))
    anyfft.fftn(FACE)

    input = to_device(FACE.astype(np.complex64))
    out = empty_like(input)

    with pytest.raises(ValueError):
        # cannot provide both output and inplace
        anyfft.fftn(input, out, inplace=True)

    with pytest.raises(ValueError):
        # cannot use inplace with numpy array
        anyfft.fftn(FACE, inplace=True)


def test_fftshift():
    scp_shift = fftpack.fftshift(FACE)
    shift = anyfft.fftshift(FACE).get()
    npt.assert_allclose(scp_shift, shift)


def test_fftconvolve_same():
    out = anyfft.fftconvolve(FACE, KERNEL, mode="same").get()
    scp_out = signal.fftconvolve(FACE, KERNEL, mode="same")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)


def test_fftconvolve_from_oclarray():
    out = anyfft.fftconvolve(FACE, KERNEL, mode="same")
    out2 = anyfft.fftconvolve(to_device(FACE.astype("complex64")), KERNEL, mode="same")
    npt.assert_allclose(out.get(), out2.get(), atol=0.2)


def test_fftconvolve_full():
    out = anyfft.fftconvolve(FACE, KERNEL, mode="full").get()
    scp_out = signal.fftconvolve(FACE, KERNEL, mode="full")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)


def test_fftconvolve_valid():
    out = anyfft.fftconvolve(FACE, KERNEL, mode="valid").get()
    scp_out = signal.fftconvolve(FACE, KERNEL, mode="valid")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)
