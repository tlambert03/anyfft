import pytest
import numpy as np
import numpy.testing as npt
from scipy import fftpack, signal
from skimage.data import grass
import anyfft
from anyfft._reikna._util import to_device, empty_like

GRASS = grass().astype("float32")
KERNEL = np.outer(
    signal.windows.gaussian(70, 8), signal.windows.gaussian(70, 8)
).astype("float32")


def test_scipy_fft():
    scipy_fgrass = fftpack.fftn(GRASS)
    scipy_ifgrass = fftpack.ifftn(scipy_fgrass)

    assert not np.allclose(scipy_fgrass, GRASS, atol=100)
    assert scipy_fgrass.dtype == np.complex64
    assert scipy_fgrass.shape == GRASS.shape
    # inverse
    npt.assert_allclose(scipy_ifgrass.real, GRASS, atol=1e-4)


def test_fft():
    gpu_fgrass = anyfft.fftn(GRASS)
    fgrass = gpu_fgrass.get()

    assert not np.allclose(fgrass, GRASS, atol=100)
    npt.assert_allclose(fgrass, fftpack.fftn(GRASS), atol=2e-1)

    gpu_ifgrss = anyfft.ifftn(gpu_fgrass)
    ifgrss = gpu_ifgrss.get()
    npt.assert_allclose(ifgrss.real, GRASS, atol=1e-4)


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
    input = to_device(GRASS.astype(np.complex64))
    out = empty_like(input)
    anyfft.fftn(input, out)
    npt.assert_allclose(out.get(), fftpack.fftn(GRASS), atol=2e-1)


def test_fft_inplace():
    input = to_device(GRASS.astype(np.complex64))
    anyfft.fftn(input, inplace=True)
    npt.assert_allclose(input.get(), fftpack.fftn(GRASS), atol=2e-1)


def test_fft_errors():

    with pytest.raises(TypeError):
        # existing OCLArray must be of complex type
        anyfft.fftn(to_device(GRASS))

    anyfft.fftn(to_device(GRASS.astype(np.complex64)))
    anyfft.fftn(GRASS)

    input = to_device(GRASS.astype(np.complex64))
    out = empty_like(input)

    with pytest.raises(ValueError):
        # cannot provide both output and inplace
        anyfft.fftn(input, out, inplace=True)

    with pytest.raises(ValueError):
        # cannot use inplace with numpy array
        anyfft.fftn(GRASS, inplace=True)


def test_fftshift():
    scp_shift = fftpack.fftshift(GRASS)
    shift = anyfft.fftshift(GRASS).get()
    npt.assert_allclose(scp_shift, shift)


def test_fftconvolve_same():
    out = anyfft.fftconvolve(GRASS, KERNEL, mode="same").get()
    scp_out = signal.fftconvolve(GRASS, KERNEL, mode="same")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)


def test_fftconvolve_from_oclarray():
    out = anyfft.fftconvolve(GRASS, KERNEL, mode="same")
    out2 = anyfft.fftconvolve(to_device(GRASS.astype("complex64")), KERNEL, mode="same")
    npt.assert_allclose(out.get(), out2.get(), atol=0.2)


def test_fftconvolve_full():
    out = anyfft.fftconvolve(GRASS, KERNEL, mode="full").get()
    scp_out = signal.fftconvolve(GRASS, KERNEL, mode="full")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)


def test_fftconvolve_valid():
    out = anyfft.fftconvolve(GRASS, KERNEL, mode="valid").get()
    scp_out = signal.fftconvolve(GRASS, KERNEL, mode="valid")
    assert out.shape == scp_out.shape
    assert out.dtype == scp_out.dtype
    npt.assert_allclose(out, scp_out, atol=0.2)
