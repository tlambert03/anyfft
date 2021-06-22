import pytest

import anyfft
import numpy as np
import numpy.testing as npt
import scipy


@pytest.fixture(scope="session")
def img():
    return scipy.misc.face(gray=True)


@pytest.fixture(
    params=[(2 ** n,) * 2 for n in range(7, 11)], scope="session", ids=lambda x: x[0]
)
def random(request):
    return np.random.random(request.param)


@pytest.fixture(scope="session")
def kernel():
    return np.outer(
        scipy.signal.windows.gaussian(70, 8), scipy.signal.windows.gaussian(70, 8)
    ).astype("float32")


def reference_func(func, img):
    return getattr(scipy.fft, func)(img)


@pytest.mark.benchmark(warmup=True)
@pytest.mark.parametrize("plugin", list(anyfft._fft._PLUGINS))
@pytest.mark.parametrize("shape", [(128, 256, 256)], ids=lambda x: x[0])
@pytest.mark.parametrize("func", ["fftn"])
def test_bench(func, plugin, shape, benchmark):
    arr = np.zeros(shape)
    benchmark(getattr(anyfft, func), arr, plugin=plugin)


@pytest.mark.parametrize("plugin", anyfft._fft._PLUGINS)
@pytest.mark.parametrize("func", ["fftn", "fft", "ifftn", "ifft", "fftshift"])
def test_accuracy(random, plugin, func):
    try:
        result = getattr(anyfft, func)(random, plugin=plugin)
    except ModuleNotFoundError as e:
        pytest.skip(str(e))
    if hasattr(result, "get"):  # FIXME: move to actual code
        result = result.get()
        if func != "fftshift":
            result = result.astype(np.complex128)
    ref = reference_func(func, random)
    assert ref.shape == result.shape
    assert ref.dtype == result.dtype
    npt.assert_allclose(ref, result, rtol=4e-2)
