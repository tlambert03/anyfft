import numpy as np
import numpy.testing as npt
import pytest
from scipy import fft, misc, signal

import anyfft.fft


@pytest.fixture(scope="session")
def img():
    return misc.face(gray=True)


@pytest.fixture(
    params=[(2 ** n,) * 2 for n in range(7, 11)], scope="session", ids=lambda x: x[0]
)
def random(request):
    return np.random.random(request.param)


@pytest.fixture(scope="session")
def kernel():
    return np.outer(
        signal.windows.gaussian(70, 8), signal.windows.gaussian(70, 8)
    ).astype("float32")


def reference_fftn(img):
    return fft.fftn(img)


@pytest.mark.benchmark(warmup='on')
@pytest.mark.parametrize("plugin", list(anyfft.fft._PLUGINS))
@pytest.mark.parametrize("shape", [(256, 512, 512)], ids=lambda x: x[0])
@pytest.mark.parametrize("func", ["fftn"])
def test_bench(func, plugin, shape, benchmark):
    arr = np.zeros(shape)
    benchmark(getattr(anyfft.fft, func), arr, plugin=plugin)


@pytest.mark.parametrize("plugin", list(anyfft.fft._PLUGINS))
def test_accuracy(random, plugin):
    result = anyfft.fft.fftn(random, plugin=plugin)
    if hasattr(result, "get"):
        result = result.get().astype(np.complex128)
    ref = reference_fftn(random)
    assert ref.shape == result.shape
    assert ref.dtype == result.dtype
    npt.assert_allclose(ref, result, rtol=4e-2)
