import pytest

import anyfft
import numpy as np
import numpy.testing as npt
import scipy


@pytest.fixture(scope="session")
def img():
    return scipy.misc.face(gray=True)


@pytest.fixture(
    params=[(2**n,) * 2 for n in range(7, 11)], scope="session", ids=lambda x: x[0]
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
@pytest.mark.parametrize("shape", [(256, 256, 256)], ids=lambda x: x[0])
@pytest.mark.parametrize("func", ["fftn"])
def test_bench(func, plugin, shape, benchmark):
    try:
        _func = getattr(anyfft, func)
        benchmark(_func, np.zeros(shape), plugin=plugin)
    except ModuleNotFoundError as e:
        pytest.xfail(str(e))


@pytest.mark.parametrize("plugin", anyfft._fft._PLUGINS)
@pytest.mark.parametrize("func", anyfft.__all__)
def test_accuracy(random, plugin, func):
    if not hasattr(anyfft, func):
        pytest.skip()
    try:
        result = getattr(anyfft, func)(random, plugin=plugin)
    except (ModuleNotFoundError, AttributeError) as e:
        pytest.xfail(str(e))
    if hasattr(result, "get"):  # FIXME: move to actual code
        result = result.get()
        if "fftshift" not in func and "irfft" not in func:
            result = result.astype(np.complex128)
    ref = reference_func(func, random)
    assert ref.shape == result.shape
    assert ref.dtype == result.dtype
    npt.assert_allclose(ref, result, rtol=4e-2)
