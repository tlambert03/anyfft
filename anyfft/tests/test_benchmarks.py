import numpy as np
import pytest

import anyfft.fft


@pytest.mark.benchmark(warmup=False)
@pytest.mark.parametrize("plugin", list(anyfft.fft._PLUGINS))
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("func", ["fftn"])
def test_bench(func, plugin, shape, benchmark):
    arr = np.zeros(shape)
    benchmark(getattr(anyfft.fft, func), arr, plugin=plugin)
