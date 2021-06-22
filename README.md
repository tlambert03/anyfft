# anyfft

[![License](https://img.shields.io/pypi/l/anyfft.svg?color=green)](https://github.com/tlambert03/anyfft/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/anyfft.svg?color=green)](https://pypi.org/project/anyfft)
[![Python Version](https://img.shields.io/pypi/pyversions/anyfft.svg?color=green)](https://python.org)
[![tests](https://github.com/tlambert03/anyfft/workflows/tests/badge.svg)](https://github.com/tlambert03/anyfft/actions)
[![codecov](https://codecov.io/gh/tlambert03/anyfft/branch/master/graph/badge.svg)](https://codecov.io/gh/tlambert03/anyfft)

anyfft is a thin compatibility layer that wraps various python FFT implementations in the `scipy.fft` API (if they do not already provide it).  my motiviation is to have a package that I can install anywhere that will take advantage of the available hardware (CUDA, OpenCL, CPU fallback) without refactoring.


```python
import anyfft
import numpy as np

array = np.random.rand(128, 128)
anyfft.fft(array, plugin='reikna')
# alternative, use namespace directly:
anyfft.reikna.fft(array)
```

### current backends:

- numpy (`np.fft`)
- scipy (`scipy.fft`)
- fftpack (`scipy.fftpack`)
- pyfftw
- reikna (OpenCL FFT)
- cupy (CUDA FFT)


### available functions:

fft, fft2, fftn, ifft, ifft2, ifftn, fftshift, ifftshift, irfft (reikna WIP), irfft2 (reikna WIP), irfftn (reikna WIP),
rfft, rfft2, rfftn

... functions are tested to match the output of the corresponding `scipy.fft` function
