# anyfft

[![License](https://img.shields.io/pypi/l/anyfft.svg?color=green)](https://github.com/tlambert03/anyfft/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/anyfft.svg?color=green)](https://pypi.org/project/anyfft)
[![Python Version](https://img.shields.io/pypi/pyversions/anyfft.svg?color=green)](https://python.org)
[![tests](https://github.com/tlambert03/anyfft/workflows/tests/badge.svg)](https://github.com/tlambert03/anyfft/actions)
[![codecov](https://codecov.io/gh/tlambert03/anyfft/branch/master/graph/badge.svg)](https://codecov.io/gh/tlambert03/anyfft)

anyfft is a thin compatibility layer that wraps various python FFT implementations in the `scipy.fft` API (if they do not already provide it).  my motiviation is to have a package that I can install anywhere that will take advantage of the available hardware (CUDA, OpenCL, CPU fallback) without refactoring.
