import importlib

import scipy.fft

__all__ = ["fftn", "ifftn", "fft", "ifft", "fftshift", "rfft"]
_PLUGINS = [
    "cupy",
    "reikna",
    "scipy",
    "numpy",
    "pyfftw",
    # "fftpack",
]


def _get_fft_module(plugin):
    try:
        return importlib.import_module(f"anyfft.{plugin}")
    except KeyError:
        raise ValueError(
            f"unrecognized plugin: {plugin!r}. Options include: {set(_PLUGINS)}"
        )


def _implements(scipy_func):
    def plugin_func(func, scipy_func=scipy_func):
        def _inner(*args, plugin="reikna", **kwargs):
            f = getattr(_get_fft_module(plugin), func.__name__)
            return f(*args, **kwargs)

        _inner.__doc__ == scipy_func.__doc__
        return _inner

    return plugin_func


@_implements(scipy.fft.fft)
def fft(*args, **kwargs):
    ...


@_implements(scipy.fft.ifft)
def ifft(*args, **kwargs):
    ...


@_implements(scipy.fft.fft2)
def fft2(*args, **kwargs):
    ...


@_implements(scipy.fft.ifft2)
def ifft2(*args, **kwargs):
    ...


@_implements(scipy.fft.fftn)
def fftn(*args, **kwargs):
    ...


@_implements(scipy.fft.ifftn)
def ifftn(*args, **kwargs):
    ...


@_implements(scipy.fft.rfft)
def rfft(*args, **kwargs):
    ...


@_implements(scipy.fft.irfft)
def irfft(*args, **kwargs):
    ...


@_implements(scipy.fft.rfft2)
def rfft2(*args, **kwargs):
    ...


@_implements(scipy.fft.irfft2)
def irfft2(*args, **kwargs):
    ...


@_implements(scipy.fft.rfftn)
def rfftn(*args, **kwargs):
    ...


@_implements(scipy.fft.irfftn)
def irfftn(*args, **kwargs):
    ...


@_implements(scipy.fft.fftshift)
def fftshift(*args, **kwargs):
    ...


@_implements(scipy.fft.ifftshift)
def ifftshift(*args, **kwargs):
    ...


# @_implements(scipy.fft.fftfreq)
# def fftfreq(*args, **kwargs):
#     ...


# @_implements(scipy.fft.rfftfreq)
# def rfftfreq(*args, **kwargs):
#     ...
