import pyfftw.interfaces
from pyfftw.interfaces.numpy_fft import fftn

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1)

__all__ = ["fftn", "fft"]
