import pyfftw.interfaces

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1)
