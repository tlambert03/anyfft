from . import reikna


class ReiknaBackend:
    """The default backend for fft calculations

    Notes
    -----
    We use the domain ``numpy.scipy`` rather than ``scipy`` because in the
    future, ``uarray`` will treat the domain as a hierarchy. This means the user
    can install a single backend for ``numpy`` and have it implement
    ``numpy.scipy.fft`` as well.
    """

    __ua_domain__ = "numpy.scipy.fft"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        fn = getattr(reikna, method.__name__, None)

        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)
