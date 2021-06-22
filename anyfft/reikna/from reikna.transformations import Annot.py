import numpy as np
import reikna.cluda as cluda
from reikna.core import Type
from reikna.fft import FFT
from reikna.transformations import Annotation, Parameter, Transformation

api = cluda.ocl_api()
thr = api.Thread.create()


def tform_r2c(arr):
    complex_dtype = cluda.dtypes.complex_for(arr.dtype)
    return Transformation(
        [
            Parameter("output", Annotation(Type(complex_dtype, arr.shape), "o")),
            Parameter("input", Annotation(arr, "i")),
        ],
        """
        ${output.store_same}(
            COMPLEX_CTR(${output.ctype})(
                ${input.load_same},
                0));
        """,
    )


def tform_c2r(arr):
    """Transform a complex array to a real one by discarding the imaginary part."""
    real_dtype = cluda.dtypes.real_for(arr.dtype)
    return Transformation(
        [
            Parameter("output", Annotation(Type(real_dtype, arr.shape), "o")),
            Parameter("input", Annotation(arr, "i")),
        ],
        """
        ${output.store_same}(${input.load_same}.x);
        """,
    )


arr = np.random.rand(8).astype(np.float32)

r2c = tform_r2c(arr)
plan = FFT(r2c.output)
plan.parameter.input.connect(r2c, r2c.output, new_input=r2c.input)
c2r = tform_c2r(plan.parameter.output)
plan.parameter.output.connect(c2r, c2r.input, new_output=c2r.output)
planc = plan.compile(thread=thr)

arr_dev = thr.to_device(arr)
out_dev = thr.array(arr.shape, np.float32)
planc(out_dev, arr_dev)

assert np.allclose(out_dev.get(), np.fft.fft(arr).real)
