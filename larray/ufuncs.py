# numpy ufuncs
# http://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

from larray.core import LArray, make_numpy_broadcastable


def wrapper(func):
    def wrapped(*args, **kwargs):
        # TODO: normalize args/kwargs like in LIAM2 so that we can also
        # broadcast if args are given via kwargs (eg out=)
        args, combined_axes = make_numpy_broadcastable(args)

        # We pass only raw numpy arrays to the ufuncs even though numpy is
        # normally meant to handle those case itself via __array_wrap__

        # There is a problem with np.clip though (and possibly other ufuncs)
        # np.clip is roughly equivalent to
        # np.maximum(np.minimum(np.asarray(la), high), low)
        # the np.asarray(la) is problematic because it lose original labels
        # and then tries to get them back from high, where they are possibly
        # incomplete if broadcasting happened

        # It fails on "np.minimum(ndarray, LArray)" because it calls
        # __array_wrap__(high, result) which cannot work if there was
        # broadcasting involved (high has potentially less labels than result).
        # it does this because numpy calls __array_wrap__ on the argument with
        # the highest __array_priority__
        raw_args = [np.asarray(a) if isinstance(a, LArray) else a
                    for a in args]
        res_data = func(*raw_args, **kwargs)
        if combined_axes:
            return LArray(res_data, combined_axes)
        else:
            return res_data
        # return func(*args, **kwargs)
    wrapped.__name__ = func.__name__
    wrapped.__doc__ = func.__doc__
    return wrapped


# Trigonometric functions

sin = wrapper(np.sin)
cos = wrapper(np.cos)
tan = wrapper(np.tan)
arcsin = wrapper(np.arcsin)
arccos = wrapper(np.arccos)
arctan = wrapper(np.arctan)
hypot = wrapper(np.hypot)
arctan2 = wrapper(np.arctan2)
degrees = wrapper(np.degrees)
radians = wrapper(np.radians)
unwrap = wrapper(np.unwrap)
# deg2rad = wrapper(np.deg2rad)
# rad2deg = wrapper(np.rad2deg)

# Hyperbolic functions

sinh = wrapper(np.sinh)
cosh = wrapper(np.cosh)
tanh = wrapper(np.tanh)
arcsinh = wrapper(np.arcsinh)
arccosh = wrapper(np.arccosh)
arctanh = wrapper(np.arctanh)

# Rounding

around = wrapper(np.around)
round_ = wrapper(np.round_)
rint = wrapper(np.rint)
fix = wrapper(np.fix)
floor = wrapper(np.floor)
ceil = wrapper(np.ceil)
trunc = wrapper(np.trunc)

# Sums, products, differences

# prod = wrapper(np.prod)
# sum = wrapper(np.sum)
# nansum = wrapper(np.nansum)
# cumprod = wrapper(np.cumprod)
# cumsum = wrapper(np.cumsum)
# diff = wrapper(np.diff)
# ediff1d = wrapper(np.ediff1d)
# gradient = wrapper(np.gradient)
# cross = wrapper(np.cross)
# trapz = wrapper(np.trapz)

# Exponents and logarithms

exp = wrapper(np.exp)
expm1 = wrapper(np.expm1)
exp2 = wrapper(np.exp2)
log = wrapper(np.log)
log10 = wrapper(np.log10)
log2 = wrapper(np.log2)
log1p = wrapper(np.log1p)
logaddexp = wrapper(np.logaddexp)
logaddexp2 = wrapper(np.logaddexp2)

# Other special functions

i0 = wrapper(np.i0)
sinc = wrapper(np.sinc)

# Floating point routines

signbit = wrapper(np.signbit)
copysign = wrapper(np.copysign)
frexp = wrapper(np.frexp)
ldexp = wrapper(np.ldexp)

# Arithmetic operations

# add = wrapper(np.add)
# reciprocal = wrapper(np.reciprocal)
# negative = wrapper(np.negative)
# multiply = wrapper(np.multiply)
# divide = wrapper(np.divide)
# power = wrapper(np.power)
# subtract = wrapper(np.subtract)
# true_divide = wrapper(np.true_divide)
# floor_divide = wrapper(np.floor_divide)
# fmod = wrapper(np.fmod)
# mod = wrapper(np.mod)
modf = wrapper(np.modf)
# remainder = wrapper(np.remainder)

# Handling complex numbers

angle = wrapper(np.angle)
real = wrapper(np.real)
imag = wrapper(np.imag)
conj = wrapper(np.conj)

# Miscellaneous

convolve = wrapper(np.convolve)
clip = wrapper(np.clip)
sqrt = wrapper(np.sqrt)
# square = wrapper(np.square)
absolute = wrapper(np.absolute)
fabs = wrapper(np.fabs)
sign = wrapper(np.sign)
maximum = wrapper(np.maximum)
minimum = wrapper(np.minimum)
fmax = wrapper(np.fmax)
fmin = wrapper(np.fmin)
nan_to_num = wrapper(np.nan_to_num)
real_if_close = wrapper(np.real_if_close)
interp = wrapper(np.interp)
