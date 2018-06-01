# numpy ufuncs
# http://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

from larray.core.array import LArray, make_numpy_broadcastable

__all__ = [
    # Trigonometric functions
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'hypot', 'arctan2', 'degrees', 'radians', 'unwrap',
    # 'deg2rad', 'rad2deg',

    # Hyperbolic functions
    'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',

    # Rounding
    'round', 'around', 'round_', 'rint', 'fix', 'floor', 'ceil', 'trunc',

    # Sums, products, differences
    # 'prod', 'sum', 'nansum', 'cumprod', 'cumsum',

    # cannot use a simple wrapped ufunc because those ufuncs do not preserve shape or dimensions so labels are wrong
    # 'diff', 'ediff1d', 'gradient', 'cross', 'trapz',

    # Exponents and logarithms
    'exp', 'expm1', 'exp2', 'log', 'log10', 'log2', 'log1p', 'logaddexp', 'logaddexp2',

    # Other special functions
    'i0', 'sinc',

    # Floating point routines
    'signbit', 'copysign', 'frexp', 'ldexp',

    # Arithmetic operations
    # 'add', 'reciprocal', 'negative', 'multiply', 'divide', 'power', 'subtract', 'true_divide', 'floor_divide',
    # 'fmod', 'mod', 'modf', 'remainder',

    # Handling complex numbers
    'angle', 'real', 'imag', 'conj',

    # Miscellaneous
    'convolve', 'clip', 'sqrt',
    # 'square',
    'absolute', 'fabs', 'sign', 'maximum', 'minimum', 'fmax', 'fmin', 'nan_to_num', 'real_if_close',
    'interp', 'where', 'isnan', 'isinf',
    'inverse',
]

def broadcastify(func):
    # intentionally not using functools.wraps, because it does not work for wrapping a function from another module
    def wrapper(*args, **kwargs):
        # TODO: normalize args/kwargs like in LIAM2 so that we can also broadcast if args are given via kwargs
        #       (eg out=)
        args, combined_axes = make_numpy_broadcastable(args)

        # We pass only raw numpy arrays to the ufuncs even though numpy is normally meant to handle those case itself
        # via __array_wrap__

        # There is a problem with np.clip though (and possibly other ufuncs): np.clip is roughly equivalent to
        # np.maximum(np.minimum(np.asarray(la), high), low)
        # the np.asarray(la) is problematic because it lose original labels
        # and then tries to get them back from high, where they are possibly
        # incomplete if broadcasting happened

        # It fails on "np.minimum(ndarray, LArray)" because it calls __array_wrap__(high, result) which cannot work if
        # there was broadcasting involved (high has potentially less labels than result).
        # it does this because numpy calls __array_wrap__ on the argument with the highest __array_priority__
        raw_args = [np.asarray(a) if isinstance(a, LArray) else a
                    for a in args]
        res_data = func(*raw_args, **kwargs)
        if combined_axes:
            return LArray(res_data, combined_axes)
        else:
            return res_data
    # copy meaningful attributes (numpy ufuncs do not have __annotations__ nor __qualname__)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    # set __qualname__ explicitly (all these functions are supposed to be top-level function in the ufuncs module)
    wrapper.__qualname__ = func.__name__
    # we should not copy __module__
    return wrapper


# Trigonometric functions

sin = broadcastify(np.sin)
cos = broadcastify(np.cos)
tan = broadcastify(np.tan)
arcsin = broadcastify(np.arcsin)
arccos = broadcastify(np.arccos)
arctan = broadcastify(np.arctan)
hypot = broadcastify(np.hypot)
arctan2 = broadcastify(np.arctan2)
degrees = broadcastify(np.degrees)
radians = broadcastify(np.radians)
unwrap = broadcastify(np.unwrap)
# deg2rad = broadcastify(np.deg2rad)
# rad2deg = broadcastify(np.rad2deg)

# Hyperbolic functions

sinh = broadcastify(np.sinh)
cosh = broadcastify(np.cosh)
tanh = broadcastify(np.tanh)
arcsinh = broadcastify(np.arcsinh)
arccosh = broadcastify(np.arccosh)
arctanh = broadcastify(np.arctanh)

# Rounding

# all 3 are equivalent, I am unsure I should support around and round_
round = broadcastify(np.round)
around = broadcastify(np.around)
round_ = broadcastify(np.round_)
rint = broadcastify(np.rint)
fix = broadcastify(np.fix)
floor = broadcastify(np.floor)
ceil = broadcastify(np.ceil)
trunc = broadcastify(np.trunc)

# Sums, products, differences

# prod = broadcastify(np.prod)
# sum = broadcastify(np.sum)
# nansum = broadcastify(np.nansum)
# cumprod = broadcastify(np.cumprod)
# cumsum = broadcastify(np.cumsum)

# cannot use a simple wrapped ufunc because those ufuncs do not preserve
# shape or dimensions so labels are wrong
# diff = broadcastify(np.diff)
# ediff1d = broadcastify(np.ediff1d)
# gradient = broadcastify(np.gradient)
# cross = broadcastify(np.cross)
# trapz = broadcastify(np.trapz)

# Exponents and logarithms

exp = broadcastify(np.exp)
expm1 = broadcastify(np.expm1)
exp2 = broadcastify(np.exp2)
log = broadcastify(np.log)
log10 = broadcastify(np.log10)
log2 = broadcastify(np.log2)
log1p = broadcastify(np.log1p)
logaddexp = broadcastify(np.logaddexp)
logaddexp2 = broadcastify(np.logaddexp2)

# Other special functions

i0 = broadcastify(np.i0)
sinc = broadcastify(np.sinc)

# Floating point routines

signbit = broadcastify(np.signbit)
copysign = broadcastify(np.copysign)
frexp = broadcastify(np.frexp)
ldexp = broadcastify(np.ldexp)

# Arithmetic operations

# add = broadcastify(np.add)
# reciprocal = broadcastify(np.reciprocal)
# negative = broadcastify(np.negative)
# multiply = broadcastify(np.multiply)
# divide = broadcastify(np.divide)
# power = broadcastify(np.power)
# subtract = broadcastify(np.subtract)
# true_divide = broadcastify(np.true_divide)
# floor_divide = broadcastify(np.floor_divide)
# fmod = broadcastify(np.fmod)
# mod = broadcastify(np.mod)
# modf = broadcastify(np.modf)
# remainder = broadcastify(np.remainder)

# Handling complex numbers

angle = broadcastify(np.angle)
real = broadcastify(np.real)
imag = broadcastify(np.imag)
conj = broadcastify(np.conj)

# Miscellaneous

convolve = broadcastify(np.convolve)
clip = broadcastify(np.clip)
sqrt = broadcastify(np.sqrt)
# square = broadcastify(np.square)
absolute = broadcastify(np.absolute)
fabs = broadcastify(np.fabs)
sign = broadcastify(np.sign)
maximum = broadcastify(np.maximum)
minimum = broadcastify(np.minimum)
fmax = broadcastify(np.fmax)
fmin = broadcastify(np.fmin)
nan_to_num = broadcastify(np.nan_to_num)
real_if_close = broadcastify(np.real_if_close)
interp = broadcastify(np.interp)
where = broadcastify(np.where)
isnan = broadcastify(np.isnan)
isinf = broadcastify(np.isinf)

inverse = broadcastify(np.linalg.inv)
