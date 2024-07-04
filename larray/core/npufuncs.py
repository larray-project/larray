# numpy ufuncs -- this module is excluded from pytest
# http://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

from larray.core.ufuncs import wrap_numpy_func
from larray.util.misc import renamed_to

# Trigonometric functions

sin = wrap_numpy_func(np.sin)
cos = wrap_numpy_func(np.cos)
tan = wrap_numpy_func(np.tan)
arcsin = wrap_numpy_func(np.arcsin)
arccos = wrap_numpy_func(np.arccos)
arctan = wrap_numpy_func(np.arctan)
hypot = wrap_numpy_func(np.hypot)
arctan2 = wrap_numpy_func(np.arctan2)
degrees = wrap_numpy_func(np.degrees)
radians = wrap_numpy_func(np.radians)
unwrap = wrap_numpy_func(np.unwrap)
# deg2rad = broadcastify(np.deg2rad)
# rad2deg = broadcastify(np.rad2deg)

# Hyperbolic functions

sinh = wrap_numpy_func(np.sinh)
cosh = wrap_numpy_func(np.cosh)
tanh = wrap_numpy_func(np.tanh)
arcsinh = wrap_numpy_func(np.arcsinh)
arccosh = wrap_numpy_func(np.arccosh)
arctanh = wrap_numpy_func(np.arctanh)

# Rounding

# TODO: add examples for round, floor, ceil and trunc
# all 3 are equivalent, I am unsure I should support around and round_
round = wrap_numpy_func(np.round)
around = wrap_numpy_func(np.around)
round_ = renamed_to(round, 'round_')
rint = wrap_numpy_func(np.rint)
fix = wrap_numpy_func(np.fix)
floor = wrap_numpy_func(np.floor)
ceil = wrap_numpy_func(np.ceil)
trunc = wrap_numpy_func(np.trunc)

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

# TODO: add examples for exp and log
exp = wrap_numpy_func(np.exp)
expm1 = wrap_numpy_func(np.expm1)
exp2 = wrap_numpy_func(np.exp2)
log = wrap_numpy_func(np.log)
log10 = wrap_numpy_func(np.log10)
log2 = wrap_numpy_func(np.log2)
log1p = wrap_numpy_func(np.log1p)
logaddexp = wrap_numpy_func(np.logaddexp)
logaddexp2 = wrap_numpy_func(np.logaddexp2)

# Other special functions

i0 = wrap_numpy_func(np.i0)
sinc = wrap_numpy_func(np.sinc)

# Floating point routines

signbit = wrap_numpy_func(np.signbit)
copysign = wrap_numpy_func(np.copysign)
frexp = wrap_numpy_func(np.frexp)
ldexp = wrap_numpy_func(np.ldexp)

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

angle = wrap_numpy_func(np.angle)
real = wrap_numpy_func(np.real)
imag = wrap_numpy_func(np.imag)
conj = wrap_numpy_func(np.conj)

# Miscellaneous

convolve = wrap_numpy_func(np.convolve)
clip = wrap_numpy_func(np.clip)
# square = broadcastify(np.square)
absolute = wrap_numpy_func(np.absolute)
fabs = wrap_numpy_func(np.fabs)
sign = wrap_numpy_func(np.sign)
fmax = wrap_numpy_func(np.fmax)
fmin = wrap_numpy_func(np.fmin)
nan_to_num = wrap_numpy_func(np.nan_to_num)
real_if_close = wrap_numpy_func(np.real_if_close)

# TODO: add examples for functions below
sqrt = wrap_numpy_func(np.sqrt)
isnan = wrap_numpy_func(np.isnan)
isinf = wrap_numpy_func(np.isinf)
inverse = wrap_numpy_func(np.linalg.inv)

# XXX: create a new Array method instead ?
# TODO: should appear in the API doc if it actually works with Arrays,
#       which I have never tested (and I doubt is the case).
#       Might be worth having specific documentation if it works well.
#       My guess is that we should rather make a new Array method for that one.
interp = wrap_numpy_func(np.interp)
