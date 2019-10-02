# numpy ufuncs -- this module is excluded from pytest
# http://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

from larray.core.ufuncs import broadcastify


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

# TODO: add examples for round, floor, ceil and trunc
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

# TODO: add examples for exp and log
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
# square = broadcastify(np.square)
absolute = broadcastify(np.absolute)
fabs = broadcastify(np.fabs)
sign = broadcastify(np.sign)
fmax = broadcastify(np.fmax)
fmin = broadcastify(np.fmin)
nan_to_num = broadcastify(np.nan_to_num)
real_if_close = broadcastify(np.real_if_close)

# TODO: add examples for functions below
sqrt = broadcastify(np.sqrt)
isnan = broadcastify(np.isnan)
isinf = broadcastify(np.isinf)
inverse = broadcastify(np.linalg.inv)

# XXX: create a new Array method instead ?
# TODO: should appear in the API doc if it actually works with Arrays,
#       which I have never tested (and I doubt is the case).
#       Might be worth having specific documentation if it works well.
#       My guess is that we should rather make a new Array method for that one.
interp = broadcastify(np.interp)
