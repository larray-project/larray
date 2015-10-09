# numpy ufuncs
# http://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

# Trigonometric functions

sin = np.sin
cos = np.cos
tan = np.tan
arcsin = np.arcsin
arccos = np.arccos
arctan = np.arctan
hypot = np.hypot
arctan2 = np.arctan2
degrees = np.degrees
radians = np.radians
unwrap = np.unwrap
# deg2rad = np.deg2rad
# rad2deg = np.rad2deg

# Hyperbolic functions

sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
arcsinh = np.arcsinh
arccosh = np.arccosh
arctanh = np.arctanh

# Rounding

around = np.around
round_ = np.round_
rint = np.rint
fix = np.fix
floor = np.floor
ceil = np.ceil
trunc = np.trunc

# Sums, products, differences

# prod = np.prod
# sum = np.sum
# nansum = np.nansum
# cumprod = np.cumprod
# cumsum = np.cumsum
# diff = np.diff
# ediff1d = np.ediff1d
# gradient = np.gradient
# cross = np.cross
# trapz = np.trapz

# Exponents and logarithms

exp = np.exp
expm1 = np.expm1
exp2 = np.exp2
log = np.log
log10 = np.log10
log2 = np.log2
log1p = np.log1p
logaddexp = np.logaddexp
logaddexp2 = np.logaddexp2

# Other special functions

i0 = np.i0
sinc = np.sinc

# Floating point routines

signbit = np.signbit
copysign = np.copysign
frexp = np.frexp
ldexp = np.ldexp

# Arithmetic operations

# add = np.add
# reciprocal = np.reciprocal
# negative = np.negative
# multiply = np.multiply
# divide = np.divide
# power = np.power
# subtract = np.subtract
# true_divide = np.true_divide
# floor_divide = np.floor_divide
# fmod = np.fmod
# mod = np.mod
modf = np.modf
# remainder = np.remainder

# Handling complex numbers

angle = np.angle
real = np.real
imag = np.imag
conj = np.conj

# Miscellaneous

convolve = np.convolve
clip = np.clip
sqrt = np.sqrt
# square = np.square
absolute = np.absolute
fabs = np.fabs
sign = np.sign
maximum = np.maximum
minimum = np.minimum
fmax = np.fmax
fmin = np.fmin
nan_to_num = np.nan_to_num
real_if_close = np.real_if_close
interp = np.interp
