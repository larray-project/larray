# numpy ufuncs -- this module is excluded from pytest
# http://docs.scipy.org/doc/numpy/reference/routines.math.html
# https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

import numbers

import numpy as np


def wrap_elementwise_array_func(func):
    r"""
    Wrap a function using numpy arrays to work with LArray arrays instead.

    Parameters
    ----------
    func : function
        A function taking numpy arrays as arguments and returning numpy arrays of the same shape. If the function
        takes several arguments, this wrapping code assumes the result will have the combination of all axes present.
        In numpy talk, arguments will be broadcasted to each other.

    Returns
    -------
    function
        A function taking LArray arguments and returning LArrays.

    Examples
    --------
    For example, if we want to apply the Hodrick-Prescott filter from statsmodels we can use this:

    >>> from statsmodels.tsa.filters.hp_filter import hpfilter         # doctest: +SKIP
    >>> hpfilter = wrap_elementwise_array_func(hpfilter)               # doctest: +SKIP

    hpfilter is now a function taking a one dimensional LArray as input and returning a one dimensional LArray as output

    Now let us suppose we have a ND array such as:

    >>> from larray.random import normal
    >>> arr = normal(axes="sex=M,F;year=2016..2018")                   # doctest: +SKIP
    >>> arr                                                            # doctest: +SKIP
    sex\year   2016   2017   2018
           M  -1.15   0.56  -1.06
           F  -0.48  -0.39  -0.98

    We can apply an Hodrick-Prescott filter to it by using:

    >>> # 6.25 is the recommended smoothing value for annual data
    >>> cycle, trend = arr.apply(hpfilter, 6.25, axes="year")          # doctest: +SKIP
    >>> trend                                                          # doctest: +SKIP
    sex\year   2016   2017   2018
           M  -0.61  -0.52  -0.52
           F  -0.37  -0.61  -0.87
    """
    def wrapper(*args, **kwargs):
        from larray.core.array import LArray, make_args_broadcastable
        raw_bcast_args, raw_bcast_kwargs, res_axes = make_args_broadcastable(args, kwargs)

        # We pass only raw numpy arrays to the ufuncs even though numpy is normally meant to handle those cases itself
        # via __array_wrap__

        # There is a problem with np.clip though (and possibly other ufuncs): np.clip is roughly equivalent to
        # np.maximum(np.minimum(np.asarray(la), high), low)
        # the np.asarray(la) is problematic because it lose original labels
        # and then tries to get them back from high, where they are possibly
        # incomplete if broadcasting happened

        # It fails on "np.minimum(ndarray, LArray)" because it calls __array_wrap__(high, result) which cannot work if
        # there was broadcasting involved (high has potentially less labels than result).
        # it does this because numpy calls __array_wrap__ on the argument with the highest __array_priority__
        res_data = func(*raw_bcast_args, **raw_bcast_kwargs)
        if res_axes:
            if isinstance(res_data, tuple):
                return tuple(LArray(res_arr, res_axes) for res_arr in res_data)
            else:
                return LArray(res_data, res_axes)
        else:
            return res_data
    # copy function name. We are intentionally not using functools.wraps, because it does not work for wrapping a
    # function from another module
    wrapper.__name__ = func.__name__
    return wrapper


class SupportsNumpyUfuncs(object):
    """
    Base class for larray types that support ufuncs.
    Used by LArray.

    Notes
    -----
    See https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#numpy.class.__array_ufunc__
    for more details about __array_ufunc__ and
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
    for an example.
    """
    _HANDLED_TYPES = (np.ndarray, np.generic, numbers.Number, bytes, str)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override the behavior of NumPy’s ufuncs.

        Parameters
        ----------
        ufunc: callable
            Ufunc object that was called.
        method: str
            String indicating which Ufunc method was called
            (one of "__call__", "reduce", "reduceat", "accumulate", "outer", "inner").
        inputs: tuple
            Input arguments to the ufunc.
        kwargs: dict
            Dictionary containing the optional input arguments of the ufunc.
            If given, any out arguments, both positional and keyword, are passed as a tuple in kwargs.
        """
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (SupportsNumpyUfuncs,)):
                return NotImplemented

        if out:
            if len(out) > 1:
                raise TypeError("Passing an iterable for the argument 'out' is not supported")
            kwargs['out'] = out[0]

        if ufunc.signature is not None:
            # In regular ufuncs, the elementary function is limited to element-by-element operations,
            # whereas the generalized version (gufuncs) supports "sub-array" by "sub-array" operations.
            raise NotImplementedError('{} not supported: larray objects do not directly implement '
                                      'generalized ufuncs.'.format(ufunc))

        if method != '__call__':
            raise NotImplemented

        wrapped_ufunc = wrap_elementwise_array_func(ufunc)
        return wrapped_ufunc(*inputs, **kwargs)


def broadcastify(npfunc, copy_numpy_doc=True):
    wrapper = wrap_elementwise_array_func(npfunc)
    if copy_numpy_doc:
        # copy meaningful attributes (numpy ufuncs do not have __annotations__ nor __qualname__)
        wrapper.__name__ = npfunc.__name__
        # update documentation by inserting a warning message after the short description of the numpy function
        # (otherwise the description of npfuncs given in the corresponding API 'autosummary' tables will always
        #  start with 'larray specific variant of ...' without giving a meaningful description of what does the npfunc)
        if npfunc.__doc__.startswith('\n'):
            # docstring starts with short description
            end_signature = 1
            end_short_desc = npfunc.__doc__.find('\n\n')
        else:
            # docstring starts with signature
            end_signature = npfunc.__doc__.find('\n\n') + 2
            end_short_desc = npfunc.__doc__.find('\n\n', end_signature)
        short_desc = npfunc.__doc__[:end_short_desc]
        numpy_doc = npfunc.__doc__[end_short_desc:]
        ident = ' ' * (len(short_desc[end_signature:]) - len(short_desc[end_signature:].lstrip()))
        wrapper.__doc__ = '{short_desc}' \
                          '\n\n{ident}larray specific variant of ``numpy.{fname}``.' \
                          '\n\n{ident}Documentation from numpy:' \
                          '{numpy_doc}'.format(short_desc=short_desc, ident=ident,
                                               fname=npfunc.__name__, numpy_doc=numpy_doc)
        # set __qualname__ explicitly (all these functions are supposed to be top-level function in the npfuncs module)
        wrapper.__qualname__ = npfunc.__name__
        # we should not copy __module__
    return wrapper


# list of available Numpy ufuncs
# https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

# Trigonometric functions

sin = broadcastify(np.sin)
cos = broadcastify(np.cos)
tan = broadcastify(np.tan)
arcsin = broadcastify(np.arcsin)
arccos = broadcastify(np.arccos)
arctan = broadcastify(np.arctan)
arctan2 = broadcastify(np.arctan2)
hypot = broadcastify(np.hypot)
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

rint = broadcastify(np.rint)
# TODO: fix fails because of its 'out' argument
fix = broadcastify(np.fix)
# TODO: add examples for round, floor, ceil and trunc
floor = broadcastify(np.floor)
ceil = broadcastify(np.ceil)
trunc = broadcastify(np.trunc)

# Exponents and logarithms

# TODO: add examples for exp and log
exp = broadcastify(np.exp)
exp2 = broadcastify(np.exp2)
expm1 = broadcastify(np.expm1)
log = broadcastify(np.log)
log10 = broadcastify(np.log10)
log2 = broadcastify(np.log2)
log1p = broadcastify(np.log1p)
logaddexp = broadcastify(np.logaddexp)
logaddexp2 = broadcastify(np.logaddexp2)

# Floating functions

signbit = broadcastify(np.signbit)
copysign = broadcastify(np.copysign)
# TODO: understand why frexp fails
frexp = broadcastify(np.frexp)
ldexp = broadcastify(np.ldexp)

# Arithmetic operations

# add = broadcastify(np.add)
# reciprocal = broadcastify(np.reciprocal)
# positive = broadcastify(np.positive)
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
# divmod = broadcastify(np.divmod)

# Handling complex numbers

conj = broadcastify(np.conj)

# Miscellaneous

sqrt = broadcastify(np.sqrt)
square = broadcastify(np.square)
absolute = broadcastify(np.absolute)
fabs = broadcastify(np.fabs)
sign = broadcastify(np.sign)


def maximum(x1, x2, out=None, dtype=None):
    r"""
    Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : LArray
        The arrays holding the elements to be compared.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.

    Returns
    -------
    y : LArray or scalar
        The maximum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.

    Notes
    -----
    The maximum is equivalent to ``where(x1 >= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster.

    Examples
    --------
    >>> from larray import LArray, zeros_like
    >>> arr1 = LArray([[10, 7, 5, 9],
    ...                [5, 8, 3, 7]], "a=a0,a1;b=b0..b3")
    >>> arr2 = LArray([[6, 2, 9, 0],
    ...                [9, 10, 5, 6]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b  b0  b1  b2  b3
     a0  10   7   5   9
     a1   5   8   3   7
    >>> arr2
    a\b  b0  b1  b2  b3
     a0   6   2   9   0
     a1   9  10   5   6

    >>> maximum(arr1, arr2)
    a\b  b0  b1  b2  b3
     a0  10   7   9   9
     a1   9  10   5   7

    With broadcasting

    >>> arr2['a0']
    b  b0  b1  b2  b3
        6   2   9   0
    >>> maximum(arr1, arr2['a0'])
    a\b  b0  b1  b2  b3
     a0  10   7   9   9
     a1   6   8   9   7
    """
    return np.maximum(x1, x2, out=out, dtype=dtype)


def minimum(x1, x2, out=None, dtype=None):
    r"""
    Element-wise minimum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : LArray
        The arrays holding the elements to be compared.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.

    Returns
    -------
    y : LArray or scalar
        The minimum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.

    Notes
    -----
    The minimum is equivalent to ``where(x1 <= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster.

    Examples
    --------
    >>> from larray import LArray
    >>> arr1 = LArray([[10, 7, 5, 9],
    ...                [5, 8, 3, 7]], "a=a0,a1;b=b0..b3")
    >>> arr2 = LArray([[6, 2, 9, 0],
    ...                [9, 10, 5, 6]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b  b0  b1  b2  b3
     a0  10   7   5   9
     a1   5   8   3   7
    >>> arr2
    a\b  b0  b1  b2  b3
     a0   6   2   9   0
     a1   9  10   5   6

    >>> minimum(arr1, arr2)
    a\b  b0  b1  b2  b3
     a0   6   2   5   0
     a1   5   8   3   6

    With broadcasting

    >>> arr2['a0']
    b  b0  b1  b2  b3
        6   2   9   0
    >>> minimum(arr1, arr2['a0'])
    a\b  b0  b1  b2  b3
     a0   6   2   5   0
     a1   5   2   3   0
    """
    return np.minimum(x1, x2, out=out, dtype=dtype)

fmax = broadcastify(np.fmax)
fmin = broadcastify(np.fmin)
isnan = broadcastify(np.isnan)
isinf = broadcastify(np.isinf)


# --------------------------------
# numpy funcs which are not ufuncs


# Rounding

# all 3 are equivalent, I am unsure I should support around and round_
around = broadcastify(np.around)
round_ = broadcastify(np.round_)
round = broadcastify(np.round)

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

# Other special functions

i0 = broadcastify(np.i0)
sinc = broadcastify(np.sinc)

# Handling complex numbers

angle = broadcastify(np.angle)
real = broadcastify(np.real)
imag = broadcastify(np.imag)

# Miscellaneous

# put clip here even it is a ufunc because it ends up within a recursion loop with LArray.clip
clip = broadcastify(np.clip)

where = broadcastify(np.where, False)
where.__doc__ = r"""
    Return elements, either from `x` or `y`, depending on `condition`.

    If only `condition` is given, return ``condition.nonzero()``.

    Parameters
    ----------
    condition : boolean LArray
        When True, yield `x`, otherwise yield `y`.
    x, y : LArray
        Values from which to choose.

    Returns
    -------
    out : LArray
        If both `x` and `y` are specified, the output array contains
        elements of `x` where `condition` is True, and elements from
        `y` elsewhere.

    Examples
    --------
    >>> from larray import LArray
    >>> arr = LArray([[10, 7, 5, 9],
    ...               [5, 8, 3, 7],
    ...               [6, 2, 0, 9],
    ...               [9, 10, 5, 6]], "a=a0..a3;b=b0..b3")
    >>> arr
    a\b  b0  b1  b2  b3
     a0  10   7   5   9
     a1   5   8   3   7
     a2   6   2   0   9
     a3   9  10   5   6

    Simple use

    >>> where(arr <= 5, 0, arr)
    a\b  b0  b1  b2  b3
     a0  10   7   0   9
     a1   0   8   0   7
     a2   6   0   0   9
     a3   9  10   0   6

    With broadcasting

    >>> mean_by_col = arr.mean('a')
    >>> mean_by_col
    b   b0    b1    b2    b3
       7.5  6.75  3.25  7.75
    >>> # for each column, set values below the mean value to the mean value
    >>> where(arr < mean_by_col, mean_by_col, arr)
    a\b    b0    b1    b2    b3
     a0  10.0   7.0   5.0   9.0
     a1   7.5   8.0  3.25  7.75
     a2   7.5  6.75  3.25   9.0
     a3   9.0  10.0   5.0  7.75
"""

nan_to_num = broadcastify(np.nan_to_num)
real_if_close = broadcastify(np.real_if_close)
convolve = broadcastify(np.convolve)
inverse = broadcastify(np.linalg.inv)

# XXX: create a new LArray method instead ?
# TODO: should appear in the API doc if it actually works with LArrays,
#       which I have never tested (and I doubt is the case).
#       Might be worth having specific documentation if it works well.
#       My guess is that we should rather make a new LArray method for that one.
interp = broadcastify(np.interp)
