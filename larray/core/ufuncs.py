# numpy ufuncs -- this module is excluded from pytest
# http://docs.scipy.org/doc/numpy/reference/routines.math.html
# https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

import numbers

import numpy as np

from larray.tests.common import needs_python36


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


def floor(x, out=None, dtype=None, where=True):
    r"""
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray or scalar
        The floor of each element in `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    ceil, trunc, rint

    Notes
    -----
    Some spreadsheet programs calculate the "floor-towards-zero", in other
    words ``floor(-2.5) == -2``.  LArray instead uses the definition of
    `floor` where `floor(-2.5) == -3`.

    Examples
    --------
    >>> from larray import LArray
    >>> arr = LArray([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], 'a=a0..a6')
    >>> floor(arr)
    a    a0    a1    a2   a3   a4   a5   a6
       -2.0  -2.0  -1.0  0.0  1.0  1.0  2.0
    """
    return np.floor(x, out=out, dtype=dtype, where=where)


def ceil(x, out=None, dtype=None, where=True):
    r"""
    Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray or scalar
        The ceiling of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    floor, trunc, rint

    Examples
    --------
    >>> from larray import LArray
    >>> arr = LArray([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], 'a=a0..a6')
    >>> ceil(arr)
    a    a0    a1    a2   a3   a4   a5   a6
       -1.0  -1.0  -0.0  1.0  2.0  2.0  2.0
    """
    return np.ceil(x, out=out, dtype=dtype, where=where)


def trunc(x, out=None, dtype=None, where=True):
    r"""
    Return the truncated value of the input, element-wise.

    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray or scalar
        The truncated value of each element in `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    ceil, floor, rint

    Examples
    --------
    >>> from larray import LArray
    >>> arr = LArray([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], 'a=a0..a6')
    >>> trunc(arr)
    a    a0    a1    a2   a3   a4   a5   a6
       -1.0  -1.0  -0.0  0.0  1.0  1.0  2.0
    """
    return np.trunc(x, out=out, dtype=dtype, where=where)


# Exponents and logarithms


def exp(x, out=None, dtype=None, where=True):
    r"""
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    out : LArray or scalar
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    expm1 : Calculate ``exp(x) - 1`` for all elements in the array.
    exp2  : Calculate ``2**x`` for all elements in the array.

    Notes
    -----
    The irrational number ``e`` is also known as Euler's number.  It is
    approximately 2.718281, and is the base of the natural logarithm,
    ``ln`` (this means that, if :math:`x = \ln y = \log_e y`,
    then :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

    For complex arguments, ``x = a + ib``, we can write
    :math:`e^x = e^a e^{ib}`.  The first term, :math:`e^a`, is already
    known (it is the real argument, described above).  The second term,
    :math:`e^{ib}`, is :math:`\cos b + i \sin b`, a function with
    magnitude 1 and a periodic phase.

    References
    ----------
    .. [1] Wikipedia, "Exponential function",
           https://en.wikipedia.org/wiki/Exponential_function
    .. [2] M. Abramovitz and I. A. Stegun, "Handbook of Mathematical Functions
           with Formulas, Graphs, and Mathematical Tables," Dover, 1964, p. 69,
           http://www.math.sfu.ca/~cbm/aands/page_69.htm

    Examples
    --------
    >>> from larray import LArray, e
    >>> exp(LArray([-1, 0, 1])) / LArray([1/e, 1, e])
    {0}*    0    1    2
          1.0  1.0  1.0
    """
    return np.exp(x, out=out, dtype=dtype, where=where)


exp2 = broadcastify(np.exp2)
expm1 = broadcastify(np.expm1)


def log(x, out=None, dtype=None, where=True):
    r"""
    Natural logarithm, element-wise.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray
        The natural logarithm of `x`, element-wise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    log10, log2, log1p

    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log` always returns real output. For
    each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    >>> from larray import LArray, e
    >>> log(LArray([1, e, e**2, 0]))
    {0}*    0    1    2     3
          0.0  1.0  2.0  -inf
    """
    return np.log(x, out=out, dtype=dtype, where=where)


log10 = broadcastify(np.log10)
log2 = broadcastify(np.log2)
log1p = broadcastify(np.log1p)
logaddexp = broadcastify(np.logaddexp)
logaddexp2 = broadcastify(np.logaddexp2)

# Floating functions

signbit = broadcastify(np.signbit)
copysign = broadcastify(np.copysign)
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


def sqrt(x, out=None, dtype=None, where=True):
    r"""
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    x : LArray
        The values whose square-roots are required.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`.  If any element in `x` is
        complex, a complex array is returned (and the square-roots of
        negative reals are calculated).  If all of the elements in `x`
        are real, so is `y`, with negative elements returning ``nan``.
        If `out` was provided, `y` is a reference to it.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    *sqrt* has--consistent with common convention--as its branch cut the
    real "interval" [`-inf`, 0), and is continuous from above on it.
    A branch cut is a curve in the complex plane across which a given
    complex function fails to be continuous.

    Examples
    --------
    >>> from larray import LArray, inf
    >>> sqrt(LArray([1, 4, 9], 'a=a0..a2'))
    a   a0   a1   a2
       1.0  2.0  3.0

    >>> sqrt(LArray([4, -1, -3+4J], 'a=a0..a2'))
    a      a0  a1      a2
       (2+0j)  1j  (1+2j)

    >>> sqrt(LArray([4, -1, inf], 'a=a0..a2'))
    a   a0   a1   a2
       2.0  nan  inf
    """
    return np.sqrt(x, out=out, dtype=dtype, where=where)


def square(x, out=None, dtype=None, where=True):
    r"""
    Return the element-wise square of the input.

    Parameters
    ----------
    x : LArray
        The values whose square are required.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    out : LArray or scalar
        Element-wise `x*x`, of the same shape and dtype as `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    sqrt

    Examples
    --------
    >>> from larray import LArray, nan, inf
    >>> square(LArray([2., 3., nan, inf], 'a=a0..a3'))
    a   a0   a1   a2   a3
       4.0  9.0  nan  inf
    """
    return np.square(x, out=out, dtype=dtype, where=where)


def absolute(x, out=None, dtype=None, where=True):
    r"""
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    absolute : LArray
        An array containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\sqrt{ a^2 + b^2 }`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    fabs

    Examples
    --------
    >>> from larray import LArray, nan, inf
    >>> absolute(LArray([1, -1, -3+4J, nan, -inf, inf], 'a=a0..a5'))
    a   a0   a1   a2   a3   a4   a5
       1.0  1.0  5.0  nan  inf  inf
    """
    return np.absolute(x, out=out, dtype=dtype, where=where)


def fabs(x, out=None, dtype=None, where=True):
    r"""
    Compute the absolute values element-wise.

    This function returns the absolute values (positive magnitude) of the
    data in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray or scalar
        The absolute values of `x`, the returned values are always floats.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    absolute : Absolute values including `complex` types.

    Examples
    --------
    >>> from larray import LArray, nan, inf
    >>> fabs(LArray([1, -1, nan, -inf, inf], 'a=a0..a4'))
    a   a0   a1   a2   a3   a4
       1.0  1.0  nan  inf  inf
    """
    return np.fabs(x, out=out, dtype=dtype, where=where)


def sign(x, out=None, dtype=None, where=True):
    r"""
    Returns an element-wise indication of the sign of a number.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.
    nan is returned for nan inputs.

    For complex inputs, the `sign` function returns
    ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.

    complex(nan, 0) is returned for complex nan inputs.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : LArray or scalar
        The sign of `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    There is more than one definition of sign in common use for complex
    numbers.  The definition used here is equivalent to :math:`x/\sqrt{x*x}`
    which is different from a common alternative, :math:`x/|x|`.

    Examples
    --------
    >>> from larray import LArray, nan, inf
    >>> sign(LArray([-5., 4.5, 0, -inf, inf, nan], 'a=a0..a5'))
    a    a0   a1   a2    a3   a4   a5
       -1.0  1.0  0.0  -1.0  1.0  nan
    >>> np.sign(LArray([5, 5-2J, 2J], 'a=a0..a2'))
    a      a0      a1      a2
       (1+0j)  (1+0j)  (1+0j)
    """
    return np.sign(x, out=out, dtype=dtype, where=where)


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
    fmax :
        Element-wise maximum of two arrays, ignores NaNs.
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.

    Notes
    -----
    The maximum is equivalent to ``where(x1 >= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster.

    Examples
    --------
    >>> from larray import LArray, nan
    >>> arr1 = LArray([[8.0, 7.0, 5.0, 6.0],
    ...                [5.0, 8.0, 3.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr2 = LArray([[10.0, 2.0, 9.0, nan],
    ...                [9.0, 10.0, 5.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b   b0   b1   b2   b3
     a0  8.0  7.0  5.0  6.0
     a1  5.0  8.0  3.0  nan
    >>> arr2
    a\b    b0    b1   b2   b3
     a0  10.0   2.0  9.0  nan
     a1   9.0  10.0  5.0  nan

    >>> maximum(arr1, arr2)
    a\b    b0    b1   b2   b3
     a0  10.0   7.0  9.0  nan
     a1   9.0  10.0  5.0  nan

    With broadcasting

    >>> arr2['a0']
    b    b0   b1   b2   b3
       10.0  2.0  9.0  nan
    >>> maximum(arr1, arr2['a0'])
    a\b    b0   b1   b2   b3
     a0  10.0  7.0  9.0  nan
     a1  10.0  8.0  9.0  nan
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
    fmax :
        Element-wise maximum of two arrays, ignores NaNs.
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.

    Notes
    -----
    The minimum is equivalent to ``where(x1 <= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster.

    Examples
    --------
    >>> from larray import LArray, nan
    >>> arr1 = LArray([[8.0, 7.0, 5.0, 6.0],
    ...                [5.0, 8.0, 3.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr2 = LArray([[10.0, 2.0, 9.0, nan],
    ...                [9.0, 10.0, 5.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b   b0   b1   b2   b3
     a0  8.0  7.0  5.0  6.0
     a1  5.0  8.0  3.0  nan
    >>> arr2
    a\b    b0    b1   b2   b3
     a0  10.0   2.0  9.0  nan
     a1   9.0  10.0  5.0  nan

    >>> minimum(arr1, arr2)
    a\b   b0   b1   b2   b3
     a0  8.0  2.0  5.0  nan
     a1  5.0  8.0  3.0  nan

    With broadcasting

    >>> arr2['a0']
    b    b0   b1   b2   b3
       10.0  2.0  9.0  nan
    >>> minimum(arr1, arr2['a0'])
    a\b   b0   b1   b2   b3
     a0  8.0  2.0  5.0  nan
     a1  5.0  2.0  3.0  nan
    """
    return np.minimum(x1, x2, out=out, dtype=dtype)


def fmax(x1, x2, out=None, dtype=None):
    r"""
    Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then the
    non-nan element is returned. If both elements are NaNs then the first
    is returned.  The latter distinction is important for complex NaNs,
    which are defined as at least one of the real or imaginary parts being
    a NaN. The net effect is that NaNs are ignored when possible.

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
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.

    Notes
    -----
    The fmax is equivalent to ``np.where(x1 >= x2, x1, x2)`` when neither
    x1 nor x2 are NaNs, but it is faster.

    Examples
    --------
    >>> from larray import LArray, nan
    >>> arr1 = LArray([[8.0, 7.0, 5.0, 6.0],
    ...                [5.0, 8.0, 3.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr2 = LArray([[10.0, 2.0, 9.0, nan],
    ...                [9.0, 10.0, 5.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b   b0   b1   b2   b3
     a0  8.0  7.0  5.0  6.0
     a1  5.0  8.0  3.0  nan
    >>> arr2
    a\b    b0    b1   b2   b3
     a0  10.0   2.0  9.0  nan
     a1   9.0  10.0  5.0  nan

    >>> fmax(arr1, arr2)
    a\b    b0    b1   b2   b3
     a0  10.0   7.0  9.0  6.0
     a1   9.0  10.0  5.0  nan

    With broadcasting

    >>> arr2['a0']
    b    b0   b1   b2   b3
       10.0  2.0  9.0  nan
    >>> fmax(arr1, arr2['a0'])
    a\b    b0   b1   b2   b3
     a0  10.0  7.0  9.0  6.0
     a1  10.0  8.0  9.0  nan
    """
    return np.fmax(x1, x2, out=out, dtype=dtype)


def fmin(x1, x2, out=None, dtype=None):
    r"""
    Element-wise minimum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then the
    non-nan element is returned. If both elements are NaNs then the first
    is returned.  The latter distinction is important for complex NaNs,
    which are defined as at least one of the real or imaginary parts being
    a NaN. The net effect is that NaNs are ignored when possible.

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
    fmax :
        Element-wise maximum of two arrays, ignores NaNs.
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.

    Notes
    -----
    The fmin is equivalent to ``np.where(x1 <= x2, x1, x2)`` when neither
    x1 nor x2 are NaNs, but it is faster.

    Examples
    --------
    >>> from larray import LArray, nan
    >>> arr1 = LArray([[8.0, 7.0, 5.0, 6.0],
    ...                [5.0, 8.0, 3.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr2 = LArray([[10.0, 2.0, 9.0, nan],
    ...                [9.0, 10.0, 5.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b   b0   b1   b2   b3
     a0  8.0  7.0  5.0  6.0
     a1  5.0  8.0  3.0  nan
    >>> arr2
    a\b    b0    b1   b2   b3
     a0  10.0   2.0  9.0  nan
     a1   9.0  10.0  5.0  nan

    >>> fmin(arr1, arr2)
    a\b   b0   b1   b2   b3
     a0  8.0  2.0  5.0  6.0
     a1  5.0  8.0  3.0  nan

    With broadcasting

    >>> arr2['a0']
    b    b0   b1   b2   b3
       10.0  2.0  9.0  nan
    >>> fmin(arr1, arr2['a0'])
    a\b   b0   b1   b2   b3
     a0  8.0  2.0  5.0  6.0
     a1  5.0  2.0  3.0  nan
    """
    return np.fmin(x1, x2, out=out, dtype=dtype)


def isnan(x, out=None, where=True):
    r"""
    Test element-wise for NaN and return result as a boolean array.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    where : LArray, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : boolean LArray or bool
        True where ``x`` is NaN, false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    isinf

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> from larray import LArray, nan, inf
    >>> arr1 = LArray([[-inf, 7.0, 5.0, inf],
    ...                [5.0, 8.0, 3.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b    b0   b1   b2   b3
     a0  -inf  7.0  5.0  inf
     a1   5.0  8.0  3.0  nan
    >>> isnan(arr1)
    a\b     b0     b1     b2     b3
     a0  False  False  False  False
     a1  False  False  False   True
    """
    return np.isnan(x, out=out, where=where)


def isinf(x, out=None, where=True):
    r"""
    Test element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, True where ``x ==
    +/-inf``, otherwise False.

    Parameters
    ----------
    x : LArray
        Input array.
    out : LArray, optional
        An array into which the result is stored.
    where : LArray, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.

    Returns
    -------
    y : boolean LArray or bool
        True where ``x`` is positive or negative infinity, false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    isnan

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> from larray import LArray, nan, inf
    >>> arr1 = LArray([[-inf, 7.0, 5.0, inf],
    ...                [5.0, 8.0, 3.0, nan]], "a=a0,a1;b=b0..b3")
    >>> arr1
    a\b    b0   b1   b2   b3
     a0  -inf  7.0  5.0  inf
     a1   5.0  8.0  3.0  nan
    >>> isinf(arr1)
    a\b     b0     b1     b2     b3
     a0   True  False  False   True
     a1  False  False  False  False
    """
    return np.isinf(x, out=out, where=where)


# --------------------------------
# numpy funcs which are not ufuncs


# Rounding

# all 3 are equivalent, I am unsure I should support around and round_
around = broadcastify(np.around)
round_ = broadcastify(np.round_)


def round(a, decimals=0, out=None):
    r"""
    Round an array to the given number of decimals.

    Parameters
    ----------
    a : LArray
        Input array.
    decimals : int, optional
        Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : LArray, optional
        An array into which the result is stored.

    Returns
    -------
    rounded_array : LArray
        An array of the same type as `a`, containing the rounded values.
        Unless `out` was specified, a new array is created.
        A reference to the result is returned.

        The real and imaginary parts of complex numbers are rounded
        separately.  The result of rounding a float is a float.

    See Also
    --------
    ceil, fix, floor, rint, trunc

    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc. Results may also be surprising due
    to the inexact representation of decimal fractions in the IEEE
    floating point standard [1]_ and errors introduced when scaling
    by powers of ten.

    References
    ----------
    .. [1] "Lecture Notes on the Status of IEEE 754", William Kahan,
           https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
    .. [2] "How Futile are Mindless Assessments of
           Roundoff in Floating-Point Computation?", William Kahan,
           https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf

    Examples
    --------
    >>> from larray import LArray
    >>> round(LArray([0.37, 1.64], 'a=a0,a1'))
    a   a0   a1
       0.0  2.0
    >>> round(LArray([0.37, 1.64], 'a=a0,a1'), decimals=1)
    a   a0   a1
       0.4  1.6
    >>> round(LArray([.5, 1.5, 2.5, 3.5, 4.5], 'a=a0..a4')) # rounds to nearest even value
    a   a0   a1   a2   a3   a4
       0.0  2.0  2.0  4.0  4.0
    >>> round(LArray([1, 2, 3, 11], 'a=a0..a3'), decimals=1) # array of ints is returned
    a  a0  a1  a2  a3
        1   2   3  11
    >>> round(LArray([1, 2, 3, 11], 'a=a0..a3'), decimals=-1)
    a  a0  a1  a2  a3
        0   0   0  10
    """
    func = broadcastify(np.round, False)
    return func(a, decimals, out)


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


def where(condition, x=None, y=None):
    r"""
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
    func = broadcastify(np.where, False)
    return func(condition, x, y)

nan_to_num = broadcastify(np.nan_to_num)
real_if_close = broadcastify(np.real_if_close)
convolve = broadcastify(np.convolve)


@needs_python36
def inverse(a):
    r"""
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : Larray
        Matrix(ces) to be inverted.

    Returns
    -------
    ainv : LArray
        Inverse of the matrix(ces) `a`.

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.

    Examples
    --------
    >>> from larray import LArray, eye
    >>> arr = LArray([[1., 2.],
    ...               [3., 4.]], "a=a0,a1;b=b0,b1")
    >>> arr
    a\b   b0   b1
     a0  1.0  2.0
     a1  3.0  4.0
    >>> arr_inv = inverse(arr)
    >>> (arr @ arr_inv).equals(eye(arr.a, arr.b), atol=1.e-10)
    True
    >>> (arr_inv @ arr).equals(eye(arr.a, arr.b), atol=1.e-10)
    True

    Inverses of several matrices can be computed at once:

    >>> arr = LArray([[[1., 2.], [3., 4.]],
    ...               [[1., 3.], [3., 5.]]], "a=a0,a1;b=b0,b1;c=c0,c1")
    >>> arr_inv = inverse(arr)
    >>> (arr_inv['a0'] @ arr['a0']).equals(eye(arr.b, arr.c), atol=1.e-10)
    True
    >>> (arr_inv['a1'] @ arr['a1']).equals(eye(arr.b, arr.c), atol=1.e-10)
    True
    """
    func = broadcastify(np.linalg.inv, False)
    return func(a)


# XXX: create a new LArray method instead ?
# TODO: should appear in the API doc if it actually works with LArrays,
#       which I have never tested (and I doubt is the case).
#       Might be worth having specific documentation if it works well.
#       My guess is that we should rather make a new LArray method for that one.
interp = broadcastify(np.interp)
