# numpy ufuncs
# http://docs.scipy.org/doc/numpy/reference/routines.math.html
import numpy as np

from larray.core.array import Array, make_args_broadcastable


def wrap_elementwise_array_func(func, doc=None):
    r"""
    Wrap a function using numpy arrays to work with LArray arrays instead.

    Parameters
    ----------
    func : function
        A function taking numpy arrays as arguments and returning numpy arrays of the same shape. If the function
        takes several arguments, this wrapping code assumes the result will have the combination of all axes present.
        In numpy talk, arguments will be broadcasted to each other.
    doc : str, optional
        The documentation (docstring) for the new function. Defaults to the documentation of the original function,
        if any.

    Returns
    -------
    function
        A function taking larray.Array arguments and returning larray.Arrays.

    Examples
    --------
    For example, if we want to apply the Hodrick-Prescott filter from statsmodels we can use this:

    >>> from statsmodels.tsa.filters.hp_filter import hpfilter         # doctest: +SKIP
    >>> hpfilter = wrap_elementwise_array_func(hpfilter)               # doctest: +SKIP

    hpfilter is now a function taking a one dimensional Array as input and returning a one dimensional Array as output

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
        raw_bcast_args, raw_bcast_kwargs, res_axes = make_args_broadcastable(args, kwargs)

        # We pass only raw numpy arrays to the ufuncs even though numpy is normally meant to handle those cases itself
        # via __array_wrap__

        # There is a problem with np.clip though (and possibly other ufuncs): np.clip is roughly equivalent to
        # np.maximum(np.minimum(np.asarray(la), high), low)
        # the np.asarray(la) is problematic because it lose original labels
        # and then tries to get them back from high, where they are possibly
        # incomplete if broadcasting happened

        # It fails on "np.minimum(ndarray, Array)" because it calls __array_wrap__(high, result) which cannot work if
        # there was broadcasting involved (high has potentially less labels than result).
        # it does this because numpy calls __array_wrap__ on the argument with the highest __array_priority__
        res_data = func(*raw_bcast_args, **raw_bcast_kwargs)
        if res_axes:
            if isinstance(res_data, tuple):
                return tuple(Array(res_arr, res_axes) for res_arr in res_data)
            else:
                return Array(res_data, res_axes)
        else:
            return res_data
    # copy function name. We are intentionally not using functools.wraps, because it does not work for wrapping a
    # function from another module
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__ if doc is None else doc
    return wrapper


def wrap_numpy_func(func, doc=None):
    # update documentation by inserting a warning message after the short description of the numpy function
    # (otherwise the description of ufuncs given in the corresponding API 'autosummary' tables will always
    #  start with 'larray specific variant of ...' without giving a meaningful description of what does the ufunc)
    if doc is None:
        if func.__doc__.startswith('\n'):
            # docstring starts with short description
            end_signature = 1
            end_short_desc = func.__doc__.find('\n\n')
        else:
            # docstring starts with signature
            end_signature = func.__doc__.find('\n\n') + 2
            end_short_desc = func.__doc__.find('\n\n', end_signature)
        short_desc = func.__doc__[:end_short_desc]
        numpy_doc = func.__doc__[end_short_desc:]
        ident = ' ' * (len(short_desc[end_signature:]) - len(short_desc[end_signature:].lstrip()))
        doc = f'{short_desc}\n\n{ident}larray specific variant of ``numpy.{func.__name__}``.\n\n' \
              f'{ident}Documentation from numpy:{numpy_doc}'
    wrapper = wrap_elementwise_array_func(func, doc)

    # set __qualname__ explicitly (all these functions are supposed to be top-level function in the ufuncs module)
    wrapper.__qualname__ = func.__name__
    # we should not copy __module__
    return wrapper


where = wrap_numpy_func(np.where, r"""
where(condition, x, y)

    Return elements, either from `x` or `y`, depending on `condition`.

    Parameters
    ----------
    condition : boolean Array
        When True, yield `x`, otherwise yield `y`.
    x, y : Array
        Values from which to choose.

    Returns
    -------
    out : Array
        If both `x` and `y` are specified, the output array contains
        elements of `x` where `condition` is True, and elements from
        `y` elsewhere.

    Examples
    --------
    >>> from larray import Array
    >>> arr = Array([[10, 7, 5, 9],
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
""")

maximum = wrap_numpy_func(np.maximum, r"""
maximum(x1, x2, out=None, dtype=None)

    Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : Array
        The arrays holding the elements to be compared.
    out : Array, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.

    Returns
    -------
    y : Array or scalar
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
    >>> from larray import Array
    >>> arr1 = Array([[10, 7, 5, 9],
    ...                [5, 8, 3, 7]], "a=a0,a1;b=b0..b3")
    >>> arr2 = Array([[6, 2, 9, 0],
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
""")

minimum = wrap_numpy_func(np.minimum, r"""
minimum(x1, x2, out=None, dtype=None)

    Element-wise minimum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : Array
        The arrays holding the elements to be compared.
    out : Array, optional
        An array into which the result is stored.
    dtype : data-type, optional
        Overrides the dtype of the output array.

    Returns
    -------
    y : Array or scalar
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
    >>> from larray import Array
    >>> arr1 = Array([[10, 7, 5, 9],
    ...                [5, 8, 3, 7]], "a=a0,a1;b=b0..b3")
    >>> arr2 = Array([[6, 2, 9, 0],
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
""")

def _generalized_isnan(arr, out=None, where=True, **kwargs):
    if isinstance(arr, np.ndarray) and arr.dtype.kind == 'O':
        if out is not None or where is not True or kwargs:
            raise ValueError("The 'out', 'where' and other keyword arguments "
                             "are not supported for object arrays.")
        return arr != arr
    else:
        return np.isnan(arr, out=out, where=where, **kwargs)

isnan = wrap_elementwise_array_func(_generalized_isnan, r"""
Test element-wise for NaN and return result as a boolean array.

Parameters
----------
x : array_like
    Input array.
out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray or bool
    True where ``x`` is NaN, false otherwise.
    This is a scalar if `x` is a scalar.

See Also
--------
isinf, isneginf, isposinf, isfinite, isnat

Notes
-----
Contrary to the numpy implementation, this function support object arrays.

NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
(IEEE 754). This means that Not a Number is not equivalent to infinity.

Examples
--------
>>> import larray as la
>>> la.isnan(la.nan)
True
>>> la.isnan(la.inf)
False
>>> arr = la.Array([la.nan, 1, la.inf], 
...                la.Axis(3, 'values'))
>>> la.isnan(arr)
values*     0      1      2
         True  False  False
>>> arr = la.Array(['abc', 1, la.nan],
...                la.Axis(3, 'values'), dtype=object)
>>> la.isnan(arr)
values*      0      1     2
         False  False  True
""")


def _generalized_nan_to_num(arr, copy=True, nan=0, posinf=None, neginf=None):
    if isinstance(arr, np.ndarray) and arr.dtype.kind == 'O':
        import sys
        if posinf is None:
            posinf = sys.float_info.max
        if neginf is None:
            neginf = -sys.float_info.max
        res = arr.copy() if copy else arr
        is_nan_value = arr != arr
        is_pos_inf_value = arr == np.inf
        is_neg_inf_value = arr == -np.inf
        if isinstance(nan, np.ndarray):
            # each array argument is reshaped to a compatible shape for
            # broadcasting by larray machinery but not actually broadcasted yet
            nan = np.broadcast_to(nan, arr.shape)[is_nan_value]
        res[is_nan_value] = nan
        if isinstance(posinf, np.ndarray):
            posinf = np.broadcast_to(posinf, arr.shape)[is_pos_inf_value]
        res[is_pos_inf_value] = posinf
        if isinstance(neginf, np.ndarray):
            neginf = np.broadcast_to(neginf, arr.shape)[is_neg_inf_value]
        res[is_neg_inf_value] = neginf
        return res
    else:
        return np.nan_to_num(arr, copy=copy, nan=nan, posinf=posinf, neginf=neginf)

nan_to_num = wrap_elementwise_array_func(_generalized_nan_to_num,r"""
Replace NaN with zero and infinity with large finite numbers (default
behaviour) or with the numbers defined by the user using the `nan`,
`posinf` and/or `neginf` keywords.

If `x` is inexact or an object array, NaN is replaced by zero or by the user
defined value in `nan` keyword, infinity is replaced by the largest finite 
floating point value representable by ``x.dtype`` or by the user defined 
value in `posinf` keyword and -infinity is replaced by the most negative 
finite floating point value representable by ``x.dtype`` or by the user 
defined value in `neginf` keyword.

For complex dtypes, the above is applied to each of the real and
imaginary components of `x` separately.

If `x` is not inexact or object, then no replacements are made.

Parameters
----------
x : scalar or array_like
    Input data.
copy : bool, optional
    Whether to create a copy of `x` (True) or to replace values
    in-place (False). The in-place operation only occurs if
    casting to an array does not require a copy.
    Default is True.
nan : int, float or array_like, optional
    Value to be used to fill NaN values. If no value is passed
    then NaN values will be replaced with 0.0.
posinf : int, float, optional
    Value to be used to fill positive infinity values. If no value is
    passed then positive infinity values will be replaced with the largest 
    finite floating point value representable by ``x.dtype``.
neginf : int, float, optional
    Value to be used to fill negative infinity values. If no value is
    passed then negative infinity values will be replaced with the most 
    negative finite floating point value representable by ``x.dtype``.

Returns
-------
out : Array or scalar
    `x`, with the non-finite values replaced. If `copy` is False, this may
    be `x` itself.

See Also
--------
isinf : Shows which elements are positive or negative infinity.
isneginf : Shows which elements are negative infinity.
isposinf : Shows which elements are positive infinity.
isnan : Shows which elements are Not a Number (NaN).
isfinite : Shows which elements are finite (not NaN, not infinity)

Notes
-----
Contrary to the numpy implementation, this function support object arrays.

NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
(IEEE 754). This means that Not a Number is not equivalent to infinity.

Examples
--------
>>> import larray as la

>>> la.nan_to_num(la.inf)
1.7976931348623157e+308
>>> la.nan_to_num(-la.inf)
-1.7976931348623157e+308
>>> la.nan_to_num(np.nan)
0.0

>>> x = la.Array([-la.inf, 1, la.nan, 2, la.inf], la.Axis(5, 'values'))
>>> la.nan_to_num(x)
values*                         0    1    2    3                        4
         -1.7976931348623157e+308  1.0  0.0  2.0  1.7976931348623157e+308
>>> la.nan_to_num(x, nan=-1, posinf=999, neginf=-999)
values*       0    1     2    3      4
         -999.0  1.0  -1.0  2.0  999.0

>>> x = la.Array([1, 'abc', la.nan, 2], la.Axis(4, 'values'), dtype=object)
>>> la.nan_to_num(x)
values*  0    1  2  3
         1  abc  0  2

>>> y = la.Array([complex(la.inf, la.nan), la.nan, complex(la.nan, la.inf)],
...              la.Axis(3, 'values'))
>>> la.nan_to_num(y)
values*                             0   1                         2
         (1.7976931348623157e+308+0j)  0j  1.7976931348623157e+308j
""")