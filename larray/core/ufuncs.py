# numpy ufuncs
# http://docs.scipy.org/doc/numpy/reference/routines.math.html
import numpy as np

from larray.core.array import Array, make_args_broadcastable


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
        A function taking LArray arrays arguments and returning LArray arrays.

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
    return wrapper


# TODO: rename to wrap_numpy_func
def broadcastify(func):
    wrapper = wrap_elementwise_array_func(func)
    # update documentation by inserting a warning message after the short description of the numpy function
    # (otherwise the description of ufuncs given in the corresponding API 'autosummary' tables will always
    #  start with 'larray specific variant of ...' without giving a meaningful description of what does the ufunc)
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
    wrapper.__doc__ = f'{short_desc}\n\n{ident}larray specific variant of ``numpy.{func.__name__}``.\n\n' \
                      f'{ident}Documentation from numpy:{numpy_doc}'
    # set __qualname__ explicitly (all these functions are supposed to be top-level function in the ufuncs module)
    wrapper.__qualname__ = func.__name__
    # we should not copy __module__
    return wrapper


where = broadcastify(np.where)
where.__doc__ = r"""
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
"""

maximum = broadcastify(np.maximum)
maximum.__doc__ = r"""
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
"""

minimum = broadcastify(np.minimum)
minimum.__doc__ = r"""
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
"""
