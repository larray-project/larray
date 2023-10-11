"""
Array class.
"""

# ? implement multi group in one axis getitem: lipro['P01,P02;P05'] <=> (lipro['P01,P02'], lipro['P05'])

# * we need an API to get to the "next" label. Sometimes, we want to use label+1, but that is problematic when labels
#   are not numeric, or have not a step of 1.
#       X.agegroup[X.agegroup.after(25):]
#       X.agegroup[X.agegroup[25].next():]

# * implement keepaxes=True for _group_aggregate instead of/in addition to group tuples

# ? implement newaxis

# * Axis.sequence? geo.seq('A31', 'A38') (equivalent to geo['A31..A38'])

# ? re-implement row_totals/col_totals? or what do we do with them?

# * time specific API so that we know if we go for a subclass or not

# * data alignment in arithmetic methods

# * test structured arrays

# * use larray "utils" in LIAM2 (to avoid duplicated code)

from itertools import product, chain, groupby
from collections.abc import Iterable, Sequence
from pathlib import Path
import builtins
import functools
import warnings

from typing import Any, Union, Tuple, List

import numpy as np
import pandas as pd

try:
    import xlwings as xw
except ImportError:
    xw = None

try:
    from numpy import nanprod as np_nanprod
except ImportError:
    np_nanprod = None

from larray.core.abstractbases import ABCArray
from larray.core.constants import nan, inf
from larray.core.metadata import Metadata
from larray.core.expr import ExprNode
from larray.core.group import (Group, IGroup, LGroup, _to_key, _to_keys,
                               _translate_sheet_name, _translate_group_key_hdf)
from larray.core.axis import Axis, AxisReference, AxisCollection, X, _make_axis         # noqa: F401
from larray.core.plot import PlotObject
from larray.util.misc import (table2str, size2str, ReprString,
                              float_error_handler_factory, light_product, common_dtype,
                              renamed_to, deprecate_kwarg, LHDFStore, lazy_attribute, unique_multi, SequenceZip,
                              Repeater, Product, ensure_no_numpy_type, exactly_one, concatenate_ndarrays)
from larray.util.options import _OPTIONS, DISPLAY_MAXLINES, DISPLAY_EDGEITEMS, DISPLAY_WIDTH, DISPLAY_PRECISION
from larray.util.types import Scalar


def all(values, axis=None) -> Union['Array', Scalar]:
    r"""
    Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    Array.all
    """
    if isinstance(values, Array):
        return values.all(axis)
    else:
        return builtins.all(values)


def any(values, axis=None) -> Union['Array', Scalar]:
    r"""
    Test whether any array elements along a given axis evaluate to True.

    See Also
    --------
    Array.any
    """
    if isinstance(values, Array):
        return values.any(axis)
    else:
        return builtins.any(values)


# commutative modulo float precision errors
def sum(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Sum of array elements.

    See Also
    --------
    Array.sum
    """
    # XXX: we might want to be more aggressive here (more types to convert), however, generators should still be
    #      computed via the builtin.
    if isinstance(array, (np.ndarray, list)):
        array = Array(array)
    if isinstance(array, Array):
        return array.sum(*args, **kwargs)
    else:
        return builtins.sum(array, *args, **kwargs)


def prod(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Product of array elements.

    See Also
    --------
    Array.prod
    """
    return array.prod(*args, **kwargs)


def cumsum(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Return the cumulative sum of array elements.

    See Also
    --------
    Array.cumsum
    """
    return array.cumsum(*args, **kwargs)


def cumprod(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Return the cumulative product of array elements.

    See Also
    --------
    Array.cumprod
    """
    return array.cumprod(*args, **kwargs)


def min(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Minimum of array elements.

    See Also
    --------
    Array.min
    """
    if isinstance(array, Array):
        return array.min(*args, **kwargs)
    else:
        return builtins.min(array, *args, **kwargs)


def max(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Maximum of array elements.

    See Also
    --------
    Array.max
    """
    if isinstance(array, Array):
        return array.max(*args, **kwargs)
    else:
        return builtins.max(array, *args, **kwargs)


def mean(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the arithmetic mean.

    See Also
    --------
    Array.mean
    """
    return array.mean(*args, **kwargs)


def median(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the median.

    See Also
    --------
    Array.median
    """
    return array.median(*args, **kwargs)


def percentile(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the qth percentile of the data along the specified axis.

    See Also
    --------
    Array.percentile
    """
    return array.percentile(*args, **kwargs)


# not commutative
def ptp(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Return the range of values (maximum - minimum).

    See Also
    --------
    Array.ptp
    """
    return array.ptp(*args, **kwargs)


def var(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the variance.

    See Also
    --------
    Array.var
    """
    return array.var(*args, **kwargs)


def std(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the standard deviation.

    See Also
    --------
    Array.std
    """
    return array.std(*args, **kwargs)


def concat(arrays, axis=0, dtype=None):
    r"""Concatenate arrays along axis.

    Parameters
    ----------
    arrays : tuple of Array
        Arrays to concatenate.
    axis : axis reference (int, str or Axis), optional
        Axis along which to concatenate. All arrays must have that axis. Defaults to the first axis.
    dtype : dtype, optional
        Result data type. Defaults to the "closest" type which can hold all arrays types without loss of information.

    Returns
    -------
    Array

    Examples
    --------
    >>> arr1 = ndtest((2, 3))
    >>> arr1
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr2 = ndtest('a=a0,a1;b=b3')
    >>> arr2
    a\b  b3
     a0   0
     a1   1
    >>> arr3 = ndtest('b=b4,b5')
    >>> arr3
    b  b4  b5
        0   1
    >>> concat((arr1, arr2, arr3), 'b')
    a\b  b0  b1  b2  b3  b4  b5
     a0   0   1   2   0   0   1
     a1   3   4   5   1   0   1
    """
    # Get axis by name, so that we do *NOT* check they are "compatible", because it makes sense to append axes of
    # different length
    name = arrays[0].axes[axis].name
    arrays_labels = [array.axes[axis].labels for array in arrays]

    # switch to object dtype if labels are of incompatible types, so that we do not implicitly convert numeric types to
    # strings (numpy should not do this in the first place but that is another story). This can happen for example when
    # we want to add a "total" tick to a numeric axis (eg age).
    combined_axis = Axis(concatenate_ndarrays(arrays_labels), name)

    # combine all axes (using labels from any side if any)
    result_axes = arrays[0].axes.replace(axis, combined_axis).union(*[array.axes - axis for array in arrays[1:]])

    if dtype is None:
        dtype = common_dtype(arrays)

    result = empty(result_axes, dtype=dtype)
    start = 0
    for labels, array in zip(arrays_labels, arrays):
        stop = start + len(labels)
        result[combined_axis.i[start:stop]] = array
        start = stop
    return result


class ArrayIterator:
    __slots__ = ('__next__',)

    def __init__(self, array):
        data_iter = iter(array.data)
        next_data_func = data_iter.__next__
        res_axes = array.axes[1:]
        # this case should not happen (handled by the fastpath in Array.__iter__)
        assert len(res_axes) > 0  # noqa: S101

        def next_func():
            return Array(next_data_func(), res_axes)

        self.__next__ = next_func

    def __iter__(self):
        return self


# TODO: rename to ArrayIndexIndexer or something like that
# TODO: the first slice in the example below should be documented
class ArrayPositionalIndexer:
    r"""
    Allows selection of a subset using indices of labels.

    Notes
    -----
    Using .i[] is equivalent to numpy indexing when indexing along a single axis. However, when indexing along multiple
    axes this indexes the cross product instead of points.

    Examples
    --------
    >>> arr = ndtest((2, 3, 4))
    >>> arr
     a  b\c  c0  c1  c2  c3
    a0   b0   0   1   2   3
    a0   b1   4   5   6   7
    a0   b2   8   9  10  11
    a1   b0  12  13  14  15
    a1   b1  16  17  18  19
    a1   b2  20  21  22  23

    >>> arr.i[:, 0:2, [0, 2]]
     a  b\c  c0  c2
    a0   b0   0   2
    a0   b1   4   6
    a1   b0  12  14
    a1   b1  16  18
    """

    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        array = self.array
        ndim = array.ndim
        full_scalar_key = (
            (isinstance(key, (int, np.integer)) and ndim == 1)
            or (isinstance(key, tuple) and len(key) == ndim and all(isinstance(k, (int, np.integer)) for k in key))
        )
        # fast path when the result is a scalar
        if full_scalar_key:
            return array.data[key]
        else:
            return array.__getitem__(key, translate_key=False)

    def __setitem__(self, key, value):
        array = self.array
        ndim = array.ndim
        full_scalar_key = (
            (isinstance(key, (int, np.integer)) and ndim == 1)
            or (isinstance(key, tuple) and len(key) == ndim and all(isinstance(k, (int, np.integer)) for k in key))
        )
        # fast path when setting a single cell
        if full_scalar_key:
            array.data[key] = value
        else:
            array.__setitem__(key, value, translate_key=False)

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        array = self.array
        # fast path for 1D arrays (where we return scalars)
        if array.ndim <= 1:
            return iter(array.data)
        else:
            return ArrayIterator(array)


class ArrayPointsIndexer:
    r"""
    Allows selection of arbitrary items in the array based on their N-dimensional label index.

    Examples
    --------
    >>> arr = ndtest((2, 3, 4))
    >>> arr
     a  b\c  c0  c1  c2  c3
    a0   b0   0   1   2   3
    a0   b1   4   5   6   7
    a0   b2   8   9  10  11
    a1   b0  12  13  14  15
    a1   b1  16  17  18  19
    a1   b2  20  21  22  23

    To select the two points with label coordinates
    [a0, b0, c0] and [a1, b2, c2], you must do:

    >>> arr.points[['a0', 'a1'], ['b0', 'b2'], ['c0', 'c2']]
    a_b_c  a0_b0_c0  a1_b2_c2
                  0        22
    >>> arr.points['a0,a1', 'b0,b2', 'c0,c2']
    a_b_c  a0_b0_c0  a1_b2_c2
                  0        22

    The number of label(s) on each dimension must be equal:

    >>> arr.points['a0,a1', 'b0,b2', 'c0,c1,c2']  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: all combined keys should have the same length
    """

    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return self.array.__getitem__(key, points=True)

    def __setitem__(self, key, value):
        self.array.__setitem__(key, value, points=True)


# TODO: add support for slices
#     To select the first 4 values across all axes:
#
#     >>> arr.iflat[:4]
#     a_b  a0_b0  a0_b1  a0_b2  a1_b0
#              0     10     20     30
class ArrayFlatIndicesIndexer:
    r"""
    Access the array by index as if it was flat (one dimensional) and all its axes were combined.

    Notes
    -----
    In general arr.iflat[key] should be equivalent to (but much faster than) arr.combine_axes().i[key]

    Examples
    --------
    >>> arr = ndtest((2, 3)) * 10
    >>> arr
    a\b  b0  b1  b2
     a0   0  10  20
     a1  30  40  50

    To select the first, second, fourth and fifth values across all axes:

    >>> arr.combine_axes().i[[0, 1, 3, 4]]
    a_b  a0_b0  a0_b1  a1_b0  a1_b1
             0     10     30     40
    >>> arr.iflat[[0, 1, 3, 4]]
    a_b  a0_b0  a0_b1  a1_b0  a1_b1
             0     10     30     40

    Set the first and sixth values to 42

    >>> arr.iflat[[0, 5]] = 42
    >>> arr
    a\b  b0  b1  b2
     a0  42  10  20
     a1  30  40  42

    When the key is an Array, the result will have the axes of the key

    >>> key = Array([0, 3], 'c=c0,c1')
    >>> key
    c  c0  c1
        0   3
    >>> arr.iflat[key]
    c  c0  c1
       42  30
    """

    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, flat_key, sep='_'):
        if isinstance(flat_key, ABCArray):
            flat_np_key = flat_key.data
            res_axes = flat_key.axes
        else:
            flat_np_key = np.asarray(flat_key)
            res_axes = self.array.axes._combined_iflat(flat_np_key, sep=sep)
        return Array(self.array.data.flat[flat_np_key], res_axes)

    def __setitem__(self, flat_key, value):
        # np.ndarray.flat is a flatiter object but it is indexable despite the name
        self.array.data.flat[flat_key] = value

    def __len__(self):
        return self.array.size


# TODO: rename to ArrayIndexPointsIndexer or something like that
# TODO: show that we need to use a "full slice" for leaving the dimension alone
# TODO: document explicitly that axes should be in the correct order and missing axes should be slice None
# (except at the end)
class ArrayPositionalPointsIndexer:
    r"""
    Allows selection of arbitrary items in the array based on their N-dimensional index.

    Examples
    --------
    >>> arr = ndtest((2, 3, 4))
    >>> arr
     a  b\c  c0  c1  c2  c3
    a0   b0   0   1   2   3
    a0   b1   4   5   6   7
    a0   b2   8   9  10  11
    a1   b0  12  13  14  15
    a1   b1  16  17  18  19
    a1   b2  20  21  22  23

    To select the two points with index coordinates
    [0, 0, 0] and [1, 2, 2], you must do:

    >>> arr.ipoints[[0, 1], [0, 2], [0, 2]]
    a_b_c  a0_b0_c0  a1_b2_c2
                  0        22

    The number of index(es) on each dimension must be equal:

    >>> arr.ipoints[[0, 1], [0, 2], [0, 1, 2]]  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: all combined keys should have the same length

    >>> arr.ipoints[[0, 1], [0, 2]]
    a_b\c  c0  c1  c2  c3
    a0_b0   0   1   2   3
    a1_b2  20  21  22  23
    """

    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return self.array.__getitem__(key, translate_key=False, points=True)

    def __setitem__(self, key, value):
        self.array.__setitem__(key, value, translate_key=False, points=True)


def get_axis(obj, i):
    r"""
    Return an axis according to its position.

    Parameters
    ----------
    obj : Array or other array
        Input Array or any array object which has a shape attribute (NumPy or Pandas array).
    i : int
        index of the axis.

    Returns
    -------
    Axis
        Axis corresponding to the given index if input `obj` is an Array. A new anonymous Axis with the length of
        the ith dimension of the input `obj` otherwise.

    Examples
    --------
    >>> arr = ndtest((2, 2, 2))
    >>> arr
     a  b\c  c0  c1
    a0   b0   0   1
    a0   b1   2   3
    a1   b0   4   5
    a1   b1   6   7
    >>> get_axis(arr, 1)
    Axis(['b0', 'b1'], 'b')
    >>> np_arr = np.zeros((2, 2, 2))
    >>> get_axis(np_arr, 1)
    Axis(2, None)
    """
    return obj.axes[i] if isinstance(obj, Array) else Axis(obj.shape[i])


_arg_agg = {
    'q': """
        q : int in range of [0,100] (or sequence of floats)
            Percentile to compute, which must be between 0 and 100 inclusive."""
}

_kwarg_agg = {
    'dtype': {'value': None, 'doc': """
        dtype : dtype, optional
            The data type of the returned array. Defaults to None (the dtype of the input array)."""},
    'out': {'value': None, 'doc': """
        out : Array, optional
            Alternate output array in which to place the result. It must have the same shape as the expected output and
            its type is preserved (e.g., if dtype(out) is float, the result will consist of 0.0's and 1.0's).
            Axes and labels can be different, only the shape matters. Defaults to None (create a new array)."""},
    'ddof': {'value': 1, 'doc': """
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is ``N - ddof``, where ``N`` represents
            the number of elements. Defaults to 1."""},
    'skipna': {'value': None, 'doc': """
        skipna : bool, optional
            Whether to skip NaN (null) values. If False, resulting cells will be NaN if any of the aggregated
            cells is NaN. Defaults to True."""},
    'keepaxes': {'value': False, 'doc': """
        keepaxes : bool or label-like, optional
            Whether reduced axes are left in the result as dimensions with size one.
            If True, reduced axes will contain a unique label representing the applied aggregation
            (e.g. 'sum', 'prod', ...). It is possible to override this label by passing a specific value
            (e.g. keepaxes='summation'). Defaults to False."""},
    'method': {'value': 'linear', 'doc': """
        method : str, optional
            This parameter specifies the method to use for estimating the
            percentile when the desired percentile lies between two indexes.
            The different methods supported are described in the Notes section. The options are:
                * 'inverted_cdf'
                * 'averaged_inverted_cdf'
                * 'closest_observation'
                * 'interpolated_inverted_cdf'
                * 'hazen'
                * 'weibull'
                * 'linear'  (default)
                * 'median_unbiased'
                * 'normal_unbiased'
                * 'lower'
                * 'higher'
                * 'midpoint'
                * 'nearest'
            The first three and last four methods are discontinuous. Defaults to 'linear'."""}
}

PERCENTILE_NOTES = """Notes
        -----
        Given a vector ``V`` of length ``n``, the q-th percentile of ``V`` is
        the value ``q/100`` of the way from the minimum to the maximum in a
        sorted copy of ``V``. The values and distances of the two nearest
        neighbors as well as the `method` parameter will determine the
        percentile if the normalized ranking does not match the location of
        ``q`` exactly. This function is the same as the median if ``q=50``, the
        same as the minimum if ``q=0`` and the same as the maximum if
        ``q=100``.
    
        The optional `method` parameter specifies the method to use when the
        desired percentile lies between two indexes ``i`` and ``j = i + 1``.
        In that case, we first determine ``i + g``, a virtual index that lies
        between ``i`` and ``j``, where  ``i`` is the floor and ``g`` is the
        fractional part of the index. The final result is, then, an interpolation
        of ``a[i]`` and ``a[j]`` based on ``g``. During the computation of ``g``,
        ``i`` and ``j`` are modified using correction constants ``alpha`` and
        ``beta`` whose choices depend on the ``method`` used. Finally, note that
        since Python uses 0-based indexing, the code subtracts another 1 from the
        index internally.
    
        The following formula determines the virtual index ``i + g``, the location
        of the percentile in the sorted sample:
    
        .. math::
            i + g = (q / 100) * ( n - alpha - beta + 1 ) + alpha
    
        The different methods then work as follows
    
        inverted_cdf:
            method 1 of H&F [1]_.
            This method gives discontinuous results:
    
            * if g > 0 ; then take j
            * if g = 0 ; then take i
    
        averaged_inverted_cdf:
            method 2 of H&F [1]_.
            This method give discontinuous results:
    
            * if g > 0 ; then take j
            * if g = 0 ; then average between bounds
    
        closest_observation:
            method 3 of H&F [1]_.
            This method give discontinuous results:
    
            * if g > 0 ; then take j
            * if g = 0 and index is odd ; then take j
            * if g = 0 and index is even ; then take i
    
        interpolated_inverted_cdf:
            method 4 of H&F [1]_.
            This method give continuous results using:
    
            * alpha = 0
            * beta = 1
    
        hazen:
            method 5 of H&F [1]_.
            This method give continuous results using:
    
            * alpha = 1/2
            * beta = 1/2
    
        weibull:
            method 6 of H&F [1]_.
            This method give continuous results using:
    
            * alpha = 0
            * beta = 0
    
        linear:
            method 7 of H&F [1]_.
            This method give continuous results using:
    
            * alpha = 1
            * beta = 1
    
        median_unbiased:
            method 8 of H&F [1]_.
            This method is probably the best method if the sample
            distribution function is unknown (see reference).
            This method give continuous results using:
    
            * alpha = 1/3
            * beta = 1/3
    
        normal_unbiased:
            method 9 of H&F [1]_.
            This method is probably the best method if the sample
            distribution function is known to be normal.
            This method give continuous results using:
    
            * alpha = 3/8
            * beta = 3/8
    
        lower:
            NumPy method kept for backwards compatibility.
            Takes ``i`` as the interpolation point.
    
        higher:
            NumPy method kept for backwards compatibility.
            Takes ``j`` as the interpolation point.
    
        nearest:
            NumPy method kept for backwards compatibility.
            Takes ``i`` or ``j``, whichever is nearest.
    
        midpoint:
            NumPy method kept for backwards compatibility.
            Uses ``(i + j) / 2``."""


def _doc_agg_method(func, by=False, long_name='', action_verb='perform', extra_args=(), kwargs=()):
    if not long_name:
        long_name = func.__name__

    _args = ','.join(extra_args) + ', ' if len(extra_args) > 0 else ''
    _kwargs = ', '.join([f"{k}={_kwarg_agg[k]['value']!r}" for k in kwargs]) + ', ' if len(kwargs) > 0 else ''
    signature = f'{func.__name__}({_args}*axes_and_groups, {_kwargs}**explicit_axes)'

    if by:
        specific_template = """The {long_name} is {action_verb}ed along all axes except the given one(s).
            For groups, {long_name} is {action_verb}ed along groups and non associated axes."""
    else:
        specific_template = "Axis(es) or group(s) along which the {long_name} is {action_verb}ed."
    doc_specific = specific_template.format(long_name=long_name, action_verb=action_verb)

    doc_args = "".join(_arg_agg[arg] for arg in extra_args)
    doc_kwargs = "".join(_kwarg_agg[kw]['doc'] for kw in kwargs)
    doc_varargs = fr"""
        \*axes_and_groups : None or int or str or Axis or Group or any combination of those
            {doc_specific}
            The default (no axis or group) is to {action_verb} the {long_name} over all the dimensions of the input
            array.

            An axis can be referred by:

            * its index (integer). Index can be a negative integer, in which case it counts from the last to the
              first axis.
            * its name (str or AxisReference). You can use either a simple string ('axis_name') or the special
              variable X (X.axis_name).
            * a variable (Axis). If the axis has been defined previously and assigned to a variable, you can pass it as
              argument.

            You may not want to {action_verb} the {long_name} over a whole axis but over a selection of specific
            labels. To do so, you have several possibilities:

            * (['a1', 'a3', 'a5'], 'b1, b3, b5') : labels separated by commas in a list or a string
            * ('a1:a5:2') : select labels using a slice (general syntax is 'start:end:step' where is 'step' is
              optional and 1 by default).
            * (a='a1, a2, a3', X.b['b1, b2, b3']) : in case of possible ambiguity, i.e. if labels can belong to more
              than one axis, you must precise the axis.
            * ('a1:a3; a5:a7', b='b0,b2; b1,b3') : create several groups with semicolons.
              Names are simply given by the concatenation of labels (here: 'a1,a2,a3', 'a5,a6,a7', 'b0,b2' and 'b1,b3')
            * ('a1:a3 >> a123', 'b[b0,b2] >> b12') : operator ' >> ' allows to rename groups."""
    parameters = f"""Parameters
        ----------{doc_args}{doc_varargs}{doc_kwargs}"""
    func.__doc__ = func.__doc__.format(signature=signature, parameters=parameters, percentile_notes=PERCENTILE_NOTES)


_always_return_float = {np.mean, np.nanmean, np.median, np.nanmedian, np.percentile, np.nanpercentile,
                        np.std, np.nanstd, np.var, np.nanvar}

obj_isnan = np.vectorize(lambda x: x != x, otypes=[bool])


def element_equal(a1, a2, rtol=0, atol=0, nan_equals=False):
    warnings.warn("element_equal() is deprecated. Use array1.eq(array2, rtol, atol, nan_equals) instead.",
                  FutureWarning, stacklevel=2)
    a1 = asarray(a1)
    return a1.eq(a2, rtol, atol, nan_equals)


def nan_equal(a1, a2):
    warnings.warn("nan_equal() is deprecated. Use array1.eq(array2, nans_equal=True) instead.",
                  FutureWarning, stacklevel=2)
    return a1.eq(a2, nans_equal=True)


def _handle_meta(meta, title):
    """
    Make sure meta is either None or a Metadata instance.
    """
    if title is not None:
        if meta is None:
            meta = Metadata()
        warnings.warn("title argument is deprecated. Please use meta argument instead", FutureWarning, stacklevel=2)
        meta['title'] = title
    if meta is None or isinstance(meta, Metadata):
        return meta
    # XXX: move this test in Metadata.__init__?
    if not isinstance(meta, (list, dict)):
        raise TypeError(f"Expected None, list of pairs, dict or Metadata object "
                        f"instead of {type(meta).__name__}")
    return Metadata(meta)

# This prevents a warning in Pandas 1.4 <= version < 2.0 for arrays with object
# dtype which contain only numeric values. We force Pandas 2.0 behavior
# (ie use object dtype instead of inferring). See issue #1061.
def np_array_to_pd_index(array, name=None, tupleize_cols=True):
    dtype = None if array.dtype.kind != 'O' else object
    return pd.Index(array, dtype=dtype, name=name, tupleize_cols=tupleize_cols)


class Array(ABCArray):
    r"""
    An Array object represents a multidimensional, homogeneous array of fixed-size items with labeled axes.

    The function :func:`asarray` can be used to convert a NumPy array or Pandas DataFrame into an Array.

    Parameters
    ----------
    data : scalar, tuple, list or NumPy ndarray
        Input data.
    axes : collection (tuple, list or AxisCollection) of axes (int, str or Axis), optional
        Axes.
    title : str, optional
        Deprecated. See 'meta' below.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.
    dtype : type, optional
        Datatype for the array. Defaults to None (inferred from the data).

    Attributes
    ----------
    data : NumPy ndarray
        Data.
    axes : AxisCollection
        Axes.
    meta : Metadata
        Metadata (title, description, author, creation_date, ...) associated with the array.

    See Also
    --------
    sequence : Create an Array by sequentially applying modifications to the array along axis.
    ndtest : Create a test Array with increasing elements.
    zeros : Create an Array, each element of which is zero.
    ones : Create an Array, each element of which is 1.
    full : Create an Array filled with a given value.
    empty : Create an Array, but leave its allocated memory unchanged (i.e., it contains “garbage”).

    Warnings
    --------
    Metadata is not kept when actions or methods are applied on an array
    except for operations modifying the object in-place, such as: `pop[age < 10] = 0`.
    Do not add metadata to an array if you know you will apply actions or methods
    on it before dumping it.

    Examples
    --------
    >>> age = Axis([10, 11, 12], 'age')
    >>> sex = Axis('sex=M,F')
    >>> time = Axis([2007, 2008, 2009], 'time')
    >>> axes = [age, sex, time]
    >>> data = np.zeros((len(axes), len(sex), len(time)))

    >>> Array(data, axes)
    age  sex\time  2007  2008  2009
     10         M   0.0   0.0   0.0
     10         F   0.0   0.0   0.0
     11         M   0.0   0.0   0.0
     11         F   0.0   0.0   0.0
     12         M   0.0   0.0   0.0
     12         F   0.0   0.0   0.0
    >>> # with metadata
    >>> arr = Array(data, axes, meta=Metadata(title='my title', author='John Smith'))

    Array creation functions

    >>> full(axes, 10.0)
    age  sex\time  2007  2008  2009
     10         M  10.0  10.0  10.0
     10         F  10.0  10.0  10.0
     11         M  10.0  10.0  10.0
     11         F  10.0  10.0  10.0
     12         M  10.0  10.0  10.0
     12         F  10.0  10.0  10.0
    >>> arr = empty(axes)
    >>> arr['F'] = 1.0
    >>> arr['M'] = -1.0
    >>> arr
    age  sex\time  2007  2008  2009
     10         M  -1.0  -1.0  -1.0
     10         F   1.0   1.0   1.0
     11         M  -1.0  -1.0  -1.0
     11         F   1.0   1.0   1.0
     12         M  -1.0  -1.0  -1.0
     12         F   1.0   1.0   1.0
    >>> bysex = sequence(sex, initial=-1, inc=2)
    >>> bysex
    sex   M  F
         -1  1
    >>> sequence(age, initial=10, inc=bysex)
    sex\age  10  11  12
          M  10   9   8
          F  10  11  12
    """

    __slots__ = ('data', 'axes', '_meta')

    def __init__(self, data, axes=None, title=None, meta=None, dtype=None):
        data = np.asarray(data, dtype=dtype)
        ndim = data.ndim
        if axes is None:
            axes = AxisCollection(data.shape)
        else:
            if not isinstance(axes, AxisCollection):
                axes = AxisCollection(axes)
            if axes.ndim != ndim:
                raise ValueError(f"number of axes ({axes.ndim}) does not match "
                                 f"number of dimensions of data ({ndim})")
            if axes.shape != data.shape:
                raise ValueError(f"length of axes {axes.shape} does not match "
                                 f"data shape {data.shape}")

        self.data = data
        self.axes = axes

        if meta is not None or title is not None:
            meta = _handle_meta(meta, title)
        self._meta = meta

    @property
    def title(self) -> str:
        warnings.warn("title attribute is deprecated. Please use meta.title instead", FutureWarning, stacklevel=2)
        return self._meta.title if self._meta is not None and 'title' in self._meta else None

    @title.setter
    def title(self, title):
        warnings.warn("title attribute is deprecated. Please use meta.title instead", FutureWarning, stacklevel=2)
        if not isinstance(title, str):
            raise TypeError(f"Expected string value, got {type(title).__name__}")
        self._meta.title = title

    @property
    def meta(self) -> Metadata:
        r"""Return metadata of the array.

        Returns
        -------
        Metadata:
            Metadata of the array.
        """
        if self._meta is None:
            self._meta = Metadata()
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = _handle_meta(meta, None)

    # TODO: rename to inonzero and implement a label version of nonzero
    # TODO: implement wildcard argument to avoid producing the combined labels
    def nonzero(self) -> Tuple[IGroup, ...]:
        r"""
        Return the indices of the elements that are non-zero.

        Specifically, it returns a tuple of arrays (one for each dimension)
        containing the indices of the non-zero elements in that dimension.

        Returns
        -------
        tuple of arrays : tuple
            Indices of elements that are non-zero.

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> cond = arr > 1
        >>> cond
        a\b     b0     b1    b2
         a0  False  False  True
         a1   True   True  True
        >>> a, b = cond.nonzero()
        >>> a
        a.i[a_b  a0_b2  a1_b0  a1_b1  a1_b2
                 0      1      1      1]
        >>> b
        b.i[a_b  a0_b2  a1_b0  a1_b1  a1_b2
                 2      0      1      2]
        >>> # equivalent to arr[cond]
        >>> arr[cond.nonzero()]
        a_b  a0_b2  a1_b0  a1_b1  a1_b2
                 2      3      4      5
        """
        # the next step will be to return a Grid instead so that cond.nonzero() *displays*
        # (however it is stored!) as something like:

        # option a)

        # a_b  a0_b2  a1_b0  a1_b1  a1_b2
        #      a0,b2  a1,b0  a1,b1  a1,b2

        # PRO: * result axes are the same as grid axes
        # CON: * does not support getting the indexing for one axis (or at least it does not make it obvious that it
        #        is supported)
        #      * in the case of ambiguous labels (same label on several axes), this is not explicit enough

        # OR

        # option b)

        # source_axis\a_b  a0_b2  a1_b0  a1_b1  a1_b2
        #               a     a0     a1     a1     a1
        #               b     b2     b0     b1     b2

        # in the presence of duplicate labels on the same axis (e.g. assuming we replace 'b2' by a duplicate 'b1' label)

        # source_axis\a_b  a0_b1#1  a1_b0  a1_b1#0  a1_b1#1
        #               a       a0     a1       a1       a1
        #               b     b1#1     b0     b1#0     b1#1

        # OR

        # option c)

        # a_b\source_axis   a   b
        #           a0_b2  a0  b2
        #           a1_b0  a1  b0
        #           a1_b1  a1  b1
        #           a1_b2  a1  b2

        # Notes
        # -----
        # dtypes of a and b column can be different but since we probably only store indices, we will not even need
        # an LFrame so this shouldn't be a problem.
        ikey = self.data.nonzero()
        la_key = self.axes._adv_keys_to_combined_axis_la_keys(ikey)
        return tuple(IGroup(axis_key, axis=axis) for axis_key, axis in zip(la_key, self.axes))

    def set_axes(self, axes_to_replace=None, new_axis=None, inplace=False, **kwargs) -> 'Array':
        r"""
        Replace one, several or all axes of the array.

        Parameters
        ----------
        axes_to_replace : axis ref or dict {axis ref: axis} or list of (tuple or Axis) or AxisCollection
            Axes to replace. If a single axis reference is given, the `new_axis` argument must be provided.
            If a list of Axis or an AxisCollection is given, all axes will be replaced by the new ones.
            In that case, the number of new axes must match the number of the old ones.
            If a list of tuple is given, it must be pairs of (reference to old axis, new axis).
        new_axis : Axis, optional
            New axis if `axes_to_replace` contains a single axis reference.
        inplace : bool, optional
            Whether to modify the original object or return a new array and leave the original intact.
            Defaults to False.
        **kwargs : Axis
            New axis for each axis to replace given as a keyword argument.

        Returns
        -------
        Array
            Array with axes replaced.

        See Also
        --------
        rename : rename one of several axes

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> row = Axis(['r0', 'r1'], 'row')
        >>> column = Axis(['c0', 'c1', 'c2'], 'column')

        Replace one axis (second argument `new_axis` must be provided)

        >>> arr.set_axes('a', row)
        row\b  b0  b1  b2
           r0   0   1   2
           r1   3   4   5

        Replace several axes (keywords, list of tuple or dictionary)

        >>> arr.set_axes(a=row, b=column) # doctest: +SKIP
        >>> # or
        >>> arr.set_axes([('a', row), ('b', column)]) # doctest: +SKIP
        >>> # or
        >>> arr.set_axes({'a': row, 'b': column})
        row\column  c0  c1  c2
                r0   0   1   2
                r1   3   4   5

        Replace all axes (list of axes or AxisCollection)

        >>> arr.set_axes([row, column])
        row\column  c0  c1  c2
                r0   0   1   2
                r1   3   4   5
        >>> arr2 = ndtest([row, column])
        >>> arr.set_axes(arr2.axes)
        row\column  c0  c1  c2
                r0   0   1   2
                r1   3   4   5
        """
        new_axes = self.axes.replace(axes_to_replace, new_axis, **kwargs)
        if inplace:
            if new_axes.ndim != self.ndim:
                raise ValueError(f"number of axes ({new_axes.ndim}) does not match number of dimensions "
                                 f"of data ({self.ndim})")
            if new_axes.shape != self.data.shape:
                raise ValueError(f"length of axes {new_axes.shape} does not match data shape {self.data.shape}")
            self.axes = new_axes
            return self
        else:
            return Array(self.data, new_axes)

    with_axes = renamed_to(set_axes, 'with_axes', raise_error=True)

    def __getattr__(self, key) -> Axis:
        if key in self.axes:
            return self.axes[key]
        else:
            class_name = self.__class__.__name__
            raise AttributeError(f"'{class_name}' object has no attribute '{key}'")

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before .axes exists, which leads to
    # an infinite recursion)
    def __getstate__(self):
        return self.data, self.axes, self._meta

    def __setstate__(self, d):
        self.data, self.axes, self._meta = d

    def __dir__(self):
        axis_names = set(axis.name for axis in self.axes if axis.name is not None)
        attributes = self.__slots__
        return list(set(dir(self.__class__)) | set(attributes) | axis_names)

    def _ipython_key_completions_(self):
        return list(chain(*[list(labels) for labels in self.axes.labels]))

    @lazy_attribute
    def i(self) -> ArrayPositionalIndexer:
        return ArrayPositionalIndexer(self)
    i.__doc__ = ArrayPositionalIndexer.__doc__

    @lazy_attribute
    def points(self) -> ArrayPointsIndexer:
        return ArrayPointsIndexer(self)
    points.__doc__ = ArrayPointsIndexer.__doc__

    @lazy_attribute
    def ipoints(self) -> ArrayPositionalPointsIndexer:
        return ArrayPositionalPointsIndexer(self)
    ipoints.__doc__ = ArrayPositionalPointsIndexer.__doc__

    def to_frame(self, fold_last_axis_name=False, dropna=None) -> pd.DataFrame:
        r"""
        Convert an Array into a Pandas DataFrame.

        Parameters
        ----------
        fold_last_axis_name : bool, optional
            Defaults to False.
        dropna : {'any', 'all', None}, optional

            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
            * None by default.

        Returns
        -------
        Pandas DataFrame

        Notes
        -----
        Since pandas does not provide a way to handle metadata (yet), all metadata associated with
        the array will be lost.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        >>> arr.to_frame()                                                             # doctest: +NORMALIZE_WHITESPACE
        c      c0  c1
        a  b
        a0 b0   0   1
           b1   2   3
        a1 b0   4   5
           b1   6   7
        >>> arr.to_frame(fold_last_axis_name=True)                                     # doctest: +NORMALIZE_WHITESPACE
                c0  c1
        a  b\c
        a0 b0    0   1
           b1    2   3
        a1 b0    4   5
           b1    6   7
        """
        last_name = self.axes[-1].name
        columns_name = None if fold_last_axis_name else last_name
        columns = np_array_to_pd_index(self.axes[-1].labels, name=columns_name)
        if self.ndim > 1:
            axes_names = self.axes.names[:-1]
            if fold_last_axis_name:
                tmp = axes_names[-1] if axes_names[-1] is not None else ''
                if last_name:
                    axes_names[-1] = f"{tmp}\\{last_name}"
            if self.ndim == 2:
                index = np_array_to_pd_index(self.axes[0].labels, name=axes_names[0])
            else:
                index = pd.MultiIndex.from_product(self.axes.labels[:-1], names=axes_names)
        else:
            index = pd.Index([''])
            if fold_last_axis_name:
                index.name = self.axes.names[-1]
        data = np.asarray(self).reshape((len(index), len(columns)))
        df = pd.DataFrame(data, index, columns)
        if dropna is not None:
            dropna = dropna if dropna is not True else 'all'
            df.dropna(inplace=True, how=dropna)
        return df
    df = property(to_frame)

    def to_series(self, name=None, dropna=False) -> pd.Series:
        r"""
        Convert an Array into a Pandas Series.

        Parameters
        ----------
        name : str, optional
            Name of the series. Defaults to None.
        dropna : bool, optional.
            False by default.

        Returns
        -------
        Pandas Series

        Notes
        -----
        Since pandas does not provide a way to handle metadata (yet), all metadata associated with
        the array will be lost.

        Examples
        --------
        >>> arr = ndtest((2, 3), dtype=float)
        >>> arr
        a\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
        >>> arr.to_series() # doctest: +NORMALIZE_WHITESPACE
        a   b
        a0  b0    0.0
            b1    1.0
            b2    2.0
        a1  b0    3.0
            b1    4.0
            b2    5.0
        dtype: float64

        Set a name

        >>> arr.to_series('my_name') # doctest: +NORMALIZE_WHITESPACE
                a   b
        a0  b0    0.0
            b1    1.0
            b2    2.0
        a1  b0    3.0
            b1    4.0
            b2    5.0
        Name: my_name, dtype: float64

        Drop NaN values

        >>> arr['b1'] = nan
        >>> arr
        a\b   b0   b1   b2
         a0  0.0  nan  2.0
         a1  3.0  nan  5.0
        >>> arr.to_series(dropna=True) # doctest: +NORMALIZE_WHITESPACE
        a   b
        a0  b0    0.0
            b2    2.0
        a1  b0    3.0
            b2    5.0
        dtype: float64
        """
        if self.ndim == 0:
            raise ValueError('cannot convert 0D array to Series')
        elif self.ndim == 1:
            axis = self.axes[0]
            # Note that string labels will be converted to object dtype in the process
            # and label arrays with object dtype containing only numeric values will keep
            # the object dtype.
            index = np_array_to_pd_index(axis.labels, name=axis.name, tupleize_cols=False)
        else:
            index = pd.MultiIndex.from_product(self.axes.labels, names=self.axes.names)
        series = pd.Series(self.data.reshape(-1), index, name=name)
        if dropna:
            series.dropna(inplace=True)
        return series
    series = property(to_series)

    def describe(self, *args, percentiles=None) -> 'Array':
        r"""
        Descriptive summary statistics, excluding NaN values.

        By default, it includes the number of non-NaN values, the mean, standard deviation, minimum, maximum and
        the 25, 50 and 75 percentiles.

        Parameters
        ----------
        *args : int or str or Axis or Group or any combination of those, optional
            Axes or groups along which to compute the aggregates. Defaults to aggregate over the whole array.
        percentiles : array-like, optional.
            List of integer percentiles to include. Defaults to [25, 50, 75].

        Returns
        -------
        Array

        See Also
        --------
        Array.describe_by

        Examples
        --------
        >>> arr = Array([0, 6, 2, 5, 4, 3, 1, 3], 'year=2013..2020')
        >>> arr
        year  2013  2014  2015  2016  2017  2018  2019  2020
                 0     6     2     5     4     3     1     3
        >>> arr.describe()
        statistic  count  mean  std  min   25%  50%   75%  max
                     8.0   3.0  2.0  0.0  1.75  3.0  4.25  6.0
        >>> arr.describe(percentiles=[50, 90])
        statistic  count  mean  std  min  50%  90%  max
                     8.0   3.0  2.0  0.0  3.0  5.3  6.0
        """
        if percentiles is None:
            percentiles = [25, 50, 75]

        # TODO: we should use the commented code below to compute all percentiles in one shot but this does not work
        #       when *args is not empty (see https://github.com/larray-project/larray/issues/192)
        # return stack({
        #     ...,
        #     **arr.percentile(percentiles, *args).set_labels({p: f'{p}%' for p in percentiles}),
        #     ...
        # }, 'statistic')
        return stack({
            # Not using la.isnan to avoid a cyclic import
            'count': Array(~np.isnan(self.data), self.axes).sum(*args),
            'mean': self.mean(*args),
            'std': self.std(*args),
            'min': self.min(*args),
            **{f'{p}%': self.percentile(p, *args) for p in percentiles},
            'max': self.max(*args)
        }, 'statistic')

    def describe_by(self, *args, percentiles=None) -> 'Array':
        r"""
        Descriptive summary statistics, excluding NaN values, along axes or for groups.

        By default, it includes the number of non-NaN values, the mean, standard deviation, minimum, maximum and
        the 25, 50 and 75 percentiles.

        Parameters
        ----------
        *args : int or str or Axis or Group or any combination of those, optional
            Axes or groups to include in the result after aggregating. Defaults to aggregate over the whole array.
        percentiles : array-like, optional.
            list of integer percentiles to include. Defaults to [25, 50, 75].

        Returns
        -------
        Array

        See Also
        --------
        Array.describe

        Examples
        --------
        >>> data = [[0, 6, 3, 5, 4, 2, 1, 3], [7, 5, 3, 2, 8, 5, 6, 4]]
        >>> arr = Array(data, 'gender=Male,Female;year=2013..2020').astype(float)
        >>> arr
        gender\year  2013  2014  2015  2016  2017  2018  2019  2020
               Male   0.0   6.0   3.0   5.0   4.0   2.0   1.0   3.0
             Female   7.0   5.0   3.0   2.0   8.0   5.0   6.0   4.0
        >>> arr.describe_by('gender')
        gender\statistic  count  mean  std  min   25%  50%   75%  max
                    Male    8.0   3.0  2.0  0.0  1.75  3.0  4.25  6.0
                  Female    8.0   5.0  2.0  2.0  3.75  5.0  6.25  8.0
        >>> arr.describe_by('gender', (X.year[:2015], X.year[2018:]))
        gender  year\statistic  count  mean  std  min  25%  50%  75%  max
          Male           :2015    3.0   3.0  3.0  0.0  1.5  3.0  4.5  6.0
          Male           2018:    3.0   2.0  1.0  1.0  1.5  2.0  2.5  3.0
        Female           :2015    3.0   5.0  2.0  3.0  4.0  5.0  6.0  7.0
        Female           2018:    3.0   5.0  1.0  4.0  4.5  5.0  5.5  6.0
        >>> arr.describe_by('gender', percentiles=[50, 90])
        gender\statistic  count  mean  std  min  50%  90%  max
                    Male    8.0   3.0  2.0  0.0  3.0  5.3  6.0
                  Female    8.0   5.0  2.0  2.0  5.0  7.3  8.0
        """
        args = self._prepare_aggregate(None, args)
        args = self._by_args_to_normal_agg_args(args)
        return self.describe(*args, percentiles=percentiles)

    def value_counts(self):
        """
        Count number of occurrences of each unique value in array.

        Returns
        -------
        Array of ints
            The number of occurrences of each unique value in the input array.

        See Also
        --------
        Array.unique

        Examples
        --------
        >>> arr = Array([5, 2, 5, 5, 2, 3, 7], "a=a0..a6")
        >>> arr
        a  a0  a1  a2  a3  a4  a5  a6
            5   2   5   5   2   3   7
        >>> arr.value_counts()
        value  2  3  5  7
               2  1  3  1
        """
        unq, counts = np.unique(self.data, return_counts=True)
        return Array(counts, Axis(unq, 'value'))

    # noinspection PyAttributeOutsideInit
    # def __array_finalize__(self, obj):
    #     """
    #     used when arrays are allocated from subclasses of ndarrays
    #     """
    #     return np.ndarray.__array_finalize__(self.data, obj)

    # def __array_prepare__(self, arr, context=None):
    #     """
    #     called before ufuncs (must return an ndarray)
    #     """
    #     return np.ndarray.__array_prepare__(self.data, arr, context)

    def __array_wrap__(self, out_arr, context=None, return_scalar=False) -> 'Array':
        r"""
        Called after numpy ufuncs. This is never called during our wrapped
        ufuncs, but if somebody uses raw numpy function, this works in some
        cases.
        """
        # as far as I understand this, this line will only ever be useful if
        # our .data attribute is not a np.ndarray but an array-ish. It gives
        # that other type (cupy array or whatever) the oportunity
        data = self.data.__array_wrap__(out_arr, context)
        return Array(data, self.axes)

    def __bool__(self):
        return bool(self.data)

    # TODO: either support a list (of axes names) as first argument here (and set_labels)
    #       or don't support that in set_axes
    def rename(self, renames=None, to=None, inplace=False, **kwargs) -> 'Array':
        r"""Rename axes of the array.

        Parameters
        ----------
        renames : axis ref or dict {axis ref: str} or list of tuple (axis ref, str)
            Rename to apply. If a single axis reference is given, the `to` argument must be used.
        to : str or Axis
            New name if `renames` contains a single axis reference.
        **kwargs : str or Axis
            New name for each axis given as a keyword argument.

        Returns
        -------
        Array
            Array with axes renamed.

        See Also
        --------
        set_axes : replace one or several axes

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> arr = ndtest([nat, sex])
        >>> arr
        nat\sex  M  F
             BE  0  1
             FO  2  3
        >>> arr.rename(nat, 'nat2')
        nat2\sex  M  F
              BE  0  1
              FO  2  3
        >>> arr.rename(nat='nat2', sex='sex2')
        nat2\sex2  M  F
               BE  0  1
               FO  2  3
        >>> arr.rename([('nat', 'nat2'), ('sex', 'sex2')])
        nat2\sex2  M  F
               BE  0  1
               FO  2  3
        >>> arr.rename({'nat': 'nat2', 'sex': 'sex2'})
        nat2\sex2  M  F
               BE  0  1
               FO  2  3
        """
        axes = self.axes.rename(renames, to, **kwargs)
        if inplace:
            self.axes = axes
            return self
        else:
            return Array(self.data, axes)

    def reindex(self, axes_to_reindex=None, new_axis=None, fill_value=nan, inplace=False, **kwargs) -> 'Array':
        r"""Reorder and/or add new labels in axes.

        Place NaN or given `fill_value` in locations having no value previously.

        Parameters
        ----------
        axes_to_reindex : axis ref or dict {axis ref: axis} or list of (axis ref, axis) or sequence of Axis
            Axis(es) to reindex. If a single axis reference is given, the `new_axis` argument must be provided.
            If string, Group or Axis object, the corresponding axis is reindexed if found among existing,
            otherwise a new axis is added.
            If a list of Axis or an AxisCollection is given, existing axes are reindexed while missing ones are added.
        new_axis : int, str, list/tuple/array of str, Group or Axis, optional
            List of new labels or new axis if `axes_to_reindex` contains a single axis reference.
        fill_value : scalar or Array, optional
            Value used to fill cells corresponding to label combinations which were not present before reindexing.
            Defaults to NaN.
        inplace : bool, optional
            Whether to modify the original object or return a new array and leave the original intact.
            Defaults to False.
        **kwargs : Axis
            New axis for each axis to reindex given as a keyword argument.

        Returns
        -------
        Array
            Array with reindexed axes.

        Notes
        -----
        When introducing NaNs into an array containing integers via reindex,
        all data will be promoted to float in order to store the NaNs.

        Examples
        --------
        >>> arr = ndtest((2, 2))
        >>> arr
        a\b  b0  b1
         a0   0   1
         a1   2   3
        >>> arr2 = ndtest('a=a1,a2;c=c0;b=b2..b0')
        >>> arr2
         a  c\b  b2  b1  b0
        a1   c0   0   1   2
        a2   c0   3   4   5

        Reindex an axis by passing labels (list or string)

        >>> arr.reindex('b', ['b1', 'b2', 'b0'])
        a\b   b1   b2   b0
         a0  1.0  nan  0.0
         a1  3.0  nan  2.0
        >>> arr.reindex('b', 'b0..b2', fill_value=-1)
        a\b  b0  b1  b2
         a0   0   1  -1
         a1   2   3  -1
        >>> arr.reindex(b='b=b0..b2', fill_value=-1)
        a\b  b0  b1  b2
         a0   0   1  -1
         a1   2   3  -1

        Reindex using an axis from another array

        >>> arr.reindex('b', arr2.b, fill_value=-1)
        a\b  b2  b1  b0
         a0  -1   1   0
         a1  -1   3   2

        Reindex using a subset of an axis

        >>> arr.reindex('b', arr2.b['b1':], fill_value=-1)
        a\b  b1  b0
         a0   1   0
         a1   3   2

        Reindex by passing an axis or a group

        >>> arr.reindex('b=b2..b0', fill_value=-1)
        a\b  b2  b1  b0
         a0  -1   1   0
         a1  -1   3   2
        >>> arr.reindex(arr2.b, fill_value=-1)
        a\b  b2  b1  b0
         a0  -1   1   0
         a1  -1   3   2
        >>> arr.reindex(arr2.b['b1':], fill_value=-1)
        a\b  b1  b0
         a0   1   0
         a1   3   2

        Reindex several axes

        >>> arr.reindex({'a': arr2.a, 'b': arr2.b}, fill_value=-1)
        a\b  b2  b1  b0
         a1  -1   3   2
         a2  -1  -1  -1
        >>> arr.reindex({'a': arr2.a, 'b': arr2.b['b1':]}, fill_value=-1)
        a\b  b1  b0
         a1   3   2
         a2  -1  -1
        >>> arr.reindex(a=arr2.a, b=arr2.b, fill_value=-1)
        a\b  b2  b1  b0
         a1  -1   3   2
         a2  -1  -1  -1

        Reindex by passing a collection of axes

        >>> arr.reindex(arr2.axes, fill_value=-1)
         a  b\c  c0
        a1   b2  -1
        a1   b1   3
        a1   b0   2
        a2   b2  -1
        a2   b1  -1
        a2   b0  -1
        >>> arr2.reindex(arr.axes, fill_value=-1)
         a  c\b  b0  b1
        a0   c0  -1  -1
        a1   c0   2   1
        """
        def labels_def_and_name_to_axis(labels_def, axis_name=None):
            # TODO: the rename functionality seems weird to me.
            #       I think we should either raise an error if the axis name
            #       is different (force using new_axis=other_axis.labels instead
            #       of new_axis=other_axis) OR do not do use the old name
            #       (and make sure this effectively does a rename).
            #       it might have been the unintended consequence of supporting a
            #       list of labels as new_axis
            axis = labels_def if isinstance(labels_def, Axis) else Axis(labels_def)
            return axis.rename(axis_name) if axis_name is not None else axis

        def axis_ref_to_axis(axes, axis_ref):
            if isinstance(axis_ref, Axis) or is_axis_ref(axis_ref):
                return axes[axis_ref]
            else:
                raise TypeError(
                    "In Array.reindex, source axes must be Axis objects or axis references ('axis name', "
                    "X.axis_name or axis_integer_position) but got object of "
                    f"type {type(axis_ref).__name__} instead."
                )

        def is_axis_ref(axis_ref):
            return isinstance(axis_ref, (int, str, AxisReference))

        def is_axis_def(axis_def):
            return ((isinstance(axis_def, str) and '=' in axis_def)
                    or isinstance(axis_def, Group))

        if new_axis is None:
            if isinstance(axes_to_reindex, Axis) and not isinstance(axes_to_reindex, AxisReference):
                axes_to_reindex = {axes_to_reindex: axes_to_reindex}
            elif is_axis_def(axes_to_reindex):
                axis = Axis(axes_to_reindex)
                axes_to_reindex = {axis: axis}
            elif is_axis_ref(axes_to_reindex):
                raise TypeError("In Array.reindex, when using an axis reference ('axis name', X.axis_name or "
                                "axis_integer_position) as axes_to_reindex, you must provide a value for `new_axis`.")
            # otherwise axes_to_reindex should be None (when kwargs are used),
            # a dict or a sequence of axes
            # axes_to_reindex can be None when kwargs are used
            assert (axes_to_reindex is None or
                    isinstance(axes_to_reindex, (tuple, list, dict, AxisCollection)))
        else:
            if not (isinstance(axes_to_reindex, Axis) or is_axis_ref(axes_to_reindex)):
                raise TypeError(
                    "In Array.reindex, when `new_axis` is used, `axes_to_reindex` "
                    "must be an Axis object or an axis reference ('axis name', "
                    f"X.axis_name or axis_integer_position) but got {axes_to_reindex} "
                    f"(which is of type {type(axes_to_reindex).__name__}) instead."
                )
            axes_to_reindex = {axes_to_reindex: new_axis}
            new_axis = None

        if isinstance(axes_to_reindex, (list, tuple)):
            axes_to_reindex = AxisCollection(axes_to_reindex)

        assert new_axis is None
        assert axes_to_reindex is None or isinstance(axes_to_reindex, (dict, AxisCollection))

        if isinstance(axes_to_reindex, AxisCollection):
            # | axes_to_reindex is needed because axes_to_reindex can contain more axes than self.axes
            res_axes = AxisCollection([axes_to_reindex.get(axis, axis) for axis in self.axes]) | axes_to_reindex
        else:
            # TODO: move this to AxisCollection.replace
            if isinstance(axes_to_reindex, dict):
                new_axes_to_reindex = {}
                for k, v in axes_to_reindex.items():
                    src_axis = axis_ref_to_axis(self.axes, k)
                    dst_axis = labels_def_and_name_to_axis(v, src_axis.name)
                    new_axes_to_reindex[src_axis] = dst_axis
                axes_to_reindex = new_axes_to_reindex

            res_axes = self.axes.replace(axes_to_reindex, **kwargs)
        res = full(res_axes, fill_value, dtype=common_dtype((self.data, fill_value)))

        def get_group(res_axes, self_axis):
            res_axis = res_axes[self_axis]
            if res_axis.equals(self_axis):
                return self_axis[:]
            else:
                return self_axis[self_axis.intersection(res_axis).labels]
        self_groups = tuple(get_group(res_axes, axis) for axis in self.axes)
        res_groups = tuple(res_axes[group.axis][group] for group in self_groups)
        res[res_groups] = self[self_groups]
        if inplace:
            self.axes = res.axes
            self.data = res.data
            return self
        else:
            return res

    def align(self, other, join='outer', fill_value=nan, axes=None) -> Tuple['Array', 'Array']:
        r"""Align two arrays on their axes with the specified join method.

        In other words, it ensure all common axes are compatible. Those arrays can then be used in binary operations.

        Parameters
        ----------
        other : Array-like
        join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
            Join method. For each axis common to both arrays:
              - outer: will use a label if it is in either arrays axis (ordered like the first array).
                       This is the default as it results in no information loss.
              - inner: will use a label if it is in both arrays axis (ordered like the first array).
              - left: will use the first array axis labels.
              - right: will use the other array axis labels.
              - exact: instead of aligning, raise an error when axes to be aligned are not equal.
        fill_value : scalar or Array, optional
            Value used to fill cells corresponding to label combinations which are not common to both arrays.
            Defaults to NaN.
        axes : AxisReference or sequence of them, optional
            Axes to align. Need to be valid in both arrays. Defaults to None (all common axes). This must be specified
            when mixing anonymous and non-anonymous axes.

        Returns
        -------
        (left, right) : (Array, Array)
            Aligned objects

        Notes
        -----
            Arrays with anonymous axes are currently not supported.

        Examples
        --------
        >>> arr1 = ndtest((2, 3))
        >>> arr1
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr2 = -ndtest((3, 2))
        >>> # reorder array to make the test more interesting
        >>> arr2 = arr2[['b1', 'b0']]
        >>> arr2
        a\b  b1  b0
         a0  -1   0
         a1  -3  -2
         a2  -5  -4

        Align arr1 and arr2

        >>> aligned1, aligned2 = arr1.align(arr2)
        >>> aligned1
        a\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
         a2  nan  nan  nan
        >>> aligned2
        a\b    b0    b1   b2
         a0   0.0  -1.0  nan
         a1  -2.0  -3.0  nan
         a2  -4.0  -5.0  nan

        After aligning all common axes, one can then do operations between the two arrays

        >>> aligned1 + aligned2
        a\b   b0   b1   b2
         a0  0.0  0.0  nan
         a1  1.0  1.0  nan
         a2  nan  nan  nan

        Other kinds of joins are supported

        >>> aligned1, aligned2 = arr1.align(arr2, join='inner')
        >>> aligned1
        a\b   b0   b1
         a0  0.0  1.0
         a1  3.0  4.0
        >>> aligned2
        a\b    b0    b1
         a0   0.0  -1.0
         a1  -2.0  -3.0
        >>> aligned1, aligned2 = arr1.align(arr2, join='left')
        >>> aligned1
        a\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
        >>> aligned2
        a\b    b0    b1   b2
         a0   0.0  -1.0  nan
         a1  -2.0  -3.0  nan
        >>> aligned1, aligned2 = arr1.align(arr2, join='right')
        >>> aligned1
        a\b   b1   b0
         a0  1.0  0.0
         a1  4.0  3.0
         a2  nan  nan
        >>> aligned2
        a\b    b1    b0
         a0  -1.0   0.0
         a1  -3.0  -2.0
         a2  -5.0  -4.0

        The fill value for missing labels defaults to nan but can be changed to any compatible value.

        >>> aligned1, aligned2 = arr1.align(arr2, fill_value=0)
        >>> aligned1
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   0   0   0
        >>> aligned2
        a\b  b0  b1  b2
         a0   0  -1   0
         a1  -2  -3   0
         a2  -4  -5   0
        >>> aligned1 + aligned2
        a\b  b0  b1  b2
         a0   0   0   2
         a1   1   1   5
         a2  -4  -5   0

        It also works when either arrays (or both) have extra axes

        >>> arr3 = ndtest((3, 2, 2))
        >>> arr1
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr3
         a  b\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        a2   b0   8   9
        a2   b1  10  11
        >>> aligned1, aligned2 = arr1.align(arr3, join='inner')
        >>> aligned1
        a\b   b0   b1
         a0  0.0  1.0
         a1  3.0  4.0
        >>> aligned2
         a  b\c   c0   c1
        a0   b0  0.0  1.0
        a0   b1  2.0  3.0
        a1   b0  4.0  5.0
        a1   b1  6.0  7.0
        >>> aligned1 + aligned2
         a  b\c    c0    c1
        a0   b0   0.0   1.0
        a0   b1   3.0   4.0
        a1   b0   7.0   8.0
        a1   b1  10.0  11.0

        One can also align only some specific axes (but in that case arrays might not be compatible)

        >>> aligned1, aligned2 = arr1.align(arr2, axes='b')
        >>> aligned1
        a\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
        >>> aligned2
        a\b    b0    b1   b2
         a0   0.0  -1.0  nan
         a1  -2.0  -3.0  nan
         a2  -4.0  -5.0  nan

        Test if two arrays are aligned

        >>> arr1.align(arr2, join='exact')   # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: Both arrays are not aligned because align method with join='exact'
        expected Axis(['a0', 'a1'], 'a') to be equal to Axis(['a0', 'a1', 'a2'], 'a')
        """
        other = asarray(other)
        # reindex does not currently support anonymous axes
        if any(name is None for name in self.axes.names) or any(name is None for name in other.axes.names):
            raise ValueError("arrays with anonymous axes are currently not supported by Array.align")
        try:
            left_axes, right_axes = self.axes.align(other.axes, join=join, axes=axes)
        except ValueError as e:
            raise ValueError(f"Both arrays are not aligned because {e}")
        return self.reindex(left_axes, fill_value=fill_value), other.reindex(right_axes, fill_value=fill_value)

    @deprecate_kwarg('reverse', 'ascending', {True: False, False: True})
    def sort_values(self, key=None, axis=None, ascending=True) -> 'Array':
        r"""Sort values of the array.

        Parameters
        ----------
        key : scalar or tuple or Group
            Key along which to sort. Must have exactly one dimension less than ndim.
            Cannot be used in combination with `axis` argument.
            If both `key` and `axis` are None, sort array with all axes combined.
            Defaults to None.
        axis : int or str or Axis
            Axis along which to sort. Cannot be used in combination with `key` argument.
            Defaults to None.
        ascending : bool, optional
            Sort values in ascending order. Defaults to True.

        Returns
        -------
        Array
            Array with sorted values.

        Examples
        --------
        sort the whole array (no key or axis given)

        >>> arr_1D = Array([10, 2, 4], 'a=a0..a2')
        >>> arr_1D
        a  a0  a1  a2
           10   2   4
        >>> arr_1D.sort_values()
        a  a1  a2  a0
            2   4  10
        >>> arr_2D = Array([[10, 2, 4], [3, 7, 1]], 'a=a0,a1; b=b0..b2')
        >>> arr_2D
        a\b  b0  b1  b2
         a0  10   2   4
         a1   3   7   1
        >>> # if the array has more than one dimension, sort array with all axes combined
        >>> arr_2D.sort_values()
        a_b  a1_b2  a0_b1  a1_b0  a0_b2  a1_b1  a0_b0
                 1      2      3      4      7     10

        Sort along a given key

        >>> # sort columns according to the values of the row associated with the label 'a1'
        >>> arr_2D.sort_values('a1')
        a\b  b2  b0  b1
         a0   4  10   2
         a1   1   3   7
        >>> arr_2D.sort_values('a1', ascending=False)
        a\b  b1  b0  b2
         a0   2  10   4
         a1   7   3   1
        >>> arr_3D = Array([[[10, 2, 4], [3, 7, 1]], [[5, 1, 6], [2, 8, 9]]],
        ...            'a=a0,a1; b=b0,b1; c=c0..c2')
        >>> arr_3D
         a  b\c  c0  c1  c2
        a0   b0  10   2   4
        a0   b1   3   7   1
        a1   b0   5   1   6
        a1   b1   2   8   9
        >>> # sort columns according to the values of the row associated with the labels 'a0' and 'b1'
        >>> arr_3D.sort_values(('a0', 'b1'))
         a  b\c  c2  c0  c1
        a0   b0   4  10   2
        a0   b1   1   3   7
        a1   b0   6   5   1
        a1   b1   9   2   8

        Sort along an axis

        >>> arr_2D
        a\b  b0  b1  b2
         a0  10   2   4
         a1   3   7   1
        >>> # sort values along axis 'a'
        >>> # equivalent to sorting the values of each column of the array
        >>> arr_2D.sort_values(axis='a')
        a*\b  b0  b1  b2
           0   3   2   1
           1  10   7   4
        >>> # sort values along axis 'b'
        >>> # equivalent to sorting the values of each row of the array
        >>> arr_2D.sort_values(axis='b')
        a\b*  0  1   2
          a0  2  4  10
          a1  1  3   7
        """
        if key is not None and axis is not None:
            raise ValueError("Arguments key and axis are exclusive and cannot be used in combination")
        if axis is not None:
            axis = self.axes[axis]
            axis_idx = self.axes.index(axis)
            data = np.sort(self.data, axis_idx)
            new_axes = self.axes.replace(axis_idx, Axis(len(axis), axis.name))
            res = Array(data, new_axes)
        elif key is not None:
            subset = self[key]
            if subset.ndim > 1:
                raise NotImplementedError("sort_values key must have one dimension less than array.ndim")
            assert subset.ndim == 1
            axis = subset.axes[0]
            indicesofsorted = subset.indicesofsorted()

            # FIXME: .data shouldn't be necessary, but currently, if we do not do it, we get
            # IGroup(nat  EU  FO  BE
            #              1   2   0, axis='nat')
            # which sorts the *data* correctly, but the labels on the nat axis are not sorted
            # (because the __getitem__ in that case reuse the key axis as-is -- like it should).
            # Both use cases have value, but I think reordering the ticks should be the default.
            # Now, I am unsure where to change this. Probably in IGroupMaker.__getitem__,
            # but then how do I get the "not reordering labels" behavior that I have now?
            # FWIW, using .data, I get IGroup([1, 2, 0], axis='nat'), which works.
            sorter = axis.i[indicesofsorted.data]
            res = self[sorter]
        else:
            res = self.combine_axes()
            indicesofsorted = np.argsort(res.data)
            res = res.i[indicesofsorted]
            axis = res.axes[0]
        return res[axis[::-1]] if not ascending else res

    @deprecate_kwarg('reverse', 'ascending', {True: False, False: True})
    def sort_labels(self, axes=None, ascending=True) -> 'Array':
        r"""Sort labels of axes of the array.

        Parameters
        ----------
        axes : axis reference (Axis, str, int) or list of them, optional
            Axes to sort the labels of. Defaults None (all axes).
        ascending : bool, optional
            Sort labels in ascending order. Defaults to True.

        Returns
        -------
        Array
            Array with sorted labels.

        Examples
        --------
        >>> a = ndtest("nat=EU,FO,BE; sex=M,F")
        >>> a
        nat\sex  M  F
             EU  0  1
             FO  2  3
             BE  4  5
        >>> a.sort_labels('sex')
        nat\sex  F  M
             EU  1  0
             FO  3  2
             BE  5  4
        >>> a.sort_labels()
        nat\sex  F  M
             BE  5  4
             EU  1  0
             FO  3  2
        >>> a.sort_labels(('sex', 'nat'))
        nat\sex  F  M
             BE  5  4
             EU  1  0
             FO  3  2
        >>> a.sort_labels(ascending=False)
        nat\sex  M  F
             FO  2  3
             EU  0  1
             BE  4  5
        """
        if axes is None:
            axes = self.axes
        elif not isinstance(axes, (tuple, list, AxisCollection)):
            axes = [axes]

        if not isinstance(axes, AxisCollection):
            axes = self.axes[axes]

        def sort_key(axis):
            key = np.argsort(axis.labels)
            if not ascending:
                key = key[::-1]
            return axis.i[key]

        return self[tuple(sort_key(axis) for axis in axes)]

    sort_axis = renamed_to(sort_labels, 'sort_axis', raise_error=True)
    sort_axes = renamed_to(sort_labels, 'sort_axes')

    # TODO: set returned type to Union['Array', np.ndarray, Scalar] ?
    def __getitem__(self, key, collapse_slices=False, translate_key=True, points=False) -> Union['Array', Scalar]:
        raw_broadcasted_key, res_axes, transpose_indices = \
            self.axes._key_to_raw_and_axes(key, collapse_slices, translate_key, points, wildcard=False)
        res_data = self.data[raw_broadcasted_key]
        if res_axes:
            # if some axes have been moved in front because of advanced indexing, we transpose them back to their
            # original position. We do not use Array.transpose because that creates another Array object which is costly
            if transpose_indices is not None:
                res_data = res_data.transpose(transpose_indices)
                res_axes = res_axes[transpose_indices]
            return Array(res_data, res_axes)
        else:
            return res_data

    def __setitem__(self, key, value, collapse_slices=True, translate_key=True, points=False) -> None:
        # TODO: if key or value has more axes than self, we could use
        # total_axes = self.axes + key.axes + value.axes
        # expanded = self.expand(total_axes)
        # data = np.asarray(expanded.data)
        raw_broadcasted_key, target_axes, _ = \
            self.axes._key_to_raw_and_axes(key, collapse_slices, translate_key, points, wildcard=True)
        if isinstance(value, Array):
            # None target_axes can happen when setting a single "cell"/value with an Array (of size 1)
            if target_axes is not None:
                value = value.broadcast_with(target_axes, check_compatible=True)
            else:
                target_axes = []
            # replace incomprehensible error message "could not broadcast input array from shape XX into shape YY"
            # for users by "incompatible axes"
            extra_axes = [axis for axis in value.axes - target_axes if len(axis) > 1]
            if extra_axes:
                extra_axes = AxisCollection(extra_axes)
                axes = AxisCollection(target_axes)
                text = 'axes are' if len(extra_axes) > 1 else 'axis is'
                raise ValueError(f"Value {extra_axes!s} {text} not present in target subset {axes!s}. A value can only "
                                 f"have the same axes or fewer axes than the subset being targeted")
            value = value.data
        self.data[raw_broadcasted_key] = value

        # concerning keys this can make sense in several cases:
        # single bool Array key with extra axes.
        # tuple of bool Array keys (eg one for each axis). each could have extra axes. Common axes between keys are
        # not a problem, we can simply "and" them. Though we should avoid explicitly "and"ing them if there is no
        # common axis because that is less efficient than the implicit "and" that is done by numpy __getitem__ (and
        # the fact we need to combine dimensions when any key has more than 1 dim).

        # the bool value represents whether the axis label is taken or not if any bool key (part) has more than one
        # axis, we get combined dimensions out of it.

    def set(self, value, **kwargs) -> None:
        r"""
        Set a subset of array to value.

        * all common axes must be either of length 1 or the same length
        * extra axes in value must be of length 1
        * extra axes in current array can have any length

        Parameters
        ----------
        value : scalar or Array

        Examples
        --------
        >>> arr = ndtest((3, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   6   7   8
        >>> arr['a1:', 'b1:'].set(10)
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3  10  10
         a2   6  10  10
        >>> arr['a1:', 'b1:'].set(ndtest("a=a1,a2;b=b1,b2"))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   0   1
         a2   6   2   3
        """
        self.__setitem__(kwargs, value)

    # TODO: this should be a private method
    def reshape(self, target_axes) -> 'Array':
        r"""
        Given a list of new axes, changes the shape of the array.
        The size of the array (= number of elements) must be equal
        to the product of length of target axes.

        Parameters
        ----------
        target_axes : iterable of Axis
            New axes. The size of the array (= number of stored data)
            must be equal to the product of length of target axes.

        Returns
        -------
        Array
            New array with new axes but same data.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        >>> new_arr = arr.reshape([Axis('a=a0,a1'),
        ...                        Axis(['b0c0', 'b0c1', 'b1c0', 'b1c1'], 'bc')])
        >>> new_arr
        a\bc  b0c0  b0c1  b1c0  b1c1
          a0     0     1     2     3
          a1     4     5     6     7
        """
        # this is a dangerous operation, because except for adding length 1 axes (which is safe), it potentially
        # modifies data
        # TODO: add a check/flag? for "unsafe" reshapes (but allow merging
        # several axes & "splitting" axes) etc.
        # eg 4, 3, 2 -> 2, 3, 4 is wrong (even if size is respected)
        #    4, 3, 2 -> 12, 2 is potentially ok (merging adjacent dimensions)
        #            -> 4, 6 is potentially ok (merging adjacent dimensions)
        #            -> 24 is potentially ok (merging adjacent dimensions)
        #            -> 3, 8 WRONG (non adjacent dimensions)
        #            -> 8, 3 WRONG
        #    4, 3, 2 -> 2, 2, 3, 2 is potentially ok (splitting dim)
        if not isinstance(target_axes, AxisCollection):
            target_axes = AxisCollection(target_axes)
        data = self.data.reshape(target_axes.shape)
        return Array(data, target_axes)

    # TODO: this should be a private method
    def reshape_like(self, target) -> 'Array':
        r"""
        Same as reshape but with an array as input.
        Total size (= number of stored data) of the two arrays must be equal.

        See Also
        --------
        reshape : returns an Array with a new shape given a list of axes.

        Examples
        --------
        >>> arr = zeros((2, 2, 2), dtype=int)
        >>> arr
        {0}*  {1}*\{2}*  0  1
           0          0  0  0
           0          1  0  0
           1          0  0  0
           1          1  0  0
        >>> new_arr = arr.reshape_like(ndtest((2, 4)))
        >>> new_arr
        a\b  b0  b1  b2  b3
         a0   0   0   0   0
         a1   0   0   0   0
        """
        return self.reshape(target.axes)

    def broadcast_with(self, target, check_compatible=False) -> 'Array':
        r"""
        Return an array that is (NumPy) broadcastable with target.

        * all common axes must be either of length 1 or the same length
        * extra axes in source can have any length and will be moved to the
          front
        * extra axes in target can have any length and the result will have axes
          of length 1 for those axes

        This is different from reshape which ensures the result has exactly the
        shape of the target.

        Parameters
        ----------
        target : Array or collection of Axis

        check_compatible : bool, optional
            Whether to check that common axes are compatible. Defaults to False.

        Returns
        -------
        Array
        """
        if isinstance(target, Array):
            target_axes = target.axes
        else:
            target_axes = target
            if not isinstance(target_axes, (tuple, list, AxisCollection)):
                target_axes = AxisCollection(target_axes)
        if self.axes == target_axes:
            return self
        # determine real target order (= left_only then target_axes)
        # (we will add length one axes to the left like numpy just below)
        target_axes = (self.axes - target_axes) | target_axes

        # XXX: this breaks la['1,5,9'] = la['2,7,3']
        # but that use case should use ignore_labels
        # self.axes.check_compatible(target_axes)

        # 1) reorder axes to target order
        array = self.transpose(target_axes & self.axes)

        # 2) add length one axes
        res_axes = array.axes.get_all(target_axes)
        if check_compatible:
            res_axes.check_compatible(target_axes)
        return array.reshape(res_axes)

    # XXX: I wonder if effectively dropping the labels is necessary or not
    # we could perfectly only mark the axis as being a wildcard axis and keep
    # the labels intact. These wildcard axes with labels
    # could be useful in a few situations. For example, Excel sheets could
    # have such behavior: you can slice columns using letters, but that
    # wouldn't prevent doing computation between arrays using different
    # columns. On the other hand, it makes wild axes less obvious and I
    # wonder if there would be a risk of wildcard axes inadvertently leaking.
    # plus it might be confusing if incompatible labels "work".
    def ignore_labels(self, axes=None) -> 'Array':
        r"""Ignore labels from axes (replace those axes by "wildcard" axes).

        Useful when you want to apply operations between two arrays
        or subarrays with same shape but incompatible axes
        (different labels).

        Parameters
        ----------
        axes : Axis or list/tuple/AxisCollection of Axis, optional
            Axis(es) on which you want to drop the labels.

        Returns
        -------
        Array

        Notes
        -----
        Use it at your own risk.

        Examples
        --------
        >>> a = Axis('a=a1,a2')
        >>> b = Axis('b=b1,b2')
        >>> b2 = Axis('b=b2,b3')
        >>> arr1 = ndtest([a, b])
        >>> arr1
        a\b  b1  b2
         a1   0   1
         a2   2   3
        >>> arr1.ignore_labels(b)
        a\b*  0  1
          a1  0  1
          a2  2  3
        >>> arr1.ignore_labels([a, b])
        a*\b*  0  1
            0  0  1
            1  2  3
        >>> arr2 = ndtest([a, b2])
        >>> arr2
        a\b  b2  b3
         a1   0   1
         a2   2   3
        >>> arr1 * arr2
        Traceback (most recent call last):
        ...
        ValueError: incompatible axes:
        Axis(['b2', 'b3'], 'b')
        vs
        Axis(['b1', 'b2'], 'b')
        >>> arr1 * arr2.ignore_labels()
        a\b  b1  b2
         a1   0   1
         a2   4   9
        >>> arr1.ignore_labels() * arr2
        a\b  b2  b3
         a1   0   1
         a2   4   9
        >>> arr1.ignore_labels('a') * arr2.ignore_labels('b')
        a\b  b1  b2
         a1   0   1
         a2   4   9
        """
        if axes is None:
            axes = self.axes
        elif not isinstance(axes, (tuple, list, AxisCollection)):
            axes = self.axes[[axes]]
        else:
            axes = self.axes[axes]
        res_axes = self.axes.replace({axis: axis.ignore_labels() for axis in axes})
        return Array(self.data, res_axes)
    drop_labels = renamed_to(ignore_labels, 'drop_labels', raise_error=True)

    def __str__(self) -> str:
        if self.ndim == 0:
            return str(self.data.item())
        elif len(self) == 0:
            return 'Array([])'
        else:
            table = self.dump(maxlines=_OPTIONS[DISPLAY_MAXLINES], edgeitems=_OPTIONS[DISPLAY_EDGEITEMS],
                              _axes_display_names=True)
            return table2str(table, 'nan', maxwidth=_OPTIONS[DISPLAY_WIDTH], keepcols=self.ndim - 1,
                             precision=_OPTIONS[DISPLAY_PRECISION])
    __repr__ = __str__

    def __iter__(self):
        # fast path for 1D arrays where we return elements
        if self.ndim <= 1:
            return iter(self.data)
        else:
            return ArrayIterator(self)

    def __contains__(self, key) -> bool:
        return any(key in axis for axis in self.axes)

    # XXX: dump as a 2D Array with row & col dims?
    def dump(self, header=True, wide=True, value_name='value', light=False, axes_names=True, na_repr='as_is',
             maxlines=-1, edgeitems=5, _axes_display_names=False) -> List[List[str]]:
        r"""dump(self, header=True, wide=True, value_name='value', light=False, axes_names=True, na_repr='as_is',
             maxlines=-1, edgeitems=5)

        Dump array as a 2D nested list. This is especially useful when writing to an Excel sheet via open_excel().

        Parameters
        ----------
        header : bool
            Whether to output axes names and labels.
        wide : boolean, optional
            Whether to write arrays in "wide" format. If True, arrays are exported with the last axis
            represented horizontally. If False, arrays are exported in "narrow" format: one column per axis plus one
            value column. Not used if header=False. Defaults to True.
        value_name : str, optional
            Name of the column containing the values (last column) when `wide=False` (see above).
            Not used if header=False. Defaults to 'value'.
        light : bool, optional
            Whether to hide repeated labels. In other words, only show a label if it is different from the
            previous one. Defaults to False.
        axes_names : bool or 'except_last', optional
            Assuming header is True, whether to include axes names. If axes_names is 'except_last',
            all axes names will be included except the last. Defaults to True.
        na_repr : any scalar, optional
            Replace missing values (NaN floats) by this value. Defaults to 'as_is' (do not do any replacement).
        maxlines : int, optional
            Maximum number of lines to show. Defaults to -1 (all lines are shown).
        edgeitems : int, optional
            If number of lines to display is greater than `maxlines`, only the first and last `edgeitems` lines are
            displayed. Only active if `maxlines` is not -1. Defaults to 5.

        Returns
        -------
        2D nested list of builtin Python values or None for 0d arrays

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr.dump()                               # doctest: +NORMALIZE_WHITESPACE
        [['a',  'b\\c', 'c0', 'c1'],
         ['a0',   'b0',    0,    1],
         ['a0',   'b1',    2,    3],
         ['a1',   'b0',    4,    5],
         ['a1',   'b1',    6,    7]]
        >>> arr.dump(axes_names=False)               # doctest: +NORMALIZE_WHITESPACE
        [['',       '', 'c0', 'c1'],
         ['a0',   'b0',    0,    1],
         ['a0',   'b1',    2,    3],
         ['a1',   'b0',    4,    5],
         ['a1',   'b1',    6,    7]]
        >>> arr.dump(axes_names='except_last')       # doctest: +NORMALIZE_WHITESPACE
        [['a',     'b', 'c0', 'c1'],
         ['a0',   'b0',    0,    1],
         ['a0',   'b1',    2,    3],
         ['a1',   'b0',    4,    5],
         ['a1',   'b1',    6,    7]]
        >>> arr.dump(light=True)                     # doctest: +NORMALIZE_WHITESPACE
        [['a',  'b\\c', 'c0', 'c1'],
         ['a0',   'b0',    0,    1],
         ['',     'b1',    2,    3],
         ['a1',   'b0',    4,    5],
         ['',     'b1',    6,    7]]
        >>> arr.dump(wide=False, value_name='data')  # doctest: +NORMALIZE_WHITESPACE
        [['a',   'b',  'c', 'data'],
         ['a0', 'b0', 'c0',      0],
         ['a0', 'b0', 'c1',      1],
         ['a0', 'b1', 'c0',      2],
         ['a0', 'b1', 'c1',      3],
         ['a1', 'b0', 'c0',      4],
         ['a1', 'b0', 'c1',      5],
         ['a1', 'b1', 'c0',      6],
         ['a1', 'b1', 'c1',      7]]
        >>> arr.dump(maxlines=3, edgeitems=1)        # doctest: +NORMALIZE_WHITESPACE
        [['a',   'b\\c',  'c0',  'c1'],
         ['a0',    'b0',     0,     1],
         ['...',  '...', '...', '...'],
         ['a1',    'b1',     6,     7]]
        """
        # _axes_display_names : bool, optional
        #    Whether to get axes names using AxisCollection.display_names instead of
        #    AxisCollection.names. Defaults to False.

        dump_axes_names = axes_names

        if not header:
            # ensure_no_numpy_type is there mostly to avoid problems with xlwings, but I am unsure where that problem
            # should be fixed: in np.array.tolist, in xlwings, here or in xw_excel.Sheet.__setitem__. Doing it here
            # is uglier than in xw_excel but is faster because nothing (extra) needs to be done when the
            # array is not of object dtype (the usual case).

            # flatten all dimensions except the last one
            res2d = ensure_no_numpy_type(self.data.reshape((-1, self.shape[-1])))
        else:
            if not self.ndim:
                return None

            if wide:
                width = self.shape[-1]
                height = int(np.prod(self.shape[:-1]))
            else:
                width = 1
                height = int(np.prod(self.shape))
            data = self.data.reshape((height, width))

            # get list of names of axes
            if _axes_display_names:
                axes_names = self.axes.display_names[:]
            else:
                axes_names = self.axes.names

            # transforms ['a', 'b', 'c', 'd'] into ['a', 'b', 'c\\d']
            if wide:
                if len(axes_names) == 1:
                    # if dump_axes_names is False or 'except_last'
                    if dump_axes_names is not True:
                        axes_names = []
                    # and do nothing when dump_axes_names is True
                elif len(axes_names) > 1:
                    if dump_axes_names is True:
                        # combine two last names
                        last_name = axes_names.pop()
                        prev_name = axes_names[-1]
                        # do not combine if last_name is None or ''
                        if last_name:
                            prev_name = prev_name if prev_name is not None else ''
                            combined_name = prev_name + '\\' + last_name
                        else:
                            # whether it is a string or None !
                            combined_name = prev_name
                        axes_names[-1] = combined_name
                    elif dump_axes_names == 'except_last':
                        axes_names = axes_names[:-1]
                    else:
                        axes_names = [''] * (len(axes_names) - 1)

            axes = self.axes[:-1] if wide else self.axes

            # get list of labels for each axis (except the last one if wide=True)
            labels = [ensure_no_numpy_type(axis.labels) for axis in axes]

            # creates vertical lines (ticks is a list of list)
            if self.ndim == 1 and wide:
                if dump_axes_names is True:
                    # There is no vertical axis, so the axis name should not have
                    # any "tick" below it and we add an empty "tick".
                    ticks = [['']]
                else:
                    # There is no vertical axis but no axis name either
                    ticks = [[]]
            elif light:
                ticks = light_product(*labels)
            else:
                ticks = Product(labels)

            # computes the first line
            other_colnames = ensure_no_numpy_type(self.axes[-1].labels) if wide else [value_name]
            res2d = [axes_names + other_colnames]

            # summary if needed
            if maxlines != -1 and height > maxlines:
                # replace middle lines of the table by '...'.
                # We show only the first and last edgeitems lines.
                res2d.extend([list(tick) + dataline
                              for tick, dataline in zip(ticks[:edgeitems], ensure_no_numpy_type(data[:edgeitems]))])
                res2d.append(["..."] * (self.ndim - 1 + width))
                res2d.extend([list(tick) + dataline
                              for tick, dataline in zip(ticks[-edgeitems:], ensure_no_numpy_type(data[-edgeitems:]))])
            else:
                # all other lines (labels of N-1 first axes + data)
                res2d.extend([list(tick) + ensure_no_numpy_type(dataline) for tick, dataline in zip(ticks, data)])

        if na_repr != 'as_is':
            res2d = [[na_repr if value != value else value
                      for value in line]
                     for line in res2d]
        return res2d
    # this is not 100% equivalent (the names of displayed axes is different) but it has been deprecated long enough
    # (since 0.30) that we can afford slightly breaking backward compatibility.
    as_table = renamed_to(dump, 'as_table')

    # XXX: should filter(geo=['W']) return a view by default? (collapse=True)
    # I think it would be dangerous to make it the default
    # behavior, because that would introduce a subtle difference between
    # filter(dim=[a, b]) and filter(dim=[a]) even though it would be faster
    # and uses less memory. Maybe I should have a "view" argument which
    # defaults to 'auto' (ie collapse by default), can be set to False to
    # force a copy and to True to raise an exception if a view is not possible.
    def filter(self, collapse=False, **kwargs) -> 'Array':
        r"""Filter the array along the axes given as keyword arguments.

        The *collapse* argument determines whether consecutive ranges should
        be collapsed to slices, which is more efficient and returns a view
        (and not a copy) if possible (if all ranges are consecutive).
        Only use this argument if you do not intent to modify the resulting
        array, or if you know what you are doing.

        It is similar to np.take but works with several axes at once.
        """
        return self.__getitem__(kwargs, collapse)

    def _axis_aggregate(self, op, axes=(), keepaxes=False, out=None, **kwargs) -> Union['Array', Scalar]:
        r"""
        Parameters
        ----------
        op : function
            An aggregate function with this signature: func(a, axis=None, dtype=None, out=None, keepdims=False)
        axes : tuple of axes, optional
            Each axis can be an Axis object, str or int.
        out : Array, optional
            Alternative output array in which to place the result. It must have the same shape as the expected output.
        keepaxes : bool or scalar, optional
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

        Returns
        -------
        Array or scalar
        """
        src_data = np.asarray(self)
        axes = self.axes[list(axes)] if axes else self.axes
        axes_indices = tuple(self.axes.index(a) for a in axes) if axes != self.axes else None
        if op.__name__ == 'ptp':
            if axes_indices is not None and len(axes) > 1:
                raise ValueError('ptp can only be applied along a single axis or all axes, not multiple arbitrary axes')
            elif axes_indices is not None:
                axes_indices = axes_indices[0]
        else:
            kwargs['keepdims'] = bool(keepaxes)
        if out is not None:
            assert isinstance(out, Array)
            kwargs['out'] = out.data
        res_data = op(src_data, axis=axes_indices, **kwargs)
        if keepaxes:
            label = op.__name__.replace('nan', '') if keepaxes is True else keepaxes
            new_axes = [Axis([label], axis.name) for axis in axes]
            res_axes = self.axes[:]
            res_axes[axes] = new_axes
        else:
            res_axes = self.axes - axes
        if not res_axes:
            # scalars don't need to be wrapped in Array
            return res_data
        else:
            return Array(res_data, res_axes)

    def _cum_aggregate(self, op, axis) -> 'Array':
        r"""
        op is a numpy cumulative aggregate function: func(arr, axis=0).
        axis is an Axis object, a str or an int. Contrary to other aggregate functions this only supports one axis at a
        time.
        """
        # TODO: accept a single group in axis, to filter & aggregate in one shot
        return Array(op(np.asarray(self), axis=self.axes.index(axis)),
                     self.axes)

    # TODO: now that items is never a (k, v), it should be renamed to something else: args?
    #       (groups would be misleading because each "item" can contain several groups)
    # TODO: experiment implementing this using ufunc.reduceat
    # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ufunc.reduceat.html
    # XXX: rename keepaxes to label=value? For group_aggregates we might want to keep the LGroup label if any
    def _group_aggregate(self, op, items, keepaxes=False, out=None, **kwargs) -> 'Array':
        assert out is None
        res = self
        # TODO: when working with several "axes" at the same times, we should not produce the intermediary result at
        #       all. It should be faster and consume a bit less memory.
        for item in items:
            res_axes = res.axes[:]
            res_shape = list(res.shape)

            if isinstance(item, tuple):
                assert all(isinstance(g, Group) for g in item)
                groups = item
                axis = groups[0].axis
                # they should all have the same axis (this is already checked
                # in _prepare_aggregate though)
                assert all(g.axis.equals(axis) for g in groups[1:])
                killaxis = False
            else:
                # item is in fact a single group
                assert isinstance(item, Group), type(item)
                groups = (item,)
                axis = item.axis
                # it is easier to kill the axis after the fact
                killaxis = True

            axis, axis_idx = res.axes[axis], res.axes.index(axis)
            # potentially translate axis reference to real axes
            groups = tuple(g.with_axis(axis) for g in groups)
            res_shape[axis_idx] = len(groups)

            # XXX: this code is fragile. I wonder if there isn't a way to ask the function what kind of dtype/shape it
            #      will return given the input we are going to give it. My first search for this found nothing. One
            #      way to do this would be to create one big mapping: {(op, input dtype): res dtype}
            res_dtype = float if op in _always_return_float else res.dtype
            if op in (np.sum, np.nansum) and res.dtype in (bool, np.bool_):
                res_dtype = int
            res_data = np.empty(res_shape, dtype=res_dtype)

            group_idx = [slice(None) for _ in res_shape]
            for i, group in enumerate(groups):
                group_idx[axis_idx] = i
                # this is only useful for ndim == 1 because a[(0,)] (equivalent to a[0] which kills the axis)
                # is different from a[[0]] (which does not kill the axis)
                idx = tuple(group_idx)

                # we need only lists of ticks, not single ticks, otherwise the dimension is discarded too early
                # (in __getitem__ instead of in the aggregate func)
                if isinstance(group, IGroup) and np.isscalar(group.key):
                    group = IGroup([group.key], axis=group.axis)
                elif isinstance(group, LGroup):
                    key = _to_key(group.key)
                    assert not isinstance(key, Group)
                    if np.isscalar(key):
                        key = [key]
                    # we do not care about the name at this point
                    group = LGroup(key, axis=group.axis)

                arr = res.__getitem__(group, collapse_slices=True)
                if res_data.ndim == 1:
                    assert len(idx) == 1 and idx[0] == i

                    # res_data[idx] but instead of returning a scalar (eg np.int32), it returns a 0d array which is a
                    # view on res_data, which can thus be used as out
                    out = res_data[i:i + 1].reshape(())
                else:
                    out = res_data[idx]

                arr = np.asarray(arr)
                op(arr, axis=axis_idx, out=out, **kwargs)
                del arr

            if killaxis:
                assert group_idx[axis_idx] == 0
                res_data = res_data[idx]
                del res_axes[axis_idx]
            else:
                # We do NOT modify the axis name (eg append "_agg" or "*") even though this creates a new axis that is
                # independent from the original one because the original name is what users will want to use to access
                # that axis (eg in .filter kwargs)
                res_axes[axis_idx] = Axis(groups, axis.name)

            if isinstance(res_data, np.ndarray):
                res = Array(res_data, res_axes)
            else:
                res = res_data
        return res

    # TODO: not sure about the returned type
    def _prepare_aggregate(self, op, args, kwargs=None, commutative=False, stack_depth=1) \
            -> Union[List[Union[LGroup, Axis]], AxisCollection]:
        r"""Convert args to keys & LGroup and kwargs to LGroup."""
        if kwargs is None:
            kwargs_items = []
        else:
            explicit_axis = kwargs.pop('axis', None)
            if explicit_axis is not None:
                explicit_axis = self.axes[explicit_axis]
                if isinstance(explicit_axis, Axis):
                    args += (explicit_axis,)
                else:
                    assert isinstance(explicit_axis, AxisCollection)
                    args += tuple(explicit_axis)
            kwargs_items = kwargs.items()

        # Sort kwargs by axis name so that we have consistent results between runs because otherwise rounding errors
        # could lead to slightly different results even for commutative operations.
        sorted_kwargs = sorted(kwargs_items)

        # convert kwargs to LGroup so that we can only use args afterwards but still keep the axis information
        def standardise_kw_arg(axis_name, key, stack_depth=1):
            if isinstance(key, str):
                key = _to_keys(key, stack_depth + 1)
            if isinstance(key, tuple):
                # XXX +2?
                return tuple(standardise_kw_arg(axis_name, k, stack_depth + 1) for k in key)
            if isinstance(key, LGroup):
                return key
            return self.axes[axis_name][key]

        def to_labelgroup(key, stack_depth=1):
            if isinstance(key, str):
                key = _to_keys(key, stack_depth + 1)
            if isinstance(key, tuple):
                # a tuple is supposed to be several groups on the same axis
                # TODO: it would be better to use self._translate_axis_key directly (so that we do not need to do the
                # label -> position translation twice) but this fails because the groups are also used as ticks on the
                # new axis, and igroups are not the same that LGroups in this regard (I wonder if ideally it shouldn't
                # be the same???)
                # groups = tuple(self._translate_axis_key(k) for k in key)
                groups = tuple(self.axes._guess_axis(_to_key(k, stack_depth + 1)) for k in key)
                first_group_axis = groups[0].axis
                if not all(g.axis.equals(first_group_axis) for g in groups[1:]):
                    raise ValueError(f"group with different axes: {key}")
                return groups
            elif isinstance(key, (Group, int, str, list, slice)):
                return self.axes._guess_axis(key)
            else:
                key_type = type(key).__name__
                raise NotImplementedError(f"{key} has invalid type ({key_type}) for a group aggregate key")

        def standardise_arg(arg, stack_depth=1):
            if self.axes.isaxis(arg):
                return self.axes[arg]
            else:
                return to_labelgroup(arg, stack_depth + 1)

        operations = [standardise_arg(a, stack_depth=stack_depth + 2) for a in args if a is not None] + \
                     [standardise_kw_arg(k, v, stack_depth=stack_depth + 2) for k, v in sorted_kwargs]
        if not operations:
            # op() without args is equal to op(all_axes)
            operations = self.axes
        return operations

    def _by_args_to_normal_agg_args(self, operations) -> List[Union[Axis, Group]]:
        # get axes to aggregate
        flat_op = chain.from_iterable([(o,) if isinstance(o, (Group, Axis)) else o
                                       for o in operations])
        axes = [o.axis if isinstance(o, Group) else o for o in flat_op]
        to_agg = self.axes - axes

        # add groups to axes to aggregate
        def is_or_contains_group(o):
            return isinstance(o, Group) or (isinstance(o, tuple) and isinstance(o[0], Group))

        return list(to_agg) + [o for o in operations if is_or_contains_group(o)]

    def _aggregate(self, op, args, kwargs=None, keepaxes=False, by_agg=False, commutative=False,
                   out=None, extra_kwargs={}) -> Union['Array', Scalar]:
        operations = self._prepare_aggregate(op, args, kwargs, commutative, stack_depth=3)

        total_len_args = len(args) + len(kwargs) if kwargs is not None else 0
        if by_agg and total_len_args:
            operations = self._by_args_to_normal_agg_args(operations)

        res = self
        # group *consecutive* same-type (group vs axis aggregates) operations
        # we do not change the order of operations since we only group consecutive operations.
        for are_axes, axes in groupby(operations, self.axes.isaxis):
            func = res._axis_aggregate if are_axes else res._group_aggregate
            res = func(op, axes, keepaxes=keepaxes, out=out, **extra_kwargs)
        return res

    def with_total(self, *args, op=sum, label='total', **kwargs) -> 'Array':
        r"""Add aggregated values (sum by default) along each axis.

        A user defined label can be given to specified the computed values.

        Parameters
        ----------
        *args : int or str or Axis or Group or any combination of those, optional
            Axes or groups along which to compute the aggregates. Passed groups should be named.
            Defaults to aggregate over the whole array.
        op : aggregate function, optional
            Available aggregate functions are: `sum`, `prod`, `min`, `max`, `mean`, `ptp`, `var`, `std`,
            `median` and `percentile`. Defaults to `sum`.
        label : scalar value, optional
            Label to use for the total. Applies only to aggregated axes, not groups. Defaults to "total".
        \**kwargs : int or str or Group or any combination of those, optional
            Axes or groups along which to compute the aggregates.

        Returns
        -------
        Array

        Examples
        --------
        >>> arr = ndtest("gender=M,F;time=2013..2016")
        >>> arr
        gender\time  2013  2014  2015  2016
                  M     0     1     2     3
                  F     4     5     6     7
        >>> arr.with_total()
        gender\time  2013  2014  2015  2016  total
                  M     0     1     2     3      6
                  F     4     5     6     7     22
              total     4     6     8    10     28

        Using another function and label

        >>> arr.with_total(op=mean, label='mean')
        gender\time  2013  2014  2015  2016  mean
                  M   0.0   1.0   2.0   3.0   1.5
                  F   4.0   5.0   6.0   7.0   5.5
               mean   2.0   3.0   4.0   5.0   3.5

        Specifying an axis and a label

        >>> arr.with_total('gender', label='U')
        gender\time  2013  2014  2015  2016
                  M     0     1     2     3
                  F     4     5     6     7
                  U     4     6     8    10

        Using groups

        >>> time_groups = (arr.time[:2014] >> 'before_2015',
        ...                arr.time[2015:] >> 'after_2015')
        >>> arr.with_total(time_groups)
        gender\time  2013  2014  2015  2016  before_2015  after_2015
                  M     0     1     2     3            1           5
                  F     4     5     6     7            9          13
        >>> # or equivalently
        >>> # arr.with_total('time[:2014] >> before_2015; time[2015:] >> after_2015')
        """
        # TODO: make label default to op.__name__
        npop = {
            sum: np.sum,
            prod: np.prod,
            min: np.min,
            max: np.max,
            mean: np.mean,
            ptp: np.ptp,
            var: np.var,
            std: np.std,
            median: np.median,
            percentile: np.percentile,
        }
        # TODO: commutative should be known for usual ops
        operations = self._prepare_aggregate(op, args, kwargs, False, stack_depth=2)
        res = self
        # TODO: we should allocate the final result directly and fill it progressively, so that the original array is
        #       only copied once
        for axis in operations:
            # TODO: append/extend first with an empty array then _aggregate with out=
            if self.axes.isaxis(axis):
                value = res._axis_aggregate(npop[op], (axis,), keepaxes=label)
            else:
                # groups
                if not isinstance(axis, tuple):
                    # assume a single group
                    axis = (axis,)
                lgkey = axis
                axis = lgkey[0].axis
                value = res._aggregate(npop[op], (lgkey,))
            res = res.append(axis, value)
        return res

    # TODO: make sure we can do
    # arr[X.sex.i[arr.indexofmin(X.sex)]] <- fails
    # and
    # arr[arr.labelofmin(X.sex)] <- fails
    # should both be equal to arr.min(X.sex)
    # the versions where axis is None already work as expected in the simple
    # case (no ambiguous labels):
    # arr.i[arr.indexofmin()]
    # arr[arr.labelofmin()]
    # for the case where axis is None, we should return an NDGroup
    # so that arr[arr.labelofmin()] works even if the minimum is on ambiguous labels
    def labelofmin(self, axis=None) -> Union['Array', Tuple[Scalar, ...]]:
        r"""Return labels of the minimum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        Array

        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = Array([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.labelofmin('sex')
        nat  BE  FR  IT
              M   F   M
        >>> arr.labelofmin()
        ('BE', 'M')
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            data = axis.labels[self.data.argmin(axis_idx)]
            return Array(data, self.axes - axis)
        else:
            return self.axes._iflat(self.data.argmin())

    argmin = renamed_to(labelofmin, 'argmin', raise_error=True)

    def indexofmin(self, axis=None) -> Union['Array', Tuple[int, ...]]:
        r"""Return indices of the minimum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        Array

        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = Array([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.indexofmin('sex')
        nat  BE  FR  IT
              0   1   0
        >>> arr.indexofmin()
        (0, 0)
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            return Array(self.data.argmin(axis_idx), self.axes - axis)
        else:
            return np.unravel_index(self.data.argmin(), self.shape)

    posargmin = renamed_to(indexofmin, 'posargmin', raise_error=True)

    def labelofmax(self, axis=None) -> Union['Array', Tuple[Scalar, ...]]:
        r"""Return labels of the maximum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        Array

        Notes
        -----
        In case of multiple occurrences of the maximum values, the labels corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = Array([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.labelofmax('sex')
        nat  BE  FR  IT
              F   M   F
        >>> arr.labelofmax()
        ('IT', 'F')
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            data = axis.labels[self.data.argmax(axis_idx)]
            return Array(data, self.axes - axis)
        else:
            return self.axes._iflat(self.data.argmax())

    argmax = renamed_to(labelofmax, 'argmax', raise_error=True)

    def indexofmax(self, axis=None) -> Union['Array', Tuple[int, ...]]:
        r"""Return indices of the maximum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        Array

        Notes
        -----
        In case of multiple occurrences of the maximum values, the labels corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = Array([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.indexofmax('sex')
        nat  BE  FR  IT
              1   0   1
        >>> arr.indexofmax()
        (2, 1)
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            return Array(self.data.argmax(axis_idx), self.axes - axis)
        else:
            return np.unravel_index(self.data.argmax(), self.shape)

    posargmax = renamed_to(indexofmax, 'posargmax', raise_error=True)

    def labelsofsorted(self, axis=None, ascending=True, kind='quicksort') -> 'Array':
        r"""Return the labels that would sort this array.

        Performs an indirect sort along the given axis using the algorithm specified by the `kind` keyword. It returns
        an array of labels of the same shape as `a` that index data along the given axis in sorted order.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to sort. This can be omitted if array has only one axis.
        ascending : bool, optional
            Sort values in ascending order. Defaults to True.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Defaults to 'quicksort'.

        Returns
        -------
        Array

        Examples
        --------
        >>> arr = Array([[0, 1], [3, 2], [2, 5]], "nat=BE,FR,IT; sex=M,F")
        >>> arr
        nat\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.labelsofsorted('sex')
        nat\sex  0  1
             BE  M  F
             FR  F  M
             IT  M  F
        >>> arr.labelsofsorted('sex', ascending=False)
        nat\sex  0  1
             BE  F  M
             FR  M  F
             IT  F  M
        """
        if axis is None:
            if self.ndim > 1:
                raise ValueError("array has ndim > 1 and no axis specified for labelsofsorted")
            axis = self.axes[0]
        axis = self.axes[axis]
        pos = self.indicesofsorted(axis, ascending=ascending, kind=kind)
        return Array(axis.labels[pos.data], pos.axes)

    argsort = renamed_to(labelsofsorted, 'argsort', raise_error=True)

    def indicesofsorted(self, axis=None, ascending=True, kind='quicksort') -> 'Array':
        r"""Return the indices that would sort this array.

        Performs an indirect sort along the given axis using the algorithm specified by the `kind` keyword. It returns
        an array of indices with the same axes as `a` that index data along the given axis in sorted order.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to sort. This can be omitted if array has only one axis.
        ascending : bool, optional
            Sort values in ascending order. Defaults to True.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Defaults to 'quicksort'.

        Returns
        -------
        Array

        Examples
        --------
        >>> arr = Array([[1, 5], [3, 2], [0, 4]], "nat=BE,FR,IT; sex=M,F")
        >>> arr
        nat\sex  M  F
             BE  1  5
             FR  3  2
             IT  0  4
        >>> arr.indicesofsorted('nat')
        nat\sex  M  F
              0  2  1
              1  0  2
              2  1  0
        >>> arr.indicesofsorted('nat', ascending=False)
        nat\sex  M  F
              0  1  0
              1  0  2
              2  2  1
        """
        if axis is None:
            if self.ndim > 1:
                raise ValueError("array has ndim > 1 and no axis specified for indicesofsorted")
            axis = self.axes[0]
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        data = self.data.argsort(axis_idx, kind=kind)
        if not ascending:
            reverser = tuple(slice(None, None, -1) if i == axis_idx else slice(None)
                             for i in range(self.ndim))
            data = data[reverser]
        new_axis = Axis(np.arange(len(axis)), axis.name)
        return Array(data, self.axes.replace(axis, new_axis))

    posargsort = renamed_to(indicesofsorted, 'posargsort', raise_error=True)

    # TODO: implement keys_by
    # XXX: implement expand=True? Unsure it is necessary now that we have zip_array_*
    # TODO: add support for groups in addition to entire axes
    def keys(self, axes=None, ascending=True) -> Product:
        r"""Return a view on the array labels along axes.

        Parameters
        ----------
        axes : int, str or Axis or tuple of them, optional
            Axis or axes along which to iterate and in which order. Defaults to None (all axes in the order they are
            in the array).
        ascending : bool, optional
            Whether to iterate the axes in ascending order (from start to end). Defaults to True.

        Returns
        -------
        Sequence
            An object you can iterate (loop) on and index by position to get the Nth label along axes.

        Examples
        --------
        First, define a small helper function to make the following examples more readable.

        >>> def str_key(key):
        ...     return tuple(str(k) for k in key)

        Then create a test array:

        >>> arr = ndtest((2, 2))
        >>> arr
        a\b  b0  b1
         a0   0   1
         a1   2   3

        By default it iterates on all axes, in the order they are in the array.

        >>> for key in arr.keys():
        ...     # print both the actual key object, and a (nicer) string representation
        ...     print(key, "->", str_key(key))
        (a.i[0], b.i[0]) -> ('a0', 'b0')
        (a.i[0], b.i[1]) -> ('a0', 'b1')
        (a.i[1], b.i[0]) -> ('a1', 'b0')
        (a.i[1], b.i[1]) -> ('a1', 'b1')
        >>> for key in arr.keys(ascending=False):
        ...     print(str_key(key))
        ('a1', 'b1')
        ('a1', 'b0')
        ('a0', 'b1')
        ('a0', 'b0')

        but you can specify another axis order:

        >>> for key in arr.keys(('b', 'a')):
        ...     print(str_key(key))
        ('b0', 'a0')
        ('b0', 'a1')
        ('b1', 'a0')
        ('b1', 'a1')

        One can specify less axes than the array has:

        >>> # iterate on the "b" axis, that is return each label along the "b" axis
        ... for key in arr.keys('b'):
        ...     print(str_key(key))
        ('b0',)
        ('b1',)

        One can also access elements of the key sequence directly, instead of iterating over it. Say we want to
        retrieve the first and last keys of our array, we could write:

        >>> keys = arr.keys()
        >>> first_key = keys[0]
        >>> str_key(first_key)
        ('a0', 'b0')
        >>> last_key = keys[-1]
        >>> str_key(last_key)
        ('a1', 'b1')
        """
        return self.axes.iter_labels(axes, ascending=ascending)

    # TODO: implement values_by
    # TODO: add support for groups in addition to entire axes
    # TODO : not sure about the returned type
    def values(self, axes=None, ascending=True) -> Union[np.ndarray, List['Array'], ArrayPositionalIndexer]:
        r"""Return a view on the values of the array along axes.

        Parameters
        ----------
        axes : int, str or Axis or tuple of them, optional
            Axis or axes along which to iterate and in which order. Defaults to None (all axes in the order they are
            in the array).
        ascending : bool, optional
            Whether to iterate the axes in ascending order (from start to end). Defaults to True.

        Returns
        -------
        Sequence
            An object you can iterate (loop) on and index by position.

        Examples
        --------
        >>> arr = ndtest((2, 2))
        >>> arr
        a\b  b0  b1
         a0   0   1
         a1   2   3

        By default it iterates on all axes, in the order they are in the array.

        >>> for value in arr.values():
        ...     print(value)
        0
        1
        2
        3
        >>> for value in arr.values(ascending=False):
        ...     print(value)
        3
        2
        1
        0

        but you can specify another axis order:

        >>> for value in arr.values(('b', 'a')):
        ...     print(value)
        0
        2
        1
        3

        When you specify less axes than the array has, you get arrays back:

        >>> # iterate on the "b" axis, that is return the (sub)array for each label along the "b" axis
        ... for value in arr.values('b'):
        ...     print(value)
        a  a0  a1
            0   2
        a  a0  a1
            1   3
        >>> # iterate on the "b" axis, that is return the (sub)array for each label along the "b" axis
        ... for value in arr.values('b', ascending=False):
        ...     print(value)
        a  a0  a1
            1   3
        a  a0  a1
            0   2

        One can also access elements of the value sequence directly, instead of iterating over it. Say we want to
        retrieve the first and last values of our array, we could write:

        >>> values = arr.values()
        >>> values[0]
        0
        >>> values[-1]
        3
        """
        if axes is None:
            combined = np.ravel(self.data)
            # combined[::-1] *is* indexable
            return combined if ascending else combined[::-1]

        if not isinstance(axes, (tuple, list, AxisCollection)):
            axes = (axes,)

        if len(axes) == 0:
            # empty axes list
            return [self]

        axes = self.axes[axes]
        # move axes in front
        transposed = self.transpose(axes)
        # combine axes if necessary
        combined = transposed.combine_axes(axes, wildcard=True) if len(axes) > 1 else transposed
        # trailing .i is to support the case where axis < self.axes (ie the elements of the result are arrays)
        return combined.i if ascending else combined.i[::-1].i

    # TODO: we currently return a tuple of groups even for 1D arrays, which can be both a bad or a good thing.
    #       if we returned an NDGroup in all cases, it would solve the problem
    def items(self, axes=None, ascending=True) -> SequenceZip:
        r"""Return a (label, value) view of the array along axes.

        Parameters
        ----------
        axes : int, str or Axis or tuple of them, optional
            Axis or axes along which to iterate and in which order. Defaults to None (all axes in the order they are
            in the array).
        ascending : bool, optional
            Whether to iterate the axes in ascending order (from start to end). Defaults to True.

        Returns
        -------
        Sequence
            An object you can iterate (loop) on and index by position to get the Nth (label, value) couple along axes.

        Examples
        --------
        First, define a small helper function to make the following examples more readable.

        >>> def str_key(key):
        ...     return tuple(str(k) for k in key)

        Then create a test array:

        >>> arr = ndtest((2, 2))
        >>> arr
        a\b  b0  b1
         a0   0   1
         a1   2   3

        By default it iterates on all axes, in the order they are in the array.

        >>> for key, value in arr.items():
        ...     print(str_key(key), "->", value)
        ('a0', 'b0') -> 0
        ('a0', 'b1') -> 1
        ('a1', 'b0') -> 2
        ('a1', 'b1') -> 3
        >>> for key, value in arr.items(ascending=False):
        ...     print(str_key(key), "->", value)
        ('a1', 'b1') -> 3
        ('a1', 'b0') -> 2
        ('a0', 'b1') -> 1
        ('a0', 'b0') -> 0

        but you can specify another axis order:

        >>> for key, value in arr.items(('b', 'a')):
        ...     print(str_key(key), "->", value)
        ('b0', 'a0') -> 0
        ('b0', 'a1') -> 2
        ('b1', 'a0') -> 1
        ('b1', 'a1') -> 3

        When you specify less axes than the array has, you get arrays back:

        >>> # iterate on the "b" axis, that is return the (sub)array for each label along the "b" axis
        ... for key, value in arr.items('b'):
        ...     print(str_key(key), value, sep="\n")
        ('b0',)
        a  a0  a1
            0   2
        ('b1',)
        a  a0  a1
            1   3

        One can also access elements of the items sequence directly, instead of iterating over it. Say we want to
        retrieve the first and last key-value pairs of our array, we could write:

        >>> items = arr.items()
        >>> first_key, first_value = items[0]
        >>> str_key(first_key)
        ('a0', 'b0')
        >>> first_value
        0
        >>> last_key, last_value = items[-1]
        >>> str_key(last_key)
        ('a1', 'b1')
        >>> last_value
        3
        """
        return SequenceZip((self.keys(axes, ascending=ascending), self.values(axes, ascending=ascending)))

    @lazy_attribute
    def iflat(self) -> ArrayFlatIndicesIndexer:
        return ArrayFlatIndicesIndexer(self)
    iflat.__doc__ = ArrayFlatIndicesIndexer.__doc__

    def copy(self) -> 'Array':
        r"""Return a copy of the array."""
        return Array(self.data.copy(), axes=self.axes[:], meta=self.meta)

    # XXX: we might want to implement this using .groupby().first()
    def unique(self, axes=None, sort=False, sep='_') -> 'Array':
        r"""Return unique values (optionally along axes).

        Parameters
        ----------
        axes : axis reference (int, str, Axis) or sequence of them, optional
            Axis or axes along which to compute unique values. Defaults to None (all axes).
        sort : bool, optional
            Whether to sort unique values. Defaults to False. Sorting is not implemented yet for unique() along
            multiple axes.
        sep : str, optional
            Separator when several labels need to be combined. Defaults to '_'.

        Returns
        -------
        Array
            array with unique values

        Examples
        --------
        >>> arr = Array([[0, 2, 0, 0],
        ...              [1, 1, 1, 0]], 'a=a0,a1;b=b0..b3')
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   2   0   0
         a1   1   1   1   0

        By default unique() returns the first occurrence of each unique value in the order it appears:

        >>> arr.unique()
        a_b  a0_b0  a0_b1  a1_b0
                 0      2      1

        To sort the unique values, use the sort argument:

        >>> arr.unique(sort=True)
        a_b  a0_b0  a1_b0  a0_b1
                 0      1      2

        One can also compute unique sub-arrays (i.e. combination of values) along axes. In our example the a0=0, a1=1
        combination appears twice along the 'b' axis, so 'b2' is not returned:

        >>> arr.unique('b')
        a\b  b0  b1  b3
         a0   0   2   0
         a1   1   1   0
        >>> arr.unique('b', sort=True)
        a\b  b3  b0  b1
         a0   0   0   2
         a1   0   1   1
        """
        if axes is not None:
            axes = self.axes[axes]

        assert axes is None or isinstance(axes, (Axis, AxisCollection))

        if not isinstance(axes, AxisCollection):
            axis_idx = self.axes.index(axes) if axes is not None else None
            # axis needs np >= 1.13
            _, unq_index = np.unique(self, axis=axis_idx, return_index=True)
            if not sort:
                unq_index = np.sort(unq_index)
            if axes is None:
                return self.iflat.__getitem__(unq_index, sep=sep)
            else:
                return self[axes.i[unq_index]]
        else:
            if sort:
                raise NotImplementedError('sort=True is not implemented for unique along multiple axes')
            unq_list = []
            seen = set()
            list_append = unq_list.append
            seen_add = seen.add
            sep_join = sep.join
            axis_name = sep_join(a.name for a in axes)
            first_axis_idx = self.axes.index(axes[0])
            # XXX: use combine_axes(axes).items() instead?
            for labels, value in self.items(axes):
                hashable_value = value.data.tobytes() if isinstance(value, Array) else value
                if hashable_value not in seen:
                    list_append((sep_join(str(label) for label in labels), value))
                    seen_add(hashable_value)
            res_arr = stack(unq_list, axis_name)
            # transpose the combined axis at the position where the first of the combined axes was
            # TODO: use res_arr.transpose(res_arr.axes.move_axis(-1, first_axis_idx)) once #564 is implemented:
            #       https://github.com/larray-project/larray/issues/564
            # stack adds the stacked axes at the end
            combined_axis = res_arr.axes[-1]
            assert combined_axis.name == axis_name
            new_axes_order = res_arr.axes - combined_axis
            new_axes_order.insert(first_axis_idx, combined_axis)
            return res_arr.transpose(new_axes_order)

    @property
    def info(self) -> str:
        r"""Describe an Array (metadata + shape and labels for each axis).

        Returns
        -------
        str
            Description of the array (metadata + shape and labels for each axis).

        Examples
        --------
        >>> mat0 = Array([[2.0, 5.0], [8.0, 6.0]], "nat=BE,FO; sex=F,M")
        >>> mat0.info
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'F' 'M'
        dtype: float64
        memory used: 32 bytes
        >>> mat0.meta.title = 'test matrix'
        >>> mat0.info
        title: test matrix
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'F' 'M'
        dtype: float64
        memory used: 32 bytes
        """
        str_info = ''
        if len(self.meta):
            str_info += f'{self.meta}\n'
        str_info += f'{self.axes.info}\ndtype: {self.dtype.name}\nmemory used: {self.memory_used}'
        return ReprString(str_info)

    def ratio(self, *axes) -> 'Array':
        r"""Return an array with all values divided by the sum of values along given axes.

        Parameters
        ----------
        *axes

        Returns
        -------
        Array
            array / array.sum(axes)

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> a = Array([[4, 6], [2, 8]], [nat, sex])
        >>> a
        nat\sex  M  F
             BE  4  6
             FO  2  8
        >>> a.sum()
        20
        >>> a.ratio()
        nat\sex    M    F
             BE  0.2  0.3
             FO  0.1  0.4
        >>> a.ratio('sex')
        nat\sex    M    F
             BE  0.4  0.6
             FO  0.2  0.8
        >>> a.ratio('M')
        nat\sex    M    F
             BE  1.0  1.5
             FO  1.0  4.0
        """
        # # this does not work, but I am unsure it should
        # # >>> a.sum(age[[0, 1]], age[2]) / a.sum(age)
        # >>> a.sum(([0, 1], 2)) / a.sum(age)
        # # >>> a / a.sum(([0, 1], 2))
        # >>> a.sum(X.sex)
        # >>> a.sum(X.age)
        # >>> a.sum(X.sex) / a.sum(X.age)
        # >>> a.ratio('F')
        # could mean
        # >>> a.sum('F') / a.sum(a.get_axis('F'))
        # >>> a.sum('F') / a.sum(X.sex)
        # age    0    1               2
        #      1.0  0.6  0.555555555556
        # OR (current meaning)
        # >>> a / a.sum('F')
        # age\sex               M    F
        #       0             0.0  1.0
        #       1  0.666666666667  1.0
        #       2             0.8  1.0
        # One solution is to add an argument
        # >>> a.ratio(what='F', by=X.sex)
        # age    0    1               2
        #      1.0  0.6  0.555555555556
        # >>> a.sum('F') / a.sum(X.sex)

        # >>> a.sum((age[[0, 1]], age[[1, 2]])) / a.sum(age)
        # >>> a.ratio((age[[0, 1]], age[[1, 2]]), by=age)

        # >>> a.sum((X.age[[0, 1]], X.age[[1, 2]])) / a.sum(X.age)
        # >>> a.ratio((X.age[[0, 1]], X.age[[1, 2]], by=X.age)

        # >>> lalala.sum(([0, 1], [1, 2])) / lalala.sum(X.age)
        # >>> lalala.ratio(([0, 1], [1, 2]), by=X.age)

        # >>> b = a.sum((age[[0, 1]], age[[1, 2]]))
        # >>> b
        # age\sex  M  F
        #   [0 1]  2  4
        #   [1 2]  6  8
        # >>> b / b.sum(X.age)
        # age\sex     M               F
        #   [0 1]  0.25  0.333333333333
        #   [1 2]  0.75  0.666666666667
        # >>> b / a.sum(X.age)
        # age\sex               M               F
        #   [0 1]  0.333333333333  0.444444444444
        #   [1 2]             1.0  0.888888888889
        # # >>> a.ratio([0, 1], [2])
        # # >>> a.ratio(X.age[[0, 1]], X.age[2])
        # >>> a.ratio((X.age[[0, 1]], X.age[2]))
        # nat\sex             M    F
        #      BE           0.0  1.0
        #      FO  0.6666666666  1.0
        return self / self.sum(*axes)

    def rationot0(self, *axes) -> 'Array':
        # part of the doctest is skipped because it produces a warning we do not want to have to handle within the
        # doctest and cannot properly ignore
        r"""Return an Array with values array / array.sum(axes) where the sum is not 0, 0 otherwise.

        Parameters
        ----------
        *axes

        Returns
        -------
        Array
            array / array.sum(axes)

        Examples
        --------
        >>> a = Axis('a=a0,a1')
        >>> b = Axis('b=b0,b1,b2')
        >>> arr = Array([[6, 0, 2],
        ...              [4, 0, 8]], [a, b])
        >>> arr
        a\b  b0  b1  b2
         a0   6   0   2
         a1   4   0   8
        >>> arr.sum()
        20
        >>> arr.rationot0()
        a\b   b0   b1   b2
         a0  0.3  0.0  0.1
         a1  0.2  0.0  0.4
        >>> arr.rationot0('a')
        a\b   b0   b1   b2
         a0  0.6  0.0  0.2
         a1  0.4  0.0  0.8

        for reference, the normal ratio method would produce a warning message and return:

        >>> arr.ratio('a')                                          # doctest: +SKIP
        a\b   b0   b1   b2
         a0  0.6  nan  0.2
         a1  0.4  nan  0.8
        """
        return self.divnot0(self.sum(*axes))

    def percent(self, *axes) -> 'Array':
        r"""Return an array with values given as percent of the total of all values along given axes.

        Parameters
        ----------
        *axes

        Returns
        -------
        Array
            array / array.sum(axes) * 100

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> a = Array([[4, 6], [2, 8]], [nat, sex])
        >>> a
        nat\sex  M  F
             BE  4  6
             FO  2  8
        >>> a.percent()
        nat\sex     M     F
             BE  20.0  30.0
             FO  10.0  40.0
        >>> a.percent('sex')
        nat\sex     M     F
             BE  40.0  60.0
             FO  20.0  80.0
        """
        return self * 100.0 / self.sum(*axes)

    # aggregate method decorator
    def _decorate_agg_method(npfunc, nanfunc=None, commutative=False, by_agg=False, extra_kwargs=[],
                             long_name='', action_verb='perform'):
        def decorated(func) -> Union['Array', Scalar]:
            _doc_agg_method(func, by_agg, long_name, action_verb, kwargs=extra_kwargs + ['out', 'skipna', 'keepaxes'])

            @functools.wraps(func)
            def wrapper(self, *args, keepaxes=_kwarg_agg['keepaxes']['value'], skipna=_kwarg_agg['skipna']['value'],
                        out=_kwarg_agg['out']['value'], **kwargs):
                if skipna is None:
                    skipna = nanfunc is not None
                if skipna and nanfunc is None:
                    raise ValueError(f"skipna is not available for {func.__name__}")
                _npfunc = nanfunc if skipna else npfunc
                _extra_kwargs = {}
                for k in extra_kwargs:
                    _extra_kwargs[k] = kwargs.pop(k, _kwarg_agg[k]['value'])
                return self._aggregate(_npfunc, args, kwargs, by_agg=by_agg, keepaxes=keepaxes,
                                       commutative=commutative, out=out, extra_kwargs=_extra_kwargs)
            return wrapper
        return decorated

    @_decorate_agg_method(np.all, commutative=True, long_name="AND reduction")
    def all(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Test whether all selected elements evaluate to True.

        {parameters}

        Returns
        -------
        Array of bool or bool

        See Also
        --------
        Array.all_by, Array.any, Array.any_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.all()
        False
        >>> # along axis 'a'
        >>> barr.all('a')
        b     b0     b1     b2     b3
           False  False  False  False
        >>> # along axis 'b'
        >>> barr.all('b')
        a    a0     a1     a2     a3
           True  False  False  False

        Select some rows only

        >>> barr.all(['a0', 'a1'])
        b    b0    b1     b2     b3
           True  True  False  False
        >>> # or equivalently
        >>> # barr.all('a0,a1')

        Split an axis in several parts

        >>> barr.all((['a0', 'a1'], ['a2', 'a3']))
          a\b     b0     b1     b2     b3
        a0,a1   True   True  False  False
        a2,a3  False  False  False  False
        >>> # or equivalently
        >>> # barr.all('a0,a1;a2,a3')

        Same with renaming

        >>> barr.all((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b     b0     b1     b2     b3
        a01   True   True  False  False
        a23  False  False  False  False
        >>> # or equivalently
        >>> # barr.all('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.all, commutative=True, by_agg=True, long_name="AND reduction")
    def all_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Test whether all selected elements evaluate to True.

        {parameters}

        Returns
        -------
        Array of bool or bool

        See Also
        --------
        Array.all, Array.any, Array.any_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.all_by()
        False
        >>> # by axis 'a'
        >>> barr.all_by('a')
        a    a0     a1     a2     a3
           True  False  False  False
        >>> # by axis 'b'
        >>> barr.all_by('b')
        b     b0     b1     b2     b3
           False  False  False  False

        Select some rows only

        >>> barr.all_by(['a0', 'a1'])
        False
        >>> # or equivalently
        >>> # barr.all_by('a0,a1')

        Split an axis in several parts

        >>> barr.all_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
           False  False
        >>> # or equivalently
        >>> # barr.all_by('a0,a1;a2,a3')

        Same with renaming

        >>> barr.all_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a    a01    a23
           False  False
        >>> # or equivalently
        >>> # barr.all_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.any, commutative=True, long_name="OR reduction")
    def any(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Test whether any selected elements evaluate to True.

        {parameters}

        Returns
        -------
        Array of bool or bool

        See Also
        --------
        Array.any_by, Array.all, Array.all_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.any()
        True
        >>> # along axis 'a'
        >>> barr.any('a')
        b    b0    b1    b2    b3
           True  True  True  True
        >>> # along axis 'b'
        >>> barr.any('b')
        a    a0    a1     a2     a3
           True  True  False  False

        Select some rows only

        >>> barr.any(['a0', 'a1'])
        b    b0    b1    b2    b3
           True  True  True  True
        >>> # or equivalently
        >>> # barr.any('a0,a1')

        Split an axis in several parts

        >>> barr.any((['a0', 'a1'], ['a2', 'a3']))
          a\b     b0     b1     b2     b3
        a0,a1   True   True   True   True
        a2,a3  False  False  False  False
        >>> # or equivalently
        >>> # barr.any('a0,a1;a2,a3')

        Same with renaming

        >>> barr.any((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b     b0     b1     b2     b3
        a01   True   True   True   True
        a23  False  False  False  False
        >>> # or equivalently
        >>> # barr.any('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.any, commutative=True, by_agg=True, long_name="OR reduction")
    def any_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Test whether any selected elements evaluate to True.

        {parameters}

        Returns
        -------
        Array of bool or bool

        See Also
        --------
        Array.any, Array.all, Array.all_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.any_by()
        True
        >>> # by axis 'a'
        >>> barr.any_by('a')
        a    a0    a1     a2     a3
           True  True  False  False
        >>> # by axis 'b'
        >>> barr.any_by('b')
        b    b0    b1    b2    b3
           True  True  True  True

        Select some rows only

        >>> barr.any_by(['a0', 'a1'])
        True
        >>> # or equivalently
        >>> # barr.any_by('a0,a1')

        Split an axis in several parts

        >>> barr.any_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
            True  False
        >>> # or equivalently
        >>> # barr.any_by('a0,a1;a2,a3')

        Same with renaming

        >>> barr.any_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a   a01    a23
           True  False
        >>> # or equivalently
        >>> # barr.any_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    # commutative modulo float precision errors

    @_decorate_agg_method(np.sum, np.nansum, commutative=True, extra_kwargs=['dtype'])
    def sum(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the sum of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.sum_by, Array.prod, Array.prod_by,
        Array.cumsum, Array.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.sum()
        120
        >>> # along axis 'a'
        >>> arr.sum('a')
        b  b0  b1  b2  b3
           24  28  32  36
        >>> # along axis 'b'
        >>> arr.sum('b')
        a  a0  a1  a2  a3
            6  22  38  54

        Select some rows only

        >>> arr.sum(['a0', 'a1'])
        b  b0  b1  b2  b3
            4   6   8  10
        >>> # or equivalently
        >>> # arr.sum('a0,a1')

        Split an axis in several parts

        >>> arr.sum((['a0', 'a1'], ['a2', 'a3']))
          a\b  b0  b1  b2  b3
        a0,a1   4   6   8  10
        a2,a3  20  22  24  26
        >>> # or equivalently
        >>> # arr.sum('a0,a1;a2,a3')

        Same with renaming

        >>> arr.sum((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b  b0  b1  b2  b3
        a01   4   6   8  10
        a23  20  22  24  26
        >>> # or equivalently
        >>> # arr.sum('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.sum, np.nansum, commutative=True, by_agg=True, extra_kwargs=['dtype'], long_name="sum")
    def sum_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the sum of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.sum, Array.prod, Array.prod_by,
        Array.cumsum, Array.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.sum_by()
        120
        >>> # along axis 'a'
        >>> arr.sum_by('a')
        a  a0  a1  a2  a3
            6  22  38  54
        >>> # along axis 'b'
        >>> arr.sum_by('b')
        b  b0  b1  b2  b3
           24  28  32  36

        Select some rows only

        >>> arr.sum_by(['a0', 'a1'])
        28
        >>> # or equivalently
        >>> # arr.sum_by('a0,a1')

        Split an axis in several parts

        >>> arr.sum_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
              28     92
        >>> # or equivalently
        >>> # arr.sum_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.sum_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a  a01  a23
            28   92
        >>> # or equivalently
        >>> # arr.sum_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    # nanprod needs numpy 1.10
    @_decorate_agg_method(np.prod, np_nanprod, commutative=True, extra_kwargs=['dtype'], long_name="product")
    def prod(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the product of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.prod_by, Array.sum, Array.sum_by,
        Array.cumsum, Array.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.prod()
        0
        >>> # along axis 'a'
        >>> arr.prod('a')
        b  b0   b1    b2    b3
            0  585  1680  3465
        >>> # along axis 'b'
        >>> arr.prod('b')
        a  a0   a1    a2     a3
            0  840  7920  32760

        Select some rows only

        >>> arr.prod(['a0', 'a1'])
        b  b0  b1  b2  b3
            0   5  12  21
        >>> # or equivalently
        >>> # arr.prod('a0,a1')

        Split an axis in several parts

        >>> arr.prod((['a0', 'a1'], ['a2', 'a3']))
          a\b  b0   b1   b2   b3
        a0,a1   0    5   12   21
        a2,a3  96  117  140  165
        >>> # or equivalently
        >>> # arr.prod('a0,a1;a2,a3')

        Same with renaming

        >>> arr.prod((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b  b0   b1   b2   b3
        a01   0    5   12   21
        a23  96  117  140  165
        >>> # or equivalently
        >>> # arr.prod('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.prod, np_nanprod, commutative=True, by_agg=True, extra_kwargs=['dtype'],
                          long_name="product")
    def prod_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the product of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.prod, Array.sum, Array.sum_by,
        Array.cumsum, Array.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.prod_by()
        0
        >>> # along axis 'a'
        >>> arr.prod_by('a')
        a  a0   a1    a2     a3
            0  840  7920  32760
        >>> # along axis 'b'
        >>> arr.prod_by('b')
        b  b0   b1    b2    b3
            0  585  1680  3465

        Select some rows only

        >>> arr.prod_by(['a0', 'a1'])
        0
        >>> # or equivalently
        >>> # arr.prod_by('a0,a1')

        Split an axis in several parts

        >>> arr.prod_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1      a2,a3
               0  259459200
        >>> # or equivalently
        >>> # arr.prod_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.prod_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a  a01        a23
             0  259459200
        >>> # or equivalently
        >>> # arr.prod_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.min, np.nanmin, commutative=True, long_name="minimum", action_verb="search")
    def min(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Get minimum of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.min_by, Array.max, Array.max_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.min()
        0
        >>> # along axis 'a'
        >>> arr.min('a')
        b  b0  b1  b2  b3
            0   1   2   3
        >>> # along axis 'b'
        >>> arr.min('b')
        a  a0  a1  a2  a3
            0   4   8  12

        Select some rows only

        >>> arr.min(['a0', 'a1'])
        b  b0  b1  b2  b3
            0   1   2   3
        >>> # or equivalently
        >>> # arr.min('a0,a1')

        Split an axis in several parts

        >>> arr.min((['a0', 'a1'], ['a2', 'a3']))
          a\b  b0  b1  b2  b3
        a0,a1   0   1   2   3
        a2,a3   8   9  10  11
        >>> # or equivalently
        >>> # arr.min('a0,a1;a2,a3')

        Same with renaming

        >>> arr.min((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b  b0  b1  b2  b3
        a01   0   1   2   3
        a23   8   9  10  11
        >>> # or equivalently
        >>> # arr.min('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.min, np.nanmin, commutative=True, by_agg=True, long_name="minimum", action_verb="search")
    def min_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Get minimum of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.min, Array.max, Array.max_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.min_by()
        0
        >>> # along axis 'a'
        >>> arr.min_by('a')
        a  a0  a1  a2  a3
            0   4   8  12
        >>> # along axis 'b'
        >>> arr.min_by('b')
        b  b0  b1  b2  b3
            0   1   2   3

        Select some rows only

        >>> arr.min_by(['a0', 'a1'])
        0
        >>> # or equivalently
        >>> # arr.min_by('a0,a1')

        Split an axis in several parts

        >>> arr.min_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
               0      8
        >>> # or equivalently
        >>> # arr.min_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.min_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a  a01  a23
             0    8
        >>> # or equivalently
        >>> # arr.min_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.max, np.nanmax, commutative=True, long_name="maximum", action_verb="search")
    def max(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Get maximum of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.max_by, Array.min, Array.min_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.max()
        15
        >>> # along axis 'a'
        >>> arr.max('a')
        b  b0  b1  b2  b3
           12  13  14  15
        >>> # along axis 'b'
        >>> arr.max('b')
        a  a0  a1  a2  a3
            3   7  11  15

        Select some rows only

        >>> arr.max(['a0', 'a1'])
        b  b0  b1  b2  b3
            4   5   6   7
        >>> # or equivalently
        >>> # arr.max('a0,a1')

        Split an axis in several parts

        >>> arr.max((['a0', 'a1'], ['a2', 'a3']))
          a\b  b0  b1  b2  b3
        a0,a1   4   5   6   7
        a2,a3  12  13  14  15
        >>> # or equivalently
        >>> # arr.max('a0,a1;a2,a3')

        Same with renaming

        >>> arr.max((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b  b0  b1  b2  b3
        a01   4   5   6   7
        a23  12  13  14  15
        >>> # or equivalently
        >>> # arr.max('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.max, np.nanmax, commutative=True, by_agg=True, long_name="maximum", action_verb="search")
    def max_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Get maximum of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.max, Array.min, Array.min_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.max_by()
        15
        >>> # along axis 'a'
        >>> arr.max_by('a')
        a  a0  a1  a2  a3
            3   7  11  15
        >>> # along axis 'b'
        >>> arr.max_by('b')
        b  b0  b1  b2  b3
           12  13  14  15

        Select some rows only

        >>> arr.max_by(['a0', 'a1'])
        7
        >>> # or equivalently
        >>> # arr.max_by('a0,a1')

        Split an axis in several parts

        >>> arr.max_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
               7     15
        >>> # or equivalently
        >>> # arr.max_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.max_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a  a01  a23
             7   15
        >>> # or equivalently
        >>> # arr.max_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.mean, np.nanmean, commutative=True, extra_kwargs=['dtype'])
    def mean(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the arithmetic mean.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.mean_by, Array.median, Array.median_by,
        Array.var, Array.var_by, Array.std, Array.std_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.mean()
        7.5
        >>> # along axis 'a'
        >>> arr.mean('a')
        b   b0   b1   b2   b3
           6.0  7.0  8.0  9.0
        >>> # along axis 'b'
        >>> arr.mean('b')
        a   a0   a1   a2    a3
           1.5  5.5  9.5  13.5

        Select some rows only

        >>> arr.mean(['a0', 'a1'])
        b   b0   b1   b2   b3
           2.0  3.0  4.0  5.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1')

        Split an axis in several parts

        >>> arr.mean((['a0', 'a1'], ['a2', 'a3']))
          a\b    b0    b1    b2    b3
        a0,a1   2.0   3.0   4.0   5.0
        a2,a3  10.0  11.0  12.0  13.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1;a2,a3')

        Same with renaming

        >>> arr.mean((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b    b0    b1    b2    b3
        a01   2.0   3.0   4.0   5.0
        a23  10.0  11.0  12.0  13.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.mean, np.nanmean, commutative=True, by_agg=True, extra_kwargs=['dtype'], long_name="mean")
    def mean_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the arithmetic mean.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.mean, Array.median, Array.median_by,
        Array.var, Array.var_by, Array.std, Array.std_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.mean()
        7.5
        >>> # along axis 'a'
        >>> arr.mean_by('a')
        a   a0   a1   a2    a3
           1.5  5.5  9.5  13.5
        >>> # along axis 'b'
        >>> arr.mean_by('b')
        b   b0   b1   b2   b3
           6.0  7.0  8.0  9.0

        Select some rows only

        >>> arr.mean_by(['a0', 'a1'])
        3.5
        >>> # or equivalently
        >>> # arr.mean_by('a0,a1')

        Split an axis in several parts

        >>> arr.mean_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
             3.5   11.5
        >>> # or equivalently
        >>> # arr.mean_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.mean_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a  a01   a23
           3.5  11.5
        >>> # or equivalently
        >>> # arr.mean_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.median, np.nanmedian, commutative=True)
    def median(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the arithmetic median.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.median_by, Array.mean, Array.mean_by,
        Array.var, Array.var_by, Array.std, Array.std_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9],
        ...             [5, 8, 3, 7],
        ...             [6, 2, 0, 9],
        ...             [9, 10, 5, 6]]
        >>> arr
        a\b  b0  b1  b2  b3
         a0  10   7   5   9
         a1   5   8   3   7
         a2   6   2   0   9
         a3   9  10   5   6
        >>> arr.median()
        6.5
        >>> # along axis 'a'
        >>> arr.median('a')
        b   b0   b1   b2   b3
           7.5  7.5  4.0  8.0
        >>> # along axis 'b'
        >>> arr.median('b')
        a   a0   a1   a2   a3
           8.0  6.0  4.0  7.5

        Select some rows only

        >>> arr.median(['a0', 'a1'])
        b   b0   b1   b2   b3
           7.5  7.5  4.0  8.0
        >>> # or equivalently
        >>> # arr.median('a0,a1')

        Split an axis in several parts

        >>> arr.median((['a0', 'a1'], ['a2', 'a3']))
          a\b   b0   b1   b2   b3
        a0,a1  7.5  7.5  4.0  8.0
        a2,a3  7.5  6.0  2.5  7.5
        >>> # or equivalently
        >>> # arr.median('a0,a1;a2,a3')

        Same with renaming

        >>> arr.median((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b   b0   b1   b2   b3
        a01  7.5  7.5  4.0  8.0
        a23  7.5  6.0  2.5  7.5
        >>> # or equivalently
        >>> # arr.median('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.median, np.nanmedian, commutative=True, by_agg=True, long_name="mediane")
    def median_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the arithmetic median.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.median, Array.mean, Array.mean_by,
        Array.var, Array.var_by, Array.std, Array.std_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9],
        ...             [5, 8, 3, 7],
        ...             [6, 2, 0, 9],
        ...             [9, 10, 5, 6]]
        >>> arr
        a\b  b0  b1  b2  b3
         a0  10   7   5   9
         a1   5   8   3   7
         a2   6   2   0   9
         a3   9  10   5   6
        >>> arr.median_by()
        6.5
        >>> # along axis 'a'
        >>> arr.median_by('a')
        a   a0   a1   a2   a3
           8.0  6.0  4.0  7.5
        >>> # along axis 'b'
        >>> arr.median_by('b')
        b   b0   b1   b2   b3
           7.5  7.5  4.0  8.0

        Select some rows only

        >>> arr.median_by(['a0', 'a1'])
        7.0
        >>> # or equivalently
        >>> # arr.median_by('a0,a1')

        Split an axis in several parts

        >>> arr.median_by((['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
             7.0   5.75
        >>> # or equivalently
        >>> # arr.median_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.median_by((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a  a01   a23
           7.0  5.75
        >>> # or equivalently
        >>> # arr.median_by('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    # XXX: for performance reasons, we should use the fact that the underlying numpy function handles multiple
    #      percentiles in one call. This is easy to implement in _axis_aggregate() but not in _group_aggregate()
    #      since in this case np.percentile() may be called several times.
    # percentile needs an explicit method because it has not the same
    # signature as other aggregate functions (extra argument)
    @deprecate_kwarg('interpolation', 'method')
    def percentile(self, q, *args,
                   out=_kwarg_agg['out']['value'],
                   method=_kwarg_agg['method']['value'],
                   skipna=_kwarg_agg['skipna']['value'],
                   keepaxes=_kwarg_agg['keepaxes']['value'],
                   **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the qth percentile of the data along the specified axis.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.percentile_by, Array.mean, Array.mean_by,
        Array.median, Array.median_by, Array.var, Array.var_by,
        Array.std, Array.std_by

        {percentile_notes}

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.percentile(25)
        3.75
        >>> # along axis 'a'
        >>> arr.percentile(25, 'a')
        b   b0   b1   b2   b3
           3.0  4.0  5.0  6.0
        >>> # along axis 'b'
        >>> arr.percentile(25, 'b')
        a    a0    a1    a2     a3
           0.75  4.75  8.75  12.75
        >>> # several percentile values
        >>> arr.percentile([25, 50, 75], 'b')
        percentile\a    a0    a1     a2     a3
                  25  0.75  4.75   8.75  12.75
                  50   1.5   5.5    9.5   13.5
                  75  2.25  6.25  10.25  14.25

        Select some rows only

        >>> arr.percentile(25, ['a0', 'a1'])
        b   b0   b1   b2   b3
           1.0  2.0  3.0  4.0
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1')

        Split an axis in several parts

        >>> arr.percentile(25, (['a0', 'a1'], ['a2', 'a3']))
          a\b   b0    b1    b2    b3
        a0,a1  1.0   2.0   3.0   4.0
        a2,a3  9.0  10.0  11.0  12.0
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1;a2,a3')

        Same with renaming

        >>> arr.percentile(25, (X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b   b0    b1    b2    b3
        a01  1.0   2.0   3.0   4.0
        a23  9.0  10.0  11.0  12.0
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1>>a01;a2,a3>>a23')

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        if skipna is None:
            skipna = True
        _npfunc = np.nanpercentile if skipna else np.percentile
        def compute_percentile(q):
            extra_kwargs = {'q': q}
            if method != 'linear':
                extra_kwargs['method'] = method
            return self._aggregate(_npfunc, args, kwargs, keepaxes=keepaxes, commutative=True,
                                   extra_kwargs=extra_kwargs)
        if isinstance(q, (list, tuple)):
            res = stack({v: compute_percentile(v) for v in q}, 'percentile')
            return res.transpose()
        else:
            return compute_percentile(q)

    _doc_agg_method(percentile, False, "qth percentile", extra_args=['q'],
                    kwargs=['out', 'method', 'skipna', 'keepaxes'])

    @deprecate_kwarg('interpolation', 'method')
    def percentile_by(self, q, *args,
                      out=_kwarg_agg['out']['value'],
                      method=_kwarg_agg['method']['value'],
                      skipna=_kwarg_agg['skipna']['value'],
                      keepaxes=_kwarg_agg['keepaxes']['value'],
                      **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the qth percentile of the data for the specified axis.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.percentile, Array.mean, Array.mean_by,
        Array.median, Array.median_by, Array.var, Array.var_by,
        Array.std, Array.std_by

        {percentile_notes}

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.percentile_by(25)
        3.75
        >>> # along axis 'a'
        >>> arr.percentile_by(25, 'a')
        a    a0    a1    a2     a3
           0.75  4.75  8.75  12.75
        >>> # along axis 'b'
        >>> arr.percentile_by(25, 'b')
        b   b0   b1   b2   b3
           3.0  4.0  5.0  6.0
        >>> # several percentile values
        >>> arr.percentile_by([25, 50, 75], 'b')
        percentile\b   b0    b1    b2    b3
                  25  3.0   4.0   5.0   6.0
                  50  6.0   7.0   8.0   9.0
                  75  9.0  10.0  11.0  12.0

        Select some rows only

        >>> arr.percentile_by(25, ['a0', 'a1'])
        1.75
        >>> # or equivalently
        >>> # arr.percentile_by('a0,a1')

        Split an axis in several parts

        >>> arr.percentile_by(25, (['a0', 'a1'], ['a2', 'a3']))
        a  a0,a1  a2,a3
            1.75   9.75
        >>> # or equivalently
        >>> # arr.percentile_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.percentile_by(25, (X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a   a01   a23
           1.75  9.75
        >>> # or equivalently
        >>> # arr.percentile_by('a0,a1>>a01;a2,a3>>a23')

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        if skipna is None:
            skipna = True
        _npfunc = np.nanpercentile if skipna else np.percentile
        def compute_percentile(q):
            extra_kwargs = {'q': q}
            if method != 'linear':
                extra_kwargs['method'] = method
            return self._aggregate(_npfunc, args, kwargs, by_agg=True, keepaxes=keepaxes, commutative=True,
                                   extra_kwargs=extra_kwargs)
        if isinstance(q, (list, tuple)):
            res = stack({v: compute_percentile(v) for v in q}, 'percentile')
            return res.transpose()
        else:
            return compute_percentile(q)

    _doc_agg_method(percentile_by, True, "qth percentile", extra_args=['q'],
                    kwargs=['out', 'method', 'skipna', 'keepaxes'])

    # not commutative

    def ptp(self, *args, out=_kwarg_agg['out']['value'], **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Return the range of values (maximum - minimum).

        The name of the function comes from the acronym for `peak to peak`.

        {parameters}

        Returns
        -------
        Array or scalar

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.ptp()
        15
        >>> # along axis 'a'
        >>> arr.ptp('a')
        b  b0  b1  b2  b3
           12  12  12  12
        >>> # along axis 'b'
        >>> arr.ptp('b')
        a  a0  a1  a2  a3
            3   3   3   3

        Select some rows only

        >>> arr.ptp(['a0', 'a1'])
        b  b0  b1  b2  b3
            4   4   4   4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1')

        Split an axis in several parts

        >>> arr.ptp((['a0', 'a1'], ['a2', 'a3']))
          a\b  b0  b1  b2  b3
        a0,a1   4   4   4   4
        a2,a3   4   4   4   4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1;a2,a3')

        Same with renaming

        >>> arr.ptp((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\b  b0  b1  b2  b3
        a01   4   4   4   4
        a23   4   4   4   4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1>>a01;a2,a3>>a23')
        """
        return self._aggregate(np.ptp, args, kwargs, out=out)

    _doc_agg_method(ptp, by=False, kwargs=['out'])

    @_decorate_agg_method(np.var, np.nanvar, extra_kwargs=['dtype', 'ddof'], long_name="variance")
    def var(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the unbiased variance.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.var_by, Array.std, Array.std_by,
        Array.mean, Array.mean_by, Array.median, Array.median_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3],
        ...             [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.var()
        4.7999999999999998
        >>> # along axis 'b'
        >>> arr.var('b')
        a   a0   a1
           4.0  4.0

        Select some columns only

        >>> arr.var(['b0', 'b1', 'b3'])
        a   a0   a1
           9.0  4.0
        >>> # or equivalently
        >>> # arr.var('b0,b1,b3')

        Split an axis in several parts

        >>> arr.var((['b0', 'b1', 'b3'], 'b5:'))
        a\b  b0,b1,b3  b5:
         a0       9.0  1.0
         a1       4.0  1.0
        >>> # or equivalently
        >>> # arr.var('b0,b1,b3;b5:')

        Same with renaming

        >>> arr.var((X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\b  b013  b567
         a0   9.0   1.0
         a1   4.0   1.0
        >>> # or equivalently
        >>> # arr.var('b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    @_decorate_agg_method(np.var, np.nanvar, by_agg=True, extra_kwargs=['dtype', 'ddof'], long_name="variance")
    def var_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the unbiased variance.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.var, Array.std, Array.std_by,
        Array.mean, Array.mean_by, Array.median, Array.median_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3],
        ...             [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.var_by()
        4.7999999999999998
        >>> # along axis 'a'
        >>> arr.var_by('a')
        a   a0   a1
           4.0  4.0

        Select some columns only

        >>> arr.var_by('a', ['b0','b1','b3'])
        a   a0   a1
           9.0  4.0
        >>> # or equivalently
        >>> # arr.var_by('a','b0,b1,b3')

        Split an axis in several parts

        >>> arr.var_by('a', (['b0', 'b1', 'b3'], 'b5:'))
        a\b  b0,b1,b3  b5:
         a0       9.0  1.0
         a1       4.0  1.0
        >>> # or equivalently
        >>> # arr.var_by('a','b0,b1,b3;b5:')

        Same with renaming

        >>> arr.var_by('a', (X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\b  b013  b567
         a0   9.0   1.0
         a1   4.0   1.0
        >>> # or equivalently
        >>> # arr.var_by('a','b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    @_decorate_agg_method(np.std, np.nanstd, extra_kwargs=['dtype', 'ddof'], long_name="standard deviation")
    def std(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the sample standard deviation.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.std_by, Array.var, Array.var_by,
        Array.mean, Array.mean_by, Array.median, Array.median_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3],
        ...             [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.std()
        2.1908902300206643
        >>> # along axis 'b'
        >>> arr.std('b')
        a   a0   a1
           2.0  2.0

        Select some columns only

        >>> arr.std(['b0', 'b1', 'b3'])
        a   a0   a1
           3.0  2.0
        >>> # or equivalently
        >>> # arr.std('b0,b1,b3')

        Split an axis in several parts

        >>> arr.std((['b0', 'b1', 'b3'], 'b5:'))
        a\b  b0,b1,b3  b5:
         a0       3.0  1.0
         a1       2.0  1.0
        >>> # or equivalently
        >>> # arr.std('b0,b1,b3;b5:')

        Same with renaming

        >>> arr.std((X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\b  b013  b567
         a0   3.0   1.0
         a1   2.0   1.0
        >>> # or equivalently
        >>> # arr.std('b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    @_decorate_agg_method(np.std, np.nanstd, by_agg=True, extra_kwargs=['dtype', 'ddof'],
                          long_name="standard deviation")
    def std_by(self, *args, **kwargs) -> Union['Array', Scalar]:
        r"""{signature}

        Compute the sample standard deviation.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.std_by, Array.var, Array.var_by,
        Array.mean, Array.mean_by, Array.median, Array.median_by,
        Array.percentile, Array.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3],
        ...             [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.std_by()
        2.1908902300206643
        >>> # along axis 'a'
        >>> arr.std_by('a')
        a   a0   a1
           2.0  2.0

        Select some columns only

        >>> arr.std_by('a', ['b0','b1','b3'])
        a   a0   a1
           3.0  2.0
        >>> # or equivalently
        >>> # arr.std_by('a','b0,b1,b3')

        Split an axis in several parts

        >>> arr.std_by('a', (['b0', 'b1', 'b3'], 'b5:'))
        a\b  b0,b1,b3  b5:
         a0       3.0  1.0
         a1       2.0  1.0
        >>> # or equivalently
        >>> # arr.std_by('a','b0,b1,b3;b5:')

        Same with renaming

        >>> arr.std_by('a', (X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\b  b013  b567
         a0   3.0   1.0
         a1   2.0   1.0
        >>> # or equivalently
        >>> # arr.std_by('a','b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    # cumulative aggregates
    def cumsum(self, axis=-1) -> Union['Array', Scalar]:
        r"""
        Return the cumulative sum of array elements along an axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to perform the cumulative sum.
            If given as position, it can be a negative integer, in which case it counts from the last to the first axis.
            By default, the cumulative sum is performed along the last axis.

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.cumprod, Array.sum, Array.sum_by,
        Array.prod, Array.prod_by

        Notes
        -----
        Cumulative aggregation functions accept only one axis

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.cumsum()
        a\b  b0  b1  b2  b3
         a0   0   1   3   6
         a1   4   9  15  22
         a2   8  17  27  38
         a3  12  25  39  54
        >>> arr.cumsum('a')
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   6   8  10
         a2  12  15  18  21
         a3  24  28  32  36
        """
        return self._cum_aggregate(np.cumsum, axis)

    def cumprod(self, axis=-1) -> Union['Array', Scalar]:
        r"""
        Return the cumulative product of array elements.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to perform the cumulative product.
            If given as position, it can be a negative integer, in which case it counts from the last to the first axis.
            By default, the cumulative product is performed along the last axis.

        Returns
        -------
        Array or scalar

        See Also
        --------
        Array.cumsum, Array.sum, Array.sum_by,
        Array.prod, Array.prod_by

        Notes
        -----
        Cumulative aggregation functions accept only one axis.

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.cumprod()
        a\b  b0   b1    b2     b3
         a0   0    0     0      0
         a1   4   20   120    840
         a2   8   72   720   7920
         a3  12  156  2184  32760
        >>> arr.cumprod('a')
        a\b  b0   b1    b2    b3
         a0   0    1     2     3
         a1   0    5    12    21
         a2   0   45   120   231
         a3   0  585  1680  3465
        """
        return self._cum_aggregate(np.cumprod, axis)

    # element-wise method factory
    def _binop(opname):
        fullname = f'__{opname}__'
        super_method = getattr(np.ndarray, fullname)

        def opmethod(self, other) -> 'Array':
            if isinstance(other, ExprNode):
                other = other.evaluate(self.axes)

            # XXX: unsure what happens for non scalar Groups.
            #      we might want to be more general than this and .eval all Groups?
            #      or (and I think it's better) define __larray__ on Group
            #      so that a non scalar Group acts like an Axis in this situation.
            if isinstance(other, Group) and np.isscalar(other.key):
                other = other.eval()

            # we could pass scalars through asarray too but it is too costly performance-wise for only suppressing one
            # isscalar test and an if statement.
            # TODO: ndarray should probably be converted to larrays too because that would harmonize broadcasting rules,
            #       but it makes some tests fail for some reason.
            if isinstance(other, (list, Axis)):
                other = asarray(other)

            if isinstance(other, Array):
                # TODO: first test if it is not already broadcastable
                if self.axes == other.axes:
                    self_data = self.data
                    other_data = other.data
                    res_axes = self.axes
                else:
                    (self_data, other_data), res_axes = raw_broadcastable((self, other))
            # We need to check for None explicitly because we consider None as a valid scalar, while numpy does not.
            # i.e. we consider "arr == None" as valid code
            elif isinstance(other, np.ndarray) or np.isscalar(other) or other is None:
                self_data, other_data = self.data, other
                res_axes = self.axes
            else:
                return NotImplemented
            return Array(super_method(self_data, other_data), res_axes)
        opmethod.__name__ = fullname
        return opmethod

    __lt__ = _binop('lt')
    __le__ = _binop('le')
    __eq__ = _binop('eq')
    __ne__ = _binop('ne')
    __gt__ = _binop('gt')
    __ge__ = _binop('ge')
    __add__ = _binop('add')
    __radd__ = _binop('radd')
    __sub__ = _binop('sub')
    __rsub__ = _binop('rsub')
    __mul__ = _binop('mul')
    __rmul__ = _binop('rmul')
    # div and rdiv are not longer used on Python3+
    __truediv__ = _binop('truediv')
    __rtruediv__ = _binop('rtruediv')
    __floordiv__ = _binop('floordiv')
    __rfloordiv__ = _binop('rfloordiv')
    __mod__ = _binop('mod')
    __rmod__ = _binop('rmod')
    __divmod__ = _binop('divmod')
    __rdivmod__ = _binop('rdivmod')
    __pow__ = _binop('pow')
    __rpow__ = _binop('rpow')
    __lshift__ = _binop('lshift')
    __rlshift__ = _binop('rlshift')
    __rshift__ = _binop('rshift')
    __rrshift__ = _binop('rrshift')
    __and__ = _binop('and')
    __rand__ = _binop('rand')
    __xor__ = _binop('xor')
    __rxor__ = _binop('rxor')
    __or__ = _binop('or')
    __ror__ = _binop('ror')

    def __matmul__(self, other) -> 'Array':
        r"""
        Override operator @ for matrix multiplication.

        Notes
        -----
        Only available with Python >= 3.5

        Examples
        --------
        >>> arr1d = ndtest(3)
        >>> arr1d
        a  a0  a1  a2
            0   1   2
        >>> arr2d = ndtest((3, 3))
        >>> arr2d
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   6   7   8
        >>> arr1d @ arr1d # doctest: +SKIP
        5
        >>> arr1d @ arr2d # doctest: +SKIP
        b  b0  b1  b2
           15  18  21
        >>> arr2d @ arr1d # doctest: +SKIP
        a  a0  a1  a2
            5  14  23
        >>> arr3d = ndtest('c=c0..c2;d=d0..d2;e=e0..e2')
        >>> arr1d @ arr3d # doctest: +SKIP
        c\e  e0  e1  e2
         c0  15  18  21
         c1  42  45  48
         c2  69  72  75
        >>> arr3d @ arr1d # doctest: +SKIP
        c\d  d0  d1  d2
         c0   5  14  23
         c1  32  41  50
         c2  59  68  77
        >>> arr3d @ arr3d # doctest: +SKIP
         c  d\e    e0    e1    e2
        c0   d0    15    18    21
        c0   d1    42    54    66
        c0   d2    69    90   111
        c1   d0   366   396   426
        c1   d1   474   513   552
        c1   d2   582   630   678
        c2   d0  1203  1260  1317
        c2   d1  1392  1458  1524
        c2   d2  1581  1656  1731
        """
        current = self[:]
        axes = self.axes
        if not isinstance(other, (Array, np.ndarray)):
            raise NotImplementedError(f"matrix multiplication not implemented for {type(other)}")
        if isinstance(other, np.ndarray):
            other = Array(other)
        other_axes = other.axes

        combined_axes = axes[:-2] + other_axes[:-2]
        if self.ndim > 2 and other.ndim > 2:
            current = current.expand(combined_axes).transpose(combined_axes)
            other = other.expand(combined_axes).transpose(combined_axes)

        # XXX : What doc of Numpy matmul says:
        # The behavior depends on the arguments in the following way:
        # * If both arguments are 2-D they are multiplied like conventional matrices.
        # * If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes
        #   and broadcast accordingly.
        # * If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix
        #   multiplication the prepended 1 is removed.
        # * If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix
        #   multiplication the appended 1 is removed.
        res_data = current.data.__matmul__(other.data)

        res_axes = list(combined_axes)
        if self.ndim > 1:
            res_axes += [axes[-2]]
        if other.ndim > 1:
            res_axes += [other_axes[-1].copy()]
        if res_axes:
            return Array(res_data, res_axes)
        else:
            return res_data

    def __rmatmul__(self, other) -> 'Array':
        if isinstance(other, np.ndarray):
            other = Array(other)
        if not isinstance(other, Array):
            raise NotImplementedError(f"matrix multiplication not implemented for {type(other)}")
        return other.__matmul__(self)

    # element-wise method factory
    def _unaryop(opname):
        fullname = f'__{opname}__'
        super_method = getattr(np.ndarray, fullname)

        def opmethod(self) -> 'Array':
            return Array(super_method(self.data), self.axes)
        opmethod.__name__ = fullname
        return opmethod

    # unary ops do not need broadcasting so do not need to be overridden
    __neg__ = _unaryop('neg')
    __pos__ = _unaryop('pos')
    __abs__ = _unaryop('abs')
    __invert__ = _unaryop('invert')

    def __round__(self, n=0):
        # XXX: use the ufuncs.round instead?
        return np.round(self, decimals=n)

    def __index__(self):
        return self.data.__index__()

    def __int__(self):
        return self.data.__int__()

    def __float__(self):
        return self.data.__float__()

    @deprecate_kwarg('nan_equals', 'nans_equal')
    def equals(self, other, rtol=0, atol=0, nans_equal=False, check_axes=False) -> bool:
        r"""
        Compare this array with another array and returns True if they have the same axes and elements,
        False otherwise.

        Parameters
        ----------
        other : Array-like
            Input array. asarray() is used on a non-Array input.
        rtol : float or int, optional
            The relative tolerance parameter (see Notes). Defaults to 0.
        atol : float or int, optional
            The absolute tolerance parameter (see Notes). Defaults to 0.
        nans_equal : boolean, optional
            Whether to consider NaN values at the same positions in the two arrays as equal.
            By default, an array containing NaN values is never equal to another array, even if that other array
            also contains NaN values at the same positions. The reason is that a NaN value is different from
            *anything*, including itself. Defaults to False.
        check_axes : boolean, optional
            Whether to check that the set of axes and their order is the same on both sides. Defaults to False.
            If False, two arrays with compatible axes (and the same data) will compare equal, even if some axis is
            missing on either side or if the axes are in a different order.

        Returns
        -------
        bool
            Return True if this array is equal to other.

        See Also
        --------
        Array.eq, Array.allclose

        Notes
        -----
        For finite values, equals uses the following equation to test whether two values are equal:

            absolute(array1 - array2) <= (atol + rtol * absolute(array2))

        The above equation is not symmetric in array1 and array2, so that array1.equals(array2)
        might be different from array2.equals(array1) in some rare cases.

        Examples
        --------
        >>> arr1 = ndtest((2, 3))
        >>> arr1
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr2 = arr1.copy()
        >>> arr2.equals(arr1)
        True
        >>> arr2['b1'] += 1
        >>> arr2.equals(arr1)
        False
        >>> arr3 = arr1.set_labels('a', ['x0', 'x1'])
        >>> arr3.equals(arr1)
        False

        Test equality between two arrays within a given tolerance range.
        Return True if absolute(array1 - array2) <= (atol + rtol * absolute(array2)).

        >>> arr1 = Array([6., 8.], "a=a0,a1")
        >>> arr1
        a   a0   a1
           6.0  8.0
        >>> arr2 = Array([5.999, 8.001], "a=a0,a1")
        >>> arr2
        a     a0     a1
           5.999  8.001
        >>> arr2.equals(arr1)
        False
        >>> arr2.equals(arr1, atol=0.01)
        True
        >>> arr2.equals(arr1, rtol=0.01)
        True

        Arrays with NaN values

        >>> arr1 = ndtest((2, 3), dtype=float)
        >>> arr1['a1', 'b1'] = nan
        >>> arr1
        a\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  nan  5.0
        >>> arr2 = arr1.copy()
        >>> # By default, an array containing NaN values is never equal to another array,
        >>> # even if that other array also contains NaN values at the same positions.
        >>> # The reason is that a NaN value is different from *anything*, including itself.
        >>> arr2.equals(arr1)
        False
        >>> # set flag nans_equal to True to overwrite this behavior
        >>> arr2.equals(arr1, nans_equal=True)
        True

        Arrays with the same data but different axes

        >>> arr1 = ndtest((2, 2))
        >>> arr1
        a\b  b0  b1
         a0   0   1
         a1   2   3
        >>> arr2 = arr1.transpose()
        >>> arr2
        b\a  a0  a1
         b0   0   2
         b1   1   3
        >>> arr2.equals(arr1)
        True
        >>> arr2.equals(arr1, check_axes=True)
        False
        >>> arr2 = arr1.expand('c=c0,c1')
        >>> arr2
         a  b\c  c0  c1
        a0   b0   0   0
        a0   b1   1   1
        a1   b0   2   2
        a1   b1   3   3
        >>> arr2.equals(arr1)
        True
        >>> arr2.equals(arr1, check_axes=True)
        False
        """
        try:
            other = asarray(other)
        except Exception:
            return False
        try:
            axes_equal = self.axes == other.axes if check_axes else True
            return axes_equal and all(self.eq(other, rtol=rtol, atol=atol, nans_equal=nans_equal))
        except ValueError:
            return False

    def allclose(self, other: Any, rtol: float = 1e-05, atol: float = 1e-08, nans_equal: bool = True,
                 check_axes: bool = False) -> bool:
        """
        Compare this array with another array and returns True if they are element-wise equal within a tolerance.

        The tolerance values are positive, typically very small numbers.
        The relative difference (rtol * abs(other)) and the absolute difference atol are added together to compare
        against the absolute difference between this array and other.

        NaN values are treated as equal if they are in the same place and if `nans_equal=True`.

        Parameters
        ----------
        other : Array-like
            Input array. asarray() is used on a non-Array input.
        rtol : float or int, optional
            The relative tolerance parameter (see Notes). Defaults to 1e-05.
        atol : float or int, optional
            The absolute tolerance parameter (see Notes). Defaults to 1e-08.
        nans_equal : boolean, optional
            Whether to consider NaN values at the same positions in the two arrays as equal.
            By default, an array containing NaN values is never equal to another array, even if that other array
            also contains NaN values at the same positions. The reason is that a NaN value is different from
            *anything*, including itself. Defaults to True.
        check_axes : boolean, optional
            Whether to check that the set of axes and their order is the same on both sides. Defaults to False.
            If False, two arrays with compatible axes (and the same data) will compare equal, even if some axis is
            missing on either side or if the axes are in a different order.

        Returns
        -------
        bool
            Return True if the two arrays are equal within the given tolerance; False otherwise.

        See Also
        --------
        Array.equals

        Notes
        -----
        If the following equation is element-wise True, then `allclose` returns True.

            absolute(array1 - array2) <= (atol + rtol * absolute(array2))

        The above equation is not symmetric in array1 and array2, so that array1.allclose(array2) might be different
        from array2.allclose(array1) in some rare cases.

        Examples
        --------
        >>> arr1 = Array([1e10, 1e-7], "a=a0,a1")
        >>> arr2 = Array([1.00001e10, 1e-8], "a=a0,a1")
        >>> arr1.allclose(arr2)
        False

        >>> arr1 = Array([1e10, 1e-8], "a=a0,a1")
        >>> arr2 = Array([1.00001e10, 1e-9], "a=a0,a1")
        >>> arr1.allclose(arr2)
        True

        >>> arr1 = Array([1e10, 1e-8], "a=a0,a1")
        >>> arr2 = Array([1.0001e10, 1e-9], "a=a0,a1")
        >>> arr1.allclose(arr2)
        False

        >>> arr1 = Array([1.0, nan], "a=a0,a1")
        >>> arr2 = Array([1.0, nan], "a=a0,a1")
        >>> arr1.allclose(arr2)
        True
        >>> arr1.allclose(arr2, nans_equal=False)
        False
        """
        return self.equals(other=other, rtol=rtol, atol=atol, nans_equal=nans_equal, check_axes=check_axes)

    @deprecate_kwarg('nan_equals', 'nans_equal')
    def eq(self, other, rtol=0, atol=0, nans_equal=False) -> 'Array':
        """
        Compare this array with another array element-wise and returns an array of booleans.

        Parameters
        ----------
        other : Array-like
            Input array. asarray() is used on a non-Array input.
        rtol : float or int, optional
            The relative tolerance parameter (see Notes). Defaults to 0.
        atol : float or int, optional
            The absolute tolerance parameter (see Notes). Defaults to 0.
        nans_equal : boolean, optional
            Whether to consider Nan values at the same positions in the two arrays as equal.
            By default, an array containing NaN values is never equal to another array, even if that other array
            also contains NaN values at the same positions. The reason is that a NaN value is different from
            *anything*, including itself. Defaults to False.

        Returns
        -------
        Array
            Boolean array where each cell tells whether corresponding elements of this array and other are equal
            within a tolerance range if given. If nans_equal=True, corresponding elements with NaN values
            will be considered as equal.

        See Also
        --------
        Array.equals, Array.isclose

        Notes
        -----
        For finite values, eq uses the following equation to test whether two values are equal:

            absolute(array1 - array2) <= (atol + rtol * absolute(array2))

        The above equation is not symmetric in array1 and array2, so that array1.eq(array2)
        might be different from array2.eq(array1) in some rare cases.

        Examples
        --------
        >>> arr1 = Array([6., np.nan, 8.], "a=a0..a2")
        >>> arr1
        a   a0   a1   a2
           6.0  nan  8.0

        Default behavior (same as == operator)

        >>> arr1.eq(arr1)
        a    a0     a1    a2
           True  False  True

        Test equality between two arrays within a given tolerance range.
        Return True if absolute(array1 - array2) <= (atol + rtol * absolute(array2)).

        >>> arr2 = Array([5.999, np.nan, 8.001], "a=a0..a2")
        >>> arr2
        a     a0   a1     a2
           5.999  nan  8.001
        >>> arr1.eq(arr2, nans_equal=True)
        a     a0    a1     a2
           False  True  False
        >>> arr1.eq(arr2, atol=0.01, nans_equal=True)
        a    a0    a1    a2
           True  True  True
        >>> arr1.eq(arr2, rtol=0.01, nans_equal=True)
        a    a0    a1    a2
           True  True  True
        """
        other = asarray(other)

        if rtol == 0 and atol == 0:
            if not nans_equal:
                return self == other
            else:
                from larray.core.npufuncs import isnan

                def general_isnan(a):
                    if issubclass(a.dtype.type, np.inexact):
                        return isnan(a)
                    elif a.dtype.type is np.object_:
                        return Array(obj_isnan(a), a.axes)
                    else:
                        return False

                return (self == other) | (general_isnan(self) & general_isnan(other))
        else:
            (a1_data, a2_data), res_axes = raw_broadcastable([self, other])
            return Array(np.isclose(a1_data, a2_data, rtol=rtol, atol=atol, equal_nan=nans_equal), res_axes)

    def isin(self, test_values, assume_unique=False, invert=False) -> 'Array':
        r"""
        Compute whether each element of this array is in `test_values`. Return a boolean array of the same shape as
        this array that is True where the array element is in `test_values` and False otherwise.

        Parameters
        ----------
        test_values : array_like or set
            The values against which to test each element of this array. If `test_values` is not a 1D array, it will be
            converted to one.
        assume_unique : bool, optional
            If True, this array and `test_values` are both assumed to be unique, which can speed up the calculation.
            Defaults to False.
        invert : bool, optional
            If True, the values in the returned array are inverted, as if calculating `element not in test_values`.
            Defaults to False. ``isin(a, b, invert=True)`` is equivalent to (but faster than) ``~isin(a, b)``.

        Returns
        -------
        Array
            boolean array of the same shape as this array that is True where the array element is in `test_values`
            and False otherwise.

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr.isin([1, 5, 7])
        a\b     b0     b1     b2
         a0  False   True  False
         a1  False  False   True
        >>> arr[arr.isin([1, 5, 7])]
        a_b  a0_b1  a1_b2
                 1      5
        """
        if isinstance(test_values, set):
            test_values = list(test_values)
        return Array(np.isin(self.data, test_values, assume_unique=assume_unique, invert=invert), self.axes)

    def divnot0(self, other) -> 'Array':
        # part of the doctest is skipped because it produces a warning we do not want to have to handle within the
        # doctest and cannot properly ignore
        r"""Divide this array by other, but return 0.0 where other is 0.

        Parameters
        ----------
        other : scalar or Array
            What to divide by.

        Returns
        -------
        Array
            Array divided by other, 0.0 where other is 0

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> a = ndtest((nat, sex))
        >>> a
        nat\sex  M  F
             BE  0  1
             FO  2  3
        >>> b = ndtest(sex)
        >>> b
        sex  M  F
             0  1
        >>> a.divnot0(b)
        nat\sex    M    F
             BE  0.0  1.0
             FO  0.0  3.0

        Compare this to:

        >>> a / b                                  # doctest: +SKIP
        nat\sex    M    F
             BE  nan  1.0
             FO  inf  3.0
        """
        if np.isscalar(other):
            if other == 0:
                return zeros_like(self, dtype=float)
            else:
                return self / other
        else:
            (self_data, other_data), res_axes = raw_broadcastable((self, other))
            other_eq0 = other_data == 0
            # numpy array division gets slower the more zeros you have in other, so we change it before the division
            # happens. This is obviously slower than doing nothing if we have very few zeros but I think it's a win
            # on average given that other is likely to contain zeros when using divnot0.
            other_data = np.where(other_eq0, 1, other_data)
            res_data = self_data / other_data
            res_data[np.broadcast_to(other_eq0, res_data.shape)] = 0.0
            return Array(res_data, res_axes)

    # XXX: rename/change to "add_axes" ?
    # TODO: add a flag copy=True to force a new array.
    def expand(self, target_axes=None, out=None, readonly=False) -> 'Array':
        r"""Expand this array to target_axes.

        Target axes will be added to this array if not present.
        In most cases this function is not needed because LArray can do operations with arrays having different
        (compatible) axes.

        Parameters
        ----------
        target_axes : string, list of Axis or AxisCollection, optional
            This array can contain axes not present in `target_axes`.
            The result axes will be: [self.axes not in target_axes] + target_axes
        out : Array, optional
            Output array, must have more axes than array. Defaults to a new array.
            arr.expand(out=out) is equivalent to out[:] = arr
        readonly : bool, optional
            Whether returning a readonly view is acceptable or not (this is much faster)
            Defaults to False.

        Returns
        -------
        Array
            Original array if possible (and out is None).

        Examples
        --------
        >>> a = Axis('a=a1,a2')
        >>> b = Axis('b=b1,b2')
        >>> arr = ndtest([a, b])
        >>> arr
        a\b  b1  b2
         a1   0   1
         a2   2   3

        Adding one or several axes will append the new axes at the end

        >>> c = Axis('c=c1,c2')
        >>> arr.expand(c)
         a  b\c  c1  c2
        a1   b1   0   0
        a1   b2   1   1
        a2   b1   2   2
        a2   b2   3   3

        If you want the new axes to be inserted in a particular order, you have to give that order

        >>> arr.expand([a, c, b])
         a  c\b  b1  b2
        a1   c1   0   1
        a1   c2   0   1
        a2   c1   2   3
        a2   c2   2   3

        But it is enough to list only the added axes and the axes after them:

        >>> arr.expand([c, b])
         a  c\b  b1  b2
        a1   c1   0   1
        a1   c2   0   1
        a2   c1   2   3
        a2   c2   2   3
        """
        if not exactly_one(target_axes is not None, out is not None):
            raise ValueError("exactly one of either `target_axes` or `out` must be defined (not both)")

        if out is not None:
            out[:] = self
        else:
            # this is not strictly necessary but avoids doing this test twice if it is True
            if self.axes == target_axes:
                return self

            if not isinstance(target_axes, (tuple, list, AxisCollection)):
                target_axes = AxisCollection(target_axes)
            target_axes = (self.axes - target_axes) | target_axes

            broadcasted = self.broadcast_with(target_axes)
            # this can only happen if only the order of axes differed and/or all extra axes have length 1
            if broadcasted.axes == target_axes:
                return broadcasted

            if readonly:
                # requires numpy 1.10
                return Array(np.broadcast_to(broadcasted, target_axes.shape), target_axes)

            out = empty(target_axes, dtype=self.dtype)
            out[:] = broadcasted
        return out

    def append(self, axis, value, label=None) -> 'Array':
        r"""Add a value to this array along an axis.

        Parameters
        ----------
        axis : axis reference
            Axis along which to append `value`.
        value : scalar or Array
            Scalar or array with compatible axes.
        label : scalar, optional
            Label for the new item in axis. When `axis` is not present in `value`, this argument should be used.
            Defaults to None.

        Returns
        -------
        Array
            Array with `value` appended along `axis`.

        Examples
        --------
        >>> arr = ones('nat=BE,FO;sex=M,F')
        >>> arr["BE", "F"] = 2.0
        >>> arr
        nat\sex    M    F
             BE  1.0  2.0
             FO  1.0  1.0
        >>> sex_total = arr.sum('sex')
        >>> sex_total
        nat   BE   FO
             3.0  2.0
        >>> arr.append('sex', sex_total, label='M+F')
        nat\sex    M    F  M+F
             BE  1.0  2.0  3.0
             FO  1.0  1.0  2.0

        The value can already have the axis along which it is appended:

        >>> sex_total = arr.sum('sex', keepaxes='M+F')
        >>> sex_total
        nat\sex  M+F
             BE  3.0
             FO  2.0
        >>> arr.append('sex', sex_total)
        nat\sex    M    F  M+F
             BE  1.0  2.0  3.0
             FO  1.0  1.0  2.0

        The value can be a scalar or an array with fewer axes than the original array.
        In this case, the appended value is expanded (repeated) as necessary:

        >>> arr.append('nat', 2, 'Other')
        nat\sex    M    F
             BE  1.0  2.0
             FO  1.0  1.0
          Other  2.0  2.0

        The value can also have extra axes (axes not present in the original array),
        in which case, the original array is expanded as necessary:

        >>> other = zeros('type=type1,type2')
        >>> other
        type  type1  type2
                0.0    0.0
        >>> arr.append('nat', other, 'Other')
          nat  sex\type  type1  type2
           BE         M    1.0    1.0
           BE         F    2.0    2.0
           FO         M    1.0    1.0
           FO         F    1.0    1.0
        Other         M    0.0    0.0
        Other         F    0.0    0.0
        """
        axis = self.axes[axis]
        if isinstance(value, Array) and axis in value.axes:
             # This is just an optimization because going via the insert path
             # for this case makes this 10x slower.
             # FIXME: we should fix insert slowness instead
             return concat((self, value), axis)
        else:
            return self.insert(value, before=IGroup(len(axis), axis=axis), label=label)
    extend = renamed_to(append, 'extend')

    def prepend(self, axis, value, label=None) -> 'Array':
        r"""Add an array before this array along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis reference
            Axis along which to prepend input array (`value`)
        value : scalar or Array
            Scalar or array with compatible axes.
        label : str, optional
            Label for the new item in axis

        Returns
        -------
        Array
            Array expanded with 'value' at the start of 'axis'.

        Examples
        --------
        >>> a = ones('nat=BE,FO;sex=M,F')
        >>> a
        nat\sex    M    F
             BE  1.0  1.0
             FO  1.0  1.0
        >>> a.prepend('sex', a.sum('sex'), 'M+F')
        nat\sex  M+F    M    F
             BE  2.0  1.0  1.0
             FO  2.0  1.0  1.0
        >>> a.prepend('nat', 2, 'Other')
        nat\sex    M    F
          Other  2.0  2.0
             BE  1.0  1.0
             FO  1.0  1.0
        >>> b = zeros('type=type1,type2')
        >>> b
        type  type1  type2
                0.0    0.0
        >>> a.prepend('sex', b, 'Other')
        nat  sex\type  type1  type2
         BE     Other    0.0    0.0
         BE         M    1.0    1.0
         BE         F    1.0    1.0
         FO     Other    0.0    0.0
         FO         M    1.0    1.0
         FO         F    1.0    1.0
        """
        return self.insert(value, before=IGroup(0, axis=axis), label=label)

    def insert(self, value, before=None, after=None, pos=None, axis=None, label=None) -> 'Array':
        r"""Insert value in array along an axis.

        Parameters
        ----------
        value : scalar or Array
            Value to insert. If an Array, it must have compatible axes. If value already has the axis along which it
            is inserted, `label` should not be used.
        before : scalar or Group
            Label or group before which to insert `value`.
        after : scalar or Group
            Label or group after which to insert `value`.
        label : str, optional
            Label for the new item in axis.

        Returns
        -------
        Array
            Array with `value` inserted along `axis`. The dtype of the returned array will be the "closest" type
            which can hold both the array values and the inserted values without loss of information. For example,
            when mixing numeric and string types, the dtype will be object.

        Examples
        --------
        >>> arr1 = ndtest((2, 3))
        >>> arr1
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr1.insert(42, before='b1', label='b0.5')
        a\b  b0  b0.5  b1  b2
         a0   0    42   1   2
         a1   3    42   4   5

        The inserted value can be an array:

        >>> arr2 = ndtest(2)
        >>> arr2
        a  a0  a1
            0   1
        >>> arr1.insert(arr2, after='b0', label='b0.5')
        a\b  b0  b0.5  b1  b2
         a0   0     0   1   2
         a1   3     1   4   5

        If you want to target positions, you have to somehow specify the axis:

        >>> a, b = arr1.axes
        >>> # arr1.insert(42, before='b.i[1]', label='b0.5')
        >>> arr1.insert(42, before=b.i[1], label='b0.5')
        a\b  b0  b0.5  b1  b2
         a0   0    42   1   2
         a1   3    42   4   5

        Insert an array which already has the axis

        >>> arr3 = ndtest('a=a0,a1;b=b0.1,b0.2') + 42
        >>> arr3
        a\b  b0.1  b0.2
         a0    42    43
         a1    44    45
        >>> arr1.insert(arr3, before='b1')
        a\b  b0  b0.1  b0.2  b1  b2
         a0   0    42    43   1   2
         a1   3    44    45   4   5
        """
        # XXX: unsure we should have arr1.insert(arr3, before='b1,b2') result in (see unit tests):

        # a\b  b0  b0.1  b1  b0.2  b2
        #  a0   0    42   1    43   2
        #  a1   3    44   4    45   5

        # we might to implement the following instead:

        # a\b  b0  b0.1  b0.2  b1  b0.1  b0.2  b2
        #  a0   0    42    43   1    42    43   2
        #  a1   3    44    45   4    44    45   5

        # The later looks less useful and could be emulated easily via:
        # arr1.insert([arr3, arr3], before='b1,b2')
        # while the above is a bit harder to achieve manually:
        # arr1.insert([arr3[[b]] for b in arr3.b], before=['b1', 'b2'])
        # but the later is *probably* more intuitive (and wouldn't suffer from the inefficiency we currently have).

        # XXX: when we have several lists, we implicitly match them by position, which we should avoid for the usual
        # reason, but I am unsure what the best syntax for that would be.

        # the goal is to get this result

        # a\b  b0  b0.5  b1  b1.5  b2
        #  a0   0     8   1     9   2
        #  a1   3     8   4     9   5

        # When the inserted arrays already contain a label, this seems reasonably readable:

        # >>> arr1 = ndtest((2, 3))
        # >>> arr1
        # a\b  b0  b1  b2
        #  a0   0   1   2
        #  a1   3   4   5
        # >>> arr2 = full('b=b0.5', 8)
        # >>> arr2
        # b  b0.5
        #       8
        # >>> arr3 = full('b=b1.5', 9)
        # >>> arr3
        # b  b1.5
        #       9
        # >>> arr1.insert(before={'b1': arr2, 'b2': arr3})
        # a\b  b0  b0.5  b1  b1.5  b2
        #  a0   0     8   1     9   2
        #  a1   3     8   4     9   5

        # When the inserted arrays/values have no label, this does not really convince me and it prevents using after
        # or pos.

        # >>> arr1.insert(value={'b0.5': ('b1', 8), 'b1.5': ('b2', 9)})
        # a\b  b0  b0.5  b1  b1.5  b2
        #  a0   0     8   1     9   2
        #  a1   3     8   4     9   5

        # This works with both after and pos and we could support it along with the above syntax when no label is
        # needed. Problem: label, value is arbitrary and as such potentially hard to remember.

        # >>> arr1.insert(before={'b1': ('b0.5', 8), 'b2': ('b1.5', 9)})
        # a\b  b0  b0.5  b1  b1.5  b2
        #  a0   0     8   1     9   2
        #  a1   3     8   4     9   5

        # This is shorter but not readable enough/even more arbitrary than the previous option.

        # >>> arr1.insert([(8, 'b1', 'b0.5'), (9, 'b2', 'b1.5')])
        # a\b  b0  b0.5  b1  b1.5  b2
        #  a0   0     8   1     9   2
        #  a1   3     8   4     9   5

        # This is readable but odd and not much gained (except efficiency) compared with multiple insert calls

        # >>> arr1.insert([(8, 'before', 'b1', 'label', 'b0.5'),
        #                  (9, 'before', 'b2', 'label', 'b1.5')])
        # >>> arr1.insert(8, before='b1', label='b0.5') \
        #         .insert(9, before='b2', label='b1.5')

        # @alixdamman suggested using a list of dictionaries {'value': XX, 'before': YY, 'label': ZZ}

        # >>> arr1.insert([{'value': 8, 'before': 'b1', 'label': 'b0.5'},
        #                  {'value': 9, 'before': 'b2', 'label': 'b1.5'}])
        # >>> arr1.insert([dict(value=8, before='b1', label='b0.5'),
        #                  dict(value=9, before='b2', label='b1.5')])

        # It would be nice to somehow support easily inserting values defined using an Array

        # >>> toinsert = Array([[8, 'b1', 'b0.5'],
        # >>>                    [9, 'b2', 'b1.5']], "row=2;column=value,before,label")
        # >>> arr1.insert(toinsert)
        # >>> arr1.insert(value=toinsert['value'], before=toinsert['before'], label=toinsert['label'])
        # >>> arr1.insert(**toinsert)
        # >>> arr1.insert(**toinsert.to_dict('column'))
        if not exactly_one(before is not None, after is not None, pos is not None):
            raise ValueError("must specify exactly one of before, after or pos")

        if pos is not None or axis is not None:
            warnings.warn("The 'pos' and 'axis' keyword arguments are deprecated, please use axis.i[pos] instead",
                          FutureWarning, stacklevel=2)
            before = IGroup(pos, axis=axis)

        if before is not None:
            axis, before_pos = self.axes._translate_axis_key(before)
        else:
            axis, after_pos = self.axes._translate_axis_key(after)
            before_pos = after_pos + 1

        def length(v):
            if isinstance(v, Array) and axis in v.axes:
                return len(v.axes[axis])
            else:
                return len(v) if isinstance(v, (tuple, list, np.ndarray)) else 1

        def expand(v, length):
            return v if isinstance(v, (tuple, list, np.ndarray)) else [v] * length

        num_inserts = max(length(before_pos), length(label), length(value))
        stops = expand(before_pos, num_inserts)

        axis_in_value = isinstance(value, Array) and axis in value.axes
        if axis_in_value:
            # FIXME: when length(before_pos) == 1 and length(label) == 1, this is inefficient
            #        in the case of extend, this is awfully inefficent (needlessly splits the value)
            value_axis = value.axes[axis]
            # This odd construction is to get a subset for each individual label of the axis
            # but keep the label AND work with ambigous labels
            # values = [value[[k]] for k in value_axis]             -> does not work for ambigous labels
            # values = [value[k] for k in value_axis]               -> does not keep the label
            # values = [value[value_axis[[k]]] for k in value_axis] -> works but is "slow"
            values = [value[IGroup([i], None, value_axis)] for i in range(len(value_axis))]
        else:
            values = expand(value, num_inserts)

        values = [asarray(v) if not isinstance(v, Array) else v
                  for v in values]

        if label is not None:
            labels = expand(label, num_inserts)
            if axis_in_value:
                values = [v.set_labels(axis, [label])
                          for v, label in zip(values, labels)]
            else:
                values = [v.expand(Axis([label], axis.name), readonly=True)
                          for v, label in zip(values, labels)]
        elif not axis_in_value:
            v_axis = Axis([None], axis.name)
            values = [v.expand(v_axis, readonly=True)
                      for v in values]
        else:
            # When label is None and axis is in value.axes, we do not need to do anything
            pass

        start = 0
        chunks = []
        for stop, value in zip(stops, values):
            chunks.append(self[axis.i[start:stop]])
            chunks.append(value)
            start = stop
        if start < len(axis):
            chunks.append(self[axis.i[start:]])
        return concat(chunks, axis)

    def drop(self, labels=None) -> 'Array':
        r"""Return array without some labels or indices along an axis.

        Parameters
        ----------
        labels : scalar, list or Group
            Label(s) or group to remove. To remove indices, one must pass an IGroup.

        Returns
        -------
        Array
            Array with `labels` removed along their axis.

        Examples
        --------
        >>> arr1 = ndtest((2, 4))
        >>> arr1
        a\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
        >>> a, b = arr1.axes

        dropping a single label

        >>> arr1.drop('b1')
        a\b  b0  b2  b3
         a0   0   2   3
         a1   4   6   7

        dropping multiple labels

        >>> # arr1.drop('b1,b3')
        >>> arr1.drop(['b1', 'b3'])
        a\b  b0  b2
         a0   0   2
         a1   4   6

        dropping a slice

        >>> # arr1.drop('b1:b3')
        >>> arr1.drop(b['b1':'b3'])
        a\b  b0
         a0   0
         a1   4

        when deleting indices instead of labels, one must specify the axis explicitly (using an IGroup):

        >>> # arr1.drop('b.i[1]')
        >>> arr1.drop(b.i[1])
        a\b  b0  b2  b3
         a0   0   2   3
         a1   4   6   7

        as when deleting ambiguous labels (which are present on several axes):

        >>> a = Axis('a=label0..label2')
        >>> b = Axis('b=label0..label2')
        >>> arr2 = ndtest((a, b))
        >>> arr2
           a\b  label0  label1  label2
        label0       0       1       2
        label1       3       4       5
        label2       6       7       8
        >>> # arr2.drop('a[label1]')
        >>> arr2.drop(a['label1'])
           a\b  label0  label1  label2
        label0       0       1       2
        label2       6       7       8
        """
        axis, indices = self.axes._translate_axis_key(labels)
        axis_idx = self.axes.index(axis)
        new_axis = Axis(np.delete(axis.labels, indices), axis.name)
        new_axes = self.axes.replace(axis, new_axis)
        return Array(np.delete(self.data, indices, axis_idx), new_axes)

    def transpose(self, *args) -> 'Array':
        r"""Reorder axes.

        By default, reverse axes, otherwise permute the axes according to the list given as argument.

        Parameters
        ----------
        *args
            Accepts either a tuple of axes specs or axes specs as `*args`. Omitted axes keep their order.
            Use ... to avoid specifying intermediate axes.

        Returns
        -------
        Array
            Array with reordered axes.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        >>> arr.transpose('b', 'c', 'a')
         b  c\a  a0  a1
        b0   c0   0   4
        b0   c1   1   5
        b1   c0   2   6
        b1   c1   3   7
        >>> arr.transpose('b')
         b  a\c  c0  c1
        b0   a0   0   1
        b0   a1   4   5
        b1   a0   2   3
        b1   a1   6   7
        >>> arr.transpose(..., 'a')  # doctest: +SKIP
         b  c\a  a0  a1
        b0   c0   0   4
        b0   c1   1   5
        b1   c0   2   6
        b1   c1   3   7
        >>> arr.transpose('c', ..., 'a')  # doctest: +SKIP
         c  b\a  a0  a1
        c0   b0   0   4
        c0   b1   2   6
        c1   b0   1   5
        c1   b1   3   7
        """
        axes = self.axes
        data = self.data
        if len(args) == 0:
            return Array(data.T, axes[::-1])
        elif len(args) == 1 and isinstance(args[0], (tuple, list, AxisCollection)):
            target_axes = args[0]
        else:
            target_axes = args

        # TODO: this shouldn't be necessary in most cases (and is expensive compared to the numpy op itself)
        #       but doing it only when ... is present breaks many tests => in which other cases is it necessary???
        target_axes = axes[target_axes]
        # if ... in target_axes:
        #     target_axes = axes[target_axes]

        # TODO: implement AxisCollection.index(sequence)
        axes_indices = [axes.index(axis) for axis in target_axes]
        # this whole mumbo jumbo is required (for now) for anonymous axes
        indices_present = set(axes_indices)
        missing_indices = [i for i in range(data.ndim) if i not in indices_present]
        axes_indices += missing_indices

        return Array(data.transpose(axes_indices), axes[axes_indices])
    T = property(transpose)

    def clip(self, minval=None, maxval=None, out=None) -> 'Array':
        r"""Clip (limit) the values in an array.

        Given an interval, values outside the interval are clipped to the interval bounds.
        For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
        and values larger than 1 become 1.

        Parameters
        ----------
        minval : scalar or array-like, optional
            Minimum value. If None, clipping is not performed on lower bound.
            Defaults to None.
        maxval : scalar or array-like, optional
            Maximum value. If None, clipping is not performed on upper bound.
            Defaults to None.
        out : Array, optional
            The results will be placed in this array.

        Returns
        -------
        Array
            An array with the elements of the current array,
            but where values < `minval` are replaced with `minval`, and those > `maxval` with `maxval`.

        Notes
        -----
        * At least either `minval` or `maxval` must be defined.
        * If `minval` and/or `maxval` are array_like, broadcast will occur between self, `minval` and `maxval`.

        Examples
        --------
        >>> arr = ndtest((3, 3)) - 3
        >>> arr
        a\b  b0  b1  b2
         a0  -3  -2  -1
         a1   0   1   2
         a2   3   4   5
        >>> arr.clip(0, 2)
        a\b  b0  b1  b2
         a0   0   0   0
         a1   0   1   2
         a2   2   2   2

        Clipping on lower bound only

        >>> arr.clip(0)
        a\b  b0  b1  b2
         a0   0   0   0
         a1   0   1   2
         a2   3   4   5

        Clipping on upper bound only

        >>> arr.clip(maxval=2)
        a\b  b0  b1  b2
         a0  -3  -2  -1
         a1   0   1   2
         a2   2   2   2

        clipping using bounds which vary along an axis

        >>> lower_bound = Array([-2, 0, 2], 'b=b0..b2')
        >>> upper_bound = Array([0, 2, 4], 'b=b0..b2')
        >>> arr.clip(lower_bound, upper_bound)
        a\b  b0  b1  b2
         a0  -2   0   2
         a1   0   1   2
         a2   0   2   4
        """
        from larray.core.npufuncs import clip
        return clip(self, minval, maxval, out)

    @deprecate_kwarg('transpose', 'wide')
    def to_csv(self, filepath, sep=',', na_rep='', wide=True, value_name='value', dropna=None,
               dialect='default', **kwargs) -> None:
        r"""
        Write array to a csv file.

        Parameters
        ----------
        filepath : str or Path
            path where the csv file has to be written.
        sep : str, optional
            separator for the csv file. Defaults to `,`.
        na_rep : str, optional
            replace NA values with na_rep. Defaults to ''.
        wide : boolean, optional
            Whether writing arrays in "wide" format. If True, arrays are exported with the last axis
            represented horizontally. If False, arrays are exported in "narrow" format: one column per axis plus one
            value column. Defaults to True.
        value_name : str, optional
            Name of the column containing the values (last column) in the csv file when `wide=False` (see above).
            Defaults to 'value'.
        dialect : 'default' | 'classic', optional
            Whether to write the last axis name (using '\' ). Defaults to 'default'.
        dropna : None, 'all', 'any' or True, optional
            Drop lines if 'all' its values are NA, if 'any' value is NA or do not drop any line (default).
            True is equivalent to 'all'.

        Examples
        --------
        >>> tmp_path = getfixture('tmp_path')
        >>> fname = tmp_path / 'test.csv'
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a
        nat\sex  M  F
             BE  0  1
             FO  2  3
        >>> a.to_csv(fname)
        >>> with open(fname) as f:
        ...     print(f.read().strip())
        nat\sex,M,F
        BE,0,1
        FO,2,3
        >>> a.to_csv(fname, sep=';', wide=False)
        >>> with open(fname) as f:
        ...     print(f.read().strip())
        nat;sex;value
        BE;M;0
        BE;F;1
        FO;M;2
        FO;F;3
        >>> a.to_csv(fname, sep=';', wide=False, value_name='population')
        >>> with open(fname) as f:
        ...     print(f.read().strip())
        nat;sex;population
        BE;M;0
        BE;F;1
        FO;M;2
        FO;F;3
        >>> a.to_csv(fname, dialect='classic')
        >>> with open(fname) as f:
        ...     print(f.read().strip())
        nat,M,F
        BE,0,1
        FO,2,3
        """
        fold = dialect == 'default'
        if wide:
            frame = self.to_frame(fold, dropna)
            frame.to_csv(filepath, sep=sep, na_rep=na_rep, **kwargs)
        else:
            series = self.to_series(value_name, dropna is not None)
            series.to_csv(filepath, sep=sep, na_rep=na_rep, header=True, **kwargs)

    def to_hdf(self, filepath, key) -> None:
        r"""
        Write array to a HDF file.

        A HDF file can contain multiple arrays.
        The 'key' parameter is a unique identifier for the array.

        Parameters
        ----------
        filepath : str or Path
            Path where the hdf file has to be written.
        key : str or Group
            Key (path) of the array within the HDF file (see Notes below).

        Notes
        -----
        Objects stored in a HDF file can be grouped together in `HDF groups`.
        If an object 'my_obj' is stored in a HDF group 'my_group',
        the key associated with this object is then 'my_group/my_obj'.
        Be aware that a HDF group can have subgroups.

        Examples
        --------
        >>> a = ndtest((2, 3))

        Save an array

        >>> a.to_hdf('test.h5', 'a')          # doctest: +SKIP

        Save an array in a specific HDF group

        >>> a.to_hdf('test.h5', 'arrays/a')   # doctest: +SKIP
        """
        key = _translate_group_key_hdf(key)
        with LHDFStore(filepath) as store:
            store.put(key, self.to_frame())
            attrs = store.get_storer(key).attrs
            attrs.type = 'Array'
            attrs.writer = 'LArray'
            self.meta.to_hdf(store, key)

    def to_stata(self, filepath_or_buffer, **kwargs) -> None:
        r"""
        Write array to a Stata .dta file.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            Path to .dta file or a file handle.

        See Also
        --------
        read_stata

        Notes
        -----
        The round trip to Stata (Array.to_stata followed by read_stata) loose the name of the "column" axis.

        Examples
        --------
        >>> axes = [Axis(3, 'row'), Axis('column=country,sex')]
        >>> arr = Array([['BE', 'F'],
        ...              ['FR', 'M'],
        ...              ['FR', 'F']], axes=axes)
        >>> arr
        row*\column  country  sex
                  0       BE    F
                  1       FR    M
                  2       FR    F
        >>> arr.to_stata('test.dta')      # doctest: +SKIP
        """
        self.to_frame().to_stata(filepath_or_buffer, **kwargs)

    @deprecate_kwarg('sheet_name', 'sheet')
    def to_excel(self, filepath=None, sheet=None, position='A1', overwrite_file=False, clear_sheet=False,
                 header=True, transpose=False, wide=True, value_name='value', engine=None, *args, **kwargs) -> None:
        r"""
        Write array in the specified sheet of specified excel workbook.

        Parameters
        ----------
        filepath : str or Path or int or None, optional
            Path where the excel file has to be written. If None (default), creates a new Excel Workbook in a live Excel
            instance (Windows only). Use -1 to use the currently active Excel Workbook. Use a name without extension
            (.xlsx) to use any unsaved* workbook.
        sheet : str or Group or int or None, optional
            Sheet where the data has to be written. Defaults to None, Excel standard name if adding a sheet to an
            existing file, "Sheet1" otherwise. sheet can also refer to the position of the sheet
            (e.g. 0 for the first sheet, -1 for the last one).
        position : str or tuple of integers, optional
            Integer position (row, column) must be 1-based. Used only if engine is 'xlwings'. Defaults to 'A1'.
        overwrite_file : bool, optional
            Whether to overwrite the existing file (or just modify the specified sheet). Defaults to False.
        clear_sheet : bool, optional
            Whether to clear the existing sheet (if any) before writing. Defaults to False.
        header : bool, optional
            Whether to write a header (axes names and labels). Defaults to True.
        transpose : bool, optional
            Whether to transpose the array over last axis.
            This is equivalent to paste with option transpose in Excel. Defaults to False.
        wide : boolean, optional
            Whether writing arrays in "wide" format. If True, arrays are exported with the last axis
            represented horizontally. If False, arrays are exported in "narrow" format: one column per axis plus one
            value column. Defaults to True.
        value_name : str, optional
            Name of the column containing the values (last column) in the Excel sheet when `wide=False` (see above).
            Defaults to 'value'.
        engine : 'xlwings' | 'openpyxl' | 'xlsxwriter' | 'xlwt' | None, optional
            Engine to use to make the output. If None (default), it will use 'xlwings' by default if the module is
            installed and relies on Pandas default writer otherwise.
        *args
        **kwargs

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> # write to a new (unnamed) sheet
        >>> a.to_excel('test.xlsx')  # doctest: +SKIP
        >>> # write to top-left corner of an existing sheet
        >>> a.to_excel('test.xlsx', 'Sheet1')  # doctest: +SKIP
        >>> # add to existing sheet starting at position A15
        >>> a.to_excel('test.xlsx', 'Sheet1', 'A15')  # doctest: +SKIP
        """
        sheet = _translate_sheet_name(sheet)

        if wide:
            pd_obj = self.to_frame(fold_last_axis_name=True)
            if transpose and self.ndim >= 2:
                names = pd_obj.index.names
                pd_obj.index.names = names[:-2] + ['\\'.join(reversed(names[-1].split('\\')))]
        else:
            pd_obj = self.to_series(value_name)

        if engine is None:
            engine = 'xlwings' if xw is not None else None

        if isinstance(filepath, str):
            filepath = Path(filepath)

        if engine == 'xlwings':
            from larray.inout.xw_excel import open_excel

            close = False
            new_workbook = False
            if filepath is None:
                new_workbook = True
            elif isinstance(filepath, Path) and filepath.suffix:
                if not filepath.is_file():
                    new_workbook = True
                close = True
            if new_workbook or overwrite_file:
                new_workbook = overwrite_file = True

            wb = open_excel(filepath, overwrite_file=overwrite_file)

            if new_workbook:
                sheetobj = wb.sheets[0]
                if sheet is not None:
                    sheetobj.name = sheet
            elif sheet is not None and sheet in wb:
                sheetobj = wb.sheets[sheet]
                if clear_sheet:
                    sheetobj.clear()
            else:
                sheetobj = wb.sheets.add(sheet, after=wb.sheets[-1])

            options = dict(header=header, index=header, transpose=transpose)
            sheetobj[position].options(**options).value = pd_obj
            # TODO: implement wide via/in dump
            # sheet[position] = self.dump(header=header, wide=wide)
            if close:
                wb.save()
                wb.close()
        else:
            if sheet is None:
                sheet = 'Sheet1'
            # TODO: implement position in this case
            # startrow, startcol
            pd_obj.to_excel(filepath, *args, sheet_name=sheet, engine=engine, **kwargs)

    def to_clipboard(self, *args, **kwargs) -> None:
        r"""Send the content of the array to the clipboard.

        Using to_clipboard() makes it possible to paste the content of the array into a file (Excel, ascii file,...).

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a.to_clipboard()  # doctest: +SKIP
        """
        self.to_frame().to_clipboard(*args, **kwargs)

    # XXX: sep argument does not seem very useful
    # def to_excel(self, filename, sep='_'):
    #     # Why xlsxwriter? Because it is faster than openpyxl and xlwt
    #     # currently does not .xlsx (only .xls).
    #     # PyExcelerate seem like a decent alternative too
    #     import xlsxwriter as xl
    #
    #     workbook = xl.Workbook(filename)
    #     if self.ndim > 2:
    #         for key in product(*[axis.labels for axis in self.axes[:-2]]):
    #             sheetname = sep.join(str(k) for k in key)
    #             # sheet names must not:
    #             # * contain any of the following characters: : \ / ? * [ ]
    #             # XXX: this will NOT work for unicode strings !
    #             table = string.maketrans('[:]', '(-)')
    #             todelete = r'\/?*'
    #             sheetname = sheetname.translate(table, todelete)
    #             # * exceed 31 characters
    #             # sheetname = sheetname[:31]
    #             # * be blank
    #             assert sheetname, "sheet name cannot be blank"
    #             worksheet = workbook.add_worksheet(sheetname)
    #             worksheet.write_row(0, 1, self.axes[-1].labels)
    #             worksheet.write_column(1, 0, self.axes[-2].labels)
    #             for row, data in enumerate(np.asarray(self[key])):
    #                 worksheet.write_row(1+row, 1, data)
    #
    #     else:
    #         worksheet = workbook.add_worksheet('Sheet1')
    #         worksheet.write_row(0, 1, self.axes[-1].labels)
    #         if self.ndim == 2:
    #             worksheet.write_column(1, 0, self.axes[-2].labels)
    #         for row, data in enumerate(np.asarray(self)):
    #             worksheet.write_row(1+row, 1, data)

    @property
    def plot(self) -> PlotObject:
        r"""Plot the data of the array into a graph (window pop-up).

        Parameters
        ----------
        kind : str
            - 'line' : line plot (default)
            - 'bar' : vertical bar plot
            - 'barh' : horizontal bar plot
            - 'hist' : histogram
            - 'box' : boxplot
            - 'kde' : Kernel Density Estimation plot
            - 'density' : same as 'kde'
            - 'area' : area plot
            - 'pie' : pie plot
            - 'scatter' : scatter plot (if array's dimensions >= 2)
            - 'hexbin' : hexbin plot (if array's dimensions >= 2)
        filepath : str or Path, default None
            Save plot as a file at `filepath`. Defaults to None (do not save).
            When saving the plot to a file, the function returns None. In other
            words, in that case, the plot is no longer available for further
            tweaking or display.
        show : bool, optional
            Whether to display the plot directly.
            Defaults to True if `filepath` is None and `ax` is None, False otherwise.
        ax : matplotlib axes object, default None
        subplots : boolean, Axis, int, str or tuple, default False
            Make several subplots.
            - if an Axis (or int or str), make subplots for each label of that axis.
            - if a tuple of Axis (or int or str), make subplots for each combination of
              labels of those axes.
            - True is equivalent to all axes except the last.
            Defaults to False.
        sharex : boolean, default True if ax is None else False
            When subplots are used, share x axis and set some x axis labels to invisible;
            defaults to True if ax is None otherwise False if an ax is passed in;
            Be aware, that passing in both an ax and sharex=True will alter all x axis
            labels for all axis in a figure!
        sharey : boolean, default False
            When subplots are used, share y axis and set some y axis labels to invisible.
        layout : tuple (optional)
            (rows, columns) for the layout of subplots
        figsize : a tuple (width, height) in inches
        use_index : boolean, default True
            Use index as ticks for x axis
        title : string
            Title to use for the plot
        grid : boolean, default None (matlab style default)
            Axis grid lines
        legend : False/True/'reverse'
            Place legend on axis subplots. Defaults to True.
        style : list or dict
            matplotlib line style per column
        logx : boolean, default False
            Use log scaling on x axis
        logy : boolean, default False
            Use log scaling on y axis
        loglog : boolean, default False
            Use log scaling on both x and y axes
        xticks : sequence
            Values to use for the xticks
        yticks : sequence
            Values to use for the yticks
        xlim : 2-tuple/list, optional
            Limits (minimum and maximum values) on x axis. If this argument is not used, or None for
            either bound, these are determined automatically from the data. Defaults to (None, None).
        ylim : 2-tuple/list, optional
            Limits (minimum and maximum values) on y axis. If this argument is not used, or None for
            either bound, these are determined automatically from the data. Defaults to (None, None).
        rot : int, default None
            Rotation for ticks (xticks for vertical, yticks for horizontal plots)
        fontsize : int, default None
            Font size for xticks and yticks
        colormap : str or matplotlib colormap object, default None
            Colormap to select colors from. If string, load colormap with that name from matplotlib.
        colorbar : boolean, optional
            If True, plot colorbar (only relevant for 'scatter' and 'hexbin' plots)
        position : float, optional
            Specify relative alignments for bar plot layout. From 0 (left/bottom-end) to 1 (right/top-end).
            Defaults to 0.5 (center).
        yerr : array-like, optional
            Error bars on y axis
        xerr : array-like, optional
            Error bars on x axis
        stack : boolean, Axis, int, str or tuple, optional
            Make a stacked plot.
            - if an Axis (or int or str), stack that axis.
            - if a tuple of Axis (or int or str), stack each combination of labels of those axes.
            - True is equivalent to all axes (not already used in other arguments) except the last.
            Defaults to False in line and bar plots, and True in area plot.
        animate : Axis, int, str or tuple, optional
            Make an animated plot.
            - if an Axis (or int or str), animate that axis (create one image per label on that axis).
              One would usually use a time-related axis.
            - if a tuple of Axis (or int or str), animate each combination of labels of those axes.
            Defaults to None.
        anim_params: dict, optional
            Optional parameters to control how animations are saved to file.
            - writer : str, optional
                Backend to use. Defaults to 'pillow' for images (.gif .png and
                .tiff), 'ffmpeg' otherwise.
            - fps : int, optional
                Animation frame rate (per second). Defaults to 5.
            - metadata : dict, optional
                Dictionary of metadata to include in the output file.
                Some keys that may be of use include: title, artist, genre,
                subject, copyright, srcform, comment. Defaults to {}.
            - bitrate : int, optional
                The bitrate of the movie, in kilobits per second.  Higher values
                means higher quality movies, but increase the file size.
                A value of -1 lets the underlying movie encoder select the
                bitrate.
        **kwargs : keywords
            Options to pass to matplotlib plotting method

        Returns
        -------
        axes : matplotlib.AxesSubplot or np.array of them

        Notes
        -----
        See Pandas documentation of `plot` function for more details on this subject

        Examples
        --------
        Let us first define an array with some made up data

        >>> import larray as la
        >>> arr = la.Array([[5, 20, 5, 10],
        ...              [6, 16, 8, 11]], 'gender=M,F;year=2018..2021')

        Simple line plot

        >>> arr.plot()
        <Axes: xlabel='year'>

        Line plot with grid and a title, saved in a file

        >>> arr.plot(grid=True, title='line plot', filepath='my_file.png')

        2 bar plots (one for each gender) sharing the same y axis, which makes sub plots
        easier to compare. By default sub plots are independant of each other and the axes
        ranges are computed to "fit" just the data for their individual plot.

        >>> arr.plot.bar(subplots='gender', sharey=True)                       # doctest: +SKIP

        A stacked bar plot (genders are stacked)

        >>> arr.plot.bar(stack='gender')

        An animated bar chart (with two bars). We set explicit y bounds via ylim so that the
        same boundaries are used for the whole animation.

        >>> arr.plot.bar(animate='year', ylim=(0, 22), filepath='myanim.avi')  # doctest: +SKIP

        Create a figure containing 2 x 2 graphs

        >>> import matplotlib.pyplot as plt
        >>> # see matplotlib.pyplot.subplots documentation for more details
        >>> fig, ax = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)   # doctest: +SKIP
        >>> # line plot with 2 curves (Males and Females) in the top left corner (0, 0)
        >>> arr.plot(ax=ax[0, 0], title='line plot')                           # doctest: +SKIP
        >>> # bar plot with stacked values in the top right corner (0, 1)
        >>> arr.plot.bar(ax=ax[0, 1], stack='gender', title='stacked bar plot')  # doctest: +SKIP
        >>> # area plot in the bottom left corner (1, 0)
        >>> arr.plot.area(ax=ax[1, 0], title='area plot')                      # doctest: +SKIP
        >>> # scatter plot in the bottom right corner (1, 1), using the year as color
        >>> # index and a specific colormap
        >>> arr.plot.scatter(ax=ax[1, 1], x='M', y='F', c=arr.year, colormap='viridis',
        ...                  title='scatter plot')                             # doctest: +SKIP
        >>> plt.show()                                                         # doctest: +SKIP
        """
        return PlotObject(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        r"""Return the shape of the array as a tuple.

        Returns
        -------
        tuple
            Tuple representing the current shape.

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F;type=type1,type2,type3')
        >>> a.shape  # doctest: +SKIP
        (2, 2, 3)
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        r"""Return the number of dimensions of the array.

        Returns
        -------
        int
            Number of dimensions of an Array.

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a.ndim
        2
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        r"""Return the number of elements in array.

        Returns
        -------
        int
            Number of elements in array.

        Examples
        --------
        >>> a = ndtest('sex=M,F;type=type1,type2,type3')
        >>> a.size
        6
        """
        return self.data.size

    @property
    def nbytes(self) -> int:
        r"""Return the number of bytes used to store the array in memory.

        Returns
        -------
        int
            Number of bytes in array.

        Examples
        --------
        >>> a = ndtest('sex=M,F;type=type1,type2,type3', dtype=float)
        >>> a.nbytes
        48
        """
        return self.data.nbytes

    @property
    def memory_used(self) -> str:
        r"""Return the memory consumed by the array in human readable form.

        Returns
        -------
        str
            Memory used by the array.

        Examples
        --------
        >>> a = ndtest('sex=M,F;type=type1,type2,type3', dtype=float)
        >>> a.memory_used
        '48 bytes'
        """
        return size2str(self.data.nbytes)

    @property
    def dtype(self) -> np.dtype:
        r"""Return the type of the data of the array.

        Returns
        -------
        dtype
            Type of the data of the array.

        Examples
        --------
        >>> a = zeros('sex=M,F;type=type1,type2,type3')
        >>> a.dtype
        dtype('float64')
        """
        return self.data.dtype

    @property
    def item(self) -> Scalar:
        return self.data.item

    def __len__(self) -> int:
        return len(self.data)

    # numpy < 2 does not use the copy argument
    def __array__(self, dtype=None, copy=None):
        if copy is None:
            # numpy < 2 does not support np.array(copy=None)
            return np.asarray(self.data, dtype=dtype)
        else:
            return np.array(self.data, dtype=dtype, copy=copy)

    __array_priority__ = 100

    # TODO: this should be a thin wrapper around a method in AxisCollection
    def set_labels(self, axis=None, labels=None, inplace=False, **kwargs) -> 'Array':
        r"""Replace the labels of one or several axes of the array.

        Parameters
        ----------
        axis : string or Axis or dict
            Axis for which we want to replace labels, or mapping {axis: changes} where changes can either be the
            complete list of labels, a mapping {old_label: new_label} or a function to transform labels.
            If there is no ambiguity (two or more axes have the same labels), `axis` can be a direct mapping
            {old_label: new_label}.
        labels : int, str, iterable or mapping or function, optional
            Integer or list of values usable as the collection of labels for an Axis. If this is mapping, it must be
            {old_label: new_label}. If it is a function, it must be a function accepting a single argument (a
            label) and returning a single value. This argument must not be used if axis is a mapping.
        inplace : bool, optional
            Whether to modify the original object or return a new array and leave the original intact.
            Defaults to False.
        **kwargs :
            `axis`=`labels` for each axis you want to set labels.

        Returns
        -------
        Array
            Array with modified labels.

        Warnings
        --------
        Not passing a mapping but the complete list of new labels as the 'labels' argument must be done with caution.
        Make sure that the order of new labels corresponds to the exact same order of previous labels.

        See Also
        --------
        AxisCollection.set_labels

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a
        nat\sex  M  F
             BE  0  1
             FO  2  3
        >>> a.set_labels('sex', ['Men', 'Women'])
        nat\sex  Men  Women
             BE    0      1
             FO    2      3

        when passing a single string as labels, it will be interpreted to create the list of labels, so that one can
        use the same syntax than during axis creation.

        >>> a.set_labels('sex', 'Men,Women')
        nat\sex  Men  Women
             BE    0      1
             FO    2      3

        to replace only some labels, one must give a mapping giving the new label for each label to replace

        >>> a.set_labels('sex', {'M': 'Men'})
        nat\sex  Men  F
             BE    0  1
             FO    2  3

        to transform labels by a function, use any function accepting and returning a single argument:

        >>> a.set_labels('nat', str.lower)
        nat\sex  M  F
             be  0  1
             fo  2  3

        to replace labels for several axes at the same time, one should give a mapping giving the new labels for each
        changed axis

        >>> a.set_labels({'sex': 'Men,Women', 'nat': 'Belgian,Foreigner'})
          nat\sex  Men  Women
          Belgian    0      1
        Foreigner    2      3

        or use keyword arguments

        >>> a.set_labels(sex='Men,Women', nat='Belgian,Foreigner')
          nat\sex  Men  Women
          Belgian    0      1
        Foreigner    2      3

        one can also replace some labels in several axes by giving a mapping of mappings

        >>> a.set_labels({'sex': {'M': 'Men'}, 'nat': {'BE': 'Belgian'}})
        nat\sex  Men  F
        Belgian    0  1
             FO    2  3

        when there is no ambiguity (two or more axes have the same labels), it is possible to give a mapping
        between old and new labels

        >>> a.set_labels({'M': 'Men', 'BE': 'Belgian'})
        nat\sex  Men  F
        Belgian    0  1
             FO    2  3
        """
        axes = self.axes.set_labels(axis, labels, **kwargs)
        if inplace:
            self.axes = axes
            return self
        else:
            return Array(self.data, axes)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True) -> 'Array':
        return Array(self.data.astype(dtype, order, casting, subok, copy), self.axes)
    astype.__doc__ = np.ndarray.astype.__doc__

    def shift(self, axis, n=1) -> 'Array':
        r"""Shift the cells of the array n-times to the right along axis.

        Parameters
        ----------
        axis : int, str or Axis
            Axis for which we want to perform the shift.
        n : int, optional
            Number of cells to shift. Defaults to 1.

        Returns
        -------
        Array

        See Also
        --------
        Array.roll : cells which are pushed "outside of the axis" are reintroduced on the opposite side of the axis
                      instead of being dropped.

        Examples
        --------
        >>> arr = ndtest('sex=M,F;year=2019..2021')
        >>> arr
        sex\year  2019  2020  2021
               M     0     1     2
               F     3     4     5
        >>> arr.shift('year')
        sex\year  2020  2021
               M     0     1
               F     3     4
        >>> arr.shift('year', n=-1)
        sex\year  2019  2020
               M     1     2
               F     4     5
        """
        axis = self.axes[axis]
        if n > 0:
            return self[axis.i[:-n]].set_labels(axis, axis.labels[n:])
        elif n < 0:
            return self[axis.i[-n:]].set_labels(axis, axis.labels[:n])
        else:
            return self[:]

    def roll(self, axis=None, n=1) -> 'Array':
        r"""Roll the cells of the array n-times to the right along axis. Cells which would be pushed "outside of the
        axis" are reintroduced on the opposite side of the axis.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis along which to roll. Defaults to None (all axes).
        n : int or Array, optional
            Number of positions to roll. Defaults to 1. Use a negative integers to roll left.
            If n is an Array the number of positions rolled can vary along the axes of n.

        Returns
        -------
        Array

        See Also
        --------
        Array.shift : cells which are pushed "outside of the axis" are dropped instead of being reintroduced on the
                       opposite side of the axis.

        Examples
        --------
        >>> arr = ndtest('sex=M,F;year=2019..2021')
        >>> arr
        sex\year  2019  2020  2021
               M     0     1     2
               F     3     4     5
        >>> arr.roll('year')
        sex\year  2019  2020  2021
               M     2     0     1
               F     5     3     4

        One can also roll by a different amount depending on another axis

        >>> # let us roll by 1 for men and by 2 for women
        >>> n = sequence(arr.sex, initial=1)
        >>> n
        sex  M  F
             1  2
        >>> arr.roll('year', n)
        sex\year  2019  2020  2021
               M     2     0     1
               F     4     5     3
        """
        if isinstance(n, (int, np.integer)):
            axis_idx = None if axis is None else self.axes.index(axis)
            return Array(np.roll(self.data, n, axis=axis_idx), self.axes)
        else:
            if not isinstance(n, Array):
                raise TypeError("n should either be an integer or an Array")
            if axis is None:
                raise TypeError("axis may not be None if n is an Array")
            axis = self.axes[axis]
            seq = sequence(axis)
            return self[axis.i[(seq - n) % len(axis)]]

    # TODO: add support for groups as axis (like aggregates)
    # eg a.diff(X.year[2018:]) instead of a[2018:].diff(X.year)
    def diff(self, axis=-1, d=1, n=1, label='upper') -> 'Array':
        r"""Compute the n-th order discrete difference along a given axis.

        The first order difference is given by out[n] = a[n + 1] - a[n] along the given axis, higher order differences
        are calculated by using diff recursively.

        Parameters
        ----------
        axis : int, str, Group or Axis, optional
            Axis or group along which the difference is taken. Defaults to the last axis.
        d : int, optional
            Periods to shift for forming difference. Defaults to 1.
        n : int, optional
            The number of times values are differenced. Defaults to 1.
        label : {'lower', 'upper'}, optional
            The new labels in `axis` will have the labels of either the array being subtracted ('lower') or the array
            it is subtracted from ('upper'). Defaults to 'upper'.

        Returns
        -------
        Array
            The n-th order differences. The shape of the output is the same as `a` except for `axis` which is smaller
            by `n` * `d`.

        Examples
        --------
        >>> a = ndtest('sex=M,F;type=type1,type2,type3').cumsum('type')
        >>> a
        sex\type  type1  type2  type3
               M      0      1      3
               F      3      7     12
        >>> a.diff()
        sex\type  type2  type3
               M      1      2
               F      4      5
        >>> a.diff(n=2)
        sex\type  type3
               M      1
               F      1
        >>> a.diff('sex')
        sex\type  type1  type2  type3
               F      3      6      9
        >>> a.diff(a.type['type2':])
        sex\type  type3
               M      2
               F      5
        """
        if isinstance(axis, Group):
            array = self[axis]
            axis = array.axes[axis.axis]
        else:
            array = self
        for _ in range(n):
            axis_obj = array.axes[axis]
            left = array[axis_obj.i[d:]]
            right = array[axis_obj.i[:-d]]
            if label == 'upper':
                right = right.ignore_labels(axis)
            else:
                left = left.ignore_labels(axis)
            array = left - right
        return array

    # XXX: this is called pct_change in Pandas (but returns the same results, not results * 100, which I find silly).
    # Maybe change_rate would be better (because growth is not always positive)?
    def growth_rate(self, axis=-1, d=1, label='upper') -> 'Array':
        r"""Compute the growth along a given axis.

        Roughly equivalent to a.diff(axis, d, label) / a[axis.i[:-d]]

        Parameters
        ----------
        axis : int, str, Group or Axis, optional
            Axis or group along which the difference is taken. Defaults to the last axis.
        d : int, optional
            Periods to shift for forming difference. Defaults to 1.
        label : {'lower', 'upper'}, optional
            The new labels in `axis` will have the labels of either
            the array being subtracted ('lower') or the array it is
            subtracted from ('upper'). Defaults to 'upper'.

        Returns
        -------
        Array

        Examples
        --------
        >>> data = [[4, 5, 4, 6, 9], [2, 4, 3, 0, 0]]
        >>> a = Array(data, "sex=F,M; year=2017..2021")
        >>> a
        sex\year  2017  2018  2019  2020  2021
               F     4     5     4     6     9
               M     2     4     3     0     0
        >>> a.growth_rate()
        sex\year  2018   2019  2020  2021
               F  0.25   -0.2   0.5   0.5
               M   1.0  -0.25  -1.0   0.0
        >>> a.growth_rate(label='lower')
        sex\year  2017   2018  2019  2020
               F  0.25   -0.2   0.5   0.5
               M   1.0  -0.25  -1.0   0.0
        >>> a.growth_rate(d=2)
        sex\year  2019  2020  2021
               F   0.0   0.2  1.25
               M   0.5  -1.0  -1.0

        It works on any axis, not just time-based axes

        >>> a.growth_rate('sex')
        sex\year  2017  2018   2019  2020  2021
               M  -0.5  -0.2  -0.25  -1.0  -1.0

        Or part of axes

        >>> a.growth_rate(a.year[2017:])
        sex\year  2018   2019  2020  2021
               F  0.25   -0.2   0.5   0.5
               M   1.0  -0.25  -1.0   0.0
        """
        if isinstance(axis, Group):
            array = self[axis]
            axis = array.axes[axis.axis]
        else:
            array = self
            axis = array.axes[axis]
        diff = array.diff(axis=axis, d=d, label=label)
        # replace 0/0 by 0/inf to avoid a nan (and a warning)
        shifted_array = np.where(diff.data == 0, inf, array.shift(axis, n=d).data)
        return Array(diff.data / shifted_array, diff.axes)

    def compact(self, display=False, name='array') -> 'Array':
        r"""Detect and remove "useless" axes (ie axes for which values are constant over the whole axis).

        Parameters
        ----------
        display : bool, optional
            Whether to display a message with the name of constant axes which were discarded. Defaults to False.
        name : str, optional
            Name to use in the message if `display` is True. Defaults to "array".

        Returns
        -------
        Array or scalar
            Array with constant axes removed.

        Examples
        --------
        >>> a = Array([[1, 2],
        ...            [1, 2]], [Axis('sex=M,F'), Axis('nat=BE,FO')])
        >>> a
        sex\nat  BE  FO
              M   1   2
              F   1   2
        >>> a.compact()
        nat  BE  FO
              1   2
        """
        res = self
        compacted_axes = []
        for axis in res.axes:
            axis_first_value = res[axis.i[0]]
            if (res == axis_first_value).all():
                res = axis_first_value
                compacted_axes.append(axis.name)
        if display and compacted_axes:
            print(f"{name} was constant over: {', '.join(compacted_axes)}")
        return res

    def combine_axes(self, axes=None, sep='_', wildcard=False) -> 'Array':
        r"""Combine several axes into one.

        Parameters
        ----------
        axes : tuple, list, AxisCollection of axes or list of combination of those or dict, optional
            axes to combine. Tuple, list or AxisCollection will combine several axes into one. To chain several axes
            combinations, pass a list of tuple/list/AxisCollection of axes. To set the name(s) of resulting axis(es),
            use a {(axes, to, combine): 'new_axis_name'} dictionary. Defaults to all axes.
        sep : str, optional
            delimiter to use for combining. Defaults to '_'.
        wildcard : bool, optional
            whether to produce a wildcard axis even if the axes to combine are not. This is much faster,
            but loose axes labels.

        Returns
        -------
        Array
            Array with combined axes.

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr.combine_axes()
        a_b  a0_b0  a0_b1  a0_b2  a1_b0  a1_b1  a1_b2
                 0      1      2      3      4      5
        >>> arr.combine_axes(sep='/')
        a/b  a0/b0  a0/b1  a0/b2  a1/b0  a1/b1  a1/b2
                 0      1      2      3      4      5
        >>> arr = ndtest((2, 2, 2, 2))
        >>> arr
         a   b  c\d  d0  d1
        a0  b0   c0   0   1
        a0  b0   c1   2   3
        a0  b1   c0   4   5
        a0  b1   c1   6   7
        a1  b0   c0   8   9
        a1  b0   c1  10  11
        a1  b1   c0  12  13
        a1  b1   c1  14  15
        >>> arr.combine_axes(('a', 'c'))
          a_c  b\d  d0  d1
        a0_c0   b0   0   1
        a0_c0   b1   4   5
        a0_c1   b0   2   3
        a0_c1   b1   6   7
        a1_c0   b0   8   9
        a1_c0   b1  12  13
        a1_c1   b0  10  11
        a1_c1   b1  14  15
        >>> arr.combine_axes({('a', 'c'): 'ac'})
           ac  b\d  d0  d1
        a0_c0   b0   0   1
        a0_c0   b1   4   5
        a0_c1   b0   2   3
        a0_c1   b1   6   7
        a1_c0   b0   8   9
        a1_c0   b1  12  13
        a1_c1   b0  10  11
        a1_c1   b1  14  15

        # make several combinations at once

        >>> arr.combine_axes([('a', 'c'), ('b', 'd')])
        a_c\b_d  b0_d0  b0_d1  b1_d0  b1_d1
          a0_c0      0      1      4      5
          a0_c1      2      3      6      7
          a1_c0      8      9     12     13
          a1_c1     10     11     14     15
        >>> arr.combine_axes({('a', 'c'): 'ac', ('b', 'd'): 'bd'})
        ac\bd  b0_d0  b0_d1  b1_d0  b1_d1
        a0_c0      0      1      4      5
        a0_c1      2      3      6      7
        a1_c0      8      9     12     13
        a1_c1     10     11     14     15
        """
        if axes is None:
            axes = {tuple(self.axes): None}
        elif isinstance(axes, AxisCollection):
            axes = {tuple(axes): None}
        elif isinstance(axes, (list, tuple)):
            # checks for nested tuple/list
            if all(isinstance(axis, (list, tuple, AxisCollection)) for axis in axes):
                axes = {tuple(axes_to_combine): None for axes_to_combine in axes}
            else:
                axes = {tuple(axes): None}
        # axes should be a dict at this time
        assert isinstance(axes, dict)

        transposed_axes = self.axes[:]
        for axes_to_combine in axes.keys():
            # transpose all axes next to each other, using index of first axis
            axes_to_combine = self.axes[axes_to_combine]
            axes_indices = [transposed_axes.index(axis) for axis in axes_to_combine]
            min_axis_index = min(axes_indices)
            transposed_axes = transposed_axes - axes_to_combine
            transposed_axes = transposed_axes[:min_axis_index] + axes_to_combine + transposed_axes[min_axis_index:]
        transposed = self.transpose(transposed_axes)
        # XXX: I think this might be problematic if axes to combine are given by position instead of by name/object
        new_axes = transposed.axes.combine_axes(axes, sep=sep, wildcard=wildcard)
        return transposed.reshape(new_axes)

    def split_axes(self, axes=None, sep='_', names=None, regex=None, sort=False, fill_value=nan) -> 'Array':
        r"""Split axes and returns a new array.

        Parameters
        ----------
        axes : int, str, Axis or any combination of those
            axes to split. All labels *must* contain the given delimiter string. To split several axes at once, pass
            a list or tuple of axes to split. To set the names of resulting axes, use a {'axis_to_split': (new, axes)}
            dictionary. Defaults to all axes whose name contains the `sep` delimiter.
        sep : str, optional
            delimiter to use for splitting. Defaults to '_'.
            When `regex` is provided, the delimiter is only used on `names` if given as one string or on axis name if
            `names` is None.
        names : str or list of str, optional
            names of resulting axes. Defaults to None.
        regex : str, optional
            use regex instead of delimiter to split labels. Defaults to None.
        sort : bool, optional
            Whether to sort the combined axis before splitting it. When all combinations of labels are present in
            the combined axis, sorting is faster than not sorting. Defaults to False.
        fill_value : scalar or Array, optional
            Value to use for missing values when the combined axis does not contain all combination of labels.
            Defaults to NaN.

        Returns
        -------
        Array

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> combined = arr.combine_axes()
        >>> combined
        a_b  a0_b0  a0_b1  a0_b2  a1_b0  a1_b1  a1_b2
                 0      1      2      3      4      5
        >>> combined.split_axes()
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5

        Split labels using regex

        >>> combined = ndtest('a_b=a0b0..a1b2')
        >>> combined
        a_b  a0b0  a0b1  a0b2  a1b0  a1b1  a1b2
                0     1     2     3     4     5
        >>> combined.split_axes('a_b', regex=r'(\w{2})(\w{2})')
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5

        Split several axes at once

        >>> combined = ndtest('a_b=a0_b0..a1_b1; c_d=c0_d0..c1_d1')
        >>> combined
        a_b\c_d  c0_d0  c0_d1  c1_d0  c1_d1
          a0_b0      0      1      2      3
          a0_b1      4      5      6      7
          a1_b0      8      9     10     11
          a1_b1     12     13     14     15
        >>> # equivalent to combined.split_axes() which split all axes whose name contains the `sep` delimiter.
        >>> combined.split_axes(['a_b', 'c_d'])
         a   b  c\d  d0  d1
        a0  b0   c0   0   1
        a0  b0   c1   2   3
        a0  b1   c0   4   5
        a0  b1   c1   6   7
        a1  b0   c0   8   9
        a1  b0   c1  10  11
        a1  b1   c0  12  13
        a1  b1   c1  14  15
        >>> combined.split_axes({'a_b': ('A', 'B'), 'c_d': ('C', 'D')})
         A   B  C\D  d0  d1
        a0  b0   c0   0   1
        a0  b0   c1   2   3
        a0  b1   c0   4   5
        a0  b1   c1   6   7
        a1  b0   c0   8   9
        a1  b0   c1  10  11
        a1  b1   c0  12  13
        a1  b1   c1  14  15
        """
        array = self.sort_labels(axes) if sort else self
        # TODO: do multiple axes split in one go
        axes = array.axes._prepare_split_axes(axes, names, sep)
        for axis, names in axes.items():
            axis = array.axes[axis]
            split_axes, split_labels = axis.split(sep, names, regex, return_labels=True)

            axis_index = array.axes.index(axis)
            new_axes = array.axes[:axis_index] + split_axes + array.axes[axis_index + 1:]
            # fast path when all combinations of labels are present in the combined axis
            all_combinations_present = AxisCollection(split_axes).size == len(np.unique(axis.labels))
            if all_combinations_present and sort:
                array = array.reshape(new_axes)
            else:
                if all_combinations_present:
                    res = empty(new_axes, dtype=array.dtype)
                else:
                    res = full(new_axes, fill_value=fill_value, dtype=common_dtype((array, fill_value)))
                if axis.name is not None:
                    if names is None:
                        names = axis.name.split(sep)
                    # Rename axis to make sure we broadcast correctly. We should NOT use sep here, but rather '_'
                    # must be kept in sync with the default sep of _adv_keys_to_combined_axis_la_keys
                    new_axis_name = '_'.join(names)
                    if new_axis_name != axis.name:
                        array = array.rename(axis, new_axis_name)
                res.points[split_labels] = array
                array = res
        return array
    split_axis = renamed_to(split_axes, 'split_axis', raise_error=True)

    def reverse(self, axes=None) -> 'Array':
        r"""
        Reverse axes of an array.

        Parameters
        ----------
        axes : int, str, Axis or any combination of those
            axes to reverse. If None, all axes are reversed. Defaults to None.

        Returns
        -------
        Array
            Array with passed `axes` reversed.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7

        Reverse one axis

        >>> arr.reverse('c')
         a  b\c  c1  c0
        a0   b0   1   0
        a0   b1   3   2
        a1   b0   5   4
        a1   b1   7   6

        Reverse several axes

        >>> arr.reverse(('a', 'c'))
         a  b\c  c1  c0
        a1   b0   5   4
        a1   b1   7   6
        a0   b0   1   0
        a0   b1   3   2

        Reverse all axes

        >>> arr.reverse()
         a  b\c  c1  c0
        a1   b1   7   6
        a1   b0   5   4
        a0   b1   3   2
        a0   b0   1   0
        """
        if axes is None:
            axes = self.axes
        else:
            axes = self.axes[axes]
        if not isinstance(axes, AxisCollection):
            axes = AxisCollection(axes)
        reversed_axes = tuple(axis[::-1] for axis in axes)
        return self[reversed_axes]

    # TODO: add excluded argument (to pass to vectorize but we must also compute res_axes / broadcasted arguments
    #       accordingly and handle it when axes is not None)
    #     excluded : set, optional
    #         Set of strings or integers representing the positional or keyword arguments for which the function
    #         will not be vectorized. These will be passed directly to the `transform` function unmodified.
    def apply(self, transform, *args, by=None, axes=None, dtype=None, ascending=True,
              **kwargs) -> Union['Array', Scalar, Tuple['Array', ...]]:
        r"""
        Apply a transformation function to array elements.

        Parameters
        ----------
        transform : function
            Function to apply. This function will be called in turn with each element of the array as the first
            argument and must return an Array, scalar or tuple.
            If returning arrays the axes of those arrays must be the same for all calls to the function.
        *args
            Extra arguments to pass to the function.
        by : str, int or Axis or tuple/list/AxisCollection of the them, optional
            Axis or axes along which to iterate. The function will thus be called with arrays having all axes not
            mentioned. Defaults to None (all axes). Mutually exclusive with the `axes` argument.
        axes : str, int or Axis or tuple/list/AxisCollection of the them, optional
            Axis or axes the arrays passed to the function will have. Defaults to None (the function is given
            scalars). Mutually exclusive with the `by` argument.
        dtype : type or list of types, optional
            Output(s) data type(s). Defaults to None (inspect all output values to infer it automatically).
        ascending : bool, optional
            Whether to iterate the axes in ascending order (from start to end). Defaults to True.
        **kwargs
            Extra keyword arguments are passed to the function (as keyword arguments).

        Returns
        -------
        Array or scalar, or tuple of them
            Axes will be the union of those in axis and those of values returned by the function.

        Examples
        --------
        First let us define a test array

        >>> arr = Array([[0, 2, 1],
        ...              [3, 1, 5]], 'a=a0,a1;b=b0..b2')
        >>> arr
        a\b  b0  b1  b2
         a0   0   2   1
         a1   3   1   5

        Here is a simple function we would like to apply to each element of the array.
        Note that this particular example should rather be written as: arr ** 2
        as it is both more concise and much faster.

        >>> def square(x):
        ...     return x ** 2
        >>> arr.apply(square)
        a\b  b0  b1  b2
         a0   0   4   1
         a1   9   1  25

        Functions can also be applied along some axes:

        >>> # this is equivalent to (but much slower than): arr.sum('a')
        ... arr.apply(sum, axes='a')
        b  b0  b1  b2
            3   3   6
        >>> # this is equivalent to (but much slower than): arr.sum_by('a')
        ... arr.apply(sum, by='a')
        a  a0  a1
            3   9

        Applying the function along some axes will return an array with the
        union of those axes and the axes of the returned values. For example,
        let us define a function which returns the k highest values of an array.

        >>> def topk(a, k=2):
        ...     return a.sort_values(ascending=False).ignore_labels().i[:k]
        >>> arr.apply(topk, by='a')
        a\b*  0  1
          a0  2  1
          a1  5  3

        Other arguments can be passed to the function:

        >>> arr.apply(topk, 3, by='a')
        a\b*  0  1  2
          a0  2  1  0
          a1  5  3  1

        or by using keyword arguments:

        >>> arr.apply(topk, by='a', k=3)
        a\b*  0  1  2
          a0  2  1  0
          a1  5  3  1

        If the function returns several values (as a tuple), the result will be a tuple of arrays. For example,
        let use define a function which decompose an array in its mean and the difference to that mean :

        >>> def mean_decompose(a):
        ...     mean = a.mean()
        ...     return mean, a - mean
        >>> mean_by_a, diff_to_mean = arr.apply(mean_decompose, by='a')
        >>> mean_by_a
        a   a0   a1
           1.0  3.0
        >>> diff_to_mean
        a\b    b0    b1   b2
         a0  -1.0   1.0  0.0
         a1   0.0  -2.0  2.0
        """
        if axes is not None:
            if by is not None:
                raise ValueError("cannot specify both `by` and `axes` arguments in Array.apply")
            by = self.axes - axes

        # XXX: we could go one step further than vectorize and support a array of callables which would be broadcasted
        #      with the other arguments. I don't know whether that would actually help because I think it always
        #      possible to emulate that with a single callable with an extra argument (eg type) which dispatches to
        #      potentially different callables. It might be more practical & efficient though.
        if by is None:
            otypes = [dtype] if isinstance(dtype, type) else dtype
            vfunc = np.vectorize(transform, otypes=otypes)
            # XXX: we should probably handle excluded here
            # raw_bcast_args, raw_bcast_kwargs, res_axes = make_args_broadcastable((self,) + args, kwargs)
            raw_bcast_args, raw_bcast_kwargs, res_axes = ((self,) + args, kwargs, self.axes)
            res_data = vfunc(*raw_bcast_args, **raw_bcast_kwargs)
            if isinstance(res_data, tuple):
                return tuple(Array(res_arr, res_axes) for res_arr in res_data)
            else:
                return Array(res_data, res_axes)
        else:
            by = self.axes[by]

            values = (self,) + args + tuple(kwargs.values())
            first_kw = 1 + len(args)
            kwnames = tuple(kwargs.keys())
            key_values = [(k, transform(*a_and_kwa[:first_kw], **dict(zip(kwnames, a_and_kwa[first_kw:]))))
                          for k, a_and_kwa in zip_array_items(values, by, ascending)]
            first_key, first_value = key_values[0]
            if isinstance(first_value, tuple):
                # assume all other values are the same shape
                tuple_length = len(first_value)
                res_arrays = [stack({key: value[i] for key, value in key_values}, axes=by, dtype=dtype,
                                    res_axes=get_axes(first_value[i]).union(by))
                              for i in range(tuple_length)]
                # transpose back axis where it was
                return tuple(res_arr.transpose(self.axes & res_arr.axes) for res_arr in res_arrays)
            else:
                res_axes = get_axes(first_value).union(by)
                res_arr = stack(key_values, axes=by, dtype=dtype, res_axes=res_axes)

                # transpose back axis where it was
                return res_arr.transpose(self.axes & res_arr.axes)

    def apply_map(self, mapping, dtype=None) -> Union['Array', Scalar, Tuple['Array', ...]]:
        r"""
        Apply a transformation mapping to array elements.

        Parameters
        ----------
        mapping : mapping (dict)
            Mapping to apply to values of the array.
            A mapping (dict) must have the values to transform as keys and the new values as values, that is:
            {<oldvalue1>: <newvalue1>, <oldvalue2>: <newvalue2>, ...}.
        dtype : type, optional
            Output dtype. Defaults to None (inspect all output values to infer it automatically).

        Returns
        -------
        Array
            Axes will be the same as the original array axes.

        Notes
        -----
        To apply a transformation given as an Array (with current values as labels on one axis of
        the array and desired values as the array values), you can use: ``mapping_arr[original_arr]``.

        Examples
        --------
        First let us define a test array

        >>> arr = Array([[0, 2, 1],
        ...              [3, 1, 5]], 'a=a0,a1;b=b0..b2')
        >>> arr
        a\b  b0  b1  b2
         a0   0   2   1
         a1   3   1   5

        Now, assuming for a moment that the values of our test array above were in fact some numeric representation of
        names and we had the correspondence to the actual names stored in a dictionary:

        >>> code_to_names = {0: 'foo', 1: 'bar', 2: 'baz',
        ...                  3: 'boo', 4: 'far', 5: 'faz'}

        We could get back an array with the actual names by using:

        >>> arr.apply_map(code_to_names)
        a\b   b0   b1   b2
         a0  foo  baz  bar
         a1  boo  bar  faz
        """
        def transform(v):
            return mapping.get(v, v)
        return self.apply(transform, dtype=dtype)


class LArray(Array):
    def __init__(self, *args, **kwargs):
        warnings.warn("LArray has been renamed as Array.", FutureWarning, stacklevel=2)
        Array.__init__(self, *args, **kwargs)


def larray_equal(a1, a2):
    msg = "larray_equal() is deprecated. Use Array.equals() instead."
    warnings.warn(msg, FutureWarning, stacklevel=2)
    try:
        a1 = asarray(a1)
    except Exception:
        return False
    return a1.equals(a2)


def larray_nan_equal(a1, a2):
    msg = "larray_nan_equal() is deprecated. Use Array.equals() instead."
    warnings.warn(msg, FutureWarning, stacklevel=2)
    try:
        a1 = asarray(a1)
    except Exception:
        return False
    return a1.equals(a2, nans_equal=True)


def asarray(a, meta=None) -> Array:
    r"""
    Convert input as Array if possible.

    Parameters
    ----------
    a : array-like
        Input array to convert into an Array.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> # NumPy array
    >>> np_arr = np.arange(6).reshape((2,3))
    >>> asarray(np_arr)
    {0}*\{1}*  0  1  2
            0  0  1  2
            1  3  4  5
    >>> # Pandas dataframe
    >>> data = {'normal'  : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    ...         'reverse' : pd.Series([3., 2., 1.], index=['a', 'b', 'c'])}
    >>> df = pd.DataFrame(data)
    >>> asarray(df)
    {0}\{1}  normal  reverse
          a     1.0      3.0
          b     2.0      2.0
          c     3.0      1.0
    """
    if isinstance(a, Array):
        if meta is not None:
            res = a.copy()
            res.meta = meta
            return res
        else:
            return a
    elif hasattr(a, '__larray__'):
        res = a.__larray__()
        if meta is not None:
            res.meta = meta
        return res
    elif isinstance(a, pd.DataFrame):
        from larray.inout.pandas import from_frame
        return from_frame(a, meta=meta)
    else:
        return Array(a, meta=meta)


aslarray = renamed_to(asarray, 'aslarray', raise_error=True)


def _check_axes_argument(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Array:
        if len(args) > 1 and isinstance(args[1], (int, Axis)):
            raise ValueError(f"If you want to pass several axes or dimension lengths to {func.__name__}, you must pass "
                             f"them as a list (using []) or tuple (using()).")
        return func(*args, **kwargs)
    return wrapper


@_check_axes_argument
def zeros(axes, title=None, dtype=float, order='C', meta=None) -> Array:
    r"""Return an array with the specified axes and filled with zeros.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`. Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> zeros('nat=BE,FO;sex=M,F')
    nat\sex    M    F
         BE  0.0  0.0
         FO  0.0  0.0
    >>> zeros([(['BE', 'FO'], 'nat'),
    ...        (['M', 'F'], 'sex')])
    nat\sex    M    F
         BE  0.0  0.0
         FO  0.0  0.0
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> zeros([nat, sex])
    nat\sex    M    F
         BE  0.0  0.0
         FO  0.0  0.0
    """
    # FIXME: the error message is wrong (stackdepth is wrong) because of _check_axes_argument
    meta = _handle_meta(meta, title)
    axes = AxisCollection(axes)
    return Array(np.zeros(axes.shape, dtype, order), axes, meta=meta)


def zeros_like(array, title=None, dtype=None, order='K', meta=None) -> Array:
    r"""Return an array with the same axes as array and filled with zeros.

    Parameters
    ----------
    array : Array
         Input array.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> a = ndtest((2, 3))
    >>> zeros_like(a)
    a\b  b0  b1  b2
     a0   0   0   0
     a1   0   0   0
    """
    meta = _handle_meta(meta, title)
    return Array(np.zeros_like(array, dtype, order), array.axes, meta=meta)


@_check_axes_argument
def ones(axes, title=None, dtype=float, order='C', meta=None) -> Array:
    r"""Return an array with the specified axes and filled with ones.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.  Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> ones([nat, sex])
    nat\sex    M    F
         BE  1.0  1.0
         FO  1.0  1.0
    """
    meta = _handle_meta(meta, title)
    axes = AxisCollection(axes)
    return Array(np.ones(axes.shape, dtype, order), axes, meta=meta)


def ones_like(array, title=None, dtype=None, order='K', meta=None) -> Array:
    r"""Return an array with the same axes as array and filled with ones.

    Parameters
    ----------
    array : Array
        Input array.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> a = ndtest((2, 3))
    >>> ones_like(a)
    a\b  b0  b1  b2
     a0   1   1   1
     a1   1   1   1
    """
    meta = _handle_meta(meta, title)
    axes = array.axes
    return Array(np.ones_like(array, dtype, order), axes, meta=meta)


@_check_axes_argument
def empty(axes, title=None, dtype=float, order='C', meta=None) -> Array:
    r"""Return an array with the specified axes and uninitialized (arbitrary) data.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.  Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> empty([nat, sex])  # doctest: +SKIP
    nat\sex                   M                   F
         BE  2.47311483356e-315  2.47498446195e-315
         FO                 0.0  6.07684618082e-31
    """
    meta = _handle_meta(meta, title)
    axes = AxisCollection(axes)
    return Array(np.empty(axes.shape, dtype, order), axes, meta=meta)


def empty_like(array, title=None, dtype=None, order='K', meta=None) -> Array:
    r"""Return an array with the same axes as array and uninitialized (arbitrary) data.

    Parameters
    ----------
    array : Array
        Input array.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the data type of array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> a = ndtest((3, 2))
    >>> empty_like(a)   # doctest: +SKIP
    a\b                  b0                  b1
     a0  2.12199579097e-314  6.36598737388e-314
     a1  1.06099789568e-313  1.48539705397e-313
     a2  1.90979621226e-313  2.33419537056e-313
    """
    meta = _handle_meta(meta, title)
    # cannot use empty() because order == 'K' is not understood
    return Array(np.empty_like(array.data, dtype, order), array.axes, meta=meta)


# We cannot use @_check_axes_argument here because an integer fill_value would be considered as an error
def full(axes, fill_value, title=None, dtype=None, order='C', meta=None) -> Array:
    r"""Return an array with the specified axes and filled with fill_value.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    fill_value : scalar or Array
        Value to fill the array
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Desired data-type for the array. Default is the data type of fill_value.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> full([nat, sex], 42.0)
    nat\sex     M     F
         BE  42.0  42.0
         FO  42.0  42.0
    >>> initial_value = ndtest([sex])
    >>> initial_value
    sex  M  F
         0  1
    >>> full([nat, sex], initial_value)
    nat\sex  M  F
         BE  0  1
         FO  0  1
    """
    meta = _handle_meta(meta, title)
    if isinstance(fill_value, Axis):
        raise ValueError("If you want to pass several axes or dimension lengths to full, you must pass them as a "
                         "list (using []) or tuple (using()).")
    if dtype is None:
        dtype = np.asarray(fill_value).dtype
    res = empty(axes, dtype=dtype, order=order, meta=meta)
    res[:] = fill_value
    return res


def full_like(array, fill_value, title=None, dtype=None, order='K', meta=None) -> Array:
    r"""Return an array with the same axes and type as input array and filled with fill_value.

    Parameters
    ----------
    array : Array
        Input array.
    fill_value : scalar or Array
        Value to fill the array
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the data type of array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> a = ndtest((2, 3))
    >>> full_like(a, 5)
    a\b  b0  b1  b2
     a0   5   5   5
     a1   5   5   5
    """
    meta = _handle_meta(meta, title)
    # cannot use full() because order == 'K' is not understood
    # cannot use np.full_like() because it would not handle Array fill_value
    res = empty_like(array, dtype=dtype, meta=meta)
    res[:] = fill_value
    return res


_integer_types = (int, np.integer)


# XXX: would it be possible to generalize to multiple axes?
def sequence(axis, initial=0, inc=None, mult=None, func=None, axes=None, title=None, meta=None) -> Array:
    r"""
    Create an array by sequentially applying modifications to the array along axis.

    The value for each label in axis will be given by sequentially transforming the value for the previous label.
    This transformation on the previous label value consists of applying the function "func" on that value if provided,
    or to multiply it by mult and increment it by inc otherwise.

    Parameters
    ----------
    axis : axis definition (Axis, str, int)
        Axis along which to apply mod. An axis definition can be passed as a string. An int will be interpreted as the
        length for a new anonymous axis.
    initial : scalar or Array, optional
        Value for the first label of axis. Defaults to 0.
    inc : scalar, Array, optional
        Value to increment the previous value by. Defaults to 1 unless mult is provided (in which case it defaults
        to 0).
    mult : scalar, Array, optional
        Value to multiply the previous value by. Defaults to None.
    func : function/callable, optional
        Function to apply to the previous value. Defaults to None.
        Note that this is much slower than using inc and/or mult.
    axes : int, tuple of int or tuple/list/AxisCollection of Axis, optional
        Axes of the result. Defaults to the union of axes present in other arguments.
    title : str, optional
        Deprecated. See 'meta' below.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Examples
    --------
    >>> year = Axis('year=2016..2019')
    >>> sex = Axis('sex=M,F')
    >>> sequence(year)
    year  2016  2017  2018  2019
             0     1     2     3
    >>> sequence('year=2016..2019')
    year  2016  2017  2018  2019
             0     1     2     3
    >>> sequence(year, 1.0, 0.5)
    year  2016  2017  2018  2019
           1.0   1.5   2.0   2.5
    >>> sequence(year, 1.0, mult=1.5)
    year  2016  2017  2018   2019
           1.0   1.5  2.25  3.375
    >>> inc = Array([1, 2], [sex])
    >>> inc
    sex  M  F
         1  2
    >>> sequence(year, 1.0, inc)
    sex\year  2016  2017  2018  2019
           M   1.0   2.0   3.0   4.0
           F   1.0   3.0   5.0   7.0
    >>> mult = Array([2, 3], [sex])
    >>> mult
    sex  M  F
         2  3
    >>> sequence(year, 1.0, mult=mult)
    sex\year  2016  2017  2018  2019
           M   1.0   2.0   4.0   8.0
           F   1.0   3.0   9.0  27.0
    >>> initial = Array([3, 4], [sex])
    >>> initial
    sex  M  F
         3  4
    >>> sequence(year, initial, 1)
    sex\year  2016  2017  2018  2019
           M     3     4     5     6
           F     4     5     6     7
    >>> sequence(year, initial, mult=2)
    sex\year  2016  2017  2018  2019
           M     3     6    12    24
           F     4     8    16    32
    >>> sequence(year, initial, inc, mult)
    sex\year  2016  2017  2018  2019
           M     3     7    15    31
           F     4    14    44   134
    >>> def modify(prev_value):
    ...     return prev_value / 2
    >>> sequence(year, 8, func=modify)
    year  2016  2017  2018  2019
             8     4     2     1
    >>> sequence(3)
    {0}*  0  1  2
          0  1  2
    >>> sequence('year', axes=(sex, year))
    sex\year  2016  2017  2018  2019
           M     0     1     2     3
           F     0     1     2     3

    sequence can be used as the inverse of growth_rate:

    >>> a = Array([1.0, 2.0, 3.0, 3.0], year)
    >>> a
    year  2016  2017  2018  2019
           1.0   2.0   3.0   3.0
    >>> g = a.growth_rate() + 1
    >>> g
    year  2017  2018  2019
           2.0   1.5   1.0
    >>> sequence(year, a[2016], mult=g)
    year  2016  2017  2018  2019
           1.0   2.0   3.0   3.0
    """
    meta = _handle_meta(meta, title)

    if inc is None:
        inc = 1 if mult is None else 0
    if mult is None:
        mult = 1

    # make sure we have an axis object
    if axes is None:
        axis = _make_axis(axis)

    no_mult = isinstance(mult, _integer_types) and mult == 1

    # fast path for the most common case (integer inc and initial value, no mult, no func, no axes)
    if (isinstance(inc, _integer_types)
            and isinstance(initial, _integer_types)
            and no_mult
            and func is None
            and axes is None):
        # stop is not included
        stop = initial + inc * len(axis)
        data = np.arange(initial, stop, inc)
        return Array(data, axis, meta=meta)

    def strip_axes(col):
        return get_axes(col) - axis

    def has_axis(a, axis):
        return isinstance(a, Array) and axis in a.axes

    def array_or_full(a, axis, initial):
        dt = common_dtype((a, initial))
        r = empty(strip_axes(initial) | strip_axes(a) | axis, dtype=dt)
        r[axis.i[0]] = initial
        if isinstance(a, Array) and axis in a.axes:
            # not using axis.i[1:] because a could have less ticks
            # on axis than axis
            r[axis.i[1:]] = a[axis[axis.labels[1]:]]
        else:
            r[axis.i[1:]] = a
        return r

    if axes is None:
        # we need to remove axis if present, because it might be incompatible
        axes = strip_axes(initial) | strip_axes(inc) | strip_axes(mult) | axis
    else:
        axes = AxisCollection(axes)
        if axis not in axes:
            axis = _make_axis(axis)
        axis = axes[axis]

    res_dtype = common_dtype((initial, inc, mult))
    res = empty(axes, dtype=res_dtype, meta=meta)

    if func is not None:
        res[axis.i[0]] = prev_value = initial
        for i in range(1, len(axis)):
            res[axis.i[i]] = prev_value = func(prev_value)
    # inc only (integer) == fastpath but with axes not None
    elif res_dtype.kind == 'i' and np.isscalar(inc) and np.isscalar(initial) and np.isscalar(mult) and mult == 1:
        res[:] = sequence(axis, initial, inc)
    # inc only (non integer scalar)
    elif np.isscalar(inc) and np.isscalar(initial) and np.isscalar(mult) and mult == 1:
        # -1 because stop is included in linspace
        stop = initial + inc * (len(axis) - 1)
        data = np.linspace(initial, stop=stop, num=len(axis))
        res[:] = Array(data, axis)
    # inc only (array)
    elif np.isscalar(mult) and mult == 1:
        inc_array = array_or_full(inc, axis, initial)
        # TODO: when axis is None, this is inefficient (inc_array.cumsum() is the result)
        res[axis.i[0]] = initial
        res[axis.i[1:]] = inc_array.cumsum(axis)[axis.i[1:]]
    # mult only (scalar or array)
    elif np.isscalar(inc) and inc == 0:
        mult_array = array_or_full(mult, axis, initial)
        res[axis.i[0]] = initial
        # TODO: when axis is None, this is inefficient (mult_array.cumprod() is the result)
        res[axis.i[1:]] = mult_array.cumprod(axis)[axis.i[1:]]
    # both inc and mult defined but constant (scalars or axis not present)
    elif not has_axis(inc, axis) and not has_axis(mult, axis):
        # FIXME: the assert is broken (not has_axis is not what we want)
        assert ((np.isscalar(inc) and inc != 0) or not has_axis(inc, axis)) and \
               (np.isscalar(mult) or not has_axis(mult, axis))
        mult_array = array_or_full(mult, axis, 1.0)
        cum_mult = mult_array.cumprod(axis)[axis.i[1:]]
        res[axis.i[0]] = initial

        # a[0] = initial
        # a[1] = initial * mult ** 1                   + inc * mult ** 0
        # a[2] = initial * mult ** 2 + inc * mult ** 1 + inc * mult ** 0
        # ...
        # each term includes the sum of a geometric series:
        # series_sum = inc + inc * mult ** 1 + ... + inc * mult ** (i-1)
        # which can be computed using:
        # series_sum = inc * ((1 - mult ** i) / (1 - mult))
        # but if mult is 1, a different formula is necessary:
        # series_sum = i * inc

        # a[i] = initial * cum_mult[i] + inc * cum_mult[i - 1]

        # the case "mult == 1" was already handled above but we still need to handle the case where mult is
        # an array and *one cell* == 1
        res_where_not_1 = ((1 - cum_mult) / (1 - mult)) * inc + initial * cum_mult
        if isinstance(mult, Array) and any(mult == 1):
            from larray.core.ufuncs import where

            res_where_1 = Array(np.linspace(initial, initial + inc * (len(axis) - 1), len(axis)), axis)
            res[axis.i[1:]] = where(mult == 1, res_where_1, res_where_not_1)
        else:
            res[axis.i[1:]] = res_where_not_1
    else:
        assert has_axis(inc, axis) or has_axis(mult, axis)
        # This case is more complicated to vectorize. It seems
        # doable (probably by adding a fictive axis), but let us wait until
        # someone requests it. The trick is to be able to write this:
        # a[i] =  initial * prod(mult[j])
        #                  j=1..i
        #      +   inc[1] * prod(mult[j])
        #                  j=2..i
        #      + ...
        #      + inc[i-2] * prod(mult[j])
        #                  j=i-1..i
        #      + inc[i-1] * mult[i]
        #      + inc[i]

        # a[0] = initial
        # a[1] = initial * mult[1]
        #      +  inc[1]
        # a[2] = initial * mult[1] * mult[2]
        #      +  inc[1]           * mult[2]
        #      +  inc[2]
        # ...
        # a[4] = initial * mult[1] * mult[2] * mult[3] * mult[4]
        #      +  inc[1]           * mult[2] * mult[3] * mult[4]
        #      +  inc[2]                     * mult[3] * mult[4]
        #      +  inc[3]                               * mult[4]
        #      +  inc[4]

        # a[1:] = initial * cumprod(mult[1:]) + ...
        def index_if_exists(a, axis, i):
            if isinstance(a, Array) and axis in a.axes:
                a_axis = a.axes[axis]
                return a[a_axis[axis.labels[i]]]
            else:
                return a
        # CHECK: try something like:
        # def index_if_exists(a, igroup):
        #     axis = igroup.axis
        #     if isinstance(a, Array) and axis in a.axes:
        #         a_axis = a.axes[axis]
        #         return a[a_axis[axis.labels[i]]]
        #     else:
        #         return a
        # for i in axis.i[1:]:
        #     i_mult = index_if_exists(mult, i)
        #     i_inc = index_if_exists(inc, i)
        #     res[i] = res[i - 1] * i_mult + i_inc
        res[axis.i[0]] = prev_value = initial
        for i in range(1, len(axis)):
            i_mult = index_if_exists(mult, axis, i)
            i_inc = index_if_exists(inc, axis, i)
            res[axis.i[i]] = prev_value = prev_value * i_mult + i_inc
    return res


create_sequential = renamed_to(sequence, 'create_sequential', raise_error=True)


@_check_axes_argument
def ndrange(axes, start=0, title=None, dtype=int):
    warnings.warn("ndrange() is deprecated. Use sequence() or ndtest() instead.", FutureWarning, stacklevel=2)
    return ndtest(axes, start=start, title=title, dtype=dtype)


@_check_axes_argument
def ndtest(shape_or_axes, start=0, label_start=0, title=None, dtype=int, meta=None) -> Array:
    r"""Return test array with given shape.

    Axes are named by single letters starting from 'a'.
    Axes labels are constructed using a '{axis_name}{label_pos}' pattern (e.g. 'a0').
    Values start from `start` increase by steps of 1.

    Parameters
    ----------
    shape_or_axes : int, tuple/list of int, str, single axis or tuple/list/AxisCollection of axes
        If int or tuple/list of int, represents the shape of the array to create.
        In that case, default axes are generated.
        If string, it is used to generate axes (see :py:class:`AxisCollection` constructor).
    start : int or float, optional
        Start value
    label_start : int, optional
        Label index for each axis is `label_start + position`. `label_start` defaults to 0.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : type or np.dtype, optional
        Type of resulting array.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    Create test array by passing a shape

    >>> ndtest(6)
    a  a0  a1  a2  a3  a4  a5
        0   1   2   3   4   5
    >>> ndtest((2, 3))
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> ndtest((2, 3), label_start=1)
    a\b  b1  b2  b3
     a1   0   1   2
     a2   3   4   5
    >>> ndtest((2, 3), start=2)
    a\b  b0  b1  b2
     a0   2   3   4
     a1   5   6   7
    >>> ndtest((2, 3), dtype=float)
    a\b   b0   b1   b2
     a0  0.0  1.0  2.0
     a1  3.0  4.0  5.0

    Create test array by passing axes

    >>> ndtest("nat=BE,FO;sex=M,F")
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> nat = Axis("nat=BE,FO")
    >>> sex = Axis("sex=M,F")
    >>> ndtest([nat, sex])
    nat\sex  M  F
         BE  0  1
         FO  2  3
    """
    meta = _handle_meta(meta, title)
    # XXX: try to come up with a syntax where start is before "end".
    # For ndim > 1, I cannot think of anything nice.
    if isinstance(shape_or_axes, int):
        shape_or_axes = (shape_or_axes,)
    if isinstance(shape_or_axes, (list, tuple)) and all([isinstance(i, int) for i in shape_or_axes]):
        # TODO: move this to a class method on AxisCollection
        assert len(shape_or_axes) <= 26
        axes_names = [chr(ord('a') + i) for i in range(len(shape_or_axes))]
        label_ranges = [range(label_start, label_start + length) for length in shape_or_axes]
        shape_or_axes = [Axis([f'{name}{i}' for i in label_range], name)
                         for name, label_range in zip(axes_names, label_ranges)]
    if isinstance(shape_or_axes, AxisCollection):
        axes = shape_or_axes
    else:
        axes = AxisCollection(shape_or_axes)
    data = np.arange(start, start + axes.size, dtype=dtype).reshape(axes.shape)
    return Array(data, axes, meta=meta)


def kth_diag_indices(shape, k):
    indices = np.diag_indices(min(shape), ndim=len(shape))
    if len(shape) == 2 and k != 0:
        rows, cols = indices
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
    elif k != 0:
        raise NotImplementedError("k != 0 and len(axes) != 2")
    else:
        return indices


def diag(a, k=0, axes=(0, 1), ndim=2, split=True) -> Array:
    r"""
    Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    a : Array
        If `a` has 2 dimensions or more, return a copy of its `k`-th diagonal.
        If `a` has 1 dimension, return an array with `ndim` dimensions on the `k`-th diagonal.
    k : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or negative.  Defaults to main diagonal (0).
    axes : tuple or list or AxisCollection of axes references, optional
        Axes along which the diagonals should be taken.  Use None for all axes. Defaults to the first two axes (0, 1).
    ndim : int, optional
        Target number of dimensions when constructing a diagonal array from an array without axes names/labels.
        Defaults to 2.
    split : bool, optional
        Whether to try to split the axis name and labels. Defaults to True.

    Returns
    -------
    Array
        The extracted diagonal or constructed diagonal array.

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> a = ndtest([nat, sex], start=1)
    >>> a
    nat\sex  M  F
         BE  1  2
         FO  3  4
    >>> d = diag(a)
    >>> d
    nat_sex  BE_M  FO_F
                1     4
    >>> diag(d)
    nat\sex  M  F
         BE  1  0
         FO  0  4
    >>> a = ndtest(sex, start=1)
    >>> a
    sex  M  F
         1  2
    >>> diag(a)
    sex\sex  M  F
          M  1  0
          F  0  2
    """
    if a.ndim == 1:
        axis = a.axes[0]
        axis_name = axis.name
        if k != 0:
            raise NotImplementedError("k != 0 not supported for 1D arrays")
        if split and isinstance(axis_name, str) and '_' in axis_name:
            axes_names = axis_name.split('_')
            axes_labels = list(zip(*np.char.split(axis.labels, '_')))
            axes = [Axis(labels, name) for labels, name in zip(axes_labels, axes_names)]
        else:
            # avoid checking the axis name and labels (it expects a value lik sex_sex=M_M,F_F instead of sex=M,F)
            # TODO: in theory, this should work, but something breaks (probably those damn axes matching rules)
            # a = a.rename(0, None).ignore_labels()
            a = a.data
            axes = [axis] + [axis.copy() for _ in range(ndim - 1)]
        res = zeros(axes, dtype=a.dtype)
        diag_indices = kth_diag_indices(res.shape, k)
        res.ipoints[diag_indices] = a
        return res
    else:
        if k != 0 and len(axes) > 2:
            raise NotImplementedError("k != 0 and len(axes) > 2")
        if axes is None:
            axes = a.axes
        else:
            axes = a.axes[axes]
        axes_indices = kth_diag_indices(axes.shape, k)
        indexer = tuple(axis.i[indices] for axis, indices in zip(axes, axes_indices))
        return a.points[indexer]


@_check_axes_argument
def labels_array(axes, title=None, meta=None) -> Array:
    r"""Return an array with specified axes and the combination of
    corresponding labels as values.

    Parameters
    ----------
    axes : Axis or collection of Axis
    title : str, optional
        Deprecated. See 'meta' below.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> labels_array(sex)
    sex  M  F
         M  F
    >>> labels_array((nat, sex))
    nat  sex\axis  nat  sex
     BE         M   BE    M
     BE         F   BE    F
     FO         M   FO    M
     FO         F   FO    F
    """
    # >>> labels_array((nat, sex))
    # nat\sex     M     F
    #      BE  BE,M  BE,F
    #      FO  FO,M  FO,F
    meta = _handle_meta(meta, title)
    axes = AxisCollection(axes)
    if len(axes) > 1:
        res_axes = axes + Axis(axes.names, 'axis')
        res_data = np.empty(res_axes.shape, dtype=object)
        res_data.flat[:] = list(product(*axes.labels))
        # XXX: I wonder if it wouldn't be better to return LGroups or a similar object which would display as "a,b" but
        #      where each label is stored separately.
        # flat_data = np.array([p for p in product(*axes.labels)])
        # res_data = flat_data.reshape(axes.shape)
    else:
        res_axes = axes
        res_data = axes[0].labels
    return Array(res_data, res_axes, meta=meta)


def identity(axis):
    raise NotImplementedError("identity(axis) is deprecated. In most cases, you can now use the axis directly. "
                              "For example, 'identity(age) < 10' can be replaced by 'age < 10'. "
                              "In other cases, you should use labels_array(axis) instead.")


def eye(rows, columns=None, k=0, title=None, dtype=None, meta=None) -> Array:
    r"""Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    rows : int or Axis or tuple or length 2 AxisCollection
        Rows of the output (if int or Axis) or rows and columns (if tuple or AxisCollection).
    columns : int or Axis, optional
        Columns of the output. Defaults to the value of `rows` if it is an int or Axis.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper
        diagonal, and a negative value to a lower diagonal.
    title : str, optional
        Deprecated. See 'meta' below.
    dtype : data-type, optional
        Data-type of the returned array. Defaults to float.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array of shape (rows, columns)
        An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.

    Examples
    --------
    >>> eye('sex=M,F')
    sex\sex    M    F
          M  1.0  0.0
          F  0.0  1.0
    >>> eye(2, dtype=int)
    {0}*\{1}*  0  1
            0  1  0
            1  0  1
    >>> age = Axis('age=0..2')
    >>> sex = Axis('sex=M,F')
    >>> eye(age, sex)
    age\sex    M    F
          0  1.0  0.0
          1  0.0  1.0
          2  0.0  0.0
    >>> eye(3, k=1)
    {0}*\{1}*    0    1    2
            0  0.0  1.0  0.0
            1  0.0  0.0  1.0
            2  0.0  0.0  0.0
    """
    meta = _handle_meta(meta, title)
    if isinstance(rows, AxisCollection):
        assert columns is None
        axes = rows
    elif isinstance(rows, (tuple, list)):
        assert columns is None
        axes = AxisCollection(rows)
    else:
        if columns is None:
            columns = rows.copy() if isinstance(rows, Axis) else rows
        axes = AxisCollection([rows, columns])
    shape = axes.shape
    data = np.eye(shape[0], shape[1], k, dtype)
    return Array(data, axes, meta=meta)


# XXX: we could change the syntax to use *args
#      => less punctuation but forces kwarg
#      => potentially longer
#      => unsure for now. The most important point is that it should be consistent with other functions.
# stack(a1, a2, axis=Axis('M,F', 'sex'))
# stack(('M', a1), ('F', a2), axis='sex')
# stack(a1, a2, axis='sex')

# we could do something like (it would make from_lists obsolete for 1D arrays):
# stack('sex', M=1, F=2)

# which is almost equivalent to:

# stack(M=1, F=2, axis='sex')

# but we cannot support the current syntax unmodified AND the first version, but second version we could.

# we would only have to explain that they cannot do:

# stack(0=1, 1=2, axis='age')
# stack(0A=1, 1B=2, axis='code')

# but should use this instead:

# stack({0: 1, 1: 2}, 'age=0,1')
# stack({'0A': 1, '1B': 2}, 'code=0A,1B')

# stack({0: 1, 1: 2}, age)
# stack({'0A': 1, '1B': 2}, code)

# or this, if we decide to support *args instead:

# stack((0, 1), (1, 2), axis='age')
# stack(('0A', 1), ('1B', 2), axis='code')

# stack(M=1, F=2, axis='sex')

# is much nicer than:

# from_lists(['sex', 'M', 'F'],
#            [   '',   1,   2])

# for 2D arrays, from_lists and stack would be mostly as ugly and for 3D+ from_lists stays nicer even though I still do
# not like it much.

# stack('nationality',
#       BE=stack('sex', M=0, F=1),
#       FR=stack('sex', M=2, F=3),
#       DE=stack('sex', M=4, F=5))
#
# from_lists([['nationality\\sex', 'M', 'F'],
#             [              'BE',   0,   1],
#             [              'FR',   2,   3],
#             [              'DE',   4,   5]])

# SUPER SLOPPY (I hate this, but I bet users would like it):

# stack(BE_M=0, BE_F=1,
#       FR_M=2, FR_F=3,
#       DE_M=4, DE_F=5, axis='nationality_sex')

# stack(('nationality', 'sex'), {
#       ('BE', 'M'): 0, ('BE', 'F'): 1,
#       ('FR', 'M'): 2, ('FR', 'F'): 3,
#       ('DE', 'M'): 4, ('DE', 'F'): 5})

# for 2D, I think the best compromise is the nested dict (especially for python 3.7+):

# stack({'BE': {'M': 0, 'F': 1},
#        'FR': {'M': 2, 'F': 3},
#        'DE': {'M': 4, 'F': 5}}, axes=('nationality', 'sex'))

# we could make this valid too (combine pos and labels) but I don't think it worth it unless it comes
# naturally from the implementation:

# stack({'BE': {'M,F': [0, 1]},
#        'FR': {'M,F': [2, 3]},
#        'DE': {'M,F': [4, 5]}}, axes=('nationality', 'sex'))

# It looks especially nice if the labels have been extracted to variables:

# BE, FR, DE = nat['BE,FR,DE']
# M, F = sex['M,F']

# stack({BE: {M: 0, F: 1},
#        FR: {M: 2, F: 3},
#        DE: {M: 4, F: 5}})

# for 3D:

# stack({'a0': {'b0': {'c0':  0, 'c1':  1},
#               'b1': {'c0':  2, 'c1':  3},
#               'b2': {'c0':  4, 'c1':  5}},
#        'a1': {'b0': {'c0':  6, 'c1':  7},
#               'b1': {'c0':  8, 'c1':  9},
#               'b2': {'c0': 10, 'c1': 11}}},
#       axes=('a', 'b', 'c'))

# a0, a1 = a['a0,a1']
# b0, b1, b2 = b['b0,b1,b2']
# c0, c1 = c['c0,c1']

# stack({a0: {b0: {c0:  0, c1:  1},
#             b1: {c0:  2, c1:  3},
#             b2: {c0:  4, c1:  5}},
#        a1: {b0: {c0:  6, c1:  7},
#             b1: {c0:  8, c1:  9},
#             b2: {c0: 10, c1: 11}}},
#       axes=(a, b, c))

# if we implement:
#     arr[key] = {'a0': 0, 'a1': 1}
# where key must not be related to the "a" axis
# if would make it relatively easy to implement the nested dict syntax I think:
# first do a pass at the structure to get axes (if not provided) then:
#     for k, v in d.items():
#         arr[k] = v
# but that syntax could be annoying if we want to have an array of dicts

# alternatives:

# arr['a0'] = 0; arr['a1'] = 1 # <-- this already works
# arr['a0,a1'] = [0, 1]        # <-- unsure if this works, but we should make it work (it is annoying if we
#                              #     have an array of lists
# arr[:] = {'a0': 0, 'a1': 1}
# arr[:] = stack({'a0': 0, 'a1': 1}) # <-- not equivalent if a has more labels

# FIXME: move this function elswhere + update returned type to Union[Array, Session]
@deprecate_kwarg('axis', 'axes')
def stack(elements=None, axes=None, title=None, meta=None, dtype=None, res_axes=None, **kwargs) -> 'Array':
    r"""
    Combine several arrays or sessions along an axis.

    Parameters
    ----------
    elements : tuple, list, dict or Session.
        Elements to stack. Elements can be scalars, arrays, sessions, (label, value) pairs or a {label: value} mapping.

        Stacking a single session will stack all its arrays in a single array.
        Stacking several sessions will take the corresponding arrays in all the sessions and stack them, returning a
        new session. An array missing in a session will be replaced by NaN.
    axes : str, Axis, Group or sequence of Axis, optional
        Axes to create. If None, defaults to a range() axis.
    title : str, optional
        Deprecated. See 'meta' below.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.
    dtype : type, optional
        Output dtype. Defaults to None (inspect all output values to infer it automatically).
    res_axes : AxisCollection, optional
        Axes of the output. Defaults to None (union of axes of all values and the stacking axes).

    Returns
    -------
    Array or Session
        A single Array combining input values, or a single Session combining input Sessions.
        The new (stacked) axes will be the last axes of the new array.

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> arr1 = ones(sex)
    >>> arr1
    sex    M    F
         1.0  1.0
    >>> arr2 = zeros(sex)
    >>> arr2
    sex    M    F
         0.0  0.0

    In case the axis to create has already been defined in a variable (Axis or Group)

    >>> stack({'BE': arr1, 'FO': arr2}, nat)
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    Otherwise (when one wants to create an axis from scratch), any of these syntaxes works:

    >>> stack([arr1, arr2], 'nat=BE,FO')
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0
    >>> stack({'BE': arr1, 'FO': arr2}, 'nat=BE,FO')
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0
    >>> stack([('BE', arr1), ('FO', arr2)], 'nat=BE,FO')
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    When stacking arrays with different axes, the result has the union of all axes present:

    >>> stack({'BE': arr1, 'FO': 0}, nat)
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    Creating an axis without name nor labels can be done using:

    >>> stack((arr1, arr2))
    sex\{1}*    0    1
           M  1.0  0.0
           F  1.0  0.0

    When labels are "simple" strings (ie no integers, no string starting with integers, etc.), using keyword
    arguments can be an attractive alternative.

    >>> stack(FO=arr2, BE=arr1, axes=nat)
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    Without passing an explicit order for labels (or an axis object like above)

    >>> stack(BE=arr1, FO=arr2, axes='nat')   # doctest: +SKIP
    sex\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    One can also stack along several axes

    >>> test = Axis('test=T1,T2')
    >>> stack({('BE', 'T1'): arr1,
    ...        ('BE', 'T2'): arr2,
    ...        ('FO', 'T1'): arr2,
    ...        ('FO', 'T2'): arr1},
    ...       (nat, test))
    sex  nat\test   T1   T2
      M        BE  1.0  0.0
      M        FO  0.0  1.0
      F        BE  1.0  0.0
      F        FO  0.0  1.0

    To stack sessions, let us first create two test sessions. For example suppose we have a session storing the results
    of a baseline simulation:

    >>> from larray import Session
    >>> baseline = Session({'arr1': arr1, 'arr2': arr2})

    and another session with a variant (here we simply added 0.5 to each array)

    >>> variant = Session({'arr1': arr1 + 0.5, 'arr2': arr2 + 0.5})

    then we stack them together

    >>> stacked = stack({'baseline': baseline, 'variant': variant}, 'sessions')
    >>> stacked
    Session(arr1, arr2)
    >>> stacked.arr1
    sex\sessions  baseline  variant
               M       1.0      1.5
               F       1.0      1.5
    >>> stacked.arr2
    sex\sessions  baseline  variant
               M       0.0      0.5
               F       0.0      0.5
    """
    from larray import Session

    axes_to_anonymize = ()

    meta = _handle_meta(meta, title)

    if elements is not None and kwargs:
        raise TypeError("stack() accepts either keyword arguments OR a collection of elements, not both")

    if isinstance(axes, str) and '=' in axes:
        axes = Axis(axes)
    elif isinstance(axes, Group):
        axes = Axis(axes)

    if axes is not None and not isinstance(axes, str):
        axes = AxisCollection(axes)

    if kwargs:
        elements = kwargs.items()

    if isinstance(elements, (dict, Session)):
        elements = elements.items()

    if isinstance(elements, Array):
        if axes is None:
            axes = -1
        axes = elements.axes[axes]
        items = elements.items(axes)
    elif isinstance(elements, Session):
        if axes is None:
            axes = 'array'
        items = elements.items()
    elif isinstance(elements, Iterable):
        if not isinstance(elements, Sequence):
            elements = list(elements)

        if all(isinstance(e, tuple) for e in elements):
            assert all(len(e) == 2 for e in elements)
            if axes is None or isinstance(axes, str):
                keys = [k for k, v in elements]
                values = [v for k, v in elements]
                # assert that all keys are indexers
                assert all(np.isscalar(k) or isinstance(k, (Group, tuple)) for k in keys)

                # we need a kludge to support stacking along an anonymous axis because AxisCollection.extend
                # (and thus AxisCollection.union) support for anonymous axes is kinda messy. This needs to happen
                # *before* we compute items, otherwise the IGroups will be on the wrong axis, making result[k] = v
                # a lot slower
                stack_axis = Axis(keys, "___anonymous___" if axes is None else axes)
                if axes is None:
                    axes_to_anonymize = (stack_axis,)
                # FIXME: if res_axes is not None, we should make sure it contains "axes" (with keys in the same order!!)
                #        and extract axes from there, before we compute items, otherwise, we do not work on the
                #        result axis objects, which makes results[k] = v a lot slower
                #        and if keys are not in the same order (or maybe do it systematically?) we will need to
                #        pass via dict like below (translate_and_sort_key) but it will break with duplicate labels,
                #        unless keys are IGroups,
                #        there ought to be a way to sort the k, v efficiently without breaking duplicate labels
                # TODO: add support for more than one axis here
                axes = AxisCollection(stack_axis)
                items = list(zip(stack_axis, values))
            else:
                def translate_and_sort_key(key, axes):
                    dict_of_indices = axes._key_to_axis_indices_dict(key)
                    return tuple(IGroup(dict_of_indices[axis], axis=axis) for axis in axes)

                # passing only via _key_to_igroup should be enough if we allow for partial axes
                dict_elements = {translate_and_sort_key(key, axes): value for key, value in elements}
                items = [(k, dict_elements[k]) for k in axes.iter_labels()]
        else:
            if axes is None or isinstance(axes, str):
                stack_axis = Axis(len(elements), "___anonymous___" if axes is None else axes)
                if axes is None:
                    axes_to_anonymize = (stack_axis,)
                axes = AxisCollection(stack_axis)
            else:
                # TODO: add support for more than one axis here
                assert axes.ndim == 1 and len(axes[0]) == len(elements)
            items = list(zip(axes[0], elements))
    else:
        elements_type = type(elements).__name__
        raise TypeError(f'unsupported type for arrays: {elements_type}')

    if any(isinstance(v, Session) for k, v in items):
        if not all(isinstance(v, Session) for k, v in items):
            raise TypeError("stack() only supports stacking Session with other Session objects")

        array_names = unique_multi(sess.keys() for sess_name, sess in items)

        def stack_one(array_name):
            try:
                return stack({sess_name: sess.get(array_name, nan)
                              for sess_name, sess in items}, axes=axes)
            # TypeError for str arrays, ValueError for incompatible axes, ...
            except Exception:
                return nan

        return Session({array_name: stack_one(array_name) for array_name in array_names}, meta=meta)
    else:
        if res_axes is None or dtype is None:
            values = [asarray(v) if not np.isscalar(v) else v
                      for k, v in items]

            if res_axes is None:
                # XXX: with the current semantics of stack, we need to compute the union of axes for values but axis
                #      needs to be added unconditionally. We *might* want to change the semantics to mean either stack
                #      or concat depending on whether the axis already exists.
                #      this would be more convenient for users I think, but would mean one class of error we cannot
                #      detect anymore: if a user unintentionally stacks an array with the axis already present.
                #      (this is very similar to the debate about combining Array.append and Array.extend)
                all_axes = [get_axes(v) for v in values] + [axes]
                res_axes = AxisCollection.union(*all_axes)
            elif not isinstance(res_axes, AxisCollection):
                res_axes = AxisCollection(res_axes)

            if dtype is None:
                # dtype = common_type(values + [fill_value])
                dtype = common_dtype(values)

        # if needs_fill:
        #     result = full(res_axes, fill_value, dtype=dtype, meta=meta)
        # else:
        result = empty(res_axes, dtype=dtype, meta=meta)

        # FIXME: this is *much* faster but it only works for scalars and not for stacking arrays
        # keys = tuple(zip(*[k for k, v in items]))
        # result.points[keys] = values
        for k, v in items:
            result[k] = v

        return result if not axes_to_anonymize else result.rename({a: None for a in axes_to_anonymize})


def get_axes(value) -> AxisCollection:
    return value.axes if isinstance(value, Array) else AxisCollection([])


def _strip_shape(shape):
    return tuple(s for s in shape if s != 1)


def _equal_modulo_len1(shape1, shape2):
    return _strip_shape(shape1) == _strip_shape(shape2)


# assigning a temporary name to anonymous axes before broadcasting and removing it afterwards is not a good idea after
# all because it copies the axes/change the object, and thus "flatten" wouldn't work with index axes:
# a[ones(a.axes[axes], dtype=bool)]
# but if we had assigned axes names from the start (without dropping them) this wouldn't be a problem.
def make_numpy_broadcastable(values, min_axes=None) -> Tuple[List[Array], AxisCollection]:
    r"""
    Return values where Arrays are (NumPy) broadcastable between them.
    For that to be possible, all common axes must be compatible (see Axis class documentation).
    Extra axes (in any array) can have any length.

    * the resulting arrays will have the combination of all axes found in the input arrays, the earlier arrays defining
      the order of axes. Axes with labels take priority over wildcard axes.
    * length 1 wildcard axes will be added for axes not present in input

    Parameters
    ----------
    values : iterable of arrays
        Arrays that requires to be (NumPy) broadcastable between them.
    min_axes : AxisCollection, optional
        Minimum axes the result should have. This argument is useful both when one wants to have extra axes in the
        result or for having resulting axes in a specific order. Defaults to all input axes, ordered by first
        appearance.

    Returns
    -------
    arrays : list of arrays
        List of arrays broadcastable between them. Arrays will have the combination of all axes found in the input
        arrays, the earlier arrays defining the order of axes.
    res_axes : AxisCollection
        Union of ``min_axes`` and the axes of all input arrays.

    See Also
    --------
    Axis.iscompatible : tests if axes are compatible between them.
    """
    axes_union = AxisCollection.union(*[get_axes(v) for v in values])
    if min_axes is not None:
        if not isinstance(min_axes, AxisCollection):
            min_axes = AxisCollection(min_axes)
        axes_union = min_axes | axes_union
    def broadcasted_value(value):
        if isinstance(value, Array):
            return value.broadcast_with(axes_union)
        elif isinstance(value, ExprNode):
            return value.evaluate(axes_union)
        else:
            return value
    return [broadcasted_value(value) for value in values], axes_union


def raw_broadcastable(values, min_axes=None) -> Tuple[Tuple[Any, ...], AxisCollection]:
    r"""
    same as make_numpy_broadcastable but returns numpy arrays.
    """
    arrays, res_axes = make_numpy_broadcastable(values, min_axes=min_axes)
    raw = tuple(a.data if isinstance(a, Array) else a
                for a in arrays)
    return raw, res_axes


def make_args_broadcastable(args, kwargs=None) -> Tuple[Any, Any, Any]:
    """
    Make args and kwargs (NumPy) broadcastable between them.
    """
    values = (args + tuple(kwargs.values())) if kwargs is not None else args
    first_kw = len(args)
    raw_bcast_values, res_axes = raw_broadcastable(values)
    raw_bcast_args = raw_bcast_values[:first_kw]
    raw_bcast_kwargs = dict(zip(kwargs.keys(), raw_bcast_values[first_kw:]))
    return raw_bcast_args, raw_bcast_kwargs, res_axes


def zip_array_values(values, axes=None, ascending=True) -> SequenceZip:
    r"""Return a sequence as if simultaneously iterating on several arrays.

    Parameters
    ----------
    values : sequence of (scalar or Array)
        Values to iterate on. Scalars are repeated as many times as necessary.
    axes : int, str or Axis or tuple of them, optional
        Axis or axes along which to iterate and in which order. All those axes must be compatible (if present) between
        the different values. Defaults to None (union of all axes present in all arrays, in the order they are found).
    ascending : bool, optional
        Whether to iterate the axes in ascending order (from start to end). Defaults to True.

    Returns
    -------
    Sequence

    Examples
    --------
    >>> arr1 = ndtest('a=a0,a1;b=b1,b2')
    >>> arr2 = ndtest('a=a0,a1;c=c1,c2')
    >>> arr1
    a\b  b1  b2
     a0   0   1
     a1   2   3
    >>> arr2
    a\c  c1  c2
     a0   0   1
     a1   2   3
    >>> for a1, a2 in zip_array_values((arr1, arr2), 'a'):
    ...     print("==")
    ...     print(a1)
    ...     print(a2)
    ==
    b  b1  b2
        0   1
    c  c1  c2
        0   1
    ==
    b  b1  b2
        2   3
    c  c1  c2
        2   3

    When the axis to iterate on (`c` in this case) is not present in one of the arrays (arr1), that array is repeated
    for each label of that axis:

    >>> for a1, a2 in zip_array_values((arr1, arr2), arr2.c):
    ...     print("==")
    ...     print(a1)
    ...     print(a2)
    ==
    a\b  b1  b2
     a0   0   1
     a1   2   3
    a  a0  a1
        0   2
    ==
    a\b  b1  b2
     a0   0   1
     a1   2   3
    a  a0  a1
        1   3

    When no `axes` are given, it iterates on the union of all compatible axes (a, b, and c in this case):

    >>> for a1, a2 in zip_array_values((arr1, arr2)):
    ...     print(f"arr1: {a1}, arr2: {a2}")
    arr1: 0, arr2: 0
    arr1: 0, arr2: 1
    arr1: 1, arr2: 0
    arr1: 1, arr2: 1
    arr1: 2, arr2: 2
    arr1: 2, arr2: 3
    arr1: 3, arr2: 2
    arr1: 3, arr2: 3
    """
    def values_with_expand(value, axes, readonly=True, ascending=True):
        if isinstance(value, Array):
            # an Axis axis is not necessarily in array.axes
            expanded = value.expand(axes, readonly=readonly)
            return expanded.values(axes, ascending=ascending)
        else:
            size = axes.size if axes.ndim else 0
            return Repeater(value, size)

    values_axes = [get_axes(v) for v in values]

    if axes is None:
        all_iter_axes = values_axes
    else:
        if not isinstance(axes, (tuple, list, AxisCollection)):
            axes = (axes,)

        # transform string axes _definitions_ to objects
        axes = [Axis(axis) if isinstance(axis, str) and '=' in axis else axis
                for axis in axes]

        # get iter axes for all values and transform string axes _references_ to objects
        all_iter_axes = [AxisCollection([value_axes[axis] for axis in axes if axis in value_axes])
                         for value_axes in values_axes]

    common_iter_axes = AxisCollection.union(*all_iter_axes)

    # sequence of tuples (of scalar or arrays)
    return SequenceZip([values_with_expand(v, common_iter_axes, ascending=ascending) for v in values])


def zip_array_items(values, axes=None, ascending=True) -> SequenceZip:
    r"""Return a sequence as if simultaneously iterating on several arrays as well as the current iteration "key".

    Broadcasts all values against each other. Scalars are simply repeated.

    Parameters
    ----------
    values : Iterable
        arrays to iterate on.
    axes : int, str or Axis or tuple of them, optional
        Axis or axes along which to iterate and in which order. Defaults to None (union of all axes present in
        all arrays, in the order they are found).
    ascending : bool, optional
        Whether to iterate the axes in ascending order (from start to end). Defaults to True.

    Returns
    -------
    Sequence

    Examples
    --------
    >>> arr1 = ndtest('a=a0,a1;b=b0,b1')
    >>> arr2 = ndtest('a=a0,a1;c=c0,c1')
    >>> arr1
    a\b  b0  b1
     a0   0   1
     a1   2   3
    >>> arr2
    a\c  c0  c1
     a0   0   1
     a1   2   3
    >>> for k, (a1, a2) in zip_array_items((arr1, arr2), 'a'):
    ...     print("==", k[0], "==")
    ...     print(a1)
    ...     print(a2)
    == a0 ==
    b  b0  b1
        0   1
    c  c0  c1
        0   1
    == a1 ==
    b  b0  b1
        2   3
    c  c0  c1
        2   3
    >>> for k, (a1, a2) in zip_array_items((arr1, arr2), arr2.c):
    ...     print("==", k[0], "==")
    ...     print(a1)
    ...     print(a2)
    == c0 ==
    a\b  b0  b1
     a0   0   1
     a1   2   3
    a  a0  a1
        0   2
    == c1 ==
    a\b  b0  b1
     a0   0   1
     a1   2   3
    a  a0  a1
        1   3
    >>> for k, (a1, a2) in zip_array_items((arr1, arr2)):
    ...     print(k, "arr1: {}, arr2: {}".format(a1, a2))
    (a.i[0], b.i[0], c.i[0]) arr1: 0, arr2: 0
    (a.i[0], b.i[0], c.i[1]) arr1: 0, arr2: 1
    (a.i[0], b.i[1], c.i[0]) arr1: 1, arr2: 0
    (a.i[0], b.i[1], c.i[1]) arr1: 1, arr2: 1
    (a.i[1], b.i[0], c.i[0]) arr1: 2, arr2: 2
    (a.i[1], b.i[0], c.i[1]) arr1: 2, arr2: 3
    (a.i[1], b.i[1], c.i[0]) arr1: 3, arr2: 2
    (a.i[1], b.i[1], c.i[1]) arr1: 3, arr2: 3
    """
    res_axes = AxisCollection.union(*[get_axes(v) for v in values])
    return SequenceZip((res_axes.iter_labels(axes, ascending=ascending),
                        zip_array_values(values, axes=axes, ascending=ascending)))


_default_float_error_handler = float_error_handler_factory(3)


original_float_error_settings = np.seterr(divide='call', invalid='call')
original_float_error_handler = np.seterrcall(_default_float_error_handler)

# excel IO tools in Python
# - openpyxl: the slowest but most-complete package but still lags behind PHPExcel from which it was ported. despite
#             the drawbacks the API is very complete.
#   biggest drawbacks:
#   * you can get either the "cached" value of cells OR their formulas but NOT BOTH and this is a file-wide setting
#     (data_only=True). if you have an excel file and want to add a sheet to it, you either loose all cached values
#     (which is problematic in many cases since you do not necessarily have linked files) or loose all formulas.
#   * it loose "charts" on read. => cannot append/update a sheet to a file with charts, which is precisely what many
#     users asked. => users need to create their charts using code.
# - xlsxwriter: faster and slightly more feature-complete than openpyxl regarding writing but does not read anything
#               => cannot update an existing file. API seems extremely complete.
# - pyexcelerate: yet faster but also write only. Didn't check whether API is more featured than xlsxwriter or not.
# - xlwings: wraps win32com & equivalent on mac, so can potentially do everything (I guess) but this is SLOW and needs
#            a running excel instance, etc.
