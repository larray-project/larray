# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

__all__ = [
    'LArray', 'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like', 'full', 'full_like', 'sequence',
    'create_sequential', 'ndrange', 'labels_array', 'ndtest', 'aslarray', 'identity', 'diag', 'eye',
    'larray_equal', 'larray_nan_equal', 'all', 'any', 'sum', 'prod', 'cumsum', 'cumprod', 'min', 'max', 'mean', 'ptp',
    'var', 'std', 'median', 'percentile', 'stack', 'nan', 'nan_equal', 'element_equal'
]

"""
Matrix class
"""

# ? implement multi group in one axis getitem: lipro['P01,P02;P05'] <=> (lipro['P01,P02'], lipro['P05'])

# * we need an API to get to the "next" label. Sometimes, we want to use label+1, but that is problematic when labels
#   are not numeric, or have not a step of 1. X.agegroup[X.agegroup.after(25):]
#                                             X.agegroup[X.agegroup[25].next():]

# * implement keepaxes=True for _group_aggregate instead of/in addition to group tuples

# ? implement newaxis

# * Axis.sequence? geo.seq('A31', 'A38') (equivalent to geo['A31..A38'])

# * re-implement row_totals/col_totals? or what do we do with them?

# * time specific API so that we know if we go for a subclass or not

# * data alignment in arithmetic methods

# * test structured arrays

# ? move "utils" to its own project (so that it is not duplicated between larray and liam2)
#   OR
#   include utils only in larray project and make larray a dependency of liam2
#   (and potentially rename it to reflect the broader scope)

from collections import Iterable, Sequence
from itertools import product, chain, groupby, islice
import os
import sys
import functools

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

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

from larray.core.abstractbases import ABCLArray
from larray.core.expr import ExprNode
from larray.core.group import (Group, IGroup, LGroup, remove_nested_groups, _to_key, _to_keys,
                               _range_to_slice, _translate_sheet_name, _translate_group_key_hdf)
from larray.core.axis import Axis, AxisReference, AxisCollection, X, _make_axis
from larray.util.misc import (table2str, size2str, basestring, izip, rproduct, ReprString, duplicates,
                              float_error_handler_factory, _isnoneslice, light_product, unique_list, common_type,
                              renamed_to, deprecate_kwarg, LHDFStore)


nan = np.nan


def all(values, axis=None):
    """
    Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    LArray.all
    """
    if isinstance(values, LArray):
        return values.all(axis)
    else:
        return builtins.all(values)


def any(values, axis=None):
    """
    Test whether any array elements along a given axis evaluate to True.

    See Also
    --------
    LArray.any
    """
    if isinstance(values, LArray):
        return values.any(axis)
    else:
        return builtins.any(values)


# commutative modulo float precision errors
def sum(array, *args, **kwargs):
    """
    Sum of array elements.

    See Also
    --------
    LArray.sum
    """
    # XXX: we might want to be more aggressive here (more types to convert), however, generators should still be
    #      computed via the builtin.
    if isinstance(array, (np.ndarray, list)):
        array = LArray(array)
    if isinstance(array, LArray):
        return array.sum(*args, **kwargs)
    else:
        return builtins.sum(array, *args, **kwargs)


def prod(array, *args, **kwargs):
    """
    Product of array elements.

    See Also
    --------
    LArray.prod
    """
    return array.prod(*args, **kwargs)


def cumsum(array, *args, **kwargs):
    """
    Returns the cumulative sum of array elements.

    See Also
    --------
    LArray.cumsum
    """
    return array.cumsum(*args, **kwargs)


def cumprod(array, *args, **kwargs):
    """
    Returns the cumulative product of array elements.

    See Also
    --------
    LArray.cumprod
    """
    return array.cumprod(*args, **kwargs)


def min(array, *args, **kwargs):
    """
    Minimum of array elements.

    See Also
    --------
    LArray.min
    """
    if isinstance(array, LArray):
        return array.min(*args, **kwargs)
    else:
        return builtins.min(array, *args, **kwargs)


def max(array, *args, **kwargs):
    """
    Maximum of array elements.

    See Also
    --------
    LArray.max
    """
    if isinstance(array, LArray):
        return array.max(*args, **kwargs)
    else:
        return builtins.max(array, *args, **kwargs)


def mean(array, *args, **kwargs):
    """
    Computes the arithmetic mean.

    See Also
    --------
    LArray.mean
    """
    return array.mean(*args, **kwargs)


def median(array, *args, **kwargs):
    """
    Computes the median.

    See Also
    --------
    LArray.median
    """
    return array.median(*args, **kwargs)


def percentile(array, *args, **kwargs):
    """
    Computes the qth percentile of the data along the specified axis.

    See Also
    --------
    LArray.percentile
    """
    return array.percentile(*args, **kwargs)


# not commutative
def ptp(array, *args, **kwargs):
    """
    Returns the range of values (maximum - minimum).

    See Also
    --------
    LArray.ptp
    """
    return array.ptp(*args, **kwargs)


def var(array, *args, **kwargs):
    """
    Computes the variance.

    See Also
    --------
    LArray.var
    """
    return array.var(*args, **kwargs)


def std(array, *args, **kwargs):
    """
    Computes the standard deviation.

    See Also
    --------
    LArray.std
    """
    return array.std(*args, **kwargs)


def concat(arrays, axis=0, dtype=None):
    """Concatenate arrays along axis

    Parameters
    ----------
    arrays : tuple of LArray
        Arrays to concatenate.
    axis : axis reference (int, str or Axis), optional
        Axis along which to concatenate. Defaults to the first axis.
    dtype : dtype, optional
        Result data type. Defaults to the "closest" type which can hold all arrays types without loss of information.

    Returns
    -------
    LArray

    Examples
    --------
    >>> arr1 = ndtest((2, 3))
    >>> arr1
    a\\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr2 = ndtest('a=a0,a1;b=b3')
    >>> arr2
    a\\b  b3
     a0   0
     a1   1
    >>> arr3 = ndtest('b=b4,b5')
    >>> arr3
    b  b4  b5
        0   1
    >>> concat((arr1, arr2, arr3), 'b')
    a\\b  b0  b1  b2  b3  b4  b5
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
    labels_type = common_type(arrays_labels)
    if labels_type is object:
        # astype always copies, while asarray only copies if necessary
        arrays_labels = [np.asarray(labels, dtype=object) for labels in arrays_labels]

    combined_axis = Axis(np.concatenate(arrays_labels), name)

    # combine all axes (using labels from any side if any)
    result_axes = arrays[0].axes.replace(axis, combined_axis).union(*[array.axes - axis for array in arrays[1:]])

    if dtype is None:
        dtype = common_type(arrays)

    result = empty(result_axes, dtype=dtype)
    start = 0
    for labels, array in zip(arrays_labels, arrays):
        stop = start + len(labels)
        result[combined_axis.i[start:stop]] = array
        start = stop
    return result


class LArrayIterator(object):
    def __init__(self, array):
        self.array = array
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        array = self.array
        if self.index == len(self.array):
            raise StopIteration
        # result = array.i[array.axes[0].i[self.index]]
        result = array.i[self.index]
        self.index += 1
        return result
    # Python 2
    next = __next__


class LArrayPositionalIndexer(object):
    def __init__(self, array):
        self.array = array

    def _translate_key(self, key):
        """
        Translates key into tuple of IGroup, i.e.
        tuple of collections of labels.
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.array.ndim:
            raise IndexError("key has too many indices (%d) for array with %d dimensions" % (len(key), self.array.ndim))
        # no need to create a full nd key as that will be done later anyway
        return tuple(axis.i[axis_key]
                     for axis_key, axis in zip(key, self.array.axes))

    def __getitem__(self, key):
        return self.array[self._translate_key(key)]

    def __setitem__(self, key, value):
        self.array[self._translate_key(key)] = value

    def __len__(self):
        return len(self.array)


class LArrayPointsIndexer(object):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        # TODO: this should generate an "intersection"/points NDGroup and simply do return self.array[nd_group]
        data = np.asarray(self.array)
        translated_key = self.array._translated_key(key, bool_stuff=True)

        axes = self.array._bool_key_new_axes(translated_key)
        data = data[translated_key]
        # drop length 1 dimensions created by scalar keys
        # data = data.reshape(tuple(len(axis) for axis in axes))
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            return LArray(data, axes)

    def __setitem__(self, key, value):
        data = np.asarray(self.array)
        translated_key = self.array._translated_key(key, bool_stuff=True)
        if isinstance(value, LArray):
            axes = self.array._bool_key_new_axes(translated_key, wildcard_allowed=True)
            value = value.broadcast_with(axes)
        data[translated_key] = value


class LArrayPositionalPointsIndexer(object):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        data = np.asarray(self.array)

        axes = self.array._bool_key_new_axes(key, wildcard_allowed=False)
        data = data[key]
        # drop length 1 dimensions created by scalar keys
        # data = data.reshape(tuple(len(axis) for axis in axes))
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            return LArray(data, axes)

    def __setitem__(self, key, value):
        data = np.asarray(self.array)
        data[key] = value


def get_axis(obj, i):
    """
    Returns an axis according to its position.

    Parameters
    ----------
    obj : LArray or other array
        Input LArray or any array object which has a shape attribute (NumPy or Pandas array).
    i : int
        index of the axis.

    Returns
    -------
    Axis
        Axis corresponding to the given index if input `obj` is a LArray. A new anonymous Axis with the length of
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
    return obj.axes[i] if isinstance(obj, LArray) else Axis(obj.shape[i])


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
        out : LArray, optional
            Alternate output array in which to place the result. It must have the same shape as the expected output and
            its type is preserved (e.g., if dtype(out) is float, the result will consist of 0.0’s and 1.0’s).
            Axes and labels can be different, only the shape matters. Defaults to None (create a new array)."""},
    'ddof': {'value': 1, 'doc': """
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is ``N - ddof``, where ``N`` represents
            the number of elements. Defaults to 1."""},
    'skipna': {'value': None, 'doc': """
        skipna : bool, optional
            Whether or not to skip NaN (null) values. If False, resulting cells will be NaN if any of the aggregated
            cells is NaN. Defaults to True."""},
    'keepaxes': {'value': False, 'doc': """
        keepaxes : bool, optional
            Whether or not reduced axes are left in the result as dimensions with size one. Defaults to False."""
    },
    'interpolation': {'value': 'linear', 'doc': """
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional
            Interpolation method to use when the desired quantile lies between two data points ``i < j``:

              * linear: ``i + (j - i) * fraction``, where ``fraction`` is the fractional part of the index surrounded
                by ``i`` and ``j``.
              * lower: ``i``.
              * higher: ``j``.
              * nearest: ``i`` or ``j``, whichever is nearest.
              * midpoint: ``(i + j) / 2``.

            Defaults to 'linear'."""
    }
}


def _doc_agg_method(func, by=False, long_name='', action_verb='perform', extra_args=[], kwargs=[]):
    if not long_name:
        long_name = func.__name__

    _args = ','.join(extra_args) + ', ' if len(extra_args) > 0 else ''
    _kwargs = ', '.join(["{}={!r}".format(k, _kwarg_agg[k]['value']) for k in kwargs]) + ', ' if len(kwargs) > 0 else ''
    signature = '{name}({args}*axes_and_groups, {kwargs}**explicit_axes)'.format(name=func.__name__,
                                                                                 args=_args, kwargs=_kwargs)

    if by:
        specific_template = """The {long_name} is {action_verb}ed along all axes except the given one(s).
            For groups, {long_name} is {action_verb}ed along groups and non associated axes."""
    else:
        specific_template = "Axis(es) or group(s) along which the {long_name} is {action_verb}ed."
    doc_specific = specific_template.format(long_name=long_name, action_verb=action_verb)

    doc_args = "".join(_arg_agg[arg] for arg in extra_args)
    doc_kwargs = "".join(_kwarg_agg[kw]['doc'] for kw in kwargs)
    doc_varargs = """
        \*axes_and_groups : None or int or str or Axis or Group or any combination of those
            {specific}
            The default (no axis or group) is to {action_verb} the {long_name} over all the dimensions of the input
            array.

            An axis can be referred by:

            * its index (integer). Index can be a negative integer, in which case it counts from the last to the
              first axis.
            * its name (str or AxisReference). You can use either a simple string ('axis_name') or the special
              variable x (x.axis_name).
            * a variable (Axis). If the axis has been defined previously and assigned to a variable, you can pass it as
              argument.

            You may not want to {action_verb} the {long_name} over a whole axis but over a selection of specific
            labels. To do so, you have several possibilities:

            * (['a1', 'a3', 'a5'], 'b1, b3, b5') : labels separated by commas in a list or a string
            * ('a1:a5:2') : select labels using a slice (general syntax is 'start:end:step' where is 'step' is
              optional and 1 by default).
            * (a='a1, a2, a3', x.b['b1, b2, b3']) : in case of possible ambiguity, i.e. if labels can belong to more
              than one axis, you must precise the axis.
            * ('a1:a3; a5:a7', b='b0,b2; b1,b3') : create several groups with semicolons.
              Names are simply given by the concatenation of labels (here: 'a1,a2,a3', 'a5,a6,a7', 'b0,b2' and 'b1,b3')
            * ('a1:a3 >> a123', 'b[b0,b2] >> b12') : operator ' >> ' allows to rename groups."""\
        .format(specific=doc_specific, action_verb=action_verb, long_name=long_name)
    parameters = """Parameters
        ----------{args}{varargs}{kwargs}""".format(args=doc_args, varargs=doc_varargs, kwargs=doc_kwargs)

    func.__doc__ = func.__doc__.format(signature=signature, parameters=parameters)


_always_return_float = {np.mean, np.nanmean, np.median, np.nanmedian, np.percentile, np.nanpercentile,
                        np.std, np.nanstd, np.var, np.nanvar}

obj_isnan = np.vectorize(lambda x: x != x, otypes=[bool])


def element_equal(a1, a2, rtol=0, atol=0, nan_equals=False):
    """
    Compares two arrays element-wise and returns array of booleans.

    Parameters
    ----------
    a1, a2 : LArray-like
        Input arrays. aslarray() is used on non-LArray inputs.
    rtol : float or int, optional
        The relative tolerance parameter (see Notes). Defaults to 0.
    atol : float or int, optional
        The absolute tolerance parameter (see Notes). Defaults to 0.
    nan_equals: boolean, optional
        Whether or not to consider nan values at the same positions in the two arrays as equal.
        By default, an array containing nan values is never equal to another array, even if that other array
        also contains nan values at the same positions. The reason is that a nan value is different from
        *anything*, including itself. Defaults to False.

    Returns
    -------
    LArray
        Boolean array of where a1 and a2 are equal within a tolerance range if given.
        If nan_equals=True, nan’s in a1 will be considered equal to nan’s in a2 in the output array.

    Notes
    -----
    For finite values, element_equal uses the following equation to test whether two values are equal:

        absolute(array1 - array2) <= (atol + rtol * absolute(array2))

    The above equation is not symmetric in array1 and array2, so that element_equal(array1, array2)
    might be different from element_equal(array2, array1) in some rare cases.

    Examples
    --------
    >>> arr1 = LArray([6., np.nan, 8.], "a=a0..a2")
    >>> arr1
    a   a0   a1   a2
       6.0  nan  8.0

    Default behavior (same as == operator)

    >>> element_equal(arr1, arr1)
    a    a0     a1    a2
       True  False  True

    Test equality between two arrays within a given tolerance range.
    Return True if absolute(array1 - array2) <= (atol + rtol * absolute(array2)).

    >>> arr2 = LArray([5.999, np.nan, 8.001], "a=a0..a2")
    >>> arr2
    a     a0   a1     a2
       5.999  nan  8.001
    >>> element_equal(arr1, arr2, nan_equals=True)
    a     a0    a1     a2
       False  True  False
    >>> element_equal(arr1, arr2, atol=0.01, nan_equals=True)
    a    a0    a1    a2
       True  True  True
    >>> element_equal(arr1, arr2, rtol=0.01, nan_equals=True)
    a    a0    a1    a2
       True  True  True
    """
    a1, a2 = aslarray(a1), aslarray(a2)

    if rtol == 0 and atol == 0:
        if not nan_equals:
            return a1 == a2
        else:
            from larray.core.ufuncs import isnan

            def general_isnan(a):
                if np.issubclass_(a.dtype.type, np.inexact):
                    return isnan(a)
                elif a.dtype.type is np.object_:
                    return LArray(obj_isnan(a), a.axes)
                else:
                    return False

            return (a1 == a2) | (general_isnan(a1) & general_isnan(a2))
    else:
        (a1, a2), res_axes = make_numpy_broadcastable([a1, a2])
        return LArray(np.isclose(a1.data, a2.data, rtol=rtol, atol=atol, equal_nan=nan_equals), res_axes)


def nan_equal(a1, a2):
    import warnings
    warnings.warn("nan_equal() is deprecated. Use equal() instead.", FutureWarning, stacklevel=2)
    return element_equal(a1, a2, nan_equals=True)


class LArray(ABCLArray):
    """
    A LArray object represents a multidimensional, homogeneous array of fixed-size items with labeled axes.

    The function :func:`aslarray` can be used to convert a NumPy array or Pandas DataFrame into a LArray.

    Parameters
    ----------
    data : scalar, tuple, list or NumPy ndarray
        Input data.
    axes : collection (tuple, list or AxisCollection) of axes (int, str or Axis), optional
        Axes.
    title : str, optional
        Title of array.

    Attributes
    ----------
    data : NumPy ndarray
        Data.
    axes : AxisCollection
        Axes.
    title : str
        Title.

    See Also
    --------
    sequence : Create a LArray by sequentially applying modifications to the array along axis.
    ndtest : Create a test LArray with increasing elements.
    zeros : Create a LArray, each element of which is zero.
    ones : Create a LArray, each element of which is 1.
    full : Create a LArray filled with a given value.
    empty : Create a LArray, but leave its allocated memory unchanged (i.e., it contains “garbage”).

    Examples
    --------
    >>> age = Axis([10, 11, 12], 'age')
    >>> sex = Axis('sex=M,F')
    >>> time = Axis([2007, 2008, 2009], 'time')
    >>> axes = [age, sex, time]
    >>> data = np.zeros((len(axes), len(sex), len(time)))
    >>> LArray(data, axes)
    age  sex\\time  2007  2008  2009
     10         M   0.0   0.0   0.0
     10         F   0.0   0.0   0.0
     11         M   0.0   0.0   0.0
     11         F   0.0   0.0   0.0
     12         M   0.0   0.0   0.0
     12         F   0.0   0.0   0.0
    >>> full(axes, 10.0)
    age  sex\\time  2007  2008  2009
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
    age  sex\\time  2007  2008  2009
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
    sex\\age  10  11  12
          M  10   9   8
          F  10  11  12
    """

    def __init__(self, data, axes=None, title=''):
        data = np.asarray(data)
        ndim = data.ndim
        if axes is None:
            axes = AxisCollection(data.shape)
        else:
            if not isinstance(axes, AxisCollection):
                axes = AxisCollection(axes)
            if axes.ndim != ndim:
                raise ValueError("number of axes (%d) does not match "
                                 "number of dimensions of data (%d)"
                                 % (axes.ndim, ndim))
            if axes.shape != data.shape:
                raise ValueError("length of axes %s does not match "
                                 "data shape %s" % (axes.shape, data.shape))

        self.data = data
        self.axes = axes
        self.title = title

    # XXX: rename to posnonzero and implement a label version of nonzero
    def nonzero(self):
        """
        Returns the indices of the elements that are non-zero.

        Specifically, it returns a tuple of arrays (one for each dimension)
        containing the indices of the non-zero elements in that dimension.

        Returns
        -------
        tuple of arrays : tuple
            Indices of elements that are non-zero.

        Examples
        --------
        >>> arr = ndtest((2, 3)) % 2
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   0
         a1   1   0   1
        >>> arr.nonzero() # doctest: +SKIP
        [array([0, 1, 1]), array([1, 0, 2])]
        """
        # FIXME: return tuple of IGroup instead (or even NDGroup) so that you
        #  can do a[a.nonzero()]
        return self.data.nonzero()

    def set_axes(self, axes_to_replace=None, new_axis=None, inplace=False, **kwargs):
        """
        Replace one, several or all axes of the array.

        Parameters
        ----------
        axes_to_replace : axis ref or dict {axis ref: axis} or list of tuple (axis ref, axis) \
                          or list of Axis or AxisCollection
            Axes to replace. If a single axis reference is given, the `new_axis` argument must be provided.
            If a list of Axis or an AxisCollection is given, all axes will be replaced by the new ones.
            In that case, the number of new axes must match the number of the old ones.
        new_axis : Axis, optional
            New axis if `axes_to_replace` contains a single axis reference.
        inplace : bool, optional
            Whether or not to modify the original object or return a new array and leave the original intact.
            Defaults to False.
        **kwargs : Axis
            New axis for each axis to replace given as a keyword argument.

        Returns
        -------
        LArray
            Array with axes replaced.

        See Also
        --------
        rename : rename one of several axes

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> row = Axis(['r0', 'r1'], 'row')
        >>> column = Axis(['c0', 'c1', 'c2'], 'column')

        Replace one axis (second argument `new_axis` must be provided)

        >>> arr.set_axes(X.a, row)
        row\\b  b0  b1  b2
           r0   0   1   2
           r1   3   4   5

        Replace several axes (keywords, list of tuple or dictionary)

        >>> arr.set_axes(a=row, b=column) # doctest: +SKIP
        >>> # or
        >>> arr.set_axes([(X.a, row), (X.b, column)]) # doctest: +SKIP
        >>> # or
        >>> arr.set_axes({X.a: row, X.b: column})
        row\\column  c0  c1  c2
                r0   0   1   2
                r1   3   4   5

        Replace all axes (list of axes or AxisCollection)

        >>> arr.set_axes([row, column])
        row\\column  c0  c1  c2
                r0   0   1   2
                r1   3   4   5
        >>> arr2 = ndtest([row, column])
        >>> arr.set_axes(arr2.axes)
        row\\column  c0  c1  c2
                r0   0   1   2
                r1   3   4   5
        """
        new_axes = self.axes.replace(axes_to_replace, new_axis, **kwargs)
        if inplace:
            if new_axes.ndim != self.ndim:
                raise ValueError("number of axes (%d) does not match number of dimensions of data (%d)"
                                 % (new_axes.ndim, self.ndim))
            if new_axes.shape != self.data.shape:
                raise ValueError("length of axes %s does not match data shape %s" % (new_axes.shape, self.data.shape))
            self.axes = new_axes
            return self
        else:
            return LArray(self.data, new_axes, title=self.title)

    with_axes = renamed_to(set_axes, 'with_axes')

    def __getattr__(self, key):
        if key in self.axes:
            return self.axes[key]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, key))

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before .axes exists, which leads to
    # an infinite recursion)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __dir__(self):
        names = set(axis.name for axis in self.axes if axis.name is not None)
        return list(set(dir(self.__class__)) | set(self.__dict__.keys()) | names)

    def _ipython_key_completions_(self):
        return list(chain(*[list(labels) for labels in self.axes.labels]))

    @property
    def i(self):
        """
        Allows selection of a subset using indices of labels.

        Examples
        --------
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a  b\\c  c0  c1  c2  c3
        a0   b0   0   1   2   3
        a0   b1   4   5   6   7
        a0   b2   8   9  10  11
        a1   b0  12  13  14  15
        a1   b1  16  17  18  19
        a1   b2  20  21  22  23

        >>> arr.i[:, 0:2, [0,2]]
         a  b\\c  c0  c2
        a0   b0   0   2
        a0   b1   4   6
        a1   b0  12  14
        a1   b1  16  18
        """
        return LArrayPositionalIndexer(self)

    @property
    def points(self):
        """
        Allows selection of arbitrary items in the array
        based on their N-dimensional label index.

        Examples
        --------
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a  b\\c  c0  c1  c2  c3
        a0   b0   0   1   2   3
        a0   b1   4   5   6   7
        a0   b2   8   9  10  11
        a1   b0  12  13  14  15
        a1   b1  16  17  18  19
        a1   b2  20  21  22  23

        To select the two points with label coordinates
        [a0, b0, c0] and [a1, b2, c2], you must do:

        >>> arr.points['a0,a1', 'b0,b2', 'c0,c2']
        a_b_c  a0_b0_c0  a1_b2_c2
                      0        22

        The number of label(s) on each dimension must be equal:

        >>> arr.points['a0,a1', 'b0,b2', 'c0,c1,c2'] # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (2,) (3,)
        """
        return LArrayPointsIndexer(self)

    @property
    def ipoints(self):
        """
        Allows selection of arbitrary items in the array based on their N-dimensional index.

        Examples
        --------
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a  b\\c  c0  c1  c2  c3
        a0   b0   0   1   2   3
        a0   b1   4   5   6   7
        a0   b2   8   9  10  11
        a1   b0  12  13  14  15
        a1   b1  16  17  18  19
        a1   b2  20  21  22  23

        To select the two points with index coordinates
        [0, 0, 0] and [1, 2, 2], you must do:

        >>> arr.ipoints[[0,1], [0,2], [0,2]]
        a_b_c  a0_b0_c0  a1_b2_c2
                      0        22

        The number of index(es) on each dimension must be equal:

        >>> arr.ipoints[[0,1], [0,2], [0,1,2]] # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (2,) (3,)
        """
        return LArrayPositionalPointsIndexer(self)

    def to_frame(self, fold_last_axis_name=False, dropna=None):
        """
        Converts LArray into Pandas DataFrame.

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

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\\c  c0  c1
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
        a  b\\c
        a0 b0    0   1
           b1    2   3
        a1 b0    4   5
           b1    6   7
        """
        columns = pd.Index(self.axes[-1].labels)
        if not fold_last_axis_name:
            columns.name = self.axes[-1].name
        if self.ndim > 1:
            axes_names = self.axes.names[:-1]
            if fold_last_axis_name:
                tmp = axes_names[-1] if axes_names[-1] is not None else ''
                if self.axes[-1].name:
                    axes_names[-1] = "{}\\{}".format(tmp, self.axes[-1].name)

            index = pd.MultiIndex.from_product(self.axes.labels[:-1], names=axes_names)
        else:
            index = pd.Index([''])
            if fold_last_axis_name:
                index.name = self.axes.names[-1]
        data = np.asarray(self).reshape(len(index), len(columns))
        df = pd.DataFrame(data, index, columns)
        if dropna is not None:
            dropna = dropna if dropna is not True else 'all'
            df.dropna(inplace=True, how=dropna)
        return df
    df = property(to_frame)

    def to_series(self, name=None, dropna=False):
        """
        Converts LArray into Pandas Series.

        Parameters
        ----------
        name : str, optional
            Name of the series. Defaults to None.
        dropna : bool, optional.
            False by default.

        Returns
        -------
        Pandas Series

        Examples
        --------
        >>> arr = ndtest((2, 3), dtype=float)
        >>> arr
        a\\b   b0   b1   b2
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

        Drop nan values

        >>> arr['b1'] = np.nan
        >>> arr
        a\\b   b0   b1   b2
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
        index = pd.MultiIndex.from_product([axis.labels for axis in self.axes], names=self.axes.names)
        series = pd.Series(np.asarray(self).reshape(self.size), index, name=name)
        if dropna:
            series.dropna(inplace=True)
        return series
    series = property(to_series)

    def describe(self, *args, **kwargs):
        """
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
        LArray

        See Also
        --------
        LArray.describe_by

        Examples
        --------
        >>> arr = LArray([0, 6, 2, 5, 4, 3, 1, 3], 'year=2013..2020')
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
        # retrieve kw-only arguments
        percentiles = kwargs.pop('percentiles', None)
        if kwargs:
            raise TypeError("describe() got an unexpected keyword argument '{}'".format(list(kwargs.keys())[0]))
        if percentiles is None:
            percentiles = [25, 50, 75]
        plabels = ['{}%'.format(p) for p in percentiles]
        labels = ['count', 'mean', 'std', 'min'] + plabels + ['max']
        percentiles = [0] + list(percentiles) + [100]
        # TODO: we should use the commented code using  *self.percentile(percentiles, *args) but this does not work
        # when *args is not empty (see https://github.com/larray-project/larray/issues/192)
        # return stack([(~np.isnan(self)).sum(*args), self.mean(*args), self.std(*args),
        #               *self.percentile(percentiles, *args)], Axis(labels, 'stats'))
        return stack([(~np.isnan(self)).sum(*args), self.mean(*args), self.std(*args)] +
                     [self.percentile(p, *args) for p in percentiles], Axis(labels, 'statistic'))

    def describe_by(self, *args, **kwargs):
        """
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
        LArray

        See Also
        --------
        LArray.describe

        Examples
        --------
        >>> data = [[0, 6, 3, 5, 4, 2, 1, 3], [7, 5, 3, 2, 8, 5, 6, 4]]
        >>> arr = LArray(data, 'gender=Male,Female;year=2013..2020').astype(float)
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
        # retrieve kw-only arguments
        percentiles = kwargs.pop('percentiles', None)
        if kwargs:
            raise TypeError("describe() got an unexpected keyword argument '{}'".format(list(kwargs.keys())[0]))
        args = self._prepare_aggregate(None, args)
        args = self._by_args_to_normal_agg_args(args)
        return self.describe(*args, percentiles=percentiles)

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

    def __array_wrap__(self, out_arr, context=None):
        """
        Called after numpy ufuncs. This is never called during our wrapped
        ufuncs, but if somebody uses raw numpy function, this works in some
        cases.
        """
        data = np.ndarray.__array_wrap__(self.data, out_arr, context)
        return LArray(data, self.axes)

    def __bool__(self):
        return bool(self.data)
    # Python 2
    __nonzero__= __bool__

    def rename(self, renames=None, to=None, inplace=False, **kwargs):
        """Renames axes of the array.

        Parameters
        ----------
        renames : axis ref or dict {axis ref: str} or list of tuple (axis ref, str)
            Renames to apply. If a single axis reference is given, the `to` argument must be used.
        to : str or Axis
            New name if `renames` contains a single axis reference.
        **kwargs : str or Axis
            New name for each axis given as a keyword argument.

        Returns
        -------
        LArray
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
        nat\\sex  M  F
             BE  0  1
             FO  2  3
        >>> arr.rename(X.nat, 'nat2')
        nat2\\sex  M  F
              BE  0  1
              FO  2  3
        >>> arr.rename(nat='nat2', sex='sex2')
        nat2\\sex2  M  F
               BE  0  1
               FO  2  3
        >>> arr.rename([('nat', 'nat2'), ('sex', 'sex2')])
        nat2\\sex2  M  F
               BE  0  1
               FO  2  3
        >>> arr.rename({'nat': 'nat2', 'sex': 'sex2'})
        nat2\\sex2  M  F
               BE  0  1
               FO  2  3
        """
        if isinstance(renames, dict):
            items = list(renames.items())
        elif isinstance(renames, list):
            items = renames[:]
        elif isinstance(renames, (str, Axis, int)):
            items = [(renames, to)]
        else:
            items = []
        items += kwargs.items()
        renames = {self.axes[k]: v for k, v in items}
        axes = [a.rename(renames[a]) if a in renames else a
                for a in self.axes]
        if inplace:
            self.axes = AxisCollection(axes)
            return self
        else:
            return LArray(self.data, axes)

    def reindex(self, axes_to_reindex=None, new_axis=None, fill_value=np.nan, inplace=False, **kwargs):
        """Reorder and/or add new labels in axes.

        Place NaN or given `fill_value` in locations having no value previously.

        Parameters
        ----------
        axes_to_reindex : axis ref or dict {axis ref: axis} or list of tuple (axis ref, axis) \
                          or list of Axis or AxisCollection
            Axes to reindex. If a single axis reference is given, the `new_axis` argument must be provided.
            If a list of Axis or an AxisCollection is given, existing axes are reindexed while missing ones are added.
        new_axis : int, str, list/tuple/array of str, Group or Axis, optional
            List of new labels or new axis if `axes_to_replace` contains a single axis reference.
        fill_value : scalar or LArray, optional
            Value used to fill cells corresponding to label combinations which were not present before reindexing.
            Defaults to NaN.
        inplace : bool, optional
            Whether or not to modify the original object or return a new array and leave the original intact.
            Defaults to False.
        **kwargs : Axis
            New axis for each axis to reindex given as a keyword argument.

        Returns
        -------
        LArray
            Array with reindexed axes.

        Notes
        -----
        When introducing NAs into an array containing integers via reindex,
        all data will be promoted to float in order to store the NAs.

        Examples
        --------
        >>> arr = ndtest((2, 2))
        >>> arr
        a\\b  b0  b1
         a0   0   1
         a1   2   3
        >>> arr2 = ndtest('a=a1,a2;c=c0;b=b2..b0')
        >>> arr2
         a  c\\b  b2  b1  b0
        a1   c0   0   1   2
        a2   c0   3   4   5

        Reindex an axis by passing labels (list or string)

        >>> arr.reindex('b', ['b1', 'b2', 'b0'])
        a\\b   b1   b2   b0
         a0  1.0  nan  0.0
         a1  3.0  nan  2.0
        >>> arr.reindex('b', 'b0..b2', fill_value=-1)
        a\\b  b0  b1  b2
         a0   0   1  -1
         a1   2   3  -1

        Reindex using an axis from another array

        >>> arr.reindex('b', arr2.b, fill_value=-1)
        a\\b  b2  b1  b0
         a0  -1   1   0
         a1  -1   3   2

        Reindex using a subset of an axis

       >>> arr.reindex('b', arr2.b['b1':], fill_value=-1)
       a\\b  b1  b0
        a0   1   0
        a1   3   2

        Reindex several axes

        >>> arr.reindex({'a': arr2.a, 'b': arr2.b}, fill_value=-1)
        a\\b  b2  b1  b0
         a1  -1   3   2
         a2  -1  -1  -1
        >>> arr.reindex({'a': arr2.a, 'b': arr2.b['b1':]}, fill_value=-1)
        a\\b  b1  b0
         a1   3   2
         a2  -1  -1

        Reindex by passing a collection of axes

        >>> arr.reindex(arr2.axes, fill_value=-1)
         a  b\\c  c0
        a1   b2  -1
        a1   b1   3
        a1   b0   2
        a2   b2  -1
        a2   b1  -1
        a2   b0  -1
        >>> arr2.reindex(arr.axes, fill_value=-1)
         a  c\\b  b0  b1
        a0   c0  -1  -1
        a1   c0   2   1
        """
        # XXX: can't we move this to AxisCollection.replace?
        if new_axis is not None and not isinstance(new_axis, Axis):
            new_axis = Axis(new_axis, self.axes[axes_to_reindex].name)
        elif isinstance(new_axis, Axis):
            new_axis = new_axis.rename(self.axes[axes_to_reindex].name)
        if isinstance(axes_to_reindex, (list, tuple)) and all([isinstance(axis, Axis) for axis in axes_to_reindex]):
            axes_to_reindex = AxisCollection(axes_to_reindex)
        if isinstance(axes_to_reindex, AxisCollection):
            assert new_axis is None
            # add extra axes if needed
            res_axes = AxisCollection([axes_to_reindex.get(axis, axis) for axis in self.axes]) | axes_to_reindex
        else:
            res_axes = self.axes.replace(axes_to_reindex, new_axis, **kwargs)
        res = full(res_axes, fill_value, dtype=common_type((self.data, fill_value)))
        def get_labels(self_axis):
            res_axis = res_axes[self_axis]
            if res_axis.equals(self_axis):
                return self_axis[:]
            else:
                return self_axis[self_axis.intersection(res_axis).labels]
        self_labels = tuple(get_labels(axis) for axis in self.axes)
        res_labels = tuple(res_axes[group.axis][group] for group in self_labels)
        res[res_labels] = self[self_labels]
        if inplace:
            self.axes = res.axes
            self.data = res.data
            return self
        else:
            return res

    def align(self, other, join='outer', fill_value=nan, axes=None):
        """Align two arrays on their axes with the specified join method.

        In other words, it ensure all common axes are compatible. Those arrays can then be used in binary operations.

        Parameters
        ----------
        other : LArray-like
        join : {'outer', 'inner', 'left', 'right'}, optional
            Join method. For each axis common to both arrays:
              - outer: will use a label if it is in either arrays axis (ordered like the first array).
                       This is the default as it results in no information loss.
              - inner: will use a label if it is in both arrays axis (ordered like the first array)
              - left: will use the first array axis labels
              - right: will use the other array axis labels.
        fill_value : scalar or LArray, optional
            Value used to fill cells corresponding to label combinations which are not common to both arrays.
            Defaults to NaN.
        axes : AxisReference or sequence of them, optional
            Axes to align. Need to be valid in both arrays. Defaults to None (all common axes). This must be specified
            when mixing anonymous and non-anonymous axes.

        Returns
        -------
        (left, right) : (LArray, LArray)
            Aligned objects

        Notes
        -----
            Arrays with anonymous axes are currently not supported.

        Examples
        --------
        >>> arr1 = ndtest((2, 3))
        >>> arr1
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr2 = -ndtest((3, 2))
        >>> # reorder array to make the test more interesting
        >>> arr2 = arr2[['b1', 'b0']]
        >>> arr2
        a\\b  b1  b0
         a0  -1   0
         a1  -3  -2
         a2  -5  -4

        Align arr1 and arr2

        >>> aligned1, aligned2 = arr1.align(arr2)
        >>> aligned1
        a\\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
         a2  nan  nan  nan
        >>> aligned2
        a\\b    b0    b1   b2
         a0   0.0  -1.0  nan
         a1  -2.0  -3.0  nan
         a2  -4.0  -5.0  nan

        After aligning all common axes, one can then do operations between the two arrays

        >>> aligned1 + aligned2
        a\\b   b0   b1   b2
         a0  0.0  0.0  nan
         a1  1.0  1.0  nan
         a2  nan  nan  nan

        Other kinds of joins are supported

        >>> aligned1, aligned2 = arr1.align(arr2, join='inner')
        >>> aligned1
        a\\b   b0   b1
         a0  0.0  1.0
         a1  3.0  4.0
        >>> aligned2
        a\\b    b0    b1
         a0   0.0  -1.0
         a1  -2.0  -3.0
        >>> aligned1, aligned2 = arr1.align(arr2, join='left')
        >>> aligned1
        a\\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
        >>> aligned2
        a\\b    b0    b1   b2
         a0   0.0  -1.0  nan
         a1  -2.0  -3.0  nan
        >>> aligned1, aligned2 = arr1.align(arr2, join='right')
        >>> aligned1
        a\\b   b1   b0
         a0  1.0  0.0
         a1  4.0  3.0
         a2  nan  nan
        >>> aligned2
        a\\b    b1    b0
         a0  -1.0   0.0
         a1  -3.0  -2.0
         a2  -5.0  -4.0

        The fill value for missing labels defaults to nan but can be changed to any compatible value.

        >>> aligned1, aligned2 = arr1.align(arr2, fill_value=0)
        >>> aligned1
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   0   0   0
        >>> aligned2
        a\\b  b0  b1  b2
         a0   0  -1   0
         a1  -2  -3   0
         a2  -4  -5   0
        >>> aligned1 + aligned2
        a\\b  b0  b1  b2
         a0   0   0   2
         a1   1   1   5
         a2  -4  -5   0

        It also works when either arrays (or both) have extra axes

        >>> arr3 = ndtest((3, 2, 2))
        >>> arr1
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr3
         a  b\\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        a2   b0   8   9
        a2   b1  10  11
        >>> aligned1, aligned2 = arr1.align(arr3, join='inner')
        >>> aligned1
        a\\b   b0   b1
         a0  0.0  1.0
         a1  3.0  4.0
        >>> aligned2
         a  b\c   c0   c1
        a0   b0  0.0  1.0
        a0   b1  2.0  3.0
        a1   b0  4.0  5.0
        a1   b1  6.0  7.0
        >>> aligned1 + aligned2
         a  b\\c    c0    c1
        a0   b0   0.0   1.0
        a0   b1   3.0   4.0
        a1   b0   7.0   8.0
        a1   b1  10.0  11.0

        One can also align only some specific axes (but in that case arrays might not be compatible)

        >>> aligned1, aligned2 = arr1.align(arr2, axes='b')
        >>> aligned1
        a\\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  4.0  5.0
        >>> aligned2
        a\\b    b0    b1   b2
         a0   0.0  -1.0  nan
         a1  -2.0  -3.0  nan
         a2  -4.0  -5.0  nan
        """
        other = aslarray(other)
        # reindex does not currently support anonymous axes
        if any(name is None for name in self.axes.names) or any(name is None for name in other.axes.names):
            raise ValueError("arrays with anonymous axes are currently not supported by LArray.align")
        left_axes, right_axes = self.axes.align(other.axes, join=join, axes=axes)
        return self.reindex(left_axes, fill_value=fill_value), other.reindex(right_axes, fill_value=fill_value)

    @deprecate_kwarg('reverse', 'ascending', {True: False, False: True})
    def sort_values(self, key=None, axis=None, ascending=True):
        """Sorts values of the array.

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
        LArray
            Array with sorted values.

        Examples
        --------
        sort the whole array (no key or axis given)

        >>> arr_1D = LArray([10, 2, 4], 'a=a0..a2')
        >>> arr_1D
        a  a0  a1  a2
           10   2   4
        >>> arr_1D.sort_values()
        a  a1  a2  a0
            2   4  10
        >>> arr_2D = LArray([[10, 2, 4], [3, 7, 1]], 'a=a0,a1; b=b0..b2')
        >>> arr_2D
        a\\b  b0  b1  b2
         a0  10   2   4
         a1   3   7   1
        >>> # if the array has more than one dimension, sort array with all axes combined
        >>> arr_2D.sort_values()
        a_b  a1_b2  a0_b1  a1_b0  a0_b2  a1_b1  a0_b0
                 1      2      3      4      7     10

        Sort along a given key

        >>> # sort columns according to the values of the row associated with the label 'a1'
        >>> arr_2D.sort_values('a1')
        a\\b  b2  b0  b1
         a0   4  10   2
         a1   1   3   7
        >>> arr_2D.sort_values('a1', ascending=False)
        a\\b  b1  b0  b2
         a0   2  10   4
         a1   7   3   1
        >>> arr_3D = LArray([[[10, 2, 4], [3, 7, 1]], [[5, 1, 6], [2, 8, 9]]],
        ...            'a=a0,a1; b=b0,b1; c=c0..c2')
        >>> arr_3D
         a  b\\c  c0  c1  c2
        a0   b0  10   2   4
        a0   b1   3   7   1
        a1   b0   5   1   6
        a1   b1   2   8   9
        >>> # sort columns according to the values of the row associated with the labels 'a0' and 'b1'
        >>> arr_3D.sort_values(('a0', 'b1'))
         a  b\\c  c2  c0  c1
        a0   b0   4  10   2
        a0   b1   1   3   7
        a1   b0   6   5   1
        a1   b1   9   2   8

        Sort along an axis

        >>> arr_2D
        a\\b  b0  b1  b2
         a0  10   2   4
         a1   3   7   1
        >>> # sort values along axis 'a'
        >>> # equivalent to sorting the values of each column of the array
        >>> arr_2D.sort_values(axis='a')
        a*\\b  b0  b1  b2
           0   3   2   1
           1  10   7   4
        >>> # sort values along axis 'b'
        >>> # equivalent to sorting the values of each row of the array
        >>> arr_2D.sort_values(axis='b')
        a\\b*  0  1   2
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
            res = LArray(data, new_axes)
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
            # which sorts the *data* correctly, but the labels on the nat axis are not sorted (because the __getitem__ in
            # that case reuse the key axis as-is -- like it should).
            # Both use cases have value, but I think reordering the ticks should be the default. Now, I am unsure where to
            # change this. Probably in IGroupMaker.__getitem__, but then how do I get the "not reordering labels" behavior
            # that I have now?
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
    def sort_axes(self, axes=None, ascending=True):
        """Sorts axes of the array.

        Parameters
        ----------
        axes : axis reference (Axis, str, int) or list of them, optional
            Axes to sort. Defaults to all axes.
        ascending : bool, optional
            Sort axes in ascending order. Defaults to True.

        Returns
        -------
        LArray
            Array with sorted axes.

        Examples
        --------
        >>> a = ndtest("nat=EU,FO,BE; sex=M,F")
        >>> a
        nat\\sex  M  F
             EU  0  1
             FO  2  3
             BE  4  5
        >>> a.sort_axes('sex')
        nat\\sex  F  M
             EU  1  0
             FO  3  2
             BE  5  4
        >>> a.sort_axes()
        nat\\sex  F  M
             BE  5  4
             EU  1  0
             FO  3  2
        >>> a.sort_axes(('sex', 'nat'))
        nat\\sex  F  M
             BE  5  4
             EU  1  0
             FO  3  2
        >>> a.sort_axes(ascending=False)
        nat\\sex  M  F
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

    sort_axis = renamed_to(sort_axes, 'sort_axis')

    def _translate_axis_key_chunk(self, axis_key, bool_passthrough=True):
        """
        Translates axis(es) key into axis(es) position(s).

        Parameters
        ----------
        axis_key : any kind of key
            Key to select axis(es).
        bool_passthrough : bool, optional
            True by default.

        Returns
        -------
        IGroup
            Positional group with valid axes (from self.axes)
        """
        axis_key = remove_nested_groups(axis_key)

        if isinstance(axis_key, Group) and axis_key.axis is not None:
            # retarget to real axis, if needed
            # only retarget IGroup and not LGroup to give the opportunity for axis.translate to try the "ticks"
            # version of the group ONLY if key.axis is not real_axis (for performance reasons)
            if isinstance(axis_key, IGroup):
                if axis_key.axis in self.axes:
                    axis_key = axis_key.retarget_to(self.axes[axis_key.axis])
                else:
                    # axis associated with axis_key may not belong to self.
                    # In that case, we translate IGroup to labels and search for a compatible axis
                    # (see end of this method)
                    axis_key = axis_key.to_label()

        # already positional
        if isinstance(axis_key, IGroup):
            if axis_key.axis is None:
                raise ValueError("positional groups without axis are not supported")
            return axis_key

        # labels but known axis
        if isinstance(axis_key, LGroup) and axis_key.axis is not None:
            try:
                real_axis = self.axes[axis_key.axis]
                try:
                    axis_pos_key = real_axis.index(axis_key, bool_passthrough)
                except KeyError:
                    raise ValueError("%r is not a valid label for any axis" % axis_key)
                return real_axis.i[axis_pos_key]
            except KeyError:
                # axis associated with axis_key may not belong to self.
                # In that case, we translate LGroup to labels and search for a compatible axis
                # (see end of this method)
                axis_key = axis_key.to_label()

        # otherwise we need to guess the axis
        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        # TODO: use axis_key dtype to only check compatible axes
        for axis in self.axes:
            try:
                axis_pos_key = axis.index(axis_key, bool_passthrough)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis" % axis_key)
        elif len(valid_axes) > 1:
            # TODO: make an AxisCollection.display_name(axis) method out of this
            # valid_axes = ', '.join(self.axes.display_name(axis) for a in valid_axes)
            valid_axes = ', '.join(a.name if a.name is not None else '{{{}}}'.format(self.axes.index(a))
                                   for a in valid_axes)
            raise ValueError('%s is ambiguous (valid in %s)' % (axis_key, valid_axes))
        return valid_axes[0].i[axis_pos_key]

    def _translate_axis_key(self, axis_key, bool_passthrough=True):
        """Same as chunk.

        Returns
        -------
        IGroup
            Positional group with valid axes (from self.axes)
        """
        if isinstance(axis_key, ExprNode):
            axis_key = axis_key.evaluate(self.axes)

        if isinstance(axis_key, LArray) and np.issubdtype(axis_key.dtype, np.bool_) and bool_passthrough:
            if len(axis_key.axes) > 1:
                raise ValueError("mixing ND boolean filters with other filters in getitem is not currently supported")
            else:
                return IGroup(axis_key.nonzero()[0], axis=axis_key.axes[0])

        # translate Axis keys to LGroup keys
        # FIXME: this should be simply:
        # if isinstance(axis_key, Axis):
        #     axis_key = axis_key[:]
        # but it does not work for some reason (the retarget does not seem to happen)
        if isinstance(axis_key, Axis):
            real_axis = self.axes[axis_key]
            if isinstance(axis_key, AxisReference) or axis_key.equals(real_axis):
                axis_key = real_axis[:]
            else:
                axis_key = axis_key.labels

        # TODO: do it for Group without axis too
        if isinstance(axis_key, (tuple, list, np.ndarray, LArray)):
            axis = None
            # TODO: I should actually do some benchmarks to see if this is useful, and estimate which numbers to use
            for size in (1, 10, 100, 1000):
                # TODO: do not recheck already checked elements
                key_chunk = axis_key.i[:size] if isinstance(axis_key, LArray) else axis_key[:size]
                try:
                    tkey = self._translate_axis_key_chunk(key_chunk, bool_passthrough)
                    axis = tkey.axis
                    break
                except ValueError:
                    continue
            # the (start of the) key match a single axis
            if axis is not None:
                # make sure we have an Axis object
                # TODO: we should make sure the tkey returned from _translate_axis_key_chunk always contains a
                # real Axis (and thus kill this line)
                axis = self.axes[axis]
                # wrap key in LGroup
                axis_key = axis[axis_key]
                # XXX: reuse tkey chunks and only translate the rest?
            return self._translate_axis_key_chunk(axis_key, bool_passthrough)
        else:
            return self._translate_axis_key_chunk(axis_key, bool_passthrough)

    def _guess_axis(self, axis_key):
        if isinstance(axis_key, Group):
            group_axis = axis_key.axis
            if group_axis is not None:
                # we have axis information but not necessarily an Axis object from self.axes
                real_axis = self.axes[group_axis]
                if group_axis is not real_axis:
                    axis_key = axis_key.with_axis(real_axis)
                return axis_key

        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # or possibly (for ambiguous labels)
        # label -> {axis: index}
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        for axis in self.axes:
            try:
                axis.index(axis_key)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis" % axis_key)
        elif len(valid_axes) > 1:
            valid_axes = ', '.join(a.name if a.name is not None else '{{{}}}'.format(self.axes.index(a))
                                   for a in valid_axes)
            raise ValueError('%s is ambiguous (valid in %s)' % (axis_key, valid_axes))
        return valid_axes[0][axis_key]

    # TODO: move this to AxisCollection
    def _translated_key(self, key, bool_stuff=False):
        """Completes and translates key

        Parameters
        ----------
        key : single axis key or tuple of keys or dict {axis_name: axis_key}
           Each axis key can be either a scalar, a list of scalars or an LGroup.

        Returns
        -------
        Returns a full N dimensional positional key.
        """

        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.bool_) and not bool_stuff:
            return key.nonzero()
        if isinstance(key, LArray) and np.issubdtype(key.dtype, np.bool_) and not bool_stuff:
            # if only the axes order is wrong, transpose
            # FIXME: if the key has both missing and extra axes, it could be the correct size (or even shape, see below)
            if key.size == self.size and key.shape != self.shape:
                return np.asarray(key.transpose(self.axes)).nonzero()
            # otherwise we need to transform the key to integer
            elif key.size != self.size:
                extra_key_axes = key.axes - self.axes
                if extra_key_axes:
                    raise ValueError("subset key %s contains more axes than array %s" % (key.axes, self.axes))

                # do I want to allow key_axis.name to match against axis.num? does not seem like a good idea.
                # but this should work
                # >>> a = ndtest((3, 4))
                # >>> x1, x2 = a.axes
                # >>> a[x2 > 2]

                # the current solution with hash = (labels, name) works but is slow for large axes and broken if axis
                # labels are modified in-place, which I am unsure I want to support anyway
                self.axes.check_compatible(key.axes)
                local_axes = [self.axes[axis] for axis in key.axes]
                map_key = dict(zip(local_axes, np.asarray(key).nonzero()))
                return tuple(map_key.get(axis, slice(None)) for axis in self.axes)
            else:
                # correct shape
                # FIXME: if the key has both missing and extra axes (at the index of the missing axes), the shape
                # could be the same while the result should not
                return np.asarray(key).nonzero()

        # convert scalar keys to 1D keys
        if not isinstance(key, (tuple, dict)):
            key = (key,)

        if isinstance(key, tuple):
            # drop slice(None) and Ellipsis since they are meaningless because of guess_axis.
            # XXX: we might want to raise an exception when we find Ellipses or (most) slice(None) because except for
            #      a single slice(None) a[:], I don't think there is any point.
            key = [axis_key for axis_key in key
                   if not _isnoneslice(axis_key) and axis_key is not Ellipsis]

            # translate all keys to IGroup
            key = [self._translate_axis_key(axis_key, bool_passthrough=not bool_stuff)
                   for axis_key in key]

            assert all(isinstance(axis_key, IGroup) for axis_key in key)

            # extract axis from Group keys
            key_items = [(k.axis, k) for k in key]
        else:
            # key axes could be strings or axis references and we want real axes
            key_items = [(self.axes[k], v) for k, v in key.items()]
            # TODO: use _translate_axis_key (to translate to IGroup here too)
            # key_items = [axis.translate(axis_key, bool_passthrough=not bool_stuff)
            #              for axis, axis_key in key_items]

        # even keys given as dict can contain duplicates (if the same axis was
        # given under different forms, e.g. name and AxisReference).
        dupe_axes = list(duplicates(axis for axis, axis_key in key_items))
        if dupe_axes:
            dupe_axes = ', '.join(str(axis) for axis in dupe_axes)
            raise ValueError("key has several values for axis: %s" % dupe_axes)

        key = dict(key_items)

        # dict -> tuple (complete and order key)
        assert all(isinstance(k, Axis) for k in key)
        key = [key[axis] if axis in key else slice(None)
               for axis in self.axes]

        # IGroup -> raw positional
        return tuple(axis.index(axis_key, bool_passthrough=not bool_stuff)
                     for axis, axis_key in zip(self.axes, key))

    # TODO: we only need axes length => move this to AxisCollection
    # (but this backend/numpy-specific so we'll probably need to create a subclass of it)
    def _cross_key(self, key):
        """
        Returns a key indexing the cross product.

        Parameters
        ----------
        key : complete (contains all dimensions) index-based key.

        Returns
        -------
        key
            A key for indexing the cross product.
        """

        # handle advanced indexing with more than one indexing array: basic indexing (only integer and slices) and
        # advanced indexing with only one indexing array are handled fine by numpy
        if self._needs_advanced_indexing(key):
            # np.ix_ wants only lists so:

            # 1) transform scalar-key to lists of 1 element. In that case, ndarray.__getitem__ leaves length 1
            #    dimensions instead of dropping them like we would like, so we will need to drop them later ourselves
            #    (via reshape)
            noscalar_key = [[axis_key] if np.isscalar(axis_key) else axis_key
                            for axis_key in key]

            # 2) expand slices to lists (ranges)
            # XXX: cache the range in the axis?
            # TODO: fork np.ix_ to allow for slices directly
            # it will be tricky to get right though because in that case the result of a[key] can have its dimensions in
            # the wrong order (if the ix_arrays are not next to each other, the corresponding dimensions are moved to
            # the front). It is probably worth the trouble though because it is much faster than the current solution
            # (~5x in my simple test) but this case (num_ix_arrays > 1) is rare in the first place (at least in demo)
            # so it is not a priority.
            listkey = tuple(np.arange(*axis_key.indices(len(axis))) if isinstance(axis_key, slice) else axis_key
                            for axis_key, axis in zip(noscalar_key, self.axes))
            # np.ix_ computes the cross product of all lists
            return np.ix_(*listkey)
        else:
            return tuple(key)

    def _needs_advanced_indexing(self, key):
        sequence = (tuple, list, np.ndarray)
        # count number of indexing arrays (ie non scalar/slices) in tuple
        num_ix_arrays = sum(isinstance(axis_key, sequence) for axis_key in key)
        num_scalars = sum(np.isscalar(axis_key) for axis_key in key)
        num_slices = sum(isinstance(axis_key, slice) for axis_key in key)
        assert len(key) == num_ix_arrays + num_scalars + num_slices
        return num_ix_arrays > 1 or (num_ix_arrays > 0 and num_scalars)

    def _collapse_slices(self, key):
        # isinstance(ndarray, collections.Sequence) is False but it
        # behaves like one
        sequence = (tuple, list, np.ndarray)
        return [_range_to_slice(axis_key, len(axis)) if isinstance(axis_key, sequence) else axis_key
                for axis_key, axis in zip(key, self.axes)]

    def _get_axes_from_translated_key(self, translated_key, include_scalar_axis_key=False):
        if include_scalar_axis_key:
            return [axis.subaxis(axis_key) if not np.isscalar(axis_key) else Axis(1, axis.name)
                    for axis, axis_key in zip(self.axes, translated_key)]
        else:
            return [axis.subaxis(axis_key)
                    for axis, axis_key in zip(self.axes, translated_key)
                    if not np.isscalar(axis_key)]

    def __getitem__(self, key, collapse_slices=False):

        if isinstance(key, ExprNode):
            key = key.evaluate(self.axes)

        data = np.asarray(self.data)
        # XXX: I think I should split this into complete_key and translate_key because for LArray keys I need a
        #      complete key with axes for subaxis
        translated_key = self._translated_key(key)

        # FIXME: I have a huge problem with boolean labels + non points
        if isinstance(key, (LArray, np.ndarray)) and np.issubdtype(key.dtype, np.bool_):
            return LArray(data[translated_key], self._bool_key_new_axes(translated_key))

        if any(isinstance(axis_key, LArray) for axis_key in translated_key):
            k2 = [k.data if isinstance(k, LArray) else k
                  for k in translated_key]
            res_data = data[k2]
            axes = self._get_axes_from_translated_key(translated_key)
            first_col = AxisCollection(axes[0])
            res_axes = first_col.union(*axes[1:])
            return LArray(res_data, res_axes)

        # TODO: if the original key was a list of labels, subaxis(translated_key).labels == orig_key, so we should use
        #       orig_axis_key.copy()
        axes = self._get_axes_from_translated_key(translated_key)

        if collapse_slices:
            translated_key = self._collapse_slices(translated_key)
        cross_key = self._cross_key(translated_key)
        data = data[cross_key]
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            # drop length 1 dimensions created by scalar keys
            res_data = data.reshape(tuple(len(axis) for axis in axes))
            assert _equal_modulo_len1(data.shape, res_data.shape)
            return LArray(res_data, axes)

    def __setitem__(self, key, value, collapse_slices=True):
        # TODO: if key or value has more axes than self, we should use
        # total_axes = self.axes + key.axes + value.axes
        # expanded = self.expand(total_axes)
        # data = np.asarray(expanded.data)

        # concerning keys this can make sense in several cases:
        # single bool LArray key with extra axes.
        # tuple of bool LArray keys (eg one for each axis). each could have extra axes. Common axes between keys are
        # not a problem, we can simply "and" them. Though we should avoid explicitly "and"ing them if there is no
        # common axis because that is less efficient than the implicit "and" that is done by numpy __getitem__ (and
        # the fact we need to combine dimensions when any key has more than 1 dim).

        # the bool value represents whether the axis label is taken or not if any bool key (part) has more than one
        # axis, we get combined dimensions out of it.

        # int LArray keys
        # the int value represent an index along ONE particular axis, even if the key has more than one axis.
        if isinstance(key, ExprNode):
            key = key.evaluate(self.axes)

        data = np.asarray(self.data)
        translated_key = self._translated_key(key)

        if isinstance(key, (LArray, np.ndarray)) and np.issubdtype(key.dtype, np.bool_):
            if isinstance(value, LArray):
                new_axes = self._bool_key_new_axes(translated_key, wildcard_allowed=True)
                value = value.broadcast_with(new_axes)
            data[translated_key] = value
            return

        if collapse_slices:
            translated_key = self._collapse_slices(translated_key)
        cross_key = self._cross_key(translated_key)

        if isinstance(value, LArray):
            # XXX: we might want to create fakes (or wildcard?) axes in this case, as we only use axes names and axes
            # length, not the ticks, and those could theoretically take a significant time to compute
            if self._needs_advanced_indexing(translated_key):
                # when adv indexing is needed, cross_key converts scalars to lists of 1 element, which does not remove
                # the dimension like scalars normally do
                axes = self._get_axes_from_translated_key(translated_key, True)
            else:
                axes = self._get_axes_from_translated_key(translated_key)
            value = value.broadcast_with(axes)
            value.axes.check_compatible(axes)

            # replace incomprehensible error message "could not broadcast input array from shape XX into shape YY"
            # for users by "incompatible axes"
            extra_axes = [axis for axis in value.axes - axes if len(axis) > 1]
            if extra_axes:
                extra_axes = AxisCollection(extra_axes)
                axes = AxisCollection(axes)
                text = 'axes are' if len(extra_axes) > 1 else 'axis is'
                raise ValueError("Value {!s} {} not present in target subset {!s}. A value can only have the same axes "
                                 "or fewer axes than the subset being targeted".format(extra_axes, text, axes))
        else:
            # if value is a "raw" ndarray we rely on numpy broadcasting
            pass

        data[cross_key] = value

    def _bool_key_new_axes(self, key, wildcard_allowed=False, sep='_'):
        """
        Returns an AxisCollection containing combined axes.
        Axes corresponding to scalar key are dropped.

        This method is used in case of boolean key.

        Parameters
        ----------
        key : tuple
            Position-based key
        wildcard_allowed : bool

        Returns
        -------
        AxisCollection

        Notes
        -----
        See examples of properties `points` and `ipoints`.
        """
        # TODO: use AxisCollection.combine_axes. The problem is that it uses product(*axes_labels)
        #       while here we need zip(*axes_labels)
        combined_axes = [axis for axis_key, axis in zip(key, self.axes)
                         if not _isnoneslice(axis_key) and
                            not np.isscalar(axis_key)]
        # scalar axes are not taken, since we want to kill them
        other_axes = [axis for axis_key, axis in zip(key, self.axes)
                      if _isnoneslice(axis_key)]
        assert len(key) > 0
        axes_indices = [self.axes.index(axis) for axis in combined_axes]
        diff = np.diff(axes_indices)
        # this can happen if key has only None slices and scalars
        if not len(combined_axes):
            combined_axis_pos = None
        elif np.any(diff > 1):
            # combined axes in front
            combined_axis_pos = 0
        else:
            combined_axis_pos = axes_indices[0]
        # all anonymous axes => anonymous combined axis
        if all(axis.name is None for axis in combined_axes):
            combined_name = None
        else:
            combined_name = sep.join(str(self.axes.axis_id(axis)) for axis in combined_axes)
        new_axes = other_axes
        if combined_axis_pos is not None:
            if wildcard_allowed:
                lengths = [len(axis_key) for axis_key in key
                           if not _isnoneslice(axis_key) and not np.isscalar(axis_key)]
                combined_axis_len = lengths[0]
                assert all(l == combined_axis_len for l in lengths)
                combined_axis = Axis(combined_axis_len, combined_name)
            else:
                # TODO: the combined keys should be objects which display as:
                # (axis1_label, axis2_label, ...) but which should also store
                # the axis (names?)
                # Q: Should it be the same object as the NDLGroup?/NDKey?
                # A: yes, probably. On the Pandas backend, we could/should have
                #    separate axes. On the numpy backend we cannot.
                axes_labels = [axis.labels[axis_key]
                               for axis_key, axis in zip(key, self.axes)
                               if not _isnoneslice(axis_key) and not np.isscalar(axis_key)]
                if len(combined_axes) == 1:
                    # Q: if axis is a wildcard axis, should the result be a
                    #    wildcard axis (and axes_labels discarded?)
                    combined_labels = axes_labels[0]
                else:
                    combined_labels = [sep.join(str(l) for l in comb)
                                       for comb in zip(*axes_labels)]

                # CRAP, this can lead to duplicate labels (especially using .points)
                combined_axis = Axis(combined_labels, combined_name)
            new_axes.insert(combined_axis_pos, combined_axis)
        return AxisCollection(new_axes)

    def set(self, value, **kwargs):
        """
        Sets a subset of array to value.

        * all common axes must be either of length 1 or the same length
        * extra axes in value must be of length 1
        * extra axes in current array can have any length

        Parameters
        ----------
        value : scalar or LArray

        Examples
        --------
        >>> arr = ndtest((3, 3))
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   6   7   8
        >>> arr['a1:', 'b1:'].set(10)
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3  10  10
         a2   6  10  10
        >>> arr['a1:', 'b1:'].set(ndtest("a=a1,a2;b=b1,b2"))
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   0   1
         a2   6   2   3
        """
        self.__setitem__(kwargs, value)

    def reshape(self, target_axes):
        """
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
        LArray
            New array with new axes but same data.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        >>> new_arr = arr.reshape([Axis('a=a0,a1'),
        ... Axis(['b0c0', 'b0c1', 'b1c0', 'b1c1'], 'bc')])
        >>> new_arr
        a\\bc  b0c0  b0c1  b1c0  b1c1
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
        data = np.asarray(self).reshape([len(axis) for axis in target_axes])
        return LArray(data, target_axes)

    def reshape_like(self, target):
        """
        Same as reshape but with an array as input.
        Total size (= number of stored data) of the two arrays must be equal.

        See Also
        --------
        reshape : returns a LArray with a new shape given a list of axes.

        Examples
        --------
        >>> arr = zeros((2, 2, 2), dtype=int)
        >>> arr
        {0}*  {1}*\\{2}*  0  1
           0          0  0  0
           0          1  0  0
           1          0  0  0
           1          1  0  0
        >>> new_arr = arr.reshape_like(ndtest((2, 4)))
        >>> new_arr
        a\\b  b0  b1  b2  b3
         a0   0   0   0   0
         a1   0   0   0   0
        """
        return self.reshape(target.axes)

    def broadcast_with(self, target):
        """
        Returns an array that is (NumPy) broadcastable with target.

        * all common axes must be either of length 1 or the same length
        * extra axes in source can have any length and will be moved to the
          front
        * extra axes in target can have any length and the result will have axes
          of length 1 for those axes

        This is different from reshape which ensures the result has exactly the
        shape of the target.

        Parameters
        ----------
        target : LArray or collection of Axis

        Returns
        -------
        LArray
        """
        if isinstance(target, LArray):
            target_axes = target.axes
        else:
            target_axes = target
            if not isinstance(target, AxisCollection):
                target_axes = AxisCollection(target_axes)
        if self.axes == target_axes:
            return self

        target_axes = (self.axes - target_axes) | target_axes

        # XXX: this breaks la['1,5,9'] = la['2,7,3']
        # but that use case should use drop_labels
        # self.axes.check_compatible(target_axes)

        # 1) reorder axes to target order
        array = self.transpose(target_axes & self.axes)

        # 2) add length one axes
        return array.reshape(array.axes.get_all(target_axes))

    # XXX: I wonder if effectively dropping the labels is necessary or not
    # we could perfectly only mark the axis as being a wildcard axis and keep
    # the labels intact. These wildcard axes with labels
    # could be useful in a few situations. For example, Excel sheets could
    # have such behavior: you can slice columns using letters, but that
    # wouldn't prevent doing computation between arrays using different
    # columns. On the other hand, it makes wild axes less obvious and I
    # wonder if there would be a risk of wildcard axes inadvertently leaking.
    # plus it might be confusing if incompatible labels "work".
    def drop_labels(self, axes=None):
        """Drops the labels from axes (replace those axes by "wildcard" axes).

        Useful when you want to apply operations between two arrays
        or subarrays with same shape but incompatible axes
        (different labels).

        Parameters
        ----------
        axes : Axis or list/tuple/AxisCollection of Axis, optional
            Axis(es) on which you want to drop the labels.

        Returns
        -------
        LArray

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
        a\\b  b1  b2
         a1   0   1
         a2   2   3
        >>> arr1.drop_labels(b)
        a\\b*  0  1
          a1  0  1
          a2  2  3
        >>> arr1.drop_labels([a, b])
        a*\\b*  0  1
            0  0  1
            1  2  3
        >>> arr2 = ndtest([a, b2])
        >>> arr2
        a\\b  b2  b3
         a1   0   1
         a2   2   3
        >>> arr1 * arr2
        Traceback (most recent call last):
        ...
        ValueError: incompatible axes:
        Axis(['b2', 'b3'], 'b')
        vs
        Axis(['b1', 'b2'], 'b')
        >>> arr1 * arr2.drop_labels()
        a\\b  b1  b2
         a1   0   1
         a2   4   9
        >>> arr1.drop_labels() * arr2
        a\\b  b2  b3
         a1   0   1
         a2   4   9
        >>> arr1.drop_labels(X.a) * arr2.drop_labels(X.b)
        a\\b  b1  b2
         a1   0   1
         a2   4   9
        """
        if axes is None:
            axes = self.axes
        if not isinstance(axes, (tuple, list, AxisCollection)):
            axes = [axes]
        old_axes = self.axes[axes]
        new_axes = [Axis(len(axis), axis.name) for axis in old_axes]
        res_axes = self.axes[:]
        res_axes[axes] = new_axes
        return LArray(self.data, res_axes)

    def __str__(self):
        if not self.ndim:
            return str(np.asscalar(self))
        elif not len(self):
            return 'LArray([])'
        else:
            table = list(self.as_table(maxlines=200, edgeitems=5))
            return table2str(table, 'nan', fullinfo=True, maxwidth=200, keepcols=self.ndim - 1)
    __repr__ = __str__

    def __iter__(self):
        return LArrayIterator(self)

    def __contains__(self, key):
        return any(key in axis for axis in self.axes)

    def as_table(self, maxlines=None, edgeitems=5, light=False):
        """
        Generator. Returns next line of the table representing an array.

        Parameters
        ----------
        maxlines : int, optional
            Maximum number of lines to show.
        edgeitems : int, optional
            If number of lines to display is greater than `maxlines`,
            only the first and last `edgeitems` lines are displayed.
            Only active if `maxlines` is not None.
            Equals to 5 by default.

        Returns
        -------
        list
            Next line of the table as a list.

        Examples
        --------
        >>> arr = ndtest((2, 2, 3))
        >>> list(arr.as_table())  # doctest: +NORMALIZE_WHITESPACE
        [['a', 'b\\\\c', 'c0', 'c1', 'c2'],
         ['a0', 'b0', 0, 1, 2],
         ['a0', 'b1', 3, 4, 5],
         ['a1', 'b0', 6, 7, 8],
         ['a1', 'b1', 9, 10, 11]]
        >>> list(arr.as_table(light=True))  # doctest: +NORMALIZE_WHITESPACE
        [['a', 'b\\\\c', 'c0', 'c1', 'c2'],
         ['a0', 'b0', 0, 1, 2],
         ['', 'b1', 3, 4, 5],
         ['a1', 'b0', 6, 7, 8],
         ['', 'b1', 9, 10, 11]]
        """
        if not self.ndim:
            return

        # ert     unit  geo\time  2012    2011    2010
        # NEER27  I05   AT        101.41  101.63  101.63
        # NEER27  I05   AU        134.86  125.29  117.08
        width = self.shape[-1]
        height = int(np.prod(self.shape[:-1]))
        data = np.asarray(self).reshape(height, width)

        # get list of names of axes
        axes_names = self.axes.display_names[:]
        # transforms ['a', 'b', 'c', 'd'] into ['a', 'b', 'c\\d']
        if len(axes_names) > 1:
            axes_names[-2] = '\\'.join(axes_names[-2:])
            axes_names.pop()
        # get list of labels for each axis except the last one.
        labels = [axis.labels.tolist() for axis in self.axes[:-1]]
        # creates vertical lines (ticks is a list of list)
        if self.ndim == 1:
            # There is no vertical axis, so the axis name should not have
            # any "tick" below it and we add an empty "tick".
            ticks = [['']]
        elif light:
            ticks = light_product(*labels)
        else:
            ticks = product(*labels)
        # returns the first line (axes names + labels of last axis)
        yield axes_names + self.axes[-1].labels.tolist()
        # summary if needed
        if maxlines is not None and height > maxlines:
            # replace middle lines of the table by '...'.
            # We show only the first and last edgeitems lines.
            startticks = islice(ticks, edgeitems)
            midticks = [["..."] * (self.ndim - 1)]
            endticks = list(islice(rproduct(*labels), edgeitems))[::-1]
            ticks = chain(startticks, midticks, endticks)
            data = chain(data[:edgeitems].tolist(),
                         [["..."] * width],
                         data[-edgeitems:].tolist())
            for tick, dataline in izip(ticks, data):
                # returns next line (labels of N-1 first axes + data)
                yield list(tick) + dataline
        else:
            for tick, dataline in izip(ticks, data):
                # returns next line (labels of N-1 first axes + data)
                yield list(tick) + dataline.tolist()

    def dump(self, header=True):
        """Dump array as a 2D nested list

        Parameters
        ----------
        header : bool
            Whether or not to output axes names and labels.

        Returns
        -------
        2D nested list
        """
        if not header:
            # flatten all dimensions except the last one
            return self.data.reshape(-1, self.shape[-1]).tolist()
        else:
            return list(self.as_table())

    # XXX: should filter(geo=['W']) return a view by default? (collapse=True)
    # I think it would be dangerous to make it the default
    # behavior, because that would introduce a subtle difference between
    # filter(dim=[a, b]) and filter(dim=[a]) even though it would be faster
    # and uses less memory. Maybe I should have a "view" argument which
    # defaults to 'auto' (ie collapse by default), can be set to False to
    # force a copy and to True to raise an exception if a view is not possible.
    def filter(self, collapse=False, **kwargs):
        """Filters the array along the axes given as keyword arguments.

        The *collapse* argument determines whether consecutive ranges should
        be collapsed to slices, which is more efficient and returns a view
        (and not a copy) if possible (if all ranges are consecutive).
        Only use this argument if you do not intent to modify the resulting
        array, or if you know what you are doing.

        It is similar to np.take but works with several axes at once.
        """
        return self.__getitem__(kwargs, collapse)

    def _axis_aggregate(self, op, axes=(), keepaxes=False, out=None, **kwargs):
        """
        Parameters
        ----------
        op : function
            An aggregate function with this signature: func(a, axis=None, dtype=None, out=None, keepdims=False)
        axes : tuple of axes, optional
            Each axis can be an Axis object, str or int.
        out : LArray, optional
            Alternative output array in which to place the result. It must have the same shape as the expected output.
        keepaxes : bool or scalar, optional
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

        Returns
        -------
        LArray or scalar
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
            assert isinstance(out, LArray)
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
            # scalars don't need to be wrapped in LArray
            return res_data
        else:
            return LArray(res_data, res_axes)

    def _cum_aggregate(self, op, axis):
        """
        op is a numpy cumulative aggregate function: func(arr, axis=0).
        axis is an Axis object, a str or an int. Contrary to other aggregate functions this only supports one axis at a
        time.
        """
        # TODO: accept a single group in axis, to filter & aggregate in one shot
        return LArray(op(np.asarray(self), axis=self.axes.index(axis)),
                      self.axes)

    # TODO: now that items is never a (k, v), it should be renamed to
    # something else: args? (groups would be misleading because each "item" can contain several groups)
    # TODO: experiment implementing this using ufunc.reduceat
    # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ufunc.reduceat.html
    # XXX: rename keepaxes to label=value? For group_aggregates we might want to keep the LGroup label if any
    def _group_aggregate(self, op, items, keepaxes=False, out=None, **kwargs):
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
            if op in (np.sum, np.nansum) and res.dtype in (np.bool, np.bool_):
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
                res = LArray(res_data, res_axes)
            else:
                res = res_data
        return res

    def _prepare_aggregate(self, op, args, kwargs=None, commutative=False, stack_depth=1):
        """converts args to keys & LGroup and kwargs to LGroup"""

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
        if not commutative and len(kwargs_items) > 1:
            # TODO: lift this restriction for python3.6+
            raise ValueError("grouping aggregates on multiple axes at the same time using keyword arguments is not "
                             "supported for '%s' (because it is not a commutative operation and keyword arguments are "
                             "*not* ordered in Python)" % op.__name__)

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
                groups = tuple(self._guess_axis(_to_key(k, stack_depth + 1)) for k in key)
                axis = groups[0].axis
                if not all(g.axis.equals(axis) for g in groups[1:]):
                    raise ValueError("group with different axes: %s" % str(key))
                return groups
            if isinstance(key, (Group, int, basestring, list, slice)):
                return self._guess_axis(key)
            else:
                raise NotImplementedError("%s has invalid type (%s) for a group aggregate key"
                                          % (key, type(key).__name__))

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

    def _by_args_to_normal_agg_args(self, operations):
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
                   out=None, extra_kwargs={}):
        operations = self._prepare_aggregate(op, args, kwargs, commutative, stack_depth=3)
        if by_agg and operations != self.axes:
            operations = self._by_args_to_normal_agg_args(operations)

        res = self
        # group *consecutive* same-type (group vs axis aggregates) operations
        # we do not change the order of operations since we only group consecutive operations.
        for are_axes, axes in groupby(operations, self.axes.isaxis):
            func = res._axis_aggregate if are_axes else res._group_aggregate
            res = func(op, axes, keepaxes=keepaxes, out=out, **extra_kwargs)
        return res

    # op=sum does not parse correctly
    def with_total(self, *args, **kwargs):
        """with_total(*args, op='sum', label='total', **kwargs)

        Add aggregated values (sum by default) along each axis.
        A user defined label can be given to specified the computed values.

        Parameters
        ----------
        *args : int or str or Axis or Group or any combination of those, optional
            Axes or groups along which to compute the aggregates. Passed groups should be named.
            Defaults to aggregate over the whole array.
        op : aggregate function, optional
            Defaults to `sum`.
        label : scalar value, optional
            Label to use for the total. Applies only to aggregated axes, not groups. Defaults to "total".
        **kwargs : int or str or Group or any combination of those, optional
            Axes or groups along which to compute the aggregates.

        Returns
        -------
        LArray

        Examples
        --------
        >>> arr = ndtest((3, 3))
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   6   7   8
        >>> arr.with_total()
          a\\b  b0  b1  b2  total
           a0   0   1   2      3
           a1   3   4   5     12
           a2   6   7   8     21
        total   9  12  15     36
        >>> arr.with_total('a', 'b0,b1 >> total_01')
          a\\b  b0  b1  b2  total_01
           a0   0   1   2         1
           a1   3   4   5         7
           a2   6   7   8        13
        total   9  12  15        21
        >>> arr.with_total(op=prod, label='product')
            a\\b  b0  b1  b2  product
             a0   0   1   2        0
             a1   3   4   5       60
             a2   6   7   8      336
        product   0  28  80        0
        """
        # TODO: default to op.__name__
        label = kwargs.pop('label', 'total')
        op = kwargs.pop('op', sum)
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
            res = res.extend(axis, value)
        return res

    # TODO: make sure we can do
    # arr[x.sex.i[arr.indexofmin(x.sex)]] <- fails
    # and
    # arr[arr.labelofmin(x.sex)] <- fails
    # should both be equal to arr.min(x.sex)
    # the versions where axis is None already work as expected in the simple
    # case (no ambiguous labels):
    # arr.i[arr.indexofmin()]
    # arr[arr.labelofmin()]
    # for the case where axis is None, we should return an NDGroup
    # so that arr[arr.labelofmin()] works even if the minimum is on ambiguous labels
    def labelofmin(self, axis=None):
        """Returns labels of the minimum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.labelofmin(X.sex)
        nat  BE  FR  IT
              M   F   M
        >>> arr.labelofmin()
        ('BE', 'M')
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            data = axis.labels[self.data.argmin(axis_idx)]
            return LArray(data, self.axes - axis)
        else:
            indices = np.unravel_index(self.data.argmin(), self.shape)
            return tuple(axis.labels[i] for i, axis in zip(indices, self.axes))

    argmin = renamed_to(labelofmin, 'argmin')

    def indexofmin(self, axis=None):
        """Returns indices of the minimum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.indexofmin(X.sex)
        nat  BE  FR  IT
              0   1   0
        >>> arr.indexofmin()
        (0, 0)
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            return LArray(self.data.argmin(axis_idx), self.axes - axis)
        else:
            return np.unravel_index(self.data.argmin(), self.shape)

    posargmin = renamed_to(indexofmin, 'posargmin')

    def labelofmax(self, axis=None):
        """Returns labels of the maximum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the maximum values, the labels corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.labelofmax(X.sex)
        nat  BE  FR  IT
              F   M   F
        >>> arr.labelofmax()
        ('IT', 'F')
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            data = axis.labels[self.data.argmax(axis_idx)]
            return LArray(data, self.axes - axis)
        else:
            indices = np.unravel_index(self.data.argmax(), self.shape)
            return tuple(axis.labels[i] for i, axis in zip(indices, self.axes))

    argmax = renamed_to(labelofmax, 'argmax')

    def indexofmax(self, axis=None):
        """Returns indices of the maximum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the maximum values, the labels corresponding to the first occurrence are
        returned.

        Examples
        --------
        >>> nat = Axis('nat=BE,FR,IT')
        >>> sex = Axis('sex=M,F')
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.indexofmax(X.sex)
        nat  BE  FR  IT
              1   0   1
        >>> arr.indexofmax()
        (2, 1)
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            return LArray(self.data.argmax(axis_idx), self.axes - axis)
        else:
            return np.unravel_index(self.data.argmax(), self.shape)

    posargmax = renamed_to(indexofmax, 'posargmax')

    def labelsofsorted(self, axis=None, ascending=True, kind='quicksort'):
        """Returns the labels that would sort this array.

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
        LArray

        Examples
        --------
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], "nat=BE,FR,IT; sex=M,F")
        >>> arr
        nat\\sex  M  F
             BE  0  1
             FR  3  2
             IT  2  5
        >>> arr.labelsofsorted('sex')
        nat\\sex  0  1
             BE  M  F
             FR  F  M
             IT  M  F
        >>> arr.labelsofsorted('sex', ascending=False)
        nat\\sex  0  1
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
        return LArray(axis.labels[pos.data], pos.axes)

    argsort = renamed_to(labelsofsorted, 'argsort')

    def indicesofsorted(self, axis=None, ascending=True, kind='quicksort'):
        """Returns the indices that would sort this array.

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
        LArray

        Examples
        --------
        >>> arr = LArray([[1, 5], [3, 2], [0, 4]], "nat=BE,FR,IT; sex=M,F")
        >>> arr
        nat\\sex  M  F
             BE  1  5
             FR  3  2
             IT  0  4
        >>> arr.indicesofsorted('nat')
        nat\\sex  M  F
              0  2  1
              1  0  2
              2  1  0
        >>> arr.indicesofsorted('nat', ascending=False)
        nat\\sex  M  F
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
        return LArray(data, self.axes.replace(axis, new_axis))

    posargsort = renamed_to(indicesofsorted, 'posargsort')

    def copy(self):
        """Returns a copy of the array.
        """
        return LArray(self.data.copy(), axes=self.axes[:], title=self.title)

    @property
    def info(self):
        """Describes a LArray (title + shape and labels for each axis).

        Returns
        -------
        str
            Description of the array (title + shape and labels for each axis).

        Examples
        --------
        >>> mat0 = LArray([[2.0, 5.0], [8.0, 6.0]], "nat=BE,FO; sex=F,M")
        >>> mat0.info
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'F' 'M'
        dtype: float64
        memory used: 32 bytes
        >>> mat1 = LArray([[2.0, 5.0], [8.0, 6.0]], "nat=BE,FO; sex=F,M", 'test matrix')
        >>> mat1.info
        test matrix
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'F' 'M'
        dtype: float64
        memory used: 32 bytes
        """
        str_info = '{}\n'.format(self.title) if self.title else ''
        str_info += '{}\ndtype: {}\nmemory used: {}'.format(self.axes.info, self.dtype.name, self.memory_used)
        return ReprString(str_info)

    def ratio(self, *axes):
        """Returns an array with all values divided by the sum of values along given axes.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            array / array.sum(axes)

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> a = LArray([[4, 6], [2, 8]], [nat, sex])
        >>> a
        nat\\sex  M  F
             BE  4  6
             FO  2  8
        >>> a.sum()
        20
        >>> a.ratio()
        nat\\sex    M    F
             BE  0.2  0.3
             FO  0.1  0.4
        >>> a.ratio(X.sex)
        nat\\sex    M    F
             BE  0.4  0.6
             FO  0.2  0.8
        >>> a.ratio('M')
        nat\\sex    M    F
             BE  1.0  1.5
             FO  1.0  4.0
        """
        # # this does not work, but I am unsure it should
        # # >>> a.sum(age[[0, 1]], age[2]) / a.sum(age)
        # >>> a.sum(([0, 1], 2)) / a.sum(age)
        # # >>> a / a.sum(([0, 1], 2))
        # >>> a.sum(x.sex)
        # >>> a.sum(x.age)
        # >>> a.sum(x.sex) / a.sum(x.age)
        # >>> a.ratio('F')
        # could mean
        # >>> a.sum('F') / a.sum(a.get_axis('F'))
        # >>> a.sum('F') / a.sum(x.sex)
        # age    0    1               2
        #      1.0  0.6  0.555555555556
        # OR (current meaning)
        # >>> a / a.sum('F')
        # age\\sex               M    F
        #       0             0.0  1.0
        #       1  0.666666666667  1.0
        #       2             0.8  1.0
        # One solution is to add an argument
        # >>> a.ratio(what='F', by=x.sex)
        # age    0    1               2
        #      1.0  0.6  0.555555555556
        # >>> a.sum('F') / a.sum(x.sex)

        # >>> a.sum((age[[0, 1]], age[[1, 2]])) / a.sum(age)
        # >>> a.ratio((age[[0, 1]], age[[1, 2]]), by=age)

        # >>> a.sum((x.age[[0, 1]], x.age[[1, 2]])) / a.sum(x.age)
        # >>> a.ratio((x.age[[0, 1]], x.age[[1, 2]], by=x.age)

        # >>> lalala.sum(([0, 1], [1, 2])) / lalala.sum(x.age)
        # >>> lalala.ratio(([0, 1], [1, 2]), by=x.age)

        # >>> b = a.sum((age[[0, 1]], age[[1, 2]]))
        # >>> b
        # age\sex  M  F
        #   [0 1]  2  4
        #   [1 2]  6  8
        # >>> b / b.sum(x.age)
        # age\\sex     M               F
        #   [0 1]  0.25  0.333333333333
        #   [1 2]  0.75  0.666666666667
        # >>> b / a.sum(x.age)
        # age\\sex               M               F
        #   [0 1]  0.333333333333  0.444444444444
        #   [1 2]             1.0  0.888888888889
        # # >>> a.ratio([0, 1], [2])
        # # >>> a.ratio(x.age[[0, 1]], x.age[2])
        # >>> a.ratio((x.age[[0, 1]], x.age[2]))
        # nat\\sex             M    F
        #      BE           0.0  1.0
        #      FO  0.6666666666  1.0
        return self / self.sum(*axes)

    def rationot0(self, *axes):
        """Returns a LArray with values array / array.sum(axes) where the sum is not 0, 0 otherwise.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            array / array.sum(axes)

        Examples
        --------
        >>> a = Axis('a=a0,a1')
        >>> b = Axis('b=b0,b1,b2')
        >>> arr = LArray([[6, 0, 2],
        ...               [4, 0, 8]], [a, b])
        >>> arr
        a\\b  b0  b1  b2
         a0   6   0   2
         a1   4   0   8
        >>> arr.sum()
        20
        >>> arr.rationot0()
        a\\b   b0   b1   b2
         a0  0.3  0.0  0.1
         a1  0.2  0.0  0.4
        >>> arr.rationot0(X.a)
        a\\b   b0   b1   b2
         a0  0.6  0.0  0.2
         a1  0.4  0.0  0.8

        for reference, the normal ratio method would return:

        >>> arr.ratio(X.a)
        a\\b   b0   b1   b2
         a0  0.6  nan  0.2
         a1  0.4  nan  0.8
        """
        return self.divnot0(self.sum(*axes))

    def percent(self, *axes):
        """Returns an array with values given as percent of the total of all values along given axes.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            array / array.sum(axes) * 100

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> a = LArray([[4, 6], [2, 8]], [nat, sex])
        >>> a
        nat\\sex  M  F
             BE  4  6
             FO  2  8
        >>> a.percent()
        nat\\sex     M     F
             BE  20.0  30.0
             FO  10.0  40.0
        >>> a.percent(X.sex)
        nat\\sex     M     F
             BE  40.0  60.0
             FO  20.0  80.0
        """
        # dividing by self.sum(*axes) * 0.01 would be faster in many cases but I suspect it loose more precision.
        return self * 100 / self.sum(*axes)

    # aggregate method decorator
    def _decorate_agg_method(npfunc, nanfunc=None, commutative=False, by_agg=False, extra_kwargs=[],
                             long_name='', action_verb='perform'):
        def decorated(func):
            _doc_agg_method(func, by_agg, long_name, action_verb, kwargs=extra_kwargs + ['out', 'skipna', 'keepaxes'])

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                keepaxes = kwargs.pop('keepaxes', _kwarg_agg['keepaxes']['value'])
                skipna = kwargs.pop('skipna', _kwarg_agg['skipna']['value'])
                out = kwargs.pop('out', _kwarg_agg['out']['value'])
                if skipna is None:
                    skipna = nanfunc is not None
                if skipna and nanfunc is None:
                    raise ValueError("skipna is not available for {}".format(func.__name__))
                _npfunc = nanfunc if skipna else npfunc
                _extra_kwargs = {}
                for k in extra_kwargs:
                    _extra_kwargs[k] = kwargs.pop(k, _kwarg_agg[k]['value'])
                return self._aggregate(_npfunc, args, kwargs, by_agg=by_agg, keepaxes=keepaxes,
                                       commutative=commutative, out=out, extra_kwargs=_extra_kwargs)
            return wrapper
        return decorated

    @_decorate_agg_method(np.all, commutative=True, long_name="AND reduction")
    def all(self, *args, **kwargs):
        """{signature}
        Test whether all selected elements evaluate to True.

        {parameters}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.all_by, LArray.any, LArray.any_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.all()
        False
        >>> # along axis 'a'
        >>> barr.all(X.a)
        b     b0     b1     b2     b3
           False  False  False  False
        >>> # along axis 'b'
        >>> barr.all(X.b)
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
          a\\b     b0     b1     b2     b3
        a0,a1   True   True  False  False
        a2,a3  False  False  False  False
        >>> # or equivalently
        >>> # barr.all('a0,a1;a2,a3')

        Same with renaming

        >>> barr.all((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b     b0     b1     b2     b3
        a01   True   True  False  False
        a23  False  False  False  False
        >>> # or equivalently
        >>> # barr.all('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.all, commutative=True, by_agg=True, long_name="AND reduction")
    def all_by(self, *args, **kwargs):
        """{signature}
        Test whether all selected elements evaluate to True.

        {parameters}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.all, LArray.any, LArray.any_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.all_by()
        False
        >>> # by axis 'a'
        >>> barr.all_by(X.a)
        a    a0     a1     a2     a3
           True  False  False  False
        >>> # by axis 'b'
        >>> barr.all_by(X.b)
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
    def any(self, *args, **kwargs):
        """{signature}
        Test whether any selected elements evaluate to True.

        {parameters}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.any_by, LArray.all, LArray.all_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.any()
        True
        >>> # along axis 'a'
        >>> barr.any(X.a)
        b    b0    b1    b2    b3
           True  True  True  True
        >>> # along axis 'b'
        >>> barr.any(X.b)
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
          a\\b     b0     b1     b2     b3
        a0,a1   True   True   True   True
        a2,a3  False  False  False  False
        >>> # or equivalently
        >>> # barr.any('a0,a1;a2,a3')

        Same with renaming

        >>> barr.any((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b     b0     b1     b2     b3
        a01   True   True   True   True
        a23  False  False  False  False
        >>> # or equivalently
        >>> # barr.any('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.any, commutative=True, by_agg=True, long_name="OR reduction")
    def any_by(self, *args, **kwargs):
        """{signature}
        Test whether any selected elements evaluate to True.

        {parameters}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.any, LArray.all, LArray.all_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> barr = arr < 6
        >>> barr
        a\\b     b0     b1     b2     b3
         a0   True   True   True   True
         a1   True   True  False  False
         a2  False  False  False  False
         a3  False  False  False  False
        >>> barr.any_by()
        True
        >>> # by axis 'a'
        >>> barr.any_by(X.a)
        a    a0    a1     a2     a3
           True  True  False  False
        >>> # by axis 'b'
        >>> barr.any_by(X.b)
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
    def sum(self, *args, **kwargs):
        """{signature}
        Computes the sum of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.sum_by, LArray.prod, LArray.prod_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.sum()
        120
        >>> # along axis 'a'
        >>> arr.sum(X.a)
        b  b0  b1  b2  b3
           24  28  32  36
        >>> # along axis 'b'
        >>> arr.sum(X.b)
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
          a\\b  b0  b1  b2  b3
        a0,a1   4   6   8  10
        a2,a3  20  22  24  26
        >>> # or equivalently
        >>> # arr.sum('a0,a1;a2,a3')

        Same with renaming

        >>> arr.sum((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b  b0  b1  b2  b3
        a01   4   6   8  10
        a23  20  22  24  26
        >>> # or equivalently
        >>> # arr.sum('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.sum, np.nansum, commutative=True, by_agg=True, extra_kwargs=['dtype'], long_name="sum")
    def sum_by(self, *args, **kwargs):
        """{signature}
        Computes the sum of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.sum, LArray.prod, LArray.prod_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.sum_by()
        120
        >>> # along axis 'a'
        >>> arr.sum_by(X.a)
        a  a0  a1  a2  a3
            6  22  38  54
        >>> # along axis 'b'
        >>> arr.sum_by(X.b)
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
    def prod(self, *args, **kwargs):
        """{signature}
        Computes the product of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.prod_by, LArray.sum, LArray.sum_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.prod()
        0
        >>> # along axis 'a'
        >>> arr.prod(X.a)
        b  b0   b1    b2    b3
            0  585  1680  3465
        >>> # along axis 'b'
        >>> arr.prod(X.b)
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
          a\\b  b0   b1   b2   b3
        a0,a1   0    5   12   21
        a2,a3  96  117  140  165
        >>> # or equivalently
        >>> # arr.prod('a0,a1;a2,a3')

        Same with renaming

        >>> arr.prod((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b  b0   b1   b2   b3
        a01   0    5   12   21
        a23  96  117  140  165
        >>> # or equivalently
        >>> # arr.prod('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.prod, np_nanprod, commutative=True, by_agg=True, extra_kwargs=['dtype'],
                          long_name="product")
    def prod_by(self, *args, **kwargs):
        """{signature}
        Computes the product of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.prod, LArray.sum, LArray.sum_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.prod_by()
        0
        >>> # along axis 'a'
        >>> arr.prod_by(X.a)
        a  a0   a1    a2     a3
            0  840  7920  32760
        >>> # along axis 'b'
        >>> arr.prod_by(X.b)
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
    def min(self, *args, **kwargs):
        """{signature}
        Get minimum of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.min_by, LArray.max, LArray.max_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.min()
        0
        >>> # along axis 'a'
        >>> arr.min(X.a)
        b  b0  b1  b2  b3
            0   1   2   3
        >>> # along axis 'b'
        >>> arr.min(X.b)
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
          a\\b  b0  b1  b2  b3
        a0,a1   0   1   2   3
        a2,a3   8   9  10  11
        >>> # or equivalently
        >>> # arr.min('a0,a1;a2,a3')

        Same with renaming

        >>> arr.min((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b  b0  b1  b2  b3
        a01   0   1   2   3
        a23   8   9  10  11
        >>> # or equivalently
        >>> # arr.min('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.min, np.nanmin, commutative=True, by_agg=True, long_name="minimum", action_verb="search")
    def min_by(self, *args, **kwargs):
        """{signature}
        Get minimum of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.min, LArray.max, LArray.max_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.min_by()
        0
        >>> # along axis 'a'
        >>> arr.min_by(X.a)
        a  a0  a1  a2  a3
            0   4   8  12
        >>> # along axis 'b'
        >>> arr.min_by(X.b)
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
    def max(self, *args, **kwargs):
        """{signature}
        Get maximum of array elements along given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.max_by, LArray.min, LArray.min_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.max()
        15
        >>> # along axis 'a'
        >>> arr.max(X.a)
        b  b0  b1  b2  b3
           12  13  14  15
        >>> # along axis 'b'
        >>> arr.max(X.b)
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
          a\\b  b0  b1  b2  b3
        a0,a1   4   5   6   7
        a2,a3  12  13  14  15
        >>> # or equivalently
        >>> # arr.max('a0,a1;a2,a3')

        Same with renaming

        >>> arr.max((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b  b0  b1  b2  b3
        a01   4   5   6   7
        a23  12  13  14  15
        >>> # or equivalently
        >>> # arr.max('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.max, np.nanmax, commutative=True, by_agg=True, long_name="maximum", action_verb="search")
    def max_by(self, *args, **kwargs):
        """{signature}
        Get maximum of array elements for the given axes/groups.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.max, LArray.min, LArray.min_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.max_by()
        15
        >>> # along axis 'a'
        >>> arr.max_by(X.a)
        a  a0  a1  a2  a3
            3   7  11  15
        >>> # along axis 'b'
        >>> arr.max_by(X.b)
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
    def mean(self, *args, **kwargs):
        """{signature}
        Computes the arithmetic mean.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.mean_by, LArray.median, LArray.median_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.mean()
        7.5
        >>> # along axis 'a'
        >>> arr.mean(X.a)
        b   b0   b1   b2   b3
           6.0  7.0  8.0  9.0
        >>> # along axis 'b'
        >>> arr.mean(X.b)
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
          a\\b    b0    b1    b2    b3
        a0,a1   2.0   3.0   4.0   5.0
        a2,a3  10.0  11.0  12.0  13.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1;a2,a3')

        Same with renaming

        >>> arr.mean((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b    b0    b1    b2    b3
        a01   2.0   3.0   4.0   5.0
        a23  10.0  11.0  12.0  13.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.mean, np.nanmean, commutative=True, by_agg=True, extra_kwargs=['dtype'], long_name="mean")
    def mean_by(self, *args, **kwargs):
        """{signature}
        Computes the arithmetic mean.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.mean, LArray.median, LArray.median_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.mean()
        7.5
        >>> # along axis 'a'
        >>> arr.mean_by(X.a)
        a   a0   a1   a2    a3
           1.5  5.5  9.5  13.5
        >>> # along axis 'b'
        >>> arr.mean_by(X.b)
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
    def median(self, *args, **kwargs):
        """{signature}
        Computes the arithmetic median.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.median_by, LArray.mean, LArray.mean_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9], \
                        [5, 8, 3, 7], \
                        [6, 2, 0, 9], \
                        [9, 10, 5, 6]]
        >>> arr
        a\\b  b0  b1  b2  b3
         a0  10   7   5   9
         a1   5   8   3   7
         a2   6   2   0   9
         a3   9  10   5   6
        >>> arr.median()
        6.5
        >>> # along axis 'a'
        >>> arr.median(X.a)
        b   b0   b1   b2   b3
           7.5  7.5  4.0  8.0
        >>> # along axis 'b'
        >>> arr.median(X.b)
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
          a\\b   b0   b1   b2   b3
        a0,a1  7.5  7.5  4.0  8.0
        a2,a3  7.5  6.0  2.5  7.5
        >>> # or equivalently
        >>> # arr.median('a0,a1;a2,a3')

        Same with renaming

        >>> arr.median((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b   b0   b1   b2   b3
        a01  7.5  7.5  4.0  8.0
        a23  7.5  6.0  2.5  7.5
        >>> # or equivalently
        >>> # arr.median('a0,a1>>a01;a2,a3>>a23')
        """
        pass

    @_decorate_agg_method(np.median, np.nanmedian, commutative=True, by_agg=True, long_name="mediane")
    def median_by(self, *args, **kwargs):
        """{signature}
        Computes the arithmetic median.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.median, LArray.mean, LArray.mean_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9], \
                        [5, 8, 3, 7], \
                        [6, 2, 0, 9], \
                        [9, 10, 5, 6]]
        >>> arr
        a\\b  b0  b1  b2  b3
         a0  10   7   5   9
         a1   5   8   3   7
         a2   6   2   0   9
         a3   9  10   5   6
        >>> arr.median_by()
        6.5
        >>> # along axis 'a'
        >>> arr.median_by(X.a)
        a   a0   a1   a2   a3
           8.0  6.0  4.0  7.5
        >>> # along axis 'b'
        >>> arr.median_by(X.b)
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
    def percentile(self, q, *args, **kwargs):
        """{signature}
        Computes the qth percentile of the data along the specified axis.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.percentile_by, LArray.mean, LArray.mean_by,
        LArray.median, LArray.median_by, LArray.var, LArray.var_by,
        LArray.std, LArray.std_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.percentile(25)
        3.75
        >>> # along axis 'a'
        >>> arr.percentile(25, X.a)
        b   b0   b1   b2   b3
           3.0  4.0  5.0  6.0
        >>> # along axis 'b'
        >>> arr.percentile(25, X.b)
        a    a0    a1    a2     a3
           0.75  4.75  8.75  12.75
        >>> # several percentile values
        >>> arr.percentile([25, 50, 75], X.b)
        percentile\\a    a0    a1     a2     a3
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
          a\\b   b0    b1    b2    b3
        a0,a1  1.0   2.0   3.0   4.0
        a2,a3  9.0  10.0  11.0  12.0
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1;a2,a3')

        Same with renaming

        >>> arr.percentile(25, (X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b   b0    b1    b2    b3
        a01  1.0   2.0   3.0   4.0
        a23  9.0  10.0  11.0  12.0
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1>>a01;a2,a3>>a23')
        """
        keepaxes = kwargs.pop('keepaxes', _kwarg_agg['keepaxes']['value'])
        skipna = kwargs.pop('skipna', _kwarg_agg['skipna']['value'])
        out = kwargs.pop('out', _kwarg_agg['out']['value'])
        if skipna is None:
            skipna = True
        _npfunc = np.nanpercentile if skipna else np.percentile
        interpolation = kwargs.pop('interpolation', _kwarg_agg['interpolation']['value'])
        if isinstance(q, (list, tuple)):
            res = stack([(v, self._aggregate(_npfunc, args, kwargs, keepaxes=keepaxes, commutative=True,
                          extra_kwargs={'q': v, 'interpolation': interpolation})) for v in q], 'percentile')
            return res.transpose()
        else :
            _extra_kwargs = {'q': q, 'interpolation': interpolation}
            return self._aggregate(_npfunc, args, kwargs, by_agg=False, keepaxes=keepaxes, commutative=True,
                                   out=out, extra_kwargs=_extra_kwargs)

    _doc_agg_method(percentile, False, "qth percentile", extra_args=['q'],
                    kwargs=['out', 'interpolation', 'skipna', 'keepaxes'])

    def percentile_by(self, q, *args, **kwargs):
        """{signature}
        Computes the qth percentile of the data for the specified axis.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.percentile, LArray.mean, LArray.mean_by,
        LArray.median, LArray.median_by, LArray.var, LArray.var_by,
        LArray.std, LArray.std_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.percentile_by(25)
        3.75
        >>> # along axis 'a'
        >>> arr.percentile_by(25, X.a)
        a    a0    a1    a2     a3
           0.75  4.75  8.75  12.75
        >>> # along axis 'b'
        >>> arr.percentile_by(25, X.b)
        b   b0   b1   b2   b3
           3.0  4.0  5.0  6.0
        >>> # several percentile values
        >>> arr.percentile_by([25, 50, 75], X.b)
        percentile\\b   b0    b1    b2    b3
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
        """
        keepaxes = kwargs.pop('keepaxes', _kwarg_agg['keepaxes']['value'])
        skipna = kwargs.pop('skipna', _kwarg_agg['skipna']['value'])
        out = kwargs.pop('out', _kwarg_agg['out']['value'])
        if skipna is None:
            skipna = True
        _npfunc = np.nanpercentile if skipna else np.percentile
        interpolation = kwargs.pop('interpolation', _kwarg_agg['interpolation']['value'])
        if isinstance(q, (list, tuple)):
            res = stack([(v, self._aggregate(_npfunc, args, kwargs, by_agg=True, keepaxes=keepaxes, commutative=True,
                          extra_kwargs={'q': v, 'interpolation': interpolation})) for v in q], 'percentile')
            return res.transpose()
        else:
            return self._aggregate(_npfunc, args, kwargs, by_agg=True, keepaxes=keepaxes, commutative=True, out=out,
                                   extra_kwargs={'q': q, 'interpolation': interpolation})

    _doc_agg_method(percentile_by, True, "qth percentile", extra_args=['q'],
                    kwargs=['out', 'interpolation', 'skipna', 'keepaxes'])

    # not commutative

    def ptp(self, *args, **kwargs):
        """{signature}
        Returns the range of values (maximum - minimum).

        The name of the function comes from the acronym for ‘peak to peak’.

        {parameters}

        Returns
        -------
        LArray or scalar

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.ptp()
        15
        >>> # along axis 'a'
        >>> arr.ptp(X.a)
        b  b0  b1  b2  b3
           12  12  12  12
        >>> # along axis 'b'
        >>> arr.ptp(X.b)
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
          a\\b  b0  b1  b2  b3
        a0,a1   4   4   4   4
        a2,a3   4   4   4   4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1;a2,a3')

        Same with renaming

        >>> arr.ptp((X.a['a0', 'a1'] >> 'a01', X.a['a2', 'a3'] >> 'a23'))
        a\\b  b0  b1  b2  b3
        a01   4   4   4   4
        a23   4   4   4   4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1>>a01;a2,a3>>a23')
        """
        out = kwargs.pop('out', _kwarg_agg['out']['value'])
        return self._aggregate(np.ptp, args, kwargs, out=out)

    _doc_agg_method(ptp, False, kwargs=['out'])

    @_decorate_agg_method(np.var, np.nanvar, extra_kwargs=['dtype', 'ddof'], long_name="variance")
    def var(self, *args, **kwargs):
        """{signature}
        Computes the unbiased variance.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.var_by, LArray.std, LArray.std_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3], \
                        [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.var()
        4.7999999999999998
        >>> # along axis 'b'
        >>> arr.var(X.b)
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
        a\\b  b0,b1,b3  b5:
         a0       9.0  1.0
         a1       4.0  1.0
        >>> # or equivalently
        >>> # arr.var('b0,b1,b3;b5:')

        Same with renaming

        >>> arr.var((X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\\b  b013  b567
         a0   9.0   1.0
         a1   4.0   1.0
        >>> # or equivalently
        >>> # arr.var('b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    @_decorate_agg_method(np.var, np.nanvar, by_agg=True, extra_kwargs=['dtype', 'ddof'], long_name="variance")
    def var_by(self, *args, **kwargs):
        """{signature}
        Computes the unbiased variance.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.var, LArray.std, LArray.std_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3], \
                        [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.var_by()
        4.7999999999999998
        >>> # along axis 'a'
        >>> arr.var_by(X.a)
        a   a0   a1
           4.0  4.0

        Select some columns only

        >>> arr.var_by(X.a, ['b0','b1','b3'])
        a   a0   a1
           9.0  4.0
        >>> # or equivalently
        >>> # arr.var_by('a','b0,b1,b3')

        Split an axis in several parts

        >>> arr.var_by(X.a, (['b0', 'b1', 'b3'], 'b5:'))
        a\\b  b0,b1,b3  b5:
         a0       9.0  1.0
         a1       4.0  1.0
        >>> # or equivalently
        >>> # arr.var_by('a','b0,b1,b3;b5:')

        Same with renaming

        >>> arr.var_by(X.a, (X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\\b  b013  b567
         a0   9.0   1.0
         a1   4.0   1.0
        >>> # or equivalently
        >>> # arr.var_by('a','b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    @_decorate_agg_method(np.std, np.nanstd, extra_kwargs=['dtype', 'ddof'], long_name="standard deviation")
    def std(self, *args, **kwargs):
        """{signature}
        Computes the sample standard deviation.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.std_by, LArray.var, LArray.var_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3],
        ...             [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.std()
        2.1908902300206643
        >>> # along axis 'b'
        >>> arr.std(X.b)
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
        a\\b  b0,b1,b3  b5:
         a0       3.0  1.0
         a1       2.0  1.0
        >>> # or equivalently
        >>> # arr.std('b0,b1,b3;b5:')

        Same with renaming

        >>> arr.std((X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\\b  b013  b567
         a0   3.0   1.0
         a1   2.0   1.0
        >>> # or equivalently
        >>> # arr.std('b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    @_decorate_agg_method(np.std, np.nanstd, by_agg=True, extra_kwargs=['dtype', 'ddof'],
                          long_name="standard deviation")
    def std_by(self, *args, **kwargs):
        """{signature}
        Computes the sample standard deviation.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        {parameters}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.std_by, LArray.var, LArray.var_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((2, 8), dtype=float)
        >>> arr[:,:] = [[0, 3, 5, 6, 4, 2, 1, 3],
        ...             [7, 3, 2, 5, 8, 5, 6, 4]]
        >>> arr
        a\\b   b0   b1   b2   b3   b4   b5   b6   b7
         a0  0.0  3.0  5.0  6.0  4.0  2.0  1.0  3.0
         a1  7.0  3.0  2.0  5.0  8.0  5.0  6.0  4.0
        >>> arr.std_by()
        2.1908902300206643
        >>> # along axis 'a'
        >>> arr.std_by(X.a)
        a   a0   a1
           2.0  2.0

        Select some columns only

        >>> arr.std_by(X.a, ['b0','b1','b3'])
        a   a0   a1
           3.0  2.0
        >>> # or equivalently
        >>> # arr.std_by('a','b0,b1,b3')

        Split an axis in several parts

        >>> arr.std_by(X.a, (['b0', 'b1', 'b3'], 'b5:'))
        a\\b  b0,b1,b3  b5:
         a0       3.0  1.0
         a1       2.0  1.0
        >>> # or equivalently
        >>> # arr.std_by('a','b0,b1,b3;b5:')

        Same with renaming

        >>> arr.std_by(X.a, (X.b['b0', 'b1', 'b3'] >> 'b013', X.b['b5:'] >> 'b567'))
        a\\b  b013  b567
         a0   3.0   1.0
         a1   2.0   1.0
        >>> # or equivalently
        >>> # arr.std_by('a','b0,b1,b3>>b013;b5:>>b567')
        """
        pass

    # cumulative aggregates
    def cumsum(self, axis=-1):
        """
        Returns the cumulative sum of array elements along an axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to perform the cumulative sum.
            If given as position, it can be a negative integer, in which case it counts from the last to the first axis.
            By default, the cumulative sum is performed along the last axis.

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.cumprod, LArray.sum, LArray.sum_by,
        LArray.prod, LArray.prod_by

        Notes
        -----
        Cumulative aggregation functions accept only one axis

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.cumsum()
        a\\b  b0  b1  b2  b3
         a0   0   1   3   6
         a1   4   9  15  22
         a2   8  17  27  38
         a3  12  25  39  54
        >>> arr.cumsum(X.a)
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   6   8  10
         a2  12  15  18  21
         a3  24  28  32  36
        """
        return self._cum_aggregate(np.cumsum, axis)

    def cumprod(self, axis=-1):
        """
        Returns the cumulative product of array elements.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to perform the cumulative product.
            If given as position, it can be a negative integer, in which case it counts from the last to the first axis.
            By default, the cumulative product is performed along the last axis.

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.cumsum, LArray.sum, LArray.sum_by,
        LArray.prod, LArray.prod_by

        Notes
        -----
        Cumulative aggregation functions accept only one axis.

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b  b0  b1  b2  b3
         a0   0   1   2   3
         a1   4   5   6   7
         a2   8   9  10  11
         a3  12  13  14  15
        >>> arr.cumprod()
        a\\b  b0   b1    b2     b3
         a0   0    0     0      0
         a1   4   20   120    840
         a2   8   72   720   7920
         a3  12  156  2184  32760
        >>> arr.cumprod(X.a)
        a\\b  b0   b1    b2    b3
         a0   0    1     2     3
         a1   0    5    12    21
         a2   0   45   120   231
         a3   0  585  1680  3465
        """
        return self._cum_aggregate(np.cumprod, axis)

    # element-wise method factory
    def _binop(opname):
        fullname = '__%s__' % opname
        super_method = getattr(np.ndarray, fullname)

        def opmethod(self, other):
            res_axes = self.axes

            if isinstance(other, ExprNode):
                other = other.evaluate(self.axes)

            # we could pass scalars through aslarray too but it is too costly performance-wise for only suppressing one
            # isscalar test and an if statement.
            # TODO: ndarray should probably be converted to larrays because that would harmonize broadcasting rules, but
            # it makes some tests fail for some reason.
            if not isinstance(other, (LArray, np.ndarray)) and not np.isscalar(other):
                other = aslarray(other)

            if isinstance(other, LArray):
                # TODO: first test if it is not already broadcastable
                (self, other), res_axes = make_numpy_broadcastable([self, other])
                other = other.data
            return LArray(super_method(self.data, other), res_axes)
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
    if sys.version < '3':
        __div__ = _binop('div')
        __rdiv__ = _binop('rdiv')
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

    def __matmul__(self, other):
        """
        Overrides operator @ for matrix multiplication.

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
        a\\b  b0  b1  b2
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
        c\\e  e0  e1  e2
         c0  15  18  21
         c1  42  45  48
         c2  69  72  75
        >>> arr3d @ arr1d # doctest: +SKIP
        c\\d  d0  d1  d2
         c0   5  14  23
         c1  32  41  50
         c2  59  68  77
        >>> arr3d @ arr3d # doctest: +SKIP
         c  d\\e    e0    e1    e2
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
        if not isinstance(other, (LArray, np.ndarray)):
            raise NotImplementedError("matrix multiplication not implemented for %s" % type(other))
        if isinstance(other, np.ndarray):
            other = LArray(other)
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
        return LArray(res_data, res_axes)

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            other = LArray(other)
        if not isinstance(other, LArray):
            raise NotImplementedError("matrix multiplication not implemented for %s" % type(other))
        return other.__matmul__(self)

    # element-wise method factory
    def _unaryop(opname):
        fullname = '__%s__' % opname
        super_method = getattr(np.ndarray, fullname)

        def opmethod(self):
            return LArray(super_method(self.data), self.axes)
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

    def equals(self, other, rtol=0, atol=0, nan_equals=False):
        """
        Compares self with another array and returns True if they have the same axes and elements, False otherwise.

        Parameters
        ----------
        other: LArray-like
            Input array. aslarray() is used on a non-LArray input.
        rtol : float or int, optional
            The relative tolerance parameter (see Notes). Defaults to 0.
        atol : float or int, optional
            The absolute tolerance parameter (see Notes). Defaults to 0.
        nan_equals: boolean, optional
            Whether or not to consider nan values at the same positions in the two arrays as equal.
            By default, an array containing nan values is never equal to another array, even if that other array
            also contains nan values at the same positions. The reason is that a nan value is different from
            *anything*, including itself. Defaults to False.

        Returns
        -------
        bool
            Returns True if self is equal to other.

        Notes
        -----
        For finite values, equals uses the following equation to test whether two values are equal:

            absolute(array1 - array2) <= (atol + rtol * absolute(array2))

        The above equation is not symmetric in array1 and array2, so that equals(array1, array2) might be different
        from equals(array2, array1) in some rare cases.

        Examples
        --------
        >>> arr1 = ndtest((2, 3))
        >>> arr1
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr2 = arr1.copy()
        >>> arr1.equals(arr2)
        True
        >>> arr2['b1'] += 1
        >>> arr1.equals(arr2)
        False
        >>> arr3 = arr1.set_labels('a', ['x0', 'x1'])
        >>> arr1.equals(arr3)
        False

        Test equality between two arrays within a given tolerance range.
        Return True if absolute(array1 - array2) <= (atol + rtol * absolute(array2)).

        >>> arr1 = LArray([6., 8.], "a=a0,a1")
        >>> arr1
        a   a0   a1
           6.0  8.0
        >>> arr2 = LArray([5.999, 8.001], "a=a0,a1")
        >>> arr2
        a     a0     a1
           5.999  8.001
        >>> arr1.equals(arr2)
        False
        >>> arr1.equals(arr2, atol=0.01)
        True
        >>> arr1.equals(arr2, rtol=0.01)
        True

        Arrays with nan values

        >>> arr1 = ndtest((2, 3), dtype=float)
        >>> arr1['a1', 'b1'] = nan
        >>> arr1
        a\\b   b0   b1   b2
         a0  0.0  1.0  2.0
         a1  3.0  nan  5.0
        >>> arr2 = arr1.copy()
        >>> # By default, an array containing nan values is never equal to another array,
        >>> # even if that other array also contains nan values at the same positions.
        >>> # The reason is that a nan value is different from *anything*, including itself.
        >>> arr1.equals(arr2)
        False
        >>> # set flag nan_equals to True to overwrite this behavior
        >>> arr1.equals(arr2, nan_equals=True)
        True
        """
        try:
            other = aslarray(other)
        except Exception:
            return False
        return self.axes == other.axes and all(element_equal(self, other, rtol=rtol, atol=atol, nan_equals=nan_equals))

    def divnot0(self, other):
        """Divides array by other, but returns 0.0 where other is 0.

        Parameters
        ----------
        other : scalar or LArray
            What to divide by.

        Returns
        -------
        LArray
            Array divided by other, 0.0 where other is 0

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> a = ndtest((nat, sex))
        >>> a
        nat\\sex  M  F
             BE  0  1
             FO  2  3
        >>> b = ndtest(sex)
        >>> b
        sex  M  F
             0  1
        >>> a / b
        nat\\sex    M    F
             BE  nan  1.0
             FO  inf  3.0
        >>> a.divnot0(b)
        nat\\sex    M    F
             BE  0.0  1.0
             FO  0.0  3.0
        """
        if np.isscalar(other):
            if other == 0:
                return zeros_like(self, dtype=float)
            else:
                return self / other
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                res = self / other
            res[other == 0] = 0
            return res

    # XXX: rename/change to "add_axes" ?
    # TODO: add a flag copy=True to force a new array.
    def expand(self, target_axes=None, out=None, readonly=False):
        """Expands array to target_axes.

        Target axes will be added to array if not present.
        In most cases this function is not needed because LArray can do operations with arrays having different
        (compatible) axes.

        Parameters
        ----------
        target_axes : list of Axis or AxisCollection, optional
            Self can contain axes not present in `target_axes`.
            The result axes will be: [self.axes not in target_axes] + target_axes
        out : LArray, optional
            Output array, must have the correct shape
        readonly : bool, optional
            Whether returning a readonly view is acceptable or not (this is much faster)

        Returns
        -------
        LArray
            Original array if possible (and out is None).

        Examples
        --------
        >>> a = Axis('a=a1,a2')
        >>> b = Axis('b=b1,b2')
        >>> arr = ndtest([a, b])
        >>> arr
        a\\b  b1  b2
         a1   0   1
         a2   2   3
        >>> c = Axis('c=c1,c2')
        >>> arr.expand([a, c, b])
         a  c\\b  b1  b2
        a1   c1   0   1
        a1   c2   0   1
        a2   c1   2   3
        a2   c2   2   3
        >>> arr.expand([b, c])
         a  b\\c  c1  c2
        a1   b1   0   0
        a1   b2   1   1
        a2   b1   2   2
        a2   b2   3   3
        """
        if target_axes is None and out is None or target_axes is not None and out is not None:
            raise ValueError("either target_axes or out must be defined (not both)")
        if out is not None:
            target_axes = out.axes
        else:
            if not isinstance(target_axes, AxisCollection):
                target_axes = AxisCollection(target_axes)
            target_axes = (self.axes - target_axes) | target_axes

        if out is None:
            # this is not strictly necessary but avoids doing this test twice if it is True
            if self.axes == target_axes:
                return self

            broadcasted = self.broadcast_with(target_axes)
            # this can only happen if only the order of axes differed and/or all extra axes have length 1
            if broadcasted.axes == target_axes:
                return broadcasted

            if readonly:
                # requires numpy 1.10
                return LArray(np.broadcast_to(broadcasted, target_axes.shape), target_axes)
            else:
                out = empty(target_axes, dtype=self.dtype)
        out[:] = broadcasted
        return out

    def append(self, axis, value, label=None):
        """Adds an array to self along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis reference
            Axis along which to append input array (`value`).
        value : scalar or LArray
            Array with compatible axes.
        label : str, optional
            Label for the new item in axis

        Returns
        -------
        LArray
            Array expanded with `value` along `axis`.

        Examples
        --------
        >>> a = ones('nat=BE,FO;sex=M,F')
        >>> a
        nat\\sex    M    F
             BE  1.0  1.0
             FO  1.0  1.0
        >>> a.append(X.sex, a.sum(X.sex), 'M+F')
        nat\\sex    M    F  M+F
             BE  1.0  1.0  2.0
             FO  1.0  1.0  2.0
        >>> a.append(X.nat, 2, 'Other')
        nat\\sex    M    F
             BE  1.0  1.0
             FO  1.0  1.0
          Other  2.0  2.0
        >>> b = zeros('type=type1,type2')
        >>> b
        type  type1  type2
                0.0    0.0
        >>> a.append(X.nat, b, 'Other')
          nat  sex\\type  type1  type2
           BE         M    1.0    1.0
           BE         F    1.0    1.0
           FO         M    1.0    1.0
           FO         F    1.0    1.0
        Other         M    0.0    0.0
        Other         F    0.0    0.0
        """
        axis = self.axes[axis]
        return self.insert(value, pos=len(axis), axis=axis, label=label)

    def prepend(self, axis, value, label=None):
        """Adds an array before self along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis reference
            Axis along which to prepend input array (`value`)
        value : LArray
            Array with compatible axes.
        label : str, optional
            Label for the new item in axis

        Returns
        -------
        LArray
            Array expanded with 'value' at the start of 'axis'.

        Examples
        --------
        >>> a = ones('nat=BE,FO;sex=M,F')
        >>> a
        nat\sex    M    F
             BE  1.0  1.0
             FO  1.0  1.0
        >>> a.prepend(X.sex, a.sum(X.sex), 'M+F')
        nat\\sex  M+F    M    F
             BE  2.0  1.0  1.0
             FO  2.0  1.0  1.0
        >>> a.prepend(X.nat, 2, 'Other')
        nat\\sex    M    F
          Other  2.0  2.0
             BE  1.0  1.0
             FO  1.0  1.0
        >>> b = zeros('type=type1,type2')
        >>> b
        type  type1  type2
                0.0    0.0
        >>> a.prepend(X.sex, b, 'Other')
        nat  sex\\type  type1  type2
         BE     Other    0.0    0.0
         BE         M    1.0    1.0
         BE         F    1.0    1.0
         FO     Other    0.0    0.0
         FO         M    1.0    1.0
         FO         F    1.0    1.0
        """
        return self.insert(value, pos=0, axis=axis, label=label)

    def extend(self, axis, other):
        """Adds an array to self along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis
            Axis along which to extend with input array (`other`)
        other : LArray
            Array with compatible axes

        Returns
        -------
        LArray
            Array expanded with 'other' along 'axis'.

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> sex2 = Axis('sex=U')
        >>> xtype = Axis('type=type1,type2')
        >>> arr1 = ones([sex, xtype])
        >>> arr1
        sex\\type  type1  type2
               M    1.0    1.0
               F    1.0    1.0
        >>> arr2 = zeros([sex2, xtype])
        >>> arr2
        sex\\type  type1  type2
               U    0.0    0.0
        >>> arr1.extend(X.sex, arr2)
        sex\\type  type1  type2
               M    1.0    1.0
               F    1.0    1.0
               U    0.0    0.0
        >>> arr3 = zeros([sex2, nat])
        >>> arr3
        sex\\nat   BE   FO
              U  0.0  0.0
        >>> arr1.extend(X.sex, arr3)
        sex  type\\nat   BE   FO
          M     type1  1.0  1.0
          M     type2  1.0  1.0
          F     type1  1.0  1.0
          F     type2  1.0  1.0
          U     type1  0.0  0.0
          U     type2  0.0  0.0
        """
        return concat((self, other), axis)

    def insert(self, value, before=None, after=None, pos=None, axis=None, label=None):
        """Inserts value in array along an axis.

        Parameters
        ----------
        value : scalar or LArray
            Value to insert. If an LArray, it must have compatible axes. If value already has the axis along which it
            is inserted, `label` should not be used.
        before : scalar or Group
            Label or group before which to insert `value`.
        after : scalar or Group
            Label or group after which to insert `value`.
        pos : int
            Index before which to insert `value`.
        axis : axis reference (int, str or Axis), optional
            Axis in which to insert `value`. This is only required when using `pos` or when before or after are
            ambiguous labels.
        label : str, optional
            Label for the new item in axis.

        Returns
        -------
        LArray
            Array with `value` inserted along `axis`. The dtype of the returned array will be the "closest" type
            which can hold both the array values and the inserted values without loss of information. For example,
            when mixing numeric and string types, the dtype will be object.

        Examples
        --------
        >>> arr1 = ndtest((2, 3))
        >>> arr1
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> arr1.insert(42, before='b1', label='b0.5')
        a\\b  b0  b0.5  b1  b2
         a0   0    42   1   2
         a1   3    42   4   5
        >>> arr2 = ndtest(2)
        >>> arr2
        a  a0  a1
            0   1
        >>> arr1.insert(arr2, after='b0', label='b0.5')
        a\\b  b0  b0.5  b1  b2
         a0   0     0   1   2
         a1   3     1   4   5
        >>> arr1.insert(42, axis='b', pos=1, label='b0.5')
        a\\b  b0  b0.5  b1  b2
         a0   0    42   1   2
         a1   3    42   4   5
        >>> arr1.insert(42, before=X.b.i[1], label='b0.5')
        a\\b  b0  b0.5  b1  b2
         a0   0    42   1   2
         a1   3    42   4   5

        insert an array which already has the axis

        >>> arr3 = ndtest('a=a0,a1;b=b0.1,b0.2') + 42
        >>> arr3
        a\\b  b0.1  b0.2
         a0    42    43
         a1    44    45
        >>> arr1.insert(arr3, before='b1')
        a\\b  b0  b0.1  b0.2  b1  b2
         a0   0    42    43   1   2
         a1   3    44    45   4   5
        """

        # XXX: unsure we should have arr1.insert(arr3, before='b1,b2') result in (see unit tests):

        # a\\b  b0  b0.1  b1  b0.2  b2
        #  a0   0    42   1    43   2
        #  a1   3    44   4    45   5

        # we might to implement the following instead:

        # a\\b  b0  b0.1  b0.2  b1  b0.1  b0.2  b2
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
        # a\\b  b0  b1  b2
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
        # a\\b  b0  b0.5  b1  b1.5  b2
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
        if sum([before is not None, after is not None, pos is not None]) != 1:
            raise ValueError("must specify exactly one of before, after or pos")

        axis = self.axes[axis] if axis is not None else None
        if before is not None:
            before = self._translate_axis_key(before) if axis is None else axis[before]
            axis = before.axis
            before_pos = axis.index(before)
        elif after is not None:
            after = self._translate_axis_key(after) if axis is None else axis[after]
            axis = after.axis
            before_pos = axis.index(after) + 1
        else:
            assert pos is not None
            if axis is None:
                raise ValueError("axis argument must be provided when using insert(pos=)")
            before_pos = pos

        def length(v):
            if isinstance(v, LArray) and axis in v.axes:
                return len(v.axes[axis])
            else:
                return len(v) if isinstance(v, (tuple, list, np.ndarray)) else 1

        def expand(v, length):
            return v if isinstance(v, (tuple, list, np.ndarray)) else [v] * length

        num_inserts = max(length(before_pos), length(label), length(value))
        stops = expand(before_pos, num_inserts)

        if isinstance(value, LArray) and axis in value.axes:
            # FIXME: when length(before_pos) == 1 and length(label) == 1, this is inefficient
            values = [value[[k]] for k in value.axes[axis]]
        else:
            values = expand(value, num_inserts)
        values = [aslarray(v) if not isinstance(v, LArray) else v
                  for v in values]

        if label is not None:
            labels = expand(label, num_inserts)
            values = [v.expand(Axis([l], axis.name), readonly=True) for v, l in zip(values, labels)]

        start = 0
        chunks = []
        for stop, value in zip(stops, values):
            chunks.append(self[axis.i[start:stop]])
            chunks.append(value)
            start = stop
        chunks.append(self[axis.i[start:]])
        return concat(chunks, axis)

    def transpose(self, *args):
        """Reorder axes.

        Parameters
        ----------
        *args
            Accepts either a tuple of axes specs or axes specs as `*args`. Omitted axes keep their order.
            Use ... to avoid specifying intermediate axes.

        Returns
        -------
        LArray
            LArray with reordered axes.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a  b\\c  c0  c1
        a0   b0   0   1
        a0   b1   2   3
        a1   b0   4   5
        a1   b1   6   7
        >>> arr.transpose('b', 'c', 'a')
         b  c\\a  a0  a1
        b0   c0   0   4
        b0   c1   1   5
        b1   c0   2   6
        b1   c1   3   7
        >>> arr.transpose('b')
         b  a\\c  c0  c1
        b0   a0   0   1
        b0   a1   4   5
        b1   a0   2   3
        b1   a1   6   7
        >>> arr.transpose(..., 'a')  # doctest: +SKIP
         b  c\\a  a0  a1
        b0   c0   0   4
        b0   c1   1   5
        b1   c0   2   6
        b1   c1   3   7
        >>> arr.transpose('c', ..., 'a')  # doctest: +SKIP
         c  b\\a  a0  a1
        c0   b0   0   4
        c0   b1   2   6
        c1   b0   1   5
        c1   b1   3   7
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list, AxisCollection)):
            axes = args[0]
        elif len(args) == 0:
            axes = self.axes[::-1]
        else:
            axes = args

        axes = self.axes[axes]
        axes_indices = [self.axes.index(axis) for axis in axes]
        # this whole mumbo jumbo is required (for now) for anonymous axes
        indices_present = set(axes_indices)
        missing_indices = [i for i in range(len(self.axes)) if i not in indices_present]
        axes_indices = axes_indices + missing_indices
        return LArray(self.data.transpose(axes_indices), self.axes[axes_indices])
    T = property(transpose)

    def clip(self, a_min, a_max, out=None):
        """Clip (limit) the values in an array.

        Given an interval, values outside the interval are clipped to the interval edges.
        For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
        and values larger than 1 become 1.

        Parameters
        ----------
        a_min : scalar or array-like
            Minimum value.
        a_max : scalar or array-like
            Maximum value.
        out : LArray, optional
            The results will be placed in this array.

        Returns
        -------
        LArray
            An array with the elements of the current array,
            but where values < `a_min` are replaced with `a_min`, and those > `a_max` with `a_max`.

        Notes
        -----
        If `a_min` and/or `a_max` are array_like, broadcast will occur between self, `a_min` and `a_max`.
        """
        from larray.core.ufuncs import clip
        return clip(self, a_min, a_max, out)

    @deprecate_kwarg('transpose', 'wide')
    def to_csv(self, filepath, sep=',', na_rep='', wide=True, value_name='value', dropna=None, dialect='default', **kwargs):
        """
        Writes array to a csv file.

        Parameters
        ----------
        filepath : str
            path where the csv file has to be written.
        sep : str, optional
            separator for the csv file. Defaults to `,`.
        na_rep : str, optional
            replace NA values with na_rep. Defaults to ''.
        wide : boolean, optional
            Whether or not writing arrays in "wide" format. If True, arrays are exported with the last axis
            represented horizontally. If False, arrays are exported in "narrow" format: one column per axis plus one
            value column. Defaults to True.
        value_name : str, optional
            Name of the column containing the values (last column) in the csv file when `wide=False` (see above).
            Defaults to 'value'.
        dialect : 'default' | 'classic', optional
            Whether or not to write the last axis name (using '\' ). Defaults to 'default'.
        dropna : None, 'all', 'any' or True, optional
            Drop lines if 'all' its values are NA, if 'any' value is NA or do not drop any line (default).
            True is equivalent to 'all'.

        Examples
        --------
        >>> tmpdir = getfixture('tmpdir')
        >>> fname = os.path.join(tmpdir.strpath, 'test.csv')
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a
        nat\\sex  M  F
             BE  0  1
             FO  2  3
        >>> a.to_csv(fname)
        >>> with open(fname) as f:
        ...     print(f.read().strip())
        nat\\sex,M,F
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

    def to_hdf(self, filepath, key):
        """
        Writes array to a HDF file.

        A HDF file can contain multiple arrays.
        The 'key' parameter is a unique identifier for the array.

        Parameters
        ----------
        filepath : str
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

        >>> a.to_hdf('test.h5', 'arrays/a')  # doctest: +SKIP
        """
        key = _translate_group_key_hdf(key)
        with LHDFStore(filepath) as store:
            store.put(key, self.to_frame())
            store.get_storer(key).attrs.type = 'Array'

    @deprecate_kwarg('sheet_name', 'sheet') 
    def to_excel(self, filepath=None, sheet=None, position='A1', overwrite_file=False, clear_sheet=False,
                 header=True, transpose=False, wide=True, value_name='value', engine=None, *args, **kwargs):
        """
        Writes array in the specified sheet of specified excel workbook.

        Parameters
        ----------
        filepath : str or int or None, optional
            Path where the excel file has to be written. If None (default), creates a new Excel Workbook in a live Excel
            instance (Windows only). Use -1 to use the currently active Excel Workbook. Use a name without extension
            (.xlsx) to use any unsaved* workbook.
        sheet : str or Group or int or None, optional
            Sheet where the data has to be written. Defaults to None, Excel standard name if adding a sheet to an
            existing file, "Sheet1" otherwise. sheet can also refer to the position of the sheet
            (e.g. 0 for the first sheet, -1 for the last one).
        position : str or tuple of integers, optional
            Integer position (row, column) must be 1-based. Defaults to 'A1'.
        overwrite_file : bool, optional
            Whether or not to overwrite the existing file (or just modify the specified sheet). Defaults to False.
        clear_sheet : bool, optional
            Whether or not to clear the existing sheet (if any) before writing. Defaults to False.
        header : bool, optional
            Whether or not to write a header (axes names and labels). Defaults to True.
        transpose : bool, optional
            Whether or not to transpose the array over last axis.
            This is equivalent to paste with option transpose in Excel. Defaults to False.
        wide : boolean, optional
            Whether or not writing arrays in "wide" format. If True, arrays are exported with the last axis
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

        if engine == 'xlwings':
            from larray.inout.xw_excel import open_excel

            close = False
            new_workbook = False
            if filepath is None:
                new_workbook = True
            elif isinstance(filepath, str):
                basename, ext = os.path.splitext(filepath)
                if ext:
                    if not os.path.isfile(filepath):
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
            pd_obj.to_excel(filepath, sheet, *args, engine=engine, **kwargs)

    def to_clipboard(self, *args, **kwargs):
        """Sends the content of the array to clipboard.

        Using to_clipboard() makes it possible to paste the content of the array into a file (Excel, ascii file,...).

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a.to_clipboard()  # doctest: +SKIP
        """
        self.to_frame().to_clipboard(*args, **kwargs)

    # XXX: sep argument does not seem very useful
    # def to_excel(self, filename, sep=None):
    #     # Why xlsxwriter? Because it is faster than openpyxl and xlwt
    #     # currently does not .xlsx (only .xls).
    #     # PyExcelerate seem like a decent alternative too
    #     import xlsxwriter as xl
    #
    #     if sep is None:
    #         sep = '_'
    #         #sep = self.sep
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
    def plot(self):
        """Plots the data of the array into a graph (window pop-up).

        The graph can be tweaked to achieve the desired formatting and can be saved to a .png file.

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
        ax : matplotlib axes object, default None
        subplots : boolean, default False
            Make separate subplots for each column
        sharex : boolean, default True if ax is None else False
            In case subplots=True, share x axis and set some x axis labels to invisible;
            defaults to True if ax is None otherwise False if an ax is passed in;
            Be aware, that passing in both an ax and sharex=True will alter all x axis labels for all axis in a figure!
        sharey : boolean, default False
            In case subplots=True, share y axis and set some y axis labels to invisible
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
            Place legend on axis subplots
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
        xlim : 2-tuple/list
        ylim : 2-tuple/list
        rot : int, default None
            Rotation for ticks (xticks for vertical, yticks for horizontal plots)
        fontsize : int, default None
            Font size for xticks and yticks
        colormap : str or matplotlib colormap object, default None
            Colormap to select colors from. If string, load colormap with that name from matplotlib.
        colorbar : boolean, optional
            If True, plot colorbar (only relevant for 'scatter' and 'hexbin' plots)
        position : float
            Specify relative alignments for bar plot layout. From 0 (left/bottom-end) to 1 (right/top-end).
            Default is 0.5 (center)
        layout : tuple (optional)
            (rows, columns) for the layout of the plot
        yerr : array-like
            Error bars on y axis
        xerr : array-like
            Error bars on x axis
        stacked : boolean, default False in line and bar plots, and True in area plot.
            If True, create stacked plot.
        \**kwargs : keywords
            Options to pass to matplotlib plotting method

        Returns
        -------
        axes : matplotlib.AxesSubplot or np.array of them

        Notes
        -----
        See Pandas documentation of `plot` function for more details on this subject

        Examples
        --------
        >>> import matplotlib.pyplot as plt # doctest: +SKIP
        >>> a = ndtest('sex=M,F;age=0..20')

        Simple line plot

        >>> a.plot() # doctest: +SKIP
        >>> # shows figure (reset the current figure after showing it! Do not call it before savefig)
        >>> plt.show() # doctest: +SKIP

        Line plot with grid, title and both axes in logscale

        >>> a.plot(grid=True, loglog=True, title='line plot') # doctest: +SKIP
        >>> # saves figure in a file (see matplotlib.pyplot.savefig documentation for more details)
        >>> plt.savefig('my_file.png') # doctest: +SKIP

        2 bar plots sharing the same x axis (one for males and one for females)

        >>> a.plot.bar(subplots=True, sharex=True) # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

        Create a figure containing 2 x 2 graphs

        >>> # see matplotlib.pyplot.subplots documentation for more details
        >>> fig, ax = plt.subplots(2, 2, figsize=(15, 15)) # doctest: +SKIP
        >>> # 2 curves : Males and Females
        >>> a.plot(ax=ax[0, 0], title='line plot') # doctest: +SKIP
        >>> # bar plot with stacked values
        >>> a.plot.bar(ax=ax[0, 1], stacked=True, title='stacked bar plot') # doctest: +SKIP
        >>> # same as previously but with colored areas instead of bars
        >>> a.plot.area(ax=ax[1, 0], title='area plot') # doctest: +SKIP
        >>> # scatter plot
        >>> a.plot.scatter(ax=ax[1, 1], x='M', y='F', title='scatter plot') # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
        """
        combined = self.combine_axes(self.axes[:-1], sep=' ') if self.ndim > 2 else self
        if combined.ndim == 1:
            return combined.to_series().plot
        else:
            return combined.transpose().to_frame().plot

    @property
    def shape(self):
        """Returns the shape of the array as a tuple.

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
    def ndim(self):
        """Returns the number of dimensions of the array.

        Returns
        -------
        int
            Number of dimensions of a LArray.

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a.ndim
        2
        """
        return self.data.ndim

    @property
    def size(self):
        """Returns the number of elements in array.

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
    def nbytes(self):
        """Returns the number of bytes used to store the array in memory.

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
    def memory_used(self):
        """Returns the memory consumed by the array in human readable form.

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
    def dtype(self):
        """Returns the type of the data of the array.

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
    def item(self):
        return self.data.item

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    __array_priority__ = 100

    # XXX: implement guess axis?
    """
    # guessing each axis
    >>> a.set_labels({'M': 'Men', 'BE': 'Belgian'})
    nat\\sex  Men  Women
    BE  0  1
    FO  2  3

    # we have to choose which one to support because it is probably not a good idea to simultaneously support the
    # following syntax (even though we *could* support both if we split values on , before we determine if the key is
    # an axis or a label by looking if the value is a list or a single string.
    >>> a.set_labels({'sex': 'Men,Women', 'BE': 'Belgian'})
    nat\\sex  Men  Women
    BE  0  1
    FO  2  3
    # this is shorter but I do not like it because string are both quoted and not quoted and you cannot have int
    # labels
    >>> a.set_labels(M='Men', BE='Belgian')
    nat\\sex  Men  Women
    BE  0  1
    FO  2  3
    """
    def set_labels(self, axis=None, labels=None, inplace=False, **kwargs):
        """Replaces the labels of an axis of array.

        Parameters
        ----------
        axis : string or Axis or dict
            Axis for which we want to replace labels, or mapping {axis: changes} where changes can either be the
            complete list of labels or a mapping {old_label: new_label}.
        labels : int, str, iterable or mapping, optional
            Integer or list of values usable as the collection of labels for an Axis. If this is mapping, it must be
            {old_label: new_label}. This argument must not be used if axis is a mapping.
        inplace : bool, optional
            Whether or not to modify the original object or return a new array and leave the original intact.
            Defaults to False.
        **kwargs :
            `axis`=`labels` for each axis you want to set labels.

        Returns
        -------
        LArray
            Array with modified labels.

        Examples
        --------
        >>> a = ndtest('nat=BE,FO;sex=M,F')
        >>> a
        nat\\sex  M  F
             BE  0  1
             FO  2  3
        >>> a.set_labels(X.sex, ['Men', 'Women'])
        nat\\sex  Men  Women
             BE    0      1
             FO    2      3

        when passing a single string as labels, it will be interpreted to create the list of labels, so that one can
        use the same syntax than during axis creation.

        >>> a.set_labels(X.sex, 'Men,Women')
        nat\\sex  Men  Women
             BE    0      1
             FO    2      3

        to replace only some labels, one must give a mapping giving the new label for each label to replace

        >>> a.set_labels(X.sex, {'M': 'Men'})
        nat\\sex  Men  F
             BE    0  1
             FO    2  3

        to replace labels for several axes at the same time, one should give a mapping giving the new labels for each
        changed axis

        >>> a.set_labels({'sex': 'Men,Women', 'nat': 'Belgian,Foreigner'})
          nat\\sex  Men  Women
          Belgian    0      1
        Foreigner    2      3

        or use keyword arguments

        >>> a.set_labels(sex='Men,Women', nat='Belgian,Foreigner')
          nat\\sex  Men  Women
          Belgian    0      1
        Foreigner    2      3

        one can also replace some labels in several axes by giving a mapping of mappings

        >>> a.set_labels({'sex': {'M': 'Men'}, 'nat': {'BE': 'Belgian'}})
        nat\\sex  Men  F
        Belgian    0  1
             FO    2  3
        """
        if axis is None:
            changes = {}
        elif isinstance(axis, dict):
            changes = axis
        elif isinstance(axis, (basestring, Axis, int)):
            changes = {axis: labels}
        else:
            raise ValueError("Expected None or a string/int/Axis/dict instance for axis argument")
        changes.update(kwargs)
        # TODO: we should implement the non-dict behavior in Axis.replace, so that we can simplify this code to:
        # new_axes = [self.axes[old_axis].replace(axis_changes) for old_axis, axis_changes in changes.items()]
        new_axes = []
        for old_axis, axis_changes in changes.items():
            real_axis = self.axes[old_axis]
            if isinstance(axis_changes, dict):
                new_axis = real_axis.replace(axis_changes)
            else:
                new_axis = Axis(axis_changes, real_axis.name)
            new_axes.append((old_axis, new_axis))
        axes = self.axes.replace(new_axes)

        if inplace:
            self.axes = axes
            return self
        else:
            return LArray(self.data, axes)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return LArray(self.data.astype(dtype, order, casting, subok, copy), self.axes)
    astype.__doc__ = np.ndarray.astype.__doc__

    def shift(self, axis, n=1):
        """Shifts the cells of the array n-times to the left along axis.

        Parameters
        ----------
        axis : int, str or Axis
            Axis for which we want to perform the shift.
        n : int
            Number of cells to shift.

        Returns
        -------
        LArray

        Examples
        --------
        >>> a = ndtest('sex=M,F;type=type1,type2,type3')
        >>> a
        sex\\type  type1  type2  type3
               M      0      1      2
               F      3      4      5
        >>> a.shift(X.type)
        sex\\type  type2  type3
               M      0      1
               F      3      4
        >>> a.shift(X.type, n=-1)
        sex\\type  type1  type2
               M      1      2
               F      4      5
        """
        axis = self.axes[axis]
        if n > 0:
            return self[axis.i[:-n]].set_labels(axis, axis.labels[n:])
        elif n < 0:
            return self[axis.i[-n:]].set_labels(axis, axis.labels[:n])
        else:
            return self[:]

    # TODO: add support for groups as axis (like aggregates)
    # eg a.diff(x.year[2018:]) instead of a[2018:].diff(x.year)
    def diff(self, axis=-1, d=1, n=1, label='upper'):
        """Calculates the n-th order discrete difference along a given axis.

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
        LArray :
            The n-th order differences. The shape of the output is the same as `a` except for `axis` which is smaller
            by `n` * `d`.

        Examples
        --------
        >>> a = ndtest('sex=M,F;type=type1,type2,type3').cumsum('type')
        >>> a
        sex\\type  type1  type2  type3
               M      0      1      3
               F      3      7     12
        >>> a.diff()
        sex\\type  type2  type3
               M      1      2
               F      4      5
        >>> a.diff(n=2)
        sex\\type  type3
               M      1
               F      1
        >>> a.diff('sex')
        sex\\type  type1  type2  type3
               F      3      6      9
        >>> a.diff(a.type['type2':])
        sex\\type  type3
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
                right = right.drop_labels(axis)
            else:
                left = left.drop_labels(axis)
            array = left - right
        return array

    # XXX: this is called pct_change in Pandas (but returns the same results, not results * 100, which I find silly).
    # Maybe change_rate would be better (because growth is not always positive)?
    def growth_rate(self, axis=-1, d=1, label='upper'):
        """Calculates the growth along a given axis.

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
        LArray

        Examples
        --------
        >>> data = [[2, 4, 5, 4, 6], [4, 6, 3, 6, 9]]
        >>> a = LArray(data, "sex=M,F; year=2016..2020")
        >>> a
        sex\\year  2016  2017  2018  2019  2020
               M     2     4     5     4     6
               F     4     6     3     6     9
        >>> a.growth_rate()
        sex\\year  2017  2018  2019  2020
               M   1.0  0.25  -0.2   0.5
               F   0.5  -0.5   1.0   0.5
        >>> a.growth_rate(label='lower')
        sex\\year  2016  2017  2018  2019
               M   1.0  0.25  -0.2   0.5
               F   0.5  -0.5   1.0   0.5
        >>> a.growth_rate(d=2)
        sex\\year   2018  2019  2020
               M    1.5   0.0   0.2
               F  -0.25   0.0   2.0
        >>> a.growth_rate('sex')
        sex\\year  2016  2017  2018  2019  2020
               F   1.0   0.5  -0.4   0.5   0.5
        >>> a.growth_rate(a.year[2017:])
        sex\\year  2018  2019  2020
               M  0.25  -0.2   0.5
               F  -0.5   1.0   0.5
        """
        if isinstance(axis, Group):
            array = self[axis]
            axis = array.axes[axis.axis]
        else:
            array = self
            axis = array.axes[axis]
        diff = array.diff(axis=axis, d=d, label=label)
        return diff / array[axis.i[:-d]].drop_labels(axis)

    def compact(self):
        """Detects and removes "useless" axes (ie axes for which values are constant over the whole axis)

        Returns
        -------
        LArray or scalar
            Array with constant axes removed.

        Examples
        --------
        >>> a = LArray([[1, 2],
        ...             [1, 2]], [Axis('sex=M,F'), Axis('nat=BE,FO')])
        >>> a
        sex\\nat  BE  FO
              M   1   2
              F   1   2
        >>> a.compact()
        nat  BE  FO
              1   2
        """
        res = self
        for axis in res.axes:
            if (res == res[axis.i[0]]).all():
                res = res[axis.i[0]]
        return res

    def combine_axes(self, axes=None, sep='_', wildcard=False):
        """Combine several axes into one.

        Parameters
        ----------
        axes : tuple, list, AxisCollection of axes or list of combination of those or dict, optional
            axes to combine. Tuple, list or AxisCollection will combine several axes into one. To chain several axes
            combinations, pass a list of tuple/list/AxisCollection of axes. To set the name(s) of resulting axis(es),
            use a {(axes, to, combine): 'new_axis_name'} dictionary. Defaults to all axes.
        sep : str, optional
            delimiter to use for combining. Defaults to '_'.
        wildcard : bool, optional
            whether or not to produce a wildcard axis even if the axes to combine are not. This is much faster,
            but loose axes labels.

        Returns
        -------
        LArray
            Array with combined axes.

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\\b  b0  b1  b2
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
         a   b  c\\d  d0  d1
        a0  b0   c0   0   1
        a0  b0   c1   2   3
        a0  b1   c0   4   5
        a0  b1   c1   6   7
        a1  b0   c0   8   9
        a1  b0   c1  10  11
        a1  b1   c0  12  13
        a1  b1   c1  14  15
        >>> arr.combine_axes(('a', 'c'))
          a_c  b\\d  d0  d1
        a0_c0   b0   0   1
        a0_c0   b1   4   5
        a0_c1   b0   2   3
        a0_c1   b1   6   7
        a1_c0   b0   8   9
        a1_c0   b1  12  13
        a1_c1   b0  10  11
        a1_c1   b1  14  15
        >>> arr.combine_axes({('a', 'c'): 'ac'})
           ac  b\\d  d0  d1
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
        a_c\\b_d  b0_d0  b0_d1  b1_d0  b1_d1
          a0_c0      0      1      4      5
          a0_c1      2      3      6      7
          a1_c0      8      9     12     13
          a1_c1     10     11     14     15
        >>> arr.combine_axes({('a', 'c'): 'ac', ('b', 'd'): 'bd'})
        ac\\bd  b0_d0  b0_d1  b1_d0  b1_d1
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
        for axes_to_combine, name in axes.items():
            # transpose all axes next to each other, using index of first axis
            axes_to_combine = self.axes[axes_to_combine]
            axes_indices = [transposed_axes.index(axis) for axis in axes_to_combine]
            min_axis_index = min(axes_indices)
            transposed_axes = transposed_axes - axes_to_combine
            transposed_axes = transposed_axes[:min_axis_index] + axes_to_combine + transposed_axes[min_axis_index:]
        transposed = self.transpose(transposed_axes)

        new_axes = transposed.axes.combine_axes(axes, sep=sep, wildcard=wildcard)
        return transposed.reshape(new_axes)

    def split_axes(self, axes=None, sep='_', names=None, regex=None, sort=False, fill_value=nan):
        """Split axes and returns a new array

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
            Whether or not to sort the combined axis before splitting it. When all combinations of labels are present in
            the combined axis, sorting is faster than not sorting. Defaults to False.
        fill_value : scalar or LArray, optional
            Value to use for missing values when the combined axis does not contain all combination of labels.
            Defaults to NaN.

        Returns
        -------
        LArray

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> combined = arr.combine_axes()
        >>> combined
        a_b  a0_b0  a0_b1  a0_b2  a1_b0  a1_b1  a1_b2
                 0      1      2      3      4      5
        >>> combined.split_axes()
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5

        Split labels using regex

        >>> combined = ndtest('a_b=a0b0..a1b2')
        >>> combined
        a_b  a0b0  a0b1  a0b2  a1b0  a1b1  a1b2
                0     1     2     3     4     5
        >>> combined.split_axes('a_b', regex='(\\\\w{2})(\\\\w{2})')
        a\\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5

        Split several axes at once

        >>> combined = ndtest('a_b=a0_b0..a1_b1; c_d=c0_d0..c1_d1')
        >>> combined
        a_b\\c_d  c0_d0  c0_d1  c1_d0  c1_d1
          a0_b0      0      1      2      3
          a0_b1      4      5      6      7
          a1_b0      8      9     10     11
          a1_b1     12     13     14     15
        >>> # equivalent to combined.split_axes() which split all axes whose name contains the `sep` delimiter.
        >>> combined.split_axes(['a_b', 'c_d'])
         a   b  c\\d  d0  d1
        a0  b0   c0   0   1
        a0  b0   c1   2   3
        a0  b1   c0   4   5
        a0  b1   c1   6   7
        a1  b0   c0   8   9
        a1  b0   c1  10  11
        a1  b1   c0  12  13
        a1  b1   c1  14  15
        >>> combined.split_axes({'a_b': ('A', 'B'), 'c_d': ('C', 'D')})
         A   B  C\\D  d0  d1
        a0  b0   c0   0   1
        a0  b0   c1   2   3
        a0  b1   c0   4   5
        a0  b1   c1   6   7
        a1  b0   c0   8   9
        a1  b0   c1  10  11
        a1  b1   c0  12  13
        a1  b1   c1  14  15
        """
        array = self.sort_axes(axes) if sort else self
        # TODO:
        # * do multiple axes split in one go
        # * somehow factorize this code with AxisCollection.split_axes
        if axes is None:
            axes = {axis: None for axis in array.axes if sep in axis.name}
        elif isinstance(axes, (int, basestring, Axis)):
            axes = {axes: None}
        elif isinstance(axes, (list, tuple)):
            if all(isinstance(axis, (int, basestring, Axis)) for axis in axes):
                axes = {axis: None for axis in axes}
            else:
                raise ValueError("Expected tuple or list of int, string or Axis instances")
        # axes should be a dict at this time
        assert isinstance(axes, dict)
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
                    res = full(new_axes, fill_value=fill_value, dtype=common_type((array, fill_value)))
                if names is None:
                    names = axis.name.split(sep)
                # Rename axis to make sure we broadcast correctly. We should NOT use sep here, but rather '_' must be
                # kept in sync with the default sep of _bool_key_new_axes
                new_axis_name = '_'.join(names)
                if new_axis_name != axis.name:
                    array = array.rename(axis, new_axis_name)
                res.points[split_labels] = array
                array = res
        return array
    split_axis = renamed_to(split_axes, 'split_axis')


def larray_equal(a1, a2):
    import warnings
    msg = "larray_equal() is deprecated. Use LArray.equals() instead."
    warnings.warn(msg, FutureWarning, stacklevel=2)
    try:
        a1 = aslarray(a1)
    except Exception:
        return False
    return a1.equals(a2)


def larray_nan_equal(a1, a2):
    import warnings
    msg = "larray_nan_equal() is deprecated. Use LArray.equals() instead."
    warnings.warn(msg, FutureWarning, stacklevel=2)
    try:
        a1 = aslarray(a1)
    except Exception:
        return False
    return a1.equals(a2, nan_equals=True)


def aslarray(a):
    """
    Converts input as LArray if possible.

    Parameters
    ----------
    a : array-like
        Input array to convert into a LArray.

    Returns
    -------
    LArray

    Examples
    --------
    >>> # NumPy array
    >>> np_arr = np.arange(6).reshape((2,3))
    >>> aslarray(np_arr)
    {0}*\{1}*  0  1  2
            0  0  1  2
            1  3  4  5
    >>> # Pandas dataframe
    >>> data = {'normal'  : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    ...         'reverse' : pd.Series([3., 2., 1.], index=['a', 'b', 'c'])}
    >>> df = pd.DataFrame(data)
    >>> aslarray(df)
    {0}\{1}  normal  reverse
          a     1.0      3.0
          b     2.0      2.0
          c     3.0      1.0
    """
    if isinstance(a, LArray):
        return a
    elif hasattr(a, '__larray__'):
        return a.__larray__()
    elif isinstance(a, pd.DataFrame):
        from larray.inout.pandas import from_frame
        return from_frame(a)
    else:
        return LArray(a)


def _check_axes_argument(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 1 and isinstance(args[1], (int, Axis)):
            raise ValueError("If you want to pass several axes or dimension lengths to {}, you must pass them as a "
                             "list (using []) or tuple (using()).".format(func.__name__))
        return func(*args, **kwargs)
    return wrapper


@_check_axes_argument
def zeros(axes, title='', dtype=float, order='C'):
    """Returns an array with the specified axes and filled with zeros.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`. Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.

    Returns
    -------
    LArray

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
    axes = AxisCollection(axes)
    return LArray(np.zeros(axes.shape, dtype, order), axes, title)


def zeros_like(array, title='', dtype=None, order='K'):
    """Returns an array with the same axes as array and filled with zeros.

    Parameters
    ----------
    array : LArray
         Input array.
    title : str, optional
        Title.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.

    Returns
    -------
    LArray

    Examples
    --------
    >>> a = ndtest((2, 3))
    >>> zeros_like(a)
    a\\b  b0  b1  b2
     a0   0   0   0
     a1   0   0   0
    """
    if not title:
        title = array.title
    return LArray(np.zeros_like(array, dtype, order), array.axes, title)


@_check_axes_argument
def ones(axes, title='', dtype=float, order='C'):
    """Returns an array with the specified axes and filled with ones.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.  Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> ones([nat, sex])
    nat\\sex    M    F
         BE  1.0  1.0
         FO  1.0  1.0
    """
    axes = AxisCollection(axes)
    return LArray(np.ones(axes.shape, dtype, order), axes, title)


def ones_like(array, title='', dtype=None, order='K'):
    """Returns an array with the same axes as array and filled with ones.

    Parameters
    ----------
    array : LArray
        Input array.
    title : str, optional
        Title.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.

    Returns
    -------
    LArray

    Examples
    --------
    >>> a = ndtest((2, 3))
    >>> ones_like(a)
    a\\b  b0  b1  b2
     a0   1   1   1
     a1   1   1   1
    """
    axes = array.axes
    if not title:
        title = array.title
    return LArray(np.ones_like(array, dtype, order), axes, title)


@_check_axes_argument
def empty(axes, title='', dtype=float, order='C'):
    """Returns an array with the specified axes and uninitialized (arbitrary) data.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.  Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> empty([nat, sex])  # doctest: +SKIP
    nat\\sex                   M                   F
         BE  2.47311483356e-315  2.47498446195e-315
         FO                 0.0  6.07684618082e-31
    """
    axes = AxisCollection(axes)
    return LArray(np.empty(axes.shape, dtype, order), axes, title)


def empty_like(array, title='', dtype=None, order='K'):
    """Returns an array with the same axes as array and uninitialized (arbitrary) data.

    Parameters
    ----------
    array : LArray
        Input array.
    title : str, optional
        Title.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the data type of array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.

    Returns
    -------
    LArray

    Examples
    --------
    >>> a = ndtest((3, 2))
    >>> empty_like(a)   # doctest: +SKIP
    a\\b                  b0                  b1
     a0  2.12199579097e-314  6.36598737388e-314
     a1  1.06099789568e-313  1.48539705397e-313
     a2  1.90979621226e-313  2.33419537056e-313
    """
    if not title:
        title = array.title
    # cannot use empty() because order == 'K' is not understood
    return LArray(np.empty_like(array.data, dtype, order), array.axes, title)


# We cannot use @_check_axes_argument here because an integer fill_value would be considered as an error
def full(axes, fill_value, title='', dtype=None, order='C'):
    """Returns an array with the specified axes and filled with fill_value.

    Parameters
    ----------
    axes : int, tuple of int, Axis or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    fill_value : scalar or LArray
        Value to fill the array
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array. Default is the data type of fill_value.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or Fortran-contiguous (row- or column-wise) order in
        memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> full([nat, sex], 42.0)
    nat\\sex     M     F
         BE  42.0  42.0
         FO  42.0  42.0
    >>> initial_value = ndtest([sex])
    >>> initial_value
    sex  M  F
         0  1
    >>> full([nat, sex], initial_value)
    nat\\sex  M  F
         BE  0  1
         FO  0  1
    """
    if isinstance(fill_value, Axis):
        raise ValueError("If you want to pass several axes or dimension lengths to full, you must pass them as a "
                         "list (using []) or tuple (using()).")
    if dtype is None:
        dtype = np.asarray(fill_value).dtype
    res = empty(axes, title, dtype, order)
    res[:] = fill_value
    return res


def full_like(array, fill_value, title='', dtype=None, order='K'):
    """Returns an array with the same axes and type as input array and filled with fill_value.

    Parameters
    ----------
    array : LArray
        Input array.
    fill_value : scalar or LArray
        Value to fill the array
    title : str, optional
        Title.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the data type of array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
        'C' means C-order, 'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise.
        'K' (default) means match the layout of `a` as closely as possible.

    Returns
    -------
    LArray

    Examples
    --------
    >>> a = ndtest((2, 3))
    >>> full_like(a, 5)
    a\\b  b0  b1  b2
     a0   5   5   5
     a1   5   5   5
    """
    if not title:
        title = array.title
    # cannot use full() because order == 'K' is not understood
    # cannot use np.full_like() because it would not handle LArray fill_value
    res = empty_like(array, title, dtype, order)
    res[:] = fill_value
    return res


# XXX: would it be possible to generalize to multiple axes?
def sequence(axis, initial=0, inc=None, mult=1, func=None, axes=None, title=''):
    """
    Creates an array by sequentially applying modifications to the array along axis.

    The value for each label in axis will be given by sequentially transforming the value for the previous label.
    This transformation on the previous label value consists of applying the function "func" on that value if provided,
    or to multiply it by mult and increment it by inc otherwise.

    Parameters
    ----------
    axis : axis definition (Axis, str, int)
        Axis along which to apply mod. An axis definition can be passed as a string. An int will be interpreted as the
        length for a new anonymous axis.
    initial : scalar or LArray, optional
        Value for the first label of axis. Defaults to 0.
    inc : scalar, LArray, optional
        Value to increment the previous value by. Defaults to 0 if mult is provided, 1 otherwise.
    mult : scalar, LArray, optional
        Value to multiply the previous value by. Defaults to 1.
    func : function/callable, optional
        Function to apply to the previous value. Defaults to None.
        Note that this is much slower than using inc and/or mult.
    axes : int, tuple of int or tuple/list/AxisCollection of Axis, optional
        Axes of the result. Defaults to the union of axes present in other arguments.
    title : str, optional
        Title.

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
    >>> inc = LArray([1, 2], [sex])
    >>> inc
    sex  M  F
         1  2
    >>> sequence(year, 1.0, inc)
    sex\\year  2016  2017  2018  2019
           M   1.0   2.0   3.0   4.0
           F   1.0   3.0   5.0   7.0
    >>> mult = LArray([2, 3], [sex])
    >>> mult
    sex  M  F
         2  3
    >>> sequence(year, 1.0, mult=mult)
    sex\\year  2016  2017  2018  2019
           M   1.0   2.0   4.0   8.0
           F   1.0   3.0   9.0  27.0
    >>> initial = LArray([3, 4], [sex])
    >>> initial
    sex  M  F
         3  4
    >>> sequence(year, initial, 1)
    sex\\year  2016  2017  2018  2019
           M     3     4     5     6
           F     4     5     6     7
    >>> sequence(year, initial, mult=2)
    sex\\year  2016  2017  2018  2019
           M     3     6    12    24
           F     4     8    16    32
    >>> sequence(year, initial, inc, mult)
    sex\\year  2016  2017  2018  2019
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
    >>> sequence(X.year, axes=(sex, year))
    sex\\year  2016  2017  2018  2019
           M     0     1     2     3
           F     0     1     2     3

    sequence can be used as the inverse of growth_rate:

    >>> a = LArray([1.0, 2.0, 3.0, 3.0], year)
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
    if inc is None:
        inc = 1 if mult is 1 else 0

    if axes is None:
        if not isinstance(axis, Axis):
            axis = _make_axis(axis)

        def strip_axes(col):
            return get_axes(col) - axis
        # we need to remove axis if present, because it might be incompatible
        axes = strip_axes(initial) | strip_axes(inc) | strip_axes(mult) | axis
    else:
        axes = AxisCollection(axes)
        axis = axes[axis]
    res_dtype = np.dtype(common_type((initial, inc, mult)))
    res = empty(axes, title=title, dtype=res_dtype)
    res[axis.i[0]] = initial
    def has_axis(a, axis):
        return isinstance(a, LArray) and axis in a.axes
    if func is not None:
        for i in range(1, len(axis)):
            res[axis.i[i]] = func(res[axis.i[i - 1]])
    elif has_axis(inc, axis) and has_axis(mult, axis):
        # This case is more complicated to vectorize. It seems
        # doable (probably by adding a fictive axis), but let us wait until
        # someone requests it. The trick is to be able to write this:
        # a[i] = initial * prod(mult[j]) + inc[1] * prod(mult[j]) + ...
        #                 j=1..i                    j=2..i
        #      + inc[i-2] * prod(mult[j]) + inc[i-1] * mult[i] + inc[i]
        #                 j=i-1..i

        # a[0] = initial
        # a[1] = initial * mult[1]
        #      +  inc[1]
        # a[2] = initial * mult[1] * mult[2]
        #      +  inc[1] * mult[2]
        #      +  inc[2]
        # a[3] = initial * mult[1] * mult[2] * mult[3]
        #      +  inc[1] * mult[2] * mult[3]
        #      +  inc[2]           * mult[3]
        #      +  inc[3]
        # a[4] = initial * mult[1] * mult[2] * mult[3] * mult[4]
        #      +  inc[1] * mult[2] * mult[3] * mult[4]
        #      +  inc[2]           * mult[3] * mult[4]
        #      +  inc[3]                     * mult[4]
        #      +  inc[4]

        # a[1:] = initial * cumprod(mult[1:]) + ...
        def index_if_exists(a, axis, i):
            if isinstance(a, LArray) and axis in a.axes:
                a_axis = a.axes[axis]
                return a[a_axis[axis.labels[i]]]
            else:
                return a
        for i in range(1, len(axis)):
            i_mult = index_if_exists(mult, axis, i)
            i_inc = index_if_exists(inc, axis, i)
            res[axis.i[i]] = res[axis.i[i - 1]] * i_mult + i_inc
    else:
        # TODO: use cumprod and cumsum to avoid the explicit loop
        # it is easy for constant inc OR constant mult.
        # it is easy for array inc OR array mult.
        # it is a bit more complicated for constant inc AND constant mult
        #
        # it gets hairy for array inc AND array mult. It seems doable but let us wait until someone requests it.
        def array_or_full(a, axis, initial):
            dt = common_type((a, initial))
            r = empty((get_axes(a) - axis) | axis, title=title, dtype=dt)
            r[axis.i[0]] = initial
            if isinstance(a, LArray) and axis in a.axes:
                # not using axis.i[1:] because a could have less ticks
                # on axis than axis
                r[axis.i[1:]] = a[axis[axis.labels[1]:]]
            else:
                r[axis.i[1:]] = a
            return r

        if isinstance(initial, LArray) and np.isscalar(inc):
            inc = full_like(initial, inc)

        # inc only (integer scalar)
        if np.isscalar(mult) and mult == 1 and np.isscalar(inc) and res_dtype.kind == 'i':
            # stop is not included
            stop = initial + inc * len(axis)
            data = np.arange(initial, stop, inc)
            res[:] = LArray(data, axis)
        # inc only (other scalar)
        elif np.isscalar(mult) and mult == 1 and np.isscalar(inc):
            # stop is included
            stop = initial + inc * (len(axis) - 1)
            data = np.linspace(initial, stop=stop, num=len(axis))
            res[:] = LArray(data, axis)
        # inc only (array)
        elif np.isscalar(mult) and mult == 1:
            inc_array = array_or_full(inc, axis, initial)
            res[axis.i[1:]] = inc_array.cumsum(axis)[axis.i[1:]]
        # mult only (scalar or array)
        elif np.isscalar(inc) and inc == 0:
            mult_array = array_or_full(mult, axis, initial)
            res[axis.i[1:]] = mult_array.cumprod(axis)[axis.i[1:]]
        # both inc and mult defined but scalars or axis not present
        else:
            mult_array = array_or_full(mult, axis, 1.0)
            cum_mult = mult_array.cumprod(axis)[axis.i[1:]]
            res[axis.i[1:]] = ((1 - cum_mult) / (1 - mult)) * inc + initial * cum_mult
    return res

create_sequential = renamed_to(sequence, 'create_sequential')

@_check_axes_argument
def ndrange(axes, start=0, title='', dtype=int):
    import warnings
    warnings.warn("ndrange() is deprecated. Use sequence() or ndtest() instead.", FutureWarning, stacklevel=2)
    return ndtest(axes, start=start, title=title, dtype=dtype)


@_check_axes_argument
def ndtest(shape_or_axes, start=0, label_start=0, title='', dtype=int):
    """Returns test array with given shape.

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
        Title.
    dtype : type or np.dtype, optional
        Type of resulting array.

    Returns
    -------
    LArray

    Examples
    --------
    Create test array by passing a shape

    >>> ndtest(6)
    a  a0  a1  a2  a3  a4  a5
        0   1   2   3   4   5
    >>> ndtest((2, 3))
    a\\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> ndtest((2, 3), label_start=1)
    a\\b  b1  b2  b3
     a1   0   1   2
     a2   3   4   5
    >>> ndtest((2, 3), start=2)
    a\\b  b0  b1  b2
     a0   2   3   4
     a1   5   6   7
    >>> ndtest((2, 3), dtype=float)
    a\\b   b0   b1   b2
     a0  0.0  1.0  2.0
     a1  3.0  4.0  5.0

    Create test array by passing axes

    >>> ndtest("nat=BE,FO;sex=M,F")
    nat\\sex  M  F
         BE  0  1
         FO  2  3
    >>> nat = Axis("nat=BE,FO")
    >>> sex = Axis("sex=M,F")
    >>> ndtest([nat, sex])
    nat\\sex  M  F
         BE  0  1
         FO  2  3
    """
    # XXX: try to come up with a syntax where start is before "end".
    # For ndim > 1, I cannot think of anything nice.
    if isinstance(shape_or_axes, int):
        shape_or_axes = (shape_or_axes,)
    if isinstance(shape_or_axes, (list, tuple)) and all([isinstance(i, int) for i in shape_or_axes]):
        # TODO: move this to a class method on AxisCollection
        assert len(shape_or_axes) <= 26
        axes_names = [chr(ord('a') + i) for i in range(len(shape_or_axes))]
        label_ranges = [range(label_start, label_start + length) for length in shape_or_axes]
        shape_or_axes = [Axis(['{}{}'.format(name, i) for i in label_range], name)
                         for name, label_range in zip(axes_names, label_ranges)]
    if isinstance(shape_or_axes, AxisCollection):
        axes = shape_or_axes
    else:
        axes = AxisCollection(shape_or_axes)
    data = np.arange(start, start + axes.size, dtype=dtype).reshape(axes.shape)
    return LArray(data, axes, title=title)


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


def diag(a, k=0, axes=(0, 1), ndim=2, split=True):
    """
    Extracts a diagonal or construct a diagonal array.

    Parameters
    ----------
    a : LArray
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
        Whether or not to try to split the axis name and labels

    Returns
    -------
    LArray
        The extracted diagonal or constructed diagonal array.

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> a = ndtest([nat, sex], start=1)
    >>> a
    nat\\sex  M  F
         BE  1  2
         FO  3  4
    >>> d = diag(a)
    >>> d
    nat_sex  BE_M  FO_F
                1     4
    >>> diag(d)
    nat\\sex  M  F
         BE  1  0
         FO  0  4
    >>> a = ndtest(sex, start=1)
    >>> a
    sex  M  F
         1  2
    >>> diag(a)
    sex\\sex  M  F
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
def labels_array(axes, title=''):
    """Returns an array with specified axes and the combination of
    corresponding labels as values.

    Parameters
    ----------
    axes : Axis or collection of Axis
    title : str, optional
        Title.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat=BE,FO')
    >>> sex = Axis('sex=M,F')
    >>> labels_array(sex)
    sex  M  F
         M  F
    >>> labels_array((nat, sex))
    nat  sex\\axis  nat  sex
     BE         M   BE    M
     BE         F   BE    F
     FO         M   FO    M
     FO         F   FO    F
    """
    # >>> labels_array((nat, sex))
    # nat\\sex     M     F
    #      BE  BE,M  BE,F
    #      FO  FO,M  FO,F
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
    return LArray(res_data, res_axes, title)


def identity(axis):
    raise NotImplementedError("identity(axis) is deprecated. In most cases, you can now use the axis directly. "
                              "For example, 'identity(age) < 10' can be replaced by 'age < 10'. "
                              "In other cases, you should use labels_array(axis) instead.")


def eye(rows, columns=None, k=0, title='', dtype=None):
    """Returns a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    rows : int or Axis
        Rows of the output.
    columns : int or Axis, optional
        Columns of the output. If None, defaults to rows.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper
        diagonal, and a negative value to a lower diagonal.
    title : str, optional
        Title.
    dtype : data-type, optional
        Data-type of the returned array. Defaults to float.

    Returns
    -------
    LArray of shape (rows, columns)
        An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.

    Examples
    --------
    >>> eye(2, dtype=int)
    {0}*\\{1}*  0  1
            0  1  0
            1  0  1
    >>> sex = Axis('sex=M,F')
    >>> eye(sex)
    sex\\sex    M    F
          M  1.0  0.0
          F  0.0  1.0
    >>> age = Axis('age=0..2')
    >>> eye(age, sex)
    age\\sex    M    F
          0  1.0  0.0
          1  0.0  1.0
          2  0.0  0.0
    >>> eye(3, k=1)
    {0}*\\{1}*    0    1    2
            0  0.0  1.0  0.0
            1  0.0  0.0  1.0
            2  0.0  0.0  0.0
    """
    if columns is None:
        columns = rows.copy() if isinstance(rows, Axis) else rows
    axes = AxisCollection([rows, columns])
    shape = axes.shape
    data = np.eye(shape[0], shape[1], k, dtype)
    return LArray(data, axes, title)


# XXX: we could change the syntax to use *args
#      => less punctuation but forces kwarg
#      => potentially longer
#      => unsure for now. The most important point is that it should be consistent with other functions.
# stack(a1, a2, axis=Axis('M,F', 'sex'))
# stack(('M', a1), ('F', a2), axis='sex')
# stack(a1, a2, axis='sex')

# on Python 3.6, we could do something like (it would make from_lists obsolete for 1D arrays):
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

def stack(elements=None, axis=None, title='', **kwargs):
    """
    Combines several arrays or sessions along an axis.

    Parameters
    ----------
    elements : tuple, list or dict.
        Elements to stack. Elements can be scalars, arrays, sessions, (label, value) pairs or a {label: value} mapping.
        In the later case, axis must be defined and cannot be a name only, because we need to have labels order,
        which the mapping does not provide.

        Stacking sessions will return a new session containing the arrays of all sessions stacked together. An array
        missing in a session will be replaced by NaN.
    axis : str or Axis or Group, optional
        Axis to create. If None, defaults to a range() axis.
    title : str, optional
        Title.

    Returns
    -------
    LArray
        A single array combining arrays.

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

    In the case the axis to create has already been defined in a variable (Axis or Group)

    >>> stack({'BE': arr1, 'FO': arr2}, nat)
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0
    >>> all_nat = Axis('nat=BE,DE,FR,NL,UK')
    >>> stack({'BE': arr1, 'DE': arr2}, all_nat[:'DE'])
    sex\\nat   BE   DE
          M  1.0  0.0
          F  1.0  0.0

    Otherwise (when one wants to create an axis from scratch), any of these syntaxes works:

    >>> stack([arr1, arr2], 'nat=BE,FO')
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0
    >>> stack({'BE': arr1, 'FO': arr2}, 'nat=BE,FO')
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0
    >>> stack([('BE', arr1), ('FO', arr2)], 'nat=BE,FO')
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    When stacking arrays with different axes, the result has the union of all axes present:

    >>> stack({'BE': arr1, 'FO': 0}, nat)
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    Creating an axis without name nor labels can be done using:

    >>> stack((arr1, arr2))
    sex\\{1}*    0    1
           M  1.0  0.0
           F  1.0  0.0

    When labels are "simple" strings (ie no integers, no string starting with integers, etc.), using keyword
    arguments can be an attractive alternative.

    >>> stack(FO=arr2, BE=arr1, axis=nat)
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    Without passing an explicit order for labels (or an axis object like above), it should only be used on Python 3.6
    or later because keyword arguments are NOT ordered on earlier Python versions.

    >>> # use this only on Python 3.6 and later
    >>> stack(BE=arr1, FO=arr2, axis='nat')   # doctest: +SKIP
    sex\\nat   BE   FO
          M  1.0  0.0
          F  1.0  0.0

    To stack sessions, let us first create two test sessions. For example suppose we have a session storing the results
    of a baseline simulation:

    >>> from larray import Session
    >>> baseline = Session([('arr1', arr1), ('arr2', arr2)])

    and another session with a variant (here we simply added 0.5 to each array)

    >>> variant = Session([('arr1', arr1 + 0.5), ('arr2', arr2 + 0.5)])

    then we stack them together

    >>> stacked = stack([('baseline', baseline), ('variant', variant)], 'sessions')
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

    if isinstance(axis, str) and '=' in axis:
        axis = Axis(axis)
    if isinstance(axis, Group):
        axis = Axis(axis)
    if elements is None:
        if not isinstance(axis, Axis) and sys.version_info[:2] < (3, 6):
            raise TypeError("axis argument should provide label order when using keyword arguments on Python < 3.6")
        elements = kwargs.items()
    elif kwargs:
        raise TypeError("stack() accept either keyword arguments OR a collection of elements, not both")

    if isinstance(axis, Axis) and all(isinstance(e, tuple) for e in elements):
        assert all(len(e) == 2 for e in elements)
        elements = {k: v for k, v in elements}

    if isinstance(elements, LArray):
        if axis is None:
            axis = -1
        axis = elements.axes[axis]
        values = [elements[k] for k in axis]
    elif isinstance(elements, dict):
        assert isinstance(axis, Axis)
        values = [elements[v] for v in axis.labels]
    elif isinstance(elements, Iterable):
        if not isinstance(elements, Sequence):
            elements = list(elements)

        if all(isinstance(e, tuple) for e in elements):
            assert all(len(e) == 2 for e in elements)
            keys = [k for k, v in elements]
            values = [v for k, v in elements]
            assert all(np.isscalar(k) for k in keys)
            # this case should already be handled
            assert not isinstance(axis, Axis)
            # axis should be None or str
            axis = Axis(keys, axis)
        else:
            values = elements
            if axis is None or isinstance(axis, basestring):
                axis = Axis(len(elements), axis)
            else:
                assert len(axis) == len(elements)
    else:
        raise TypeError('unsupported type for arrays: %s' % type(elements).__name__)

    if any(isinstance(v, Session) for v in values):
        sessions = values
        if not all(isinstance(s, Session) for s in sessions):
            raise TypeError("stack() only supports stacking Session with other Session objects")

        seen = set()
        all_keys = []
        for s in sessions:
            unique_list(s.keys(), all_keys, seen)
        res = []
        for name in all_keys:
            try:
                stacked = stack([s.get(name, np.nan) for s in sessions], axis=axis)
            # TypeError for str arrays, ValueError for incompatible axes, ...
            except Exception:
                stacked = np.nan
            res.append((name, stacked))
        return Session(res)
    else:
        # XXX : use concat?
        result_axes = AxisCollection.union(*[get_axes(v) for v in values])
        result_axes.append(axis)
        result = empty(result_axes, title=title, dtype=common_type(values))
        for k, v in zip(axis, values):
            result[k] = v
        return result


def get_axes(value):
    return value.axes if isinstance(value, LArray) else AxisCollection([])


def _strip_shape(shape):
    return tuple(s for s in shape if s != 1)


def _equal_modulo_len1(shape1, shape2):
    return _strip_shape(shape1) == _strip_shape(shape2)


# assigning a temporary name to anonymous axes before broadcasting and removing it afterwards is not a good idea after
# all because it copies the axes/change the object, and thus "flatten" wouldn't work with index axes:
# a[ones(a.axes[axes], dtype=bool)]
# but if we had assigned axes names from the start (without dropping them) this wouldn't be a problem.
def make_numpy_broadcastable(values):
    """
    Returns values where LArrays are (NumPy) broadcastable between them.
    For that to be possible, all common axes must be compatible (see Axis class documentation).
    Extra axes (in any array) can have any length.

    * the resulting arrays will have the combination of all axes found in the input arrays, the earlier arrays defining
      the order of axes. Axes with labels take priority over wildcard axes.
    * length 1 wildcard axes will be added for axes not present in input

    Parameters
    ----------
    values : iterable of arrays
        Arrays that requires to be (NumPy) broadcastable between them.

    Returns
    -------
    list of arrays
        List of arrays broadcastable between them. Arrays will have the combination of all axes found in the input
        arrays, the earlier arrays defining the order of axes.
    AxisCollection
        Collection of axes of all input arrays.

    See Also
    --------
    Axis.iscompatible : tests if axes are compatible between them.
    """
    all_axes = AxisCollection.union(*[get_axes(v) for v in values])
    return [v.broadcast_with(all_axes) if isinstance(v, LArray) else v
            for v in values], all_axes


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
