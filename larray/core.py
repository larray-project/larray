# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

__version__ = "0.11.1"

__all__ = [
    'LArray', 'Axis', 'AxisCollection', 'LGroup',
    'union', 'stack',
    'read_csv', 'read_eurostat', 'read_excel', 'read_hdf', 'read_tsv',
    'read_sas',
    'x',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like',
    'full', 'full_like', 'create_sequential', 'ndrange', 'identity',
    'larray_equal',
    'all', 'any', 'sum', 'prod', 'cumsum', 'cumprod', 'min', 'max', 'mean',
    'ptp', 'var', 'std', 'median', 'percentile',
    '__version__'
]

"""
Matrix class
"""
# TODO
# * axes with no name should display as (or even have their name assigned to?)
#   their position. Assigning does not work because after aggregating axis 0,
#   we get the first axis named "1" which is a no-go.
#   it would be much easier to have a .id attribute/property on axis with
#   either the name or position in it, but this requires that axes know about
#   their AxisCollection. id might not be defined when axis is not attached
#   to a Collection

# * add check there is no duplicate label in axes!

# * for NDGroups, we have two options: cross product or intersection.
#   Technically, this is easy, we just need to store a boolean and in getitem
#   act accordingly (use ix_ or not), but what is the best API for users?
#   a different class or a flag? In fact, the same question applies to
#   positional vs label (in total, we got 4 different possibilities).

# ? how do you combine a cross-product group with an intersection group?
#   a[cpgroup, igroup]
#   -> no problem if they are on different dimensions: the igroup
#      dimension(s) are collapsed into one, pgroup dimension(s) stay.
#      The index need to be constructed carefully, but it can be done. See
#      np_indexing.py
#   -> if they are on the same dimensions, we have two options:
#      * apply one then the other (left to right)
#      * fail <-- I think this is safer at least to implement. One after the
#                 other can still be achieved by a[pgroup][igroup]

# API for ND groups (my example is mixing label with positional):

# cross: x.axis1[5:10] * x.axis2.i[3:4]
# intersection: x.axis1[5:10] & x.axis2.i[3:4]
#
# this is very nice, but prevents:
# * normal element-wise multiplication (which has little use in most cases --
#   at least for string labels)
# * set-like operations, which could make a lot of sense. set do not use
#   +, *, /
#   cross: x.axis1[5:10] * x.axis2.i[2:5]
#   intersect: x.axis1[5:10] + x.axis2.i[2:5]

# or we could only allow set operations on groups from axes with the same
# name (including None)

#   union: x.axis1[5:10] | x.axis1.i[2:5]
#   set intersect: x.axis1[5:10] & x.axis1.i[2:5]

#   cross: x.axis1[5:10] | x.axis2.i[2:5]
#   intersect: x.axis1[5:10] & x.axis2.i[2:5]

# Note that cross sections is the default and it is useless to introduce
# another API **except to give a name**, so the * or | syntaxes are useless.
# hmmm, maybe not if we name it after the fact

# => NDGroup((x.axis1[5:10], x.axis2.i[2.5]), 'exports')
# => Group((x.axis1[5:10], x.axis2.i[2.5]), 'exports')
# => (x.axis1[5:10] | x.axis2.i[2.5]).named('exports')

# generalizing "named" and suppressing .group seems like a good idea!
# => x.axis1.group([5, 7, 10], name='brussels')
# => x.axis1[[5, 7, 10]].named('brussels')

# http://xarray.pydata.org/en/stable/indexing.html#pointwise-indexing
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.lookup.html#pandas.DataFrame.lookup

# I think I would go for
# ARRAY.points[dim0_labels, ..., dimX_labels]
# and
# ARRAY.ipoints[dim0_indices, ..., dimX_indices]
# so that we can have a symmetrical API for get and set.

# I wonder if, for axes subscripting, I could not allow tuples as sequences,
# which would make it a bit nicer:
# x.axis1[5, 7, 10].named('brussels')
# instead of
# x.axis1[[5, 7, 10]].named('brussels')
# since axes are always 1D, this is not a direct problem. However the
# question is whether this would lead to an inconsistent API/confuse users
# because they would still have to write the brackets when no axis is present
# a[[5, 7, 9]]
# a[x.axis1[5, 7, 9]]
# in practice, this syntax is little used anyway


# * when trying to aggregate on an non existing Axis (using x.blabla),
#   the error message is awful

# ? implement named groups in strings
#   eg "vla=A01,A02;bru=A21;wal=A55,A56"

# ? implement multi group in one axis getitem:
#   lipro['P01,P02;P05'] <=> (lipro.group('P01,P02'), lipro.group('P05'))
#                        <=> (lipro['P01,P02'], lipro['P05'])

# discuss VG with Geert:
# I do not "expand" key (eg :) upon group creation for perf reason
# VG[:] is much faster than [A01,A02,...,A99]
# I could make that all "contiguous" ranges are conv to slices (return views)
# but that might introduce confusing differences if they update/setitem their
# arrays

# * implement keepaxes=True for _group_aggregate instead of/in addition to
#   group tuples

# ? implement newaxis

# * split unit tests

# * reindex array (ie make it conform to another index, eg of another
#   array). This can be used both for doing operations (add, divide, ...)
#   involving arrays with incompatible axes and to (manually) reorder one axis
#   labels

# * test to_csv: does it consume too much mem?
#   ---> test pandas (one dimension horizontally)

# * add labels in LGroups.__str__

# * xlsx export workbook without overwriting some sheets (charts)

# ? allow naming "one-shot" groups? e.g:
#   regsum = bel.sum(lipro='P01,P02 = P01P02; : = all')

# * docstring for all methods

# * IO functions: csv/hdf/excel?/...?
#   >> needs discussion of the formats (users involved in the discussion?)
#      + check pandas dialects
# * plotting (see plot.py)
#   >> check pandas API
# * implement more Axis functions:
#   - arithmetic operations: + -
#   - regexp functions: geo.group('A3*')
#   - sequence?: geo.seq('A31', 'A38')
#     this NOT exactly equivalent to geo['A31':'A38'] because the later
#     can contain A22 if it is defined between A31 and A38
# * re-implement row_totals/col_totals? or what do we do with them?
# * all the other TODO/XXX in the code
# * time specific API so that we know if we go for a subclass or not
# * data alignment in arithmetic methods
# * test structured arrays
# * review all method & argument names
# ? move "utils" to its own project (so that it is not duplicated between
#   larray and liam2)
#   OR
#   include utils only in larray project and make larray a dependency of liam2
#   (and potentially rename it to reflect the broader scope)
# ? move "excelcom" to its own project (so that it is not duplicated between
#   potential projects using it)

import csv
import os
from itertools import product, chain, groupby, islice
import sys
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

import numpy as np
import pandas as pd

from larray.utils import (table2str, unique, csv_open, unzip, long,
                          decode, basestring, izip, rproduct, ReprString,
                          duplicates, array_lookup2, skip_comment_cells,
                          strip_rows, PY3)

try:
    import xlwings as xw
except ImportError:
    xw = None

try:
    from numpy import nanprod as np_nanprod
except ImportError:
    np_nanprod = None

# TODO: return a generator, not a list
def srange(*args):
    return list(map(str, range(*args)))


def range_to_slice(seq, length=None):
    """
    seq is a sequence-like (list, tuple or ndarray) of integers
    returns a slice if possible (including for sequences of 1 element)
    otherwise returns the input sequence itself

    >>> range_to_slice([3, 4, 5])
    slice(3, 6, None)
    >>> range_to_slice([3, 5, 7])
    slice(3, 9, 2)
    >>> range_to_slice([-3, -2])
    slice(-3, -1, None)
    >>> range_to_slice([-1, -2])
    slice(-1, -3, -1)
    >>> range_to_slice([2, 1])
    slice(2, 0, -1)
    >>> range_to_slice([1, 0], 4)
    slice(-3, -5, -1)
    >>> range_to_slice([1, 0])
    [1, 0]
    >>> range_to_slice([1])
    slice(1, 2, None)
    >>> range_to_slice([])
    []
    """
    if len(seq) < 1:
        return seq
    start = seq[0]
    if len(seq) == 1:
        return slice(start, start + 1)
    second = seq[1]
    step = second - start
    prev_value = second
    for value in seq[2:]:
        if value != prev_value + step:
            return seq
        prev_value = value
    stop = prev_value + step
    if prev_value == 0 and step < 0:
        if length is None:
            return seq
        else:
            stop -= length
            start -= length
    if step == 1:
        step = None
    return slice(start, stop, step)


def slice_to_str(key, use_repr=False):
    """
    converts a slice to a string
    >>> slice_to_str(slice(None))
    ':'
    >>> slice_to_str(slice(24))
    ':24'
    >>> slice_to_str(slice(25, None))
    '25:'
    >>> slice_to_str(slice(5, 10))
    '5:10'
    >>> slice_to_str(slice(None, 5, 2))
    ':5:2'
    """
    # examples of result: ":24" "25:" ":" ":5:2"
    func = repr if use_repr else str
    start = func(key.start) if key.start is not None else ''
    stop = func(key.stop) if key.stop is not None else ''
    step = (":" + func(key.step)) if key.step is not None else ''
    return '%s:%s%s' % (start, stop, step)


def slice_str_to_range(s):
    """
    converts a slice string to a list of (string) values. The end point is
    included.
    >>> slice_str_to_range(':3')
    ['0', '1', '2', '3']
    >>> slice_str_to_range('2:5')
    ['2', '3', '4', '5']
    >>> slice_str_to_range('2:6:2')
    ['2', '4', '6']
    """
    numcolons = s.count(':')
    assert 1 <= numcolons <= 2
    fullstr = s + ':1' if numcolons == 1 else s
    start, stop, step = [int(a) if a else None for a in fullstr.split(':')]
    if start is None:
        start = 0
    if stop is None:
        raise ValueError("no stop bound provided in range: %r" % s)
    stop += 1
    return srange(start, stop, step)


def to_string(v):
    """
    converts a (group of) tick(s) to a string
    """
    if isinstance(v, slice):
        return slice_to_str(v)
    elif isinstance(v, (tuple, list)):
        if len(v) == 1:
            return str(v) + ','
        else:
            return ','.join(str(k) for k in v)
    else:
        return str(v)


def to_tick(e):
    """
    make it hashable, and acceptable as an ndarray element
    scalar & VG -> not modified
    slice -> 'start:stop'
    list|tuple -> 'v1,v2,v3'
    other -> str(v)
    """
    # the fact that an "aggregated tick" is passed as a LGroup or as a
    # string should be as irrelevant as possible. The thing is that we cannot
    # (currently) use the more elegant to_tick(e.key) that means the
    # LGroup is not available in Axis.__init__ after to_ticks, and we
    # need it to update the mapping if it was named. Effectively,
    # this creates two entries in the mapping for a single tick. Besides,
    # I like having the LGroup as the tick, as it provides extra info as
    # to where it comes from.
    if np.isscalar(e) or isinstance(e, LGroup):
        return e
    else:
        return to_string(e)


def to_ticks(s):
    """
    Makes a (list of) value(s) usable as the collection of labels for an
    Axis (ie hashable). Strip strings, split them on ',' and translate
    "range strings" to list of values **including the end point** !
    This function is only used in Axis.__init__ and union().

    >>> to_ticks('H , F')
    ['H', 'F']

    # XXX: we might want to return real int instead, because if we ever
    # want to have more complex queries, such as:
    # arr.filter(age > 10 and age < 20)
    # this would break for string values (because '10' < '2')
    >>> to_ticks(':3')
    ['0', '1', '2', '3']
    """
    if isinstance(s, Group):
        # a single LGroup used for all ticks of an Axis
        raise NotImplementedError("not sure what to do with it yet")
    elif isinstance(s, pd.Index):
        return s.values
    elif isinstance(s, np.ndarray):
        # we assume it has already been translated
        # XXX: Is it a safe assumption?
        return s
    elif isinstance(s, (list, tuple)):
        return [to_tick(e) for e in s]
    elif sys.version >= '3' and isinstance(s, range):
        return list(s)
    else:
        assert isinstance(s, basestring), "%s is not a supported type for " \
                                          "ticks" % type(s)

    if ':' in s:
        return slice_str_to_range(s)
    else:
        return [v.strip() for v in s.split(',')]


def to_key(v):
    """
    Converts a value to a key usable for indexing (slice object, list of values,
    ...). Strings are split on ',' and stripped. Colons (:) are interpreted
    as slices. "int strings" are not converted to int.
    >>> to_key('a:c')
    slice('a', 'c', None)
    >>> to_key('a, b,c ,')
    ['a', 'b', 'c']
    >>> to_key('a,')
    ['a']
    >>> to_key(' a ')
    'a'
    >>> to_key(10)
    10
    """
    if isinstance(v, tuple):
        return list(v)
    elif isinstance(v, Group):
        return v.__class__(to_key(v.key), v.name, v.axis)
    elif v is Ellipsis or isinstance(v, (int, list, slice, LArray)):
        return v
    elif isinstance(v, basestring):
        numcolons = v.count(':')
        if numcolons:
            assert numcolons <= 2
            # can be of len 2 or 3 (if step is provided)
            bounds = [a if a else None for a in v.split(':')]
            return slice(*bounds)
        else:
            if ',' in v:
                # strip extremity commas to avoid empty string keys
                v = v.strip(',')
                return [v.strip() for v in v.split(',')]
            else:
                return v.strip()
    else:
        raise TypeError("%s has an invalid type (%s) for a key"
                        % (v, type(v).__name__))


def to_keys(value):
    """
    converts a (collection of) group(s) to a structure usable for indexing.
    'label' or ['l1', 'l2'] or [['l1', 'l2'], ['l3']]

    It is only used for .sum(axis=xxx)
    >>> to_keys('P01,P02')  # <-- one group => collapse dimension
    ['P01', 'P02']
    >>> to_keys(('P01,P02',))  # <-- do not collapse dimension
    (['P01', 'P02'],)
    >>> to_keys('P01;P02;:')
    ('P01', 'P02', slice(None, None, None))

    # >>> to_keys('P01,P02,:') # <-- INVALID !
    # it should have an explicit failure

    # we allow this, even though it is a dubious syntax
    >>> to_keys(('P01', 'P02', ':'))
    ('P01', 'P02', slice(None, None, None))

    # it is better to use explicit groups
    >>> to_keys(('P01,', 'P02,', ':'))
    (['P01'], ['P02'], slice(None, None, None))

    # or even the ugly duck...
    >>> to_keys((('P01',), ('P02',), ':'))
    (['P01'], ['P02'], slice(None, None, None))
    """
    if isinstance(value, basestring):
        if ';' in value:
            return tuple([to_key(group) for group in value.split(';')])
        else:
            return to_key(value)
    elif isinstance(value, tuple):
        return tuple([to_key(group) for group in value])
    else:
        return to_key(value)


def union(*args):
    # TODO: add support for LGroup and lists
    """
    returns the union of several "value strings" as a list
    """
    if args:
        return list(unique(chain(*(to_ticks(arg) for arg in args))))
    else:
        return []


def larray_equal(first, other):
    if not isinstance(first, LArray) or not isinstance(other, LArray):
        return False
    return (first.axes == other.axes and
            np.array_equal(np.asarray(first), np.asarray(other)))


def isnoneslice(v):
    return isinstance(v, slice) and v == slice(None)

def seq_summary(seq, num=3, func=repr):
    def shorten(l):
        return l if len(l) <= 2 * num else l[:num] + ['...'] + list(l[-num:])

    return ' '.join(shorten([func(l) for l in seq]))


class PGroupMaker(object):
    def __init__(self, axis):
        assert isinstance(axis, Axis)
        self.axis = axis

    def __getitem__(self, key):
        return PGroup(key, None, self.axis)


class Axis(object):
    # ticks instead of labels?
    # XXX: make name and labels optional?
    def __init__(self, name, labels):
        """
        labels should be an array-like (convertible to an ndarray)
        or a int (the size of the Axis)
        """
        if isinstance(name, Axis):
            name = name.name
        self.name = name
        self._labels = None
        self.__mapping = None
        self.__sorted_keys = None
        self.__sorted_values = None
        self._length = None
        self._iswildcard = False
        self.labels = labels
        self.collection = None

    @property
    def _mapping(self):
        mapping = self.__mapping
        if mapping is None:
            labels = self._labels
            # TODO: this would be more efficient for wildcard axes but
            # does not work in all cases
            # mapping = labels
            mapping = {label: i for i, label in enumerate(labels)}
            if not self._iswildcard:
                # we have no choice but to do that!
                # otherwise we could not make geo['Brussels'] work efficiently
                # (we could have to traverse the whole mapping checking for each
                # name, which is not an option)
                # TODO: only do this if labels.dtype is object, or add "contains_lgroup"
                # flag in above code (if any(...))
                # 0.179
                mapping.update({label.name: i for i, label in enumerate(labels)
                                if isinstance(label, Group)})
            self.__mapping = mapping
        return mapping

    def _update_key_values(self):
        mapping = self._mapping
        if mapping:
            sorted_keys, sorted_values = tuple(zip(*sorted(mapping.items())))
        else:
            sorted_keys, sorted_values = (), ()
        keys, values = np.array(sorted_keys), np.array(sorted_values)
        self.__sorted_keys = keys
        self.__sorted_values = values
        return keys, values

    @property
    def _sorted_keys(self):
        if self.__sorted_keys is None:
            keys, _ = self._update_key_values()
        return self.__sorted_keys

    @property
    def _sorted_values(self):
        values = self.__sorted_values
        if values is None:
            _, values = self._update_key_values()
        return values

    @property
    def i(self):
        return PGroupMaker(self)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            raise TypeError("labels should be ndarray or int")
        if isinstance(labels, (int, long)):
            length = labels
            labels = np.arange(length)
            iswildcard = True
        else:
            # TODO: move this to to_ticks????
            # we convert to an ndarray to save memory for scalar ticks (for
            # LGroup ticks, it does not make a difference since a list of VG
            # and an ndarray of VG are both arrays of pointers)
            ticks = to_ticks(labels)
            object_array = isinstance(ticks, np.ndarray) and \
                           ticks.dtype.type == np.object_
            can_have_groups = object_array or isinstance(ticks, (tuple, list))
            if can_have_groups and any(
                    isinstance(tick, LGroup) for tick in ticks):
                # avoid getting a 2d array if all LGroup have the same length
                labels = np.empty(len(ticks), dtype=object)
                labels[:] = ticks
            else:
                labels = np.asarray(ticks)
            length = len(labels)
            iswildcard = False

        self._length = length
        self._labels = labels
        self._iswildcard = iswildcard

    @property
    def iswildcard(self):
        return self._iswildcard

    # XXX: not sure I should offer an *args version
    def group(self, *args, **kwargs):
        """
        key is label-based (slice and fancy indexing are supported)
        returns a LGroup usable in .sum or .filter
        """
        name = kwargs.pop('name', None)
        if kwargs:
            raise ValueError("invalid keyword argument(s): %s"
                             % list(kwargs.keys()))
        key = args[0] if len(args) == 1 else args
        if isinstance(key, LGroup):
            # XXX: I am not sure this test even makes sense. eg if we have two
            # axes arr_from and arr_to, we might want to reuse groups
            if key.axis != self.name:
                raise ValueError("cannot subset an axis with a LGroup of "
                                 "an incompatible axis")
            # FIXME: we should respect the given name (overrides key.name)
            return key
        return LGroup(key, name, self)

    def all(self, name=None):
        return self.group(slice(None), name=name if name is not None else "all")

    def subaxis(self, key, name=None):
        """
        key is index-based (slice and fancy indexing are supported)
        returns an Axis for a sub-array
        """
        if (isinstance(key, slice) and
                key.start is None and key.stop is None and key.step is None):
            return self
        # we must NOT modify the axis name, even though this creates a new axis
        # that is independent from the original one because the original
        # name is probably what users will want to use to filter
        if name is None:
            name = self.name
        if isinstance(key, LArray):
            return tuple(key.axes)
        return Axis(name, self.labels[key])

    def iscompatible(self, other):
        if not isinstance(other, Axis) or self.name != other.name:
            return False
        # wildcard axes of length 1 match with anything
        if self.iswildcard:
            return len(self) == 1 or len(self) == len(other)
        elif other.iswildcard:
            return len(other) == 1 or len(self) == len(other)
        else:
            return np.array_equal(self.labels, other.labels)

    def __eq__(self, other):
        return (isinstance(other, Axis) and self.name == other.name and
                self.iswildcard == other.iswildcard and
                np.array_equal(self.labels, other.labels))

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter(self.labels)

    def __getitem__(self, key):
        """
        key is a label-based key (slice and fancy indexing are supported)
        """
        return self.group(key)

    def __contains__(self, key):
        return to_tick(key) in self._mapping

    def __hash__(self):
        return id(self)

    def _is_key_type_compatible(self, key):
        label_kind = self.labels.dtype.kind
        key_kind = np.dtype(type(key)).kind
        str_key = key_kind in ('S', 'U')
        # on Python2, ascii-only unicode string can match byte strings,
        # so we shouldn't be more picky here than dict hashing
        allowed_str_kinds = ('O',) if PY3 else ('O', 'S', 'U')
        str_match = str_key and label_kind in allowed_str_kinds
        return key_kind == label_kind or str_match

    def translate(self, key, bool_passthrough=True):
        """
        translates a label key to its numerical index counterpart
        fancy index with boolean vectors are passed through unmodified
        """
        mapping = self._mapping

        # first, try the key as-is, so that we can target elements in aggregated
        # arrays (those are either strings containing comas or LGroups)
        try:
            # avoid matching 0 against False or 0.0
            if self._is_key_type_compatible(key):
                return mapping[key]
        # we must catch TypeError because key might not be hashable (eg slice)
        # IndexError is for when mapping is an ndarray
        except (KeyError, TypeError, IndexError):
            pass

        if isinstance(key, PGroup):
            return key.key

        if isinstance(key, LGroup):
            # at this point we do not care about the axis nor the name
            key = key.key

        if isinstance(key, basestring):
            # transform "specially formatted strings" for slices and lists to
            # actual objects
            key = to_key(key)

        if isinstance(key, slice):
            start = mapping[key.start] if key.start is not None else None
            # stop is inclusive in the input key and exclusive in the output !
            stop = mapping[key.stop] + 1 if key.stop is not None else None
            return slice(start, stop, key.step)
        # XXX: bool LArray do not pass through???
        elif isinstance(key, np.ndarray) and key.dtype.kind is 'b' and \
                bool_passthrough:
            return key
        elif isinstance(key, (tuple, list)):
            # TODO: the result should be cached
            # Note that this is faster than array_lookup(np.array(key), mapping)
            res = np.empty(len(key), int)
            for i, label in enumerate(key):
                res[i] = mapping[label]
            return res
        elif isinstance(key, np.ndarray):
            # handle fancy indexing with a ndarray of labels
            # TODO: the result should be cached
            # TODO: benchmark this against the tuple/list version above when
            # mapping is large
            # array_lookup is O(len(key) * log(len(mapping)))
            # vs
            # tuple/list version is O(len(key)) (dict.getitem is O(1))
            # XXX: we might want to special case dtype bool, because in that
            # case the mapping will in most case be {False: 0, True: 1} or
            # {False: 1, True: 0} and in those case key.astype(int) and
            # (~key).astype(int) are MUCH faster
            # see C:\Users\gdm\devel\lookup_methods.py and
            #     C:\Users\gdm\Desktop\lookup_methods.html
            return array_lookup2(key, self._sorted_keys, self._sorted_values)
        elif isinstance(key, LArray):
            pkey = array_lookup2(key.data, self._sorted_keys, self._sorted_values)
            return LArray(pkey, key.axes)
        else:
            # the first mapping[key] above will cover most cases. This code
            # path is only used if the key was given in "non normalized form"
            assert np.isscalar(key), "%s (%s) is not scalar" % (key, type(key))
            # key is scalar (integer, float, string, ...)
            if np.dtype(type(key)).kind == self.labels.dtype.kind:
                return mapping[key]
            else:
                # print("diff dtype", )
                raise KeyError(key)

    @property
    def id(self):
        if self.name is not None:
            return self.name
        elif self.collection is not None:
            return self.collection.index(self)
        else:
            raise ValueError('Axis has no name nor collection, so no id')

    @property
    def display_name(self):
        if self.name is not None:
            name = self.name
        elif self.collection is not None:
            name = 'axis%d' % self.collection.index(self)
        else:
            name = '-'
        return (name + '*') if self.iswildcard else name

    def __str__(self):
        return self.display_name

    def __repr__(self):
        labels = len(self) if self.iswildcard else list(self.labels)
        return 'Axis(%r, %r)' % (self.name, labels)

    def labels_summary(self):
        def repr_on_strings(v):
            if isinstance(v, str):
                return repr(v)
            else:
                return str(v)
        return seq_summary(self.labels, func=repr_on_strings)

    # XXX: we might want to use | for union (like set)
    def __add__(self, other):
        if isinstance(other, Axis):
            if self.name != other.name:
                raise ValueError('cannot add Axes with different names')
            return Axis(self.name, union(self.labels, other.labels))
        else:
            try:
                return Axis(self.name, self.labels + other)
            except Exception:
                raise ValueError

    # XXX: sub between two axes could mean set - set or elementwise -
    def __sub__(self, other):
        if isinstance(other, Axis):
            if self.name != other.name:
                raise ValueError('cannot subtract Axes with different names')
            return Axis(self.name,
                        [l for l in self.labels if l not in other.labels])
        else:
            try:
                return Axis(self.name, self.labels - other)
            except Exception:
                raise ValueError

    def copy(self):
        new_axis = Axis(self.name, [])
        # XXX: I wonder if we should make a copy of the labels + mapping.
        # There should at least be an option.
        new_axis._labels = self._labels
        new_axis.__mapping = self.__mapping
        new_axis._length = self._length
        new_axis._iswildcard = self._iswildcard
        new_axis.__sorted_keys = self.__sorted_keys
        new_axis.__sorted_values = self.__sorted_values
        # collection is intentionally not copied
        return new_axis

    def rename(self, name):
        """Renames the axis.

        Parameters
        ----------
        newname : str
            the new name for the axis.

        Returns
        -------
        Axis
            a new Axis with the same labels but a different name.

        Example
        -------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> sex
        Axis('sex', ['M', 'F'])
        >>> sex.rename('gender')
        Axis('gender', ['M', 'F'])
        """
        res = self.copy()
        res.name = name
        return res

    def _rename(self, name):
        return Axis(name, self.labels)


# We need a separate class for LGroup and cannot simply create a
# new Axis with a subset of values/ticks/labels: the subset of
# ticks/labels of the LGroup need to correspond to its *Axis*
# indices
class Group(object):
    def __init__(self, key, name=None, axis=None):
        if isinstance(key, tuple):
            key = list(key)
        self.key = key

        # we do NOT assign a name automatically when missing because that
        # makes it impossible to know whether a name was explicitly given or
        # not
        self.name = name
        assert axis is None or isinstance(axis, (basestring, int, Axis)), \
            "invalid axis '%s' (%s)" % (axis, type(axis).__name__)

        # we could check the key is valid but this can be slow and could be
        # useless
        # TODO: for performance reasons, we should cache the result. This will
        # need to be invalidated correctly
        # axis.translate(key)

        # we store the Axis object and not its name like we did previously
        # so that groups on anonymous axes are more meaningful and that we
        # can iterate on a slice of an axis (an LGroup). The reason to store
        # the name instead of the object was to make sure that a Group from an
        # axis (or without axis) could be used on another axis with the same
        # name. See test_la.py:test_...
        self.axis = axis

    def __repr__(self):
        name = ", name=%r" % self.name if self.name is not None else ''
        axis = ", axis=%r" % self.axis_id if self.axis is not None else ''
        return "%s(%r%s%s)" % (self.__class__.__name__, self.key, name, axis)

    def __len__(self):
        return len(self.key)

    @property
    def axis_id(self):
        return self.axis.id if isinstance(self.axis, Axis) else self.axis


# TODO: factorize as much as possible between LGroup & PGroup (move stuff to
#       Group)
class LGroup(Group):
    """
    key should be either a sequence of labels, a slice with label bounds
    or a string
    axis can be an int, str or Axis
    """
    # this makes range(LGroup(int)) possible
    def __index__(self):
        return self.key.__index__()

    def __int__(self):
        return self.key.__int__()

    def __float__(self):
        return self.key.__float__()

    def __hash__(self):
        # to_tick & to_key are partially opposite operations but this
        # standardize on a single notation so that they can all target each
        # other. eg, this removes spaces in "list strings", instead of
        # hashing them directly
        # XXX: but we might want to include that normalization feature in
        #      to_tick directly, instead of using to_key explicitly here
        # XXX: we probably want to include this normalization in __init__
        #      instead
        return hash(to_tick(to_key(self.key)))

    def __eq__(self, other):
        # different name or axis compare equal !
        # XXX: we might want to compare "expanded" keys, so that slices
        # can match lists and vice-versa.
        other_key = other.key if isinstance(other, LGroup) else other
        return to_tick(to_key(self.key)) == to_tick(to_key(other_key))

    def __str__(self):
        key = to_key(self.key)
        if isinstance(key, slice):
            str_key = slice_to_str(key, use_repr=True)
        elif isinstance(key, (tuple, list, np.ndarray)):
            str_key = '[%s]' % seq_summary(key, 1)
        else:
            str_key = repr(key)

        return '%r (%s)' % (self.name, str_key) if self.name is not None \
            else str_key

    def __lt__(self, other):
        other_key = other.key if isinstance(other, LGroup) else other
        return self.key.__lt__(other_key)

    def __gt__(self, other):
        other_key = other.key if isinstance(other, LGroup) else other
        return self.key.__gt__(other_key)

    def __iter__(self):
        return iter(self.key)

    def __getitem__(self, key):
        return self.key[key]


class PGroup(Group):
    """
    Positional Group
    """
    pass


def index_by_id(seq, value):
    for i, item in enumerate(seq):
        if item is value:
            return i
    raise ValueError("%s is not in list" % value)


# not using OrderedDict because it does not support indices-based getitem
# not using namedtuple because we have to know the fields in advance (it is a
# one-off class)
class AxisCollection(object):
    def __init__(self, axes=None):
        """
        :param axes: sequence of Axis (or int) objects
        """
        if axes is None:
            axes = []
        if isinstance(axes, int):
            axes = [axes]
        axes = [Axis(None, range(axis)) if isinstance(axis, (int, long))
                    else axis.copy()
                for axis in axes]
        assert all(isinstance(a, Axis) for a in axes)
        for a in axes:
            a.collection = self
        self._list = axes
        self._map = {axis.name: axis for axis in axes if axis.name is not None}

    def __getattr__(self, key):
        try:
            return self._map[key]
        except KeyError:
            return self.__getattribute__(key)

    def __getitem__(self, key):
        if isinstance(key, Axis):
            if key.name is None:
                try:
                    key = self.index(key)
                # transform ValueError to KeyError
                except ValueError:
                    raise KeyError("axis %s not found in %s" % (key, self))
            else:
                # we should NOT check that the object is the same, so that we can
                # use AxisReference objects to target real axes
                key = key.name

        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, (tuple, list, AxisCollection)):
            return AxisCollection([self[k] for k in key])
        elif isinstance(key, slice):
            return AxisCollection(self._list[key])
        elif key is None:
            raise KeyError("axis %s not found in %s" % (key, self))
        else:
            assert isinstance(key, basestring), type(key)
            if key in self._map:
                return self._map[key]
            else:
                raise KeyError("axis '%s' not found in %s" % (key, self))

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            assert isinstance(value, (tuple, list, AxisCollection))
            def slice_bound(bound):
                if bound is None or isinstance(bound, int):
                    # out of bounds integer bounds are allowed in slice setitem
                    # so we cannot use .index
                    return bound
                else:
                    return self.index(bound)
            start_idx = slice_bound(key.start)
            # XXX: we might want to make the stop bound inclusive, which makes
            # more sense for label bounds (but prevents inserts via setitem)
            stop_idx = slice_bound(key.stop)
            old = self._list[start_idx:stop_idx:key.step]
            for axis in old:
                axis.collection = None
                if axis.name is not None:
                    del self._map[axis.name]
            # XXX: in all "append"-like operations, we might want to only make a
            # copy if axis.collection is not None
            new = [axis.copy() for axis in value]
            for axis in new:
                axis.collection = self
                if axis.name is not None:
                    self._map[axis.name] = axis
            self._list[start_idx:stop_idx:key.step] = new
        elif isinstance(key, (tuple, list, AxisCollection)):
            assert isinstance(value, (tuple, list, AxisCollection))
            if len(key) != len(value):
                raise ValueError('must have as many old axes as new axes')
            for k, v in zip(key, value):
                self[k] = v
        else:
            assert isinstance(value, Axis)
            idx = self.index(key)
            step = 1 if idx >= 0 else -1
            self[idx:idx + step:step] = [value]

    def __delitem__(self, key):
        if isinstance(key, slice):
            self[key] = []
        else:
            idx = self.index(key)
            axis = self._list.pop(idx)
            if axis.name is not None:
                del self._map[axis.name]
            axis.collection = None

    def union(self, *args):
        result = self[:]
        for a in args:
            if isinstance(a, Axis):
                a = [a]
            result.extend(a, replace_wildcards=True)
        return result
    __or__ = union
    __add__ = union

    def __eq__(self, other):
        """
        other collection compares equal if all axes compare equal and in the
        same order. Works with a list.
        """
        if not isinstance(other, list):
            other = list(other)
        return self._list == other

    # for python2, we need to define it explicitly
    def __ne__(self, other):
        return not self == other

    def __contains__(self, key):
        if isinstance(key, int):
            return -len(self) <= key < len(self)
        elif isinstance(key, Axis):
            if key.name is None:
                # XXX: does this compare each axis by value (by labels)?
                return key in self._list
            key = key.name
        return key in self._map

    def isaxis(self, value):
        # this is tricky. 0 and 1 can be both axes indices and axes ticks.
        # not sure what's worse:
        # 1) disallow aggregates(axis_num)
        #    users could still use arr.sum(arr.axes[0])
        #    we could also provide an explicit kwarg (ie this would
        #    effectively forbid having an axis named "axis").
        #    arr.sum(axis=0). I think this is the sanest option. The
        #    error message in case we use it without the keyword needs to
        #    be clearer though.
        return isinstance(value, Axis) or (isinstance(value, basestring) and
                                           value in self)
        # 2) slightly inconsistent API: allow aggregate over single labels
        #    if they are string, but not int
        #    arr.sum(0) would sum on the first axis, but arr.sum('H') would
        #    sum a single tick. I don't like this option.
        # 3) disallow single tick aggregates. Single labels make little
        #    sense in the context of an aggregate, but you don't always
        #    know/want to differenciate the code in that case anyway.
        #    It would be annoying for e.g. Brussels
        # 4) give priority to axes,
        #    arr.sum(0) would sum on the first axis but arr.sum(5) would
        #    sum a single tick (assuming there is a int axis and less than
        #    six axes).
        # return value in self

    def __len__(self):
        return len(self._list)
    ndim = property(__len__)

    def __str__(self):
        return "{%s}" % ', '.join(self.display_names)

    def __repr__(self):
        axes_repr = (repr(axis) for axis in self._list)
        return "AxisCollection([\n    %s\n])" % ',\n    '.join(axes_repr)

    def __and__(self, other):
        """
        returns the intersection of this collection and other
        """
        if isinstance(other, basestring):
            other = set(other.split(','))
        elif isinstance(other, Axis):
            other = set(other.name)
        return AxisCollection([axis for axis in self if axis.name in other])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return [a.name for a in self._list]

    def pop(self, axis=-1):
        axis = self[axis]
        del self[axis]
        assert axis.collection is None
        return axis

    def append(self, axis):
        """
        append axis at the end of the collection
        """
        self[len(self):len(self)] = [axis]

    def check_compatible(self, axes):
        for i, axis in enumerate(axes):
            if axis.name is not None:
                local_axis = self._map.get(axis.name)
            else:
                if i < len(self):
                    local_axis = self[i]
                else:
                    local_axis = None
            if local_axis is not None:
                if not local_axis.iscompatible(axis):
                    raise ValueError("incompatible axes:\n%r\nvs\n%r"
                                     % (axis, local_axis))

    def extend(self, axes, validate=True, replace_wildcards=False):
        """
        extend the collection by appending the axes from axes
        """
        # axes should be a sequence
        if not isinstance(axes, (tuple, list, AxisCollection)):
            raise TypeError("AxisCollection can only be extended by a "
                            "sequence of Axis, not %s" % type(axes).__name__)
        # check that common axes are the same
        if validate:
            self.check_compatible(axes)
        for axis in axes:
            if axis not in self:
                # append axis
                self[len(self):len(self)] = [axis]
            elif replace_wildcards and self[axis].iswildcard:
                self[axis] = axis

    def index(self, axis):
        """
        returns the index of axis.

        axis can be a name or an Axis object (or an index)
        if the Axis object itself exists in the list, index() will return it
        if the Axis object is from another LArray, index() will return the
        index of the local axis with the same name, whether it is compatible
        (has the same ticks) or not.

        Raises ValueError if the axis is not present.
        """
        if isinstance(axis, int):
            if -len(self) <= axis < len(self):
                return axis
            else:
                raise ValueError("axis %d is not in collection" % axis)
        elif isinstance(axis, Axis) and axis.name is None:
            try:
                # first look by id. This avoids testing labels of each axis
                # and makes sure the result is correct even if there are
                # several axes with no name and the same labels.
                return index_by_id(self._list, axis)
            except ValueError:
                # fallback on comparing by values (ie testing labels)
                return self._list.index(axis)
        name = axis.name if isinstance(axis, Axis) else axis
        return self.names.index(name)

    # XXX: we might want to return a new AxisCollection (same question for
    # other inplace operations: append, extend, pop, __delitem__, __setitem__)
    def insert(self, index, axis):
        """
        insert axis before index
        """
        self[index:index] = [axis]

    def copy(self):
        return self[:]

    def replace(self, old, new):
        res = self[:]
        res[old] = new
        return res

    # XXX: kill this method?
    def without(self, axes):
        """
        returns a new collection without some axes
        you can use a comma separated list of names
        set operations so axes can contain axes not present in self
        """
        return self - axes

    def __sub__(self, axes):
        """
        returns a new collection without some axes
        you can use a comma separated list of names
        set operations so axes can contain axes not present in self
        """
        if isinstance(axes, basestring):
            axes = axes.split(',')
        elif isinstance(axes, Axis):
            axes = [axes]

        # only keep indices (as this works for unnamed axes too)
        to_remove = set(self.index(axis) for axis in axes if axis in self)
        return AxisCollection([axis for i, axis in enumerate(self)
                               if i not in to_remove])

    def translate_full_key(self, key):
        """
        Parameters
        ----------
        key : tuple
            a full label-based key. All dimensions must be present and in
            the correct order.

        Returns
        -------
        tuple
            a full positional key
        """
        assert len(key) == len(self)
        return tuple(axis.translate(axis_key)
                     for axis_key, axis in zip(key, self))

    @property
    def labels(self):
        """Returns the list of labels of the axes"""
        return [axis.labels for axis in self._list]

    @property
    def names(self):
        """Returns the list of (raw) names of the axes

        Returns
        -------
        List
            List of names of the axes

        Example
        -------
        >>> a = Axis('a', ['a1', 'a2'])
        >>> b = Axis('b', 2)
        >>> c = Axis(None, ['c1', 'c2'])
        >>> arr = zeros([a, b, c])
        >>> arr.axes.names
        ['a', 'b', None]
        """
        return [axis.name for axis in self._list]

    @property
    def display_names(self):
        """Returns the list of (display) names of the axes

        Returns
        -------
        List
            List of names of the axes

        Example
        -------
        >>> a = Axis('a', ['a1', 'a2'])
        >>> b = Axis('b', 2)
        >>> c = Axis(None, ['c1', 'c2'])
        >>> arr = zeros([a, b, c])
        >>> arr.axes.display_names
        ['a', 'b*', 'axis2']
        """
        return [axis.display_name for axis in self._list]

    @property
    def shape(self):
        return tuple(len(axis) for axis in self._list)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def info(self):
        """Describes an AxisCollection (shape and labels for each axis).

        Returns
        -------
        String
            Description of the AxisCollection (shape and labels for each axis).

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> axes = AxisCollection([xnat, xsex])
        >>> axes.info
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'H' 'F'
        """
        lines = [" %s [%d]: %s" % (axis.display_name, len(axis),
                                   axis.labels_summary())
                 for axis in self]
        shape = " x ".join(str(s) for s in self.shape)
        return ReprString('\n'.join([shape] + lines))


def all(values, axis=None):
    """Test whether all array elements along given axes evaluate to True.

    Parameters
    ----------
    axis : None, int, str or Axis, tuple of int, str or Axis, optional
        axes over which to aggregate. Defaults to None (all axes).

    Returns
    -------
    LArray or scalar

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> a = ndrange([xnat, xsex]) >= 1
    >>> a
    nat\\sex |     H |    F
         BE | False | True
         FO |  True | True
    >>> all(a)
    False
    >>> all(a, xnat)
    sex |     H |    F
        | False | True
    """
    if isinstance(values, LArray):
        return values.all(axis)
    else:
        return builtins.all(values)


def any(values, axis=None):
    """Test whether any array elements along given axes evaluate to True.

    Parameters
    ----------
    axis : int, str or Axis, tuple of int, str or Axis, optional
        axes over which to aggregate. Defaults to None (all axes).

    Returns
    -------
    LArray or scalar

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> a = ndrange([xnat, xsex]) >= 3
    >>> a
    nat\\sex |     H |     F
         BE | False | False
         FO | False |  True
    >>> any(a)
    True
    >>> any(a, xnat)
    sex |     H |    F
        | False | True
    """
    if isinstance(values, LArray):
        return values.any(axis)
    else:
        return builtins.any(values)


# commutative modulo float precision errors
def sum(array, *args, **kwargs):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    array : iterable or array-like or LArray
        Elements to sum.
    axis : None or int or str or Axis or tuple of those, optional
        Axis or axes along which a sum is performed.
        The default (`axis` = `None`) is to perform a sum over all
        axes of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        If this is a tuple, a sum is performed on multiple axes.

    Returns
    -------
    LArray

    See Also
    --------
    LArray.sum : Equivalent method.

    Notes
    -----
    The sum of an empty array is the neutral element 0:

    >>> sum([])
    0.0

    Examples
    --------
    >>> a = ndrange((2, 3))
    >>> a
    axis0\\axis1 | 0 | 1 | 2
              0 | 0 | 1 | 2
              1 | 3 | 4 | 5
    >>> sum(a)
    15
    >>> sum(a, axis=0)
    axis0 | 0 | 1 | 2
          | 3 | 5 | 7
    >>> sum(a, axis=1)
    axis0 | 0 |  1
          | 3 | 12
    """
    # XXX: we might want to be more aggressive here (more types to convert),
    #      however, generators should still be computed via the builtin.
    if isinstance(array, (np.ndarray, list)):
        array = LArray(array)
    if isinstance(array, LArray):
        return array.sum(*args, **kwargs)
    else:
        return builtins.sum(array)


def prod(array, *args, **kwargs):
    return array.prod(*args, **kwargs)


def cumsum(array, *args, **kwargs):
    return array.cumsum(*args, **kwargs)


def cumprod(array, *args, **kwargs):
    return array.cumprod(*args, **kwargs)


def min(array, *args, **kwargs):
    return array.min(*args, **kwargs)


def max(array, *args, **kwargs):
    return array.max(*args, **kwargs)


def mean(array, *args, **kwargs):
    return array.mean(*args, **kwargs)


def median(array, *args, **kwargs):
    """
    Parameters
    ----------
    array : iterable or array-like or LArray
    axis : None or int or str or Axis or tuple of those, optional

    Returns
    -------
    LArray

    See Also
    --------
    LArray.median : Equivalent method.

    Examples
    --------

    >>> a = LArray([[10, 7, 4], [3, 2, 1]])
    >>> a
    axis0\\axis1 |  0 | 1 | 2
              0 | 10 | 7 | 4
              1 |  3 | 2 | 1
    >>> median(a)
    3.5
    >>> median(a, axis=0)
    axis0 |   0 |   1 |   2
          | 6.5 | 4.5 | 2.5
    >>> median(a, axis=1)
    axis0 |   0 |   1
          | 7.0 | 2.0
    """
    return array.median(*args, **kwargs)


def percentile(array, *args, **kwargs):
    """
    Examples
    --------

    >>> a = LArray([[10, 7, 4], [3, 2, 1]])
    >>> a
    axis0\\axis1 |  0 | 1 | 2
              0 | 10 | 7 | 4
              1 |  3 | 2 | 1
    >>> # this is a bug in numpy: np.nanpercentile(all axes) returns an ndarray,
    >>> # instead of a scalar.
    >>> percentile(a, 50)
    array(3.5)
    >>> percentile(a, 50, axis=0)
    axis0 |   0 |   1 |   2
          | 6.5 | 4.5 | 2.5
    >>> percentile(a, 50, axis=1)
    axis0 |   0 |   1
          | 7.0 | 2.0
    """
    return array.percentile(*args, **kwargs)


# not commutative
def ptp(array, *args, **kwargs):
    return array.ptp(*args, **kwargs)


def var(array, *args, **kwargs):
    return array.var(*args, **kwargs)


def std(array, *args, **kwargs):
    return array.std(*args, **kwargs)


def concat_empty(axis, array_axes, other_axes, dtype):
    array_axis = array_axes[axis]
    # Get axis by name, so that we do *NOT* check they are "compatible",
    # because it makes sense to append axes of different length
    other_axis = other_axes[axis]
    new_labels = np.append(array_axis.labels, other_axis.labels)
    new_axis = Axis(array_axis.name, new_labels)
    array_axes = array_axes.replace(array_axis, new_axis)
    other_axes = other_axes.replace(other_axis, new_axis)
    # combine axes from both sides (using labels from either side if any)
    result_axes = array_axes | other_axes
    result_data = np.empty(result_axes.shape, dtype=dtype)
    result = LArray(result_data, result_axes)
    l = len(array_axis)
    # XXX: wouldn't it be nice to be able to say that? ie translation
    # from position to label on the original axis then translation to
    # position on the actual result axis?
    # result[:axis.i[-1]]
    return result, result[new_axis.i[:l]], result[new_axis.i[l:]]


class LArrayIterator(object):
    def __init__(self, array):
        self.array = array
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        array = self.array
        if self.position == len(self.array):
            raise StopIteration
        # result = array.i[array.axes[0].i[self.position]]
        result = array.i[self.position]
        self.position += 1
        return result
    # Python 2
    next = __next__


class LArrayPositionalIndexer(object):
    def __init__(self, array):
        self.array = array

    def translate_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.array.ndim:
            raise IndexError("key has too many indices (%d) for array with %d "
                             "dimensions" % (len(key), self.array.ndim))
        # no need to create a full nd key as that will be done later anyway
        return tuple(axis.i[axis_key]
                     for axis_key, axis in zip(key, self.array.axes))

    def __getitem__(self, key):
        return self.array[self.translate_key(key)]

    def __setitem__(self, key, value):
        self.array[self.translate_key(key)] = value


class LArrayPointsIndexer(object):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        # TODO: this should generate an "intersection"/points NDGroup and simply
        #       do return self.array[nd_group]
        data = np.asarray(self.array)
        translated_key = self.array.translated_key(key, bool_stuff=True)

        # FIXME: ideally we should use wildcard_allowed=False, but this is
        #        currently too slow
        axes = self.array._bool_key_new_axes(translated_key,
                                             wildcard_allowed=True)
        data = data[translated_key]
        # drop length 1 dimensions created by scalar keys
        # data = data.reshape(tuple(len(axis) for axis in axes))
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            return LArray(data, axes)

    # FIXME
    def __setitem__(self, key, value):
        raise NotImplementedError()


class LArray(object):
    """
    LArray class
    """
    def __init__(self, data, axes=None):
        data = np.asarray(data)
        ndim = data.ndim
        if axes is None:
            axes = AxisCollection(data.shape)
        else:
            if len(axes) != ndim:
                raise ValueError("number of axes (%d) does not match "
                                 "number of dimensions of data (%d)"
                                 % (len(axes), ndim))
            shape = tuple(len(axis) for axis in axes)
            if shape != data.shape:
                raise ValueError("length of axes %s does not match "
                                 "data shape %s" % (shape, data.shape))

            if not isinstance(axes, AxisCollection):
                axes = AxisCollection(axes)
        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'axes', axes)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        # XXX: maybe I should only catch KeyError here and be more aggressive
        #  in __getitem__ to raise KeyError on any exception
        except Exception:
            return self.__getattribute__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    @property
    def i(self):
        return LArrayPositionalIndexer(self)

    @property
    def points(self):
        return LArrayPointsIndexer(self)

    @property
    def ipoints(self):
        return LArrayPositionalPointsIndexer(self)

    def to_frame(self, fold_last_axis_name=False, dropna=None):
        columns = pd.Index(self.axes[-1].labels)
        if not fold_last_axis_name:
            columns.name = self.axes[-1].name
        if self.ndim > 1:
            axes_names = self.axes.names[:-1]
            if fold_last_axis_name:
                tmp = axes_names[-1] if axes_names[-1] is not None else ''
                if self.axes[-1].name:
                    axes_names[-1] = "{}\\{}".format(tmp, self.axes[-1].name)

            index = pd.MultiIndex.from_product(self.axes.labels[:-1],
                                               names=axes_names)
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

    def to_series(self, dropna=False):
        index = pd.MultiIndex.from_product([axis.labels for axis in self.axes],
                                           names=self.axes.names)
        series = pd.Series(np.asarray(self).reshape(self.size), index)
        if dropna:
            series.dropna(inplace=True)
        return series
    series = property(to_series)

    #noinspection PyAttributeOutsideInit
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

    def rename(self, axis, newname):
        """Renames an axis of a LArray.

        Parameters
        ----------
        axis
            axis.
        newname
            string -> the new name for the axis.
        Returns
        -------
        LArray
            LArray with one of the axis renamed.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> a = ones([xnat, xsex])
        >>> a
        nat\\sex |   H |   F
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
        >>> a.rename('nat', 'newnat')
        newnat\\sex |   H |   F
                BE | 1.0 | 1.0
                FO | 1.0 | 1.0
        """
        axis = self.axes[axis]
        axes = [axis.rename(newname) if a is axis else a
                for a in self.axes]
        return LArray(self.data, axes)

    def sort_values(self, key):
        """Sorts values of the LArray.

        Parameters
        ----------
        key : scalar or tuple or Group
            key along which to sort. Must have exactly one dimension less than
            ndim.

        Returns
        -------
        LArray
            LArray with sorted values.

        Example
        -------
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xnat = Axis('nat', ['EU', 'FO', 'BE'])
        >>> xtype = Axis('type', ['type1', 'type2'])
        >>> a = LArray([[10, 2, 4], [3, 7, 1]], [xsex, xnat])
        >>> a
        sex\\nat | EU | FO | BE
              H | 10 |  2 |  4
              F |  3 |  7 |  1
        >>> a.sort_values('F')
        sex\\nat | BE | EU | FO
              H |  4 | 10 |  2
              F |  1 |  3 |  7
        >>> b = LArray([[[10, 2, 4], [3, 7, 1]], [[5, 1, 6], [2, 8, 9]]],
        ...            [xsex, xtype, xnat])
        >>> b
        sex | type\\nat | EU | FO | BE
          H |    type1 | 10 |  2 |  4
          H |    type2 |  3 |  7 |  1
          F |    type1 |  5 |  1 |  6
          F |    type2 |  2 |  8 |  9
        >>> b.sort_values(('H', 'type2'))
        sex | type\\nat | BE | EU | FO
          H |    type1 |  4 | 10 |  2
          H |    type2 |  1 |  3 |  7
          F |    type1 |  6 |  5 |  1
          F |    type2 |  9 |  2 |  8
        """
        subset = self[key]
        if subset.ndim > 1:
            raise NotImplementedError("sort_values key must have one "
                                      "dimension less than array.ndim")
        assert subset.ndim == 1
        axis = subset.axes[0]
        posargsort = subset.posargsort()

        # FIXME: .data shouldn't be necessary, but currently, if we do not do
        #  it, we get
        # PGroup(nat | EU | FO | BE
        #            |  1 |  2 |  0, axis='nat')
        # which sorts the *data* correctly, but the labels on the nat axis are
        # not sorted (because the __getitem__ in that case reuse the key
        # axis as-is -- like it should).
        # Both use cases have value, but I think reordering the ticks
        # should be the default. Now, I am unsure where to change this.
        # Probably in PGroupMaker.__getitem__, but then how do I get the
        # "not reordering labels" behavior that I have now?
        # FWIW, using .data, I get PGroup([1, 2, 0], axis='nat'), which works.
        sorter = axis.i[posargsort.data]
        return self[sorter]

    # XXX: rename to sort_axes?
    def sort_axis(self, axes=None, reverse=False):
        """Sorts axes of the LArray.

        Parameters
        ----------
        axes : axis reference (Axis, string, int) or list of them
            axis to sort. If None, sorts all axes.
        reverse : bool
            descending sort (default: False -- ascending)

        Returns
        -------
        LArray
            LArray with sorted axes.

        Example
        -------
        >>> xnat = Axis('nat', ['EU', 'FO', 'BE'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> a = ndrange([xnat, xsex])
        >>> a
        nat\\sex | H | F
             EU | 0 | 1
             FO | 2 | 3
             BE | 4 | 5
        >>> a.sort_axis(x.sex)
        nat\\sex | F | H
             EU | 1 | 0
             FO | 3 | 2
             BE | 5 | 4
        >>> a.sort_axis()
        nat\\sex | F | H
             BE | 5 | 4
             EU | 1 | 0
             FO | 3 | 2
        >>> a.sort_axis((x.sex, x.nat))
        nat\\sex | F | H
             BE | 5 | 4
             EU | 1 | 0
             FO | 3 | 2
        >>> a.sort_axis(reverse=True)
        nat\\sex | H | F
             FO | 2 | 3
             EU | 0 | 1
             BE | 4 | 5
        """
        if axes is None:
            axes = self.axes
        elif not isinstance(axes, (tuple, list, AxisCollection)):
            axes = [axes]

        if not isinstance(axes, AxisCollection):
            axes = self.axes[axes]

        def sort_key(axis):
            key = np.argsort(axis.labels)
            if reverse:
                key = key[::-1]
            return axis.i[key]

        return self[tuple(sort_key(axis) for axis in axes)]

    def _translate_axis_key_chunk(self, axis_key, bool_passthrough=True):
        if isinstance(axis_key, Group):
            return axis_key

        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        # TODO: use axis_key dtype to only check compatible axes
        for axis in self.axes:
            try:
                axis_pos_key = axis.translate(axis_key, bool_passthrough)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis"
                             % axis_key)
        elif len(valid_axes) > 1:
            valid_axes = ', '.join(str(a.id) for a in valid_axes)
            raise ValueError('%s is ambiguous (valid in %s)' %
                             (axis_key, valid_axes))
        return valid_axes[0].i[axis_pos_key]

    def _translate_axis_key(self, axis_key, bool_passthrough=True):
        # TODO: do it for LArray key too (but using .i[] instead)
        if isinstance(axis_key, (tuple, list, np.ndarray)):
            axis = None
            for size in (1, 10, 100, 1000):
                # TODO: do not recheck already checked elements
                key_chunk = axis_key[:size]
                try:
                    tkey = self._translate_axis_key_chunk(key_chunk,
                                                          bool_passthrough)
                    axis = tkey.axis
                    break
                except ValueError:
                    continue
            if axis is not None:
                # make sure we have an Axis object
                axis = self.axes[axis]
                # wrap key in LGroup
                axis_key = axis[axis_key]
                # XXX: reuse tkey chunks and only translate the rest?
            return self._translate_axis_key_chunk(axis_key,
                                                  bool_passthrough)
        else:
            return self._translate_axis_key_chunk(axis_key, bool_passthrough)

    def _guess_axis(self, axis_key):
        if isinstance(axis_key, Group):
            return axis_key

        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        for axis in self.axes:
            try:
                axis.translate(axis_key)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis"
                             % axis_key)
        elif len(valid_axes) > 1:
            valid_axes = ', '.join(str(a.id) for a in valid_axes)
            raise ValueError('%s is ambiguous (valid in %s)' %
                             (axis_key, valid_axes))
        return valid_axes[0][axis_key]

    # TODO: move this to AxisCollection
    def translated_key(self, key, bool_stuff=False):
        """Complete and translate key

        Parameters
        ----------
        key : single axis key or tuple of keys or dict {axis_name: axis_key}
           each axis key can be either a scalar, a list of scalars or
           an LKey

        Returns
        -------
        Returns a full N dimensional positional key
        """

        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.bool_) \
                and not bool_stuff:
            return key.nonzero()
        if isinstance(key, LArray) and np.issubdtype(key.dtype, np.bool_) \
                and not bool_stuff:
            # if only the axes order is wrong, transpose
            if key.size == self.size and key.shape != self.shape:
                return np.asarray(key.transpose(self.axes)).nonzero()
            # otherwise we need to transform the key to integer
            elif key.size != self.size:
                map_key = dict(zip(key.axes.names, np.asarray(key).nonzero()))
                return tuple(map_key[name] if name in map_key else slice(None)
                             for name in self.axes.names)
            else:
                return np.asarray(key).nonzero()

        # convert scalar keys to 1D keys
        if not isinstance(key, (tuple, dict)):
            key = (key,)

        if isinstance(key, tuple):
            # handle keys containing an Ellipsis
            # cannot use key.count(Ellipsis) because that calls == on each
            # element and this breaks if there are ndarrays/LArrays in there.
            num_ellipses = sum(isinstance(k, type(Ellipsis)) for k in key)
            if num_ellipses > 1:
                raise ValueError("cannot use more than one Ellipsis (...)")
            elif num_ellipses == 1:
                # XXX: we might want to just ignore them, since their
                # position do not matter anymore (guess axis)
                pos = key.index(Ellipsis)
                none_slices = (slice(None),) * (self.ndim - len(key) + 1)
                key = key[:pos] + none_slices + key[pos + 1:]

            # translate non LKey to PGroup and drop slice(None) since
            # they are meaningless at this point
            # XXX: we might want to raise an exception when we find (most)
            # slice(None) because except for a single slice(None) a[:], I don't
            # think there is any point.
            key = tuple(
                self._translate_axis_key(axis_key,
                                         bool_passthrough=not bool_stuff)
                for axis_key in key if not isnoneslice(axis_key))

            assert all(isinstance(axis_key, Group) for axis_key in key)

            # handle keys containing LGroups (at potentially wrong places)

            # XXX: support LGroup without axis?
            # extract axis name from LGroup keys

            dupe_axes = list(duplicates(axis_key.axis for axis_key in key))
            if dupe_axes:
                dupe_axes = ', '.join(str(axis) for axis in dupe_axes)
                raise ValueError("key with duplicate axis: %s" % dupe_axes)
            key = dict((axis_key.axis_id, axis_key) for axis_key in key)

        # dict -> tuple (complete and order key)
        assert isinstance(key, dict)
        for axis_id in key:
            if axis_id not in self.axes:
                raise KeyError("{} is not a valid axis".format(repr(axis_id)))
        key = tuple(key[axis.id] if axis.id in key else slice(None)
                    for axis in self.axes)

        # label -> raw positional
        return tuple(axis.translate(axis_key, bool_passthrough=not bool_stuff)
                     for axis, axis_key in zip(self.axes, key))

    # TODO: we only need axes length => move this to AxisCollection
    # (but this backend/numpy-specific so we'll probably need to create a
    #  subclass of it)
    def cross_key(self, key, collapse_slices=False):
        """
        :param key: a complete (contains all dimensions) index-based key
        :param collapse_slices: convert contiguous ranges to slices
        :return: a key for indexing the cross product
        """
        # isinstance(ndarray, collections.Sequence) is False but it
        # behaves like one
        sequence = (tuple, list, np.ndarray)
        if collapse_slices:
            key = [range_to_slice(axis_key, len(axis))
                   if isinstance(axis_key, sequence)
                   else axis_key
                   for axis_key, axis in zip(key, self.axes)]

        # count number of indexing arrays (ie non scalar/slices) in tuple
        num_ix_arrays = sum(isinstance(axis_key, sequence) for axis_key in key)
        num_scalars = sum(np.isscalar(axis_key) for axis_key in key)
        num_slices = sum(isinstance(axis_key, slice) for axis_key in key)
        assert len(key) == num_ix_arrays + num_scalars + num_slices

        # handle advanced indexing with more than one indexing array:
        # basic indexing (only integer and slices) and advanced indexing
        # with only one indexing array are handled fine by numpy
        if num_ix_arrays > 1 or (num_ix_arrays > 0 and num_scalars):
            # np.ix_ wants only lists so:

            # 1) transform scalar-key to lists of 1 element. In that case,
            #    ndarray.__getitem__ leaves length 1 dimensions instead of
            #    dropping them like we would like so we will need to drop
            #    them later ourselves (via reshape)
            noscalar_key = [[axis_key] if np.isscalar(axis_key) else axis_key
                            for axis_key in key]

            # 2) expand slices to lists (ranges)
            # XXX: cache the range in the axis?
            # TODO: fork np.ix_ to allow for slices directly
            # it will be tricky to get right though because in that case the
            # a[key] can have its dimensions in the wrong order (if the
            # ix_arrays are not next to each other, the corresponding
            # dimensions are moved to the front). It is probably worth the
            # trouble though because it is much faster than the current
            # solution (~5x in my simple test) but this case (num_ix_arrays >
            # 1) is rare in the first place (at least in demo) so it is not a
            # priority.
            listkey = tuple(np.arange(*axis_key.indices(len(axis)))
                            if isinstance(axis_key, slice)
                            else axis_key
                            for axis_key, axis in zip(noscalar_key, self.axes))
            # np.ix_ computes the cross product of all lists
            return np.ix_(*listkey)
        else:
            return tuple(key)

    def __getitem__(self, key, collapse_slices=False):
        # move this to getattr
        # if isinstance(key, str) and key in ('__array_struct__',
        #                               '__array_interface__'):
        #     raise KeyError("bla")
        data = np.asarray(self.data)
        translated_key = self.translated_key(key)

        # TODO: make the combined keys should be objects which display as:
        # (axis1_label, axis2_label, ...) but should also store the axis (names)
        # Q: Should it be the same object as the NDLGroup?/NDKey?
        # A: yes, probably. On the Pandas backend, we could/should have
        #    separate axes. On the numpy backend we cannot.

        # FIXME: I have a huge problem with boolean labels + non points
        if isinstance(key, (LArray, np.ndarray)) and \
                np.issubdtype(key.dtype, np.bool_):
            return LArray(data[translated_key],
                          self._bool_key_new_axes(translated_key))

        if any(isinstance(axis_key, LArray) for axis_key in translated_key):
            k2 = [k.data if isinstance(k, LArray) else k
                  for k in translated_key]
            data = data[k2]
            axes = [axis.subaxis(axis_key)
                    for axis, axis_key in zip(self.axes, translated_key)
                    if not np.isscalar(axis_key)]

            # subaxis can return tuple of axes and we want a single list of axes
            # not a nested structure. We could do both in one step but it's
            # awfully unreadable.
            def flatten(l):
                return [e for sublist in l for e in sublist]

            def to2d(l):
                return [mixed if isinstance(mixed, (tuple, list)) else [mixed]
                        for mixed in l]

            axes = flatten(to2d(axes))
            return LArray(data, axes)

        # TODO: if the original key was a label key, subaxis(translated_key) == orig_key, so we should use
        # orig_axis_key.copy()
        axes = [axis.subaxis(axis_key)
                for axis, axis_key in zip(self.axes, translated_key)
                if not np.isscalar(axis_key)]

        cross_key = self.cross_key(translated_key, collapse_slices)
        data = data[cross_key]
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            # drop length 1 dimensions created by scalar keys
            data = data.reshape(tuple(len(axis) for axis in axes))
            return LArray(data, axes)

    def __setitem__(self, key, value, collapse_slices=True):
        data = np.asarray(self.data)
        translated_key = self.translated_key(key)

        if isinstance(key, (LArray, np.ndarray)) and \
                np.issubdtype(key.dtype, np.bool_):
            if isinstance(value, LArray):
                new_axes = self._bool_key_new_axes(translated_key,
                                                   wildcard_allowed=True)
                value = value.broadcast_with(new_axes)
            data[translated_key] = value
            return

        # XXX: we might want to create fakes axes in this case, as we only
        # use axes names and axes length, not the ticks, and those could
        # theoretically take a significant time to compute
        axes = [axis.subaxis(axis_key)
                for axis, axis_key in zip(self.axes, translated_key)
                if not np.isscalar(axis_key)]

        cross_key = self.cross_key(translated_key, collapse_slices)

        # if value is a "raw" ndarray we rely on numpy broadcasting
        data[cross_key] = value.broadcast_with(axes) \
            if isinstance(value, LArray) else value

    def _bool_key_new_axes(self, key, wildcard_allowed=False):
        combined_axes = [axis for axis_key, axis in zip(key, self.axes)
                         if not isnoneslice(axis_key) and
                            not np.isscalar(axis_key)]
        # scalar axes are not taken, since we want to kill them
        other_axes = [axis for axis_key, axis in zip(key, self.axes)
                      if isnoneslice(axis_key)]
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
        combined_name = ','.join(str(axis.id) for axis in combined_axes)
        new_axes = other_axes
        if combined_axis_pos is not None:
            if wildcard_allowed:
                lengths = [len(axis_key) for axis_key in key
                           if not isnoneslice(axis_key) and
                              not np.isscalar(axis_key)]
                combined_axis_len = lengths[0]
                assert all(l == combined_axis_len for l in lengths)
                combined_axis = Axis(combined_name, combined_axis_len)
            else:
                axes_labels = [axis.labels[axis_key]
                               for axis_key, axis in zip(key, self.axes)
                               if not isnoneslice(axis_key) and
                                  not np.isscalar(axis_key)]
                if len(combined_axes) == 1:
                    combined_labels = axes_labels[0]
                else:
                    combined_labels = list(zip(*axes_labels))

                # CRAP, this can lead to duplicate labels (especially using .points)
                combined_axis = Axis(combined_name, combined_labels)
            new_axes.insert(combined_axis_pos, combined_axis)
        return AxisCollection(new_axes)

    def set(self, value, **kwargs):
        """
        sets a subset of LArray to value

        * all common axes must be either 1 or the same length
        * extra axes in value must be of length 1
        * extra axes in self can have any length
        """
        self.__setitem__(kwargs, value)

    def reshape(self, target_axes):
        """
        self.size must be equal to prod([len(axis) for axis in target_axes])
        """
        # this is a dangerous operation, because except for adding
        # length 1 axes (which is safe), it potentially modifies data
        # TODO: add a check/flag? for "unsafe" reshapes (but allow merging
        # several axes & "splitting" axes) etc.
        # eg 4, 3, 2 -> 2, 3, 4 is wrong (even if size is respected)
        #    4, 3, 2 -> 12, 2 is potentially ok (merging adjacent dimensions)
        #            -> 4, 6 is potentially ok (merging adjacent dimensions)
        #            -> 24 is potentially ok (merging adjacent dimensions)
        #            -> 3, 8 WRONG (non adjacent dimentsions)
        #            -> 8, 3 WRONG
        #    4, 3, 2 -> 2, 2, 3, 2 is potentially ok (splitting dim)
        data = np.asarray(self).reshape([len(axis) for axis in target_axes])
        return LArray(data, target_axes)

    def reshape_like(self, target):
        """
        target is an LArray, total size must be compatible
        """
        return self.reshape(target.axes)

    def broadcast_with(self, other):
        """
        returns an LArray that is (numpy) broadcastable with target
        target can be either an LArray or any collection of Axis

        * all common axes must be either 1 or the same length
        * extra axes in source can have any length and will be moved to the
          front
        * extra axes in target can have any length and the result will have axes
          of length 1 for those axes

        this is different from reshape which ensures the result has exactly the
        shape of the target.
        """
        if isinstance(other, LArray):
            other_axes = other.axes
        else:
            other_axes = other
            if not isinstance(other, AxisCollection):
                other_axes = AxisCollection(other_axes)
        other_names = [a.name for a in other_axes]

        # XXX: this breaks la['1,5,9'] = la['2,7,3']
        # but that use case should use drop_labels
        # self.axes.check_compatible(other_axes)

        # 1) append length-1 axes for other-only axes
        # TODO: factorize with make_numpy_broadcastable
        otheronly_axes = [Axis(axis.name, 1) if len(axis) != 1 else axis
                          for axis in other_axes if axis not in self.axes]
        array = self.reshape(self.axes + otheronly_axes)
        # 2) reorder axes to target order (move source-only axes to the front)
        sourceonly_axes = self.axes - other_axes
        axes_other_order = [array.axes[name] for name in other_names]
        return array.transpose(sourceonly_axes + axes_other_order)

    def drop_labels(self, axes=None):
        """drop the labels from axes (replace those axes by "wildcard" axes)

        Parameters
        ----------
        axes : Axis or list/tuple/AxisCollection of Axis

        Returns
        -------
        LArray

        Examples
        --------
        >>> a = Axis('a', ['a1', 'a2'])
        >>> b = Axis('b', ['b1', 'b2'])
        >>> b2 = Axis('b', ['b2', 'b3'])
        >>> arr1 = ndrange([a, b])
        >>> arr1
        a\\b | b1 | b2
         a1 |  0 |  1
         a2 |  2 |  3
        >>> arr1.drop_labels(b)
        a\\b* | 0 | 1
          a1 | 0 | 1
          a2 | 2 | 3
        >>> arr1.drop_labels([a, b])
        a*\\b* | 0 | 1
            0 | 0 | 1
            1 | 2 | 3
        >>> arr2 = ndrange([a, b2])
        >>> arr2
        a\\b | b2 | b3
         a1 |  0 |  1
         a2 |  2 |  3
        >>> arr1 * arr2
        Traceback (most recent call last):
        ...
        ValueError: incompatible axes:
        Axis('b', ['b2', 'b3'])
        vs
        Axis('b', ['b1', 'b2'])
        >>> arr1 * arr2.drop_labels()
        a\\b | b1 | b2
         a1 |  0 |  1
         a2 |  4 |  9
        >>> arr1.drop_labels() * arr2
        a\\b | b2 | b3
         a1 |  0 |  1
         a2 |  4 |  9
        >>> arr1.drop_labels('a') * arr2.drop_labels('b')
        a\\b | b1 | b2
         a1 |  0 |  1
         a2 |  4 |  9
        """
        if axes is None:
            axes = self.axes
        if not isinstance(axes, (tuple, list, AxisCollection)):
            axes = [axes]
        old_axes = self.axes[axes]
        new_axes = [Axis(axis.name, len(axis)) for axis in old_axes]
        res_axes = self.axes.replace(axes, new_axes)
        return LArray(self.data, res_axes)

    def __str__(self):
        if not self.ndim:
            return str(np.asscalar(self))
        elif not len(self):
            return 'LArray([])'
        else:
            return table2str(list(self.as_table()), 'nan', True,
                             keepcols=self.ndim - 1)
    __repr__ = __str__

    def __iter__(self):
        return LArrayIterator(self)

    def as_table(self, maxlines=200, edgeitems=5):
        if not self.ndim:
            return

        # ert    | unit | geo\time | 2012   | 2011   | 2010
        # NEER27 | I05  | AT       | 101.41 | 101.63 | 101.63
        # NEER27 | I05  | AU       | 134.86 | 125.29 | 117.08
        width = self.shape[-1]
        height = int(np.prod(self.shape[:-1]))
        data = np.asarray(self).reshape(height, width)

        axes_names = self.axes.display_names[:]
        if len(axes_names) > 1:
            axes_names[-2] = '\\'.join(axes_names[-2:])
            axes_names.pop()
        labels = self.axes.labels[:-1]
        if self.ndim == 1:
            # There is no vertical axis, so the axis name should not have
            # any "tick" below it and we add an empty "tick".
            ticks = [['']]
        else:
            ticks = product(*labels)
        yield axes_names + list(self.axes.labels[-1])

        # summary if needed
        if height > maxlines:
            data = chain(data[:edgeitems], [["..."] * width], data[-edgeitems:])
            if height > maxlines:
                startticks = islice(ticks, edgeitems)
                midticks = [["..."] * (self.ndim - 1)]
                endticks = list(islice(rproduct(*labels), edgeitems))[::-1]
                ticks = chain(startticks, midticks, endticks)

        for tick, dataline in izip(ticks, data):
            yield list(tick) + list(dataline)

    # XXX: should filter(geo=['W']) return a view by default? (collapse=True)
    # I think it would be dangerous to make it the default
    # behavior, because that would introduce a subtle difference between
    # filter(dim=[a, b]) and filter(dim=[a]) even though it would be faster
    # and uses less memory. Maybe I should have a "view" argument which
    # defaults to 'auto' (ie collapse by default), can be set to False to
    # force a copy and to True to raise an exception if a view is not possible.
    def filter(self, collapse=False, **kwargs):
        """
        filters the array along the axes given as keyword arguments.
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
            a aggregate function with this signature:
            func(a, axis=None, dtype=None, out=None, keepdims=False)
        axes : tuple of axes, optional
            each axis can be an Axis object, str or int
        out : LArray, optional
        keepaxes : bool or scalar, optional

        Returns
        -------
        LArray or scalar
        """
        src_data = np.asarray(self)
        axes = list(axes) if axes else self.axes
        axes_indices = tuple(self.axes.index(a) for a in axes)
        keepdims = bool(keepaxes)
        if out is not None:
            assert isinstance(out, LArray)
            kwargs['out'] = out.data
        res_data = op(src_data, axis=axes_indices, keepdims=keepdims, **kwargs)

        if keepaxes:
            label = op.__name__.replace('nan', '') if keepaxes is True \
                else keepaxes
            axes_to_kill = [self.axes[axis] for axis in axes]
            new_axes = [Axis(axis.name, [label]) for axis in axes_to_kill]
            res_axes = self.axes.replace(axes_to_kill, new_axes)
        else:
            res_axes = self.axes - axes_indices
        if not res_axes:
            # scalars don't need to be wrapped in LArray
            return res_data
        else:
            return LArray(res_data, res_axes)

    def _cum_aggregate(self, op, axis):
        """
        op is a numpy cumulative aggregate function: func(arr, axis=0)
        axis is an Axis object, a str or an int. Contrary to other aggregate
        functions this only supports one axis at a time.
        """
        # TODO: accept a single group in axis, to filter & aggregate in one shot
        return LArray(op(np.asarray(self), axis=self.axes.index(axis)),
                      self.axes)

    # TODO: now that items is never a (k, v), it should be renamed to
    # something else: args? (groups would be misleading because each "item"
    # can contain several groups)
    # TODO: experiment implementing this using ufunc.reduceat
    # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ufunc.reduceat.html
    # XXX: rename keepaxes to label=value? For group_aggregates we might
    # want to keep the VG label if any
    def _group_aggregate(self, op, items, keepaxes=False, out=None, **kwargs):
        assert out is None
        res = self
        # TODO: when working with several "axes" at the same times, we should
        # not produce the intermediary result at all. It should be faster and
        # consume a bit less memory.
        for item in items:
            res_axes = res.axes[:]
            res_shape = list(res.shape)

            if isinstance(item, tuple):
                assert all(isinstance(g, Group) for g in item)
                groups = item
                axis = groups[0].axis
                killaxis = False
            else:
                # item is in fact a single group
                assert isinstance(item, Group), type(item)
                groups = (item,)
                axis = item.axis
                # it is easier to kill the axis after the fact
                killaxis = True

            axis, axis_idx = res.axes[axis], res.axes.index(axis)
            res_shape[axis_idx] = len(groups)
            res_dtype = res.dtype if op not in (np.mean, np.nanmean) else float
            res_data = np.empty(res_shape, dtype=res_dtype)

            group_idx = [slice(None) for _ in res_shape]
            for i, group in enumerate(groups):
                group_idx[axis_idx] = i
                # this is only useful for ndim == 1 because
                # a[(0,)] (equivalent to a[0] which kills the axis)
                # is different from a[[0]] (which does not kill the axis)
                idx = tuple(group_idx)

                # we need only lists of ticks, not single ticks, otherwise the
                # dimension is discarded too early (in __getitem__ instead of in
                # the aggregate func)
                if isinstance(group, PGroup) and np.isscalar(group.key):
                    group = PGroup([group.key], axis=group.axis)
                elif isinstance(group, LGroup):
                    key = to_key(group.key)
                    if np.isscalar(key):
                        key = [key]
                    # we do not care about the name at this point
                    group = LGroup(key, axis=group.axis)

                arr = res.__getitem__({axis.name: group}, collapse_slices=True)
                if res_data.ndim == 1:
                    assert len(idx) == 1 and idx[0] == i

                    # res_data[idx] but instead of returning a scalar (eg
                    # np.int32), it returns a 0d array which is a view on
                    # res_data, which can thus be used as out
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
                # We do NOT modify the axis name (eg append "_agg" or "*") even
                # though this creates a new axis that is independent from the
                # original one because the original name is what users will
                # want to use to access that axis (eg in .filter kwargs)
                res_axes[axis_idx] = Axis(axis.name, groups)

            if isinstance(res_data, np.ndarray):
                res = LArray(res_data, res_axes)
            else:
                res = res_data
        return res

    def _prepare_aggregate(self, op, args, kwargs=None, commutative=False):
        """converts args to keys & VG and kwargs to VG"""

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
            raise ValueError("grouping aggregates on multiple axes at the same "
                             "time using keyword arguments is not supported "
                             "for '%s' (because it is not a commutative"
                             "operation and keyword arguments are *not* "
                             "ordered in Python)" % op.__name__)

        # Sort kwargs by axis name so that we have consistent results
        # between runs because otherwise rounding errors could lead to
        # slightly different results even for commutative operations.
        sorted_kwargs = sorted(kwargs_items)

        # convert kwargs to LGroup so that we can only use args afterwards
        # but still keep the axis information
        def standardise_kw_arg(axis_name, key):
            if isinstance(key, str):
                key = to_keys(key)
            if isinstance(key, tuple):
                return tuple(standardise_kw_arg(axis_name, k) for k in key)
            if isinstance(key, LGroup):
                return key
            return self.axes[axis_name][key]

        def to_labelgroup(key):
            if isinstance(key, str):
                key = to_keys(key)
            if isinstance(key, tuple):
                # a tuple is supposed to be several groups on the same axis
                # XXX: we might want to use self._translate_axis_key directly
                # (so that we do not need to do the label -> position
                #  translation twice)
                groups = tuple(self._guess_axis(k) for k in key)
                axis = groups[0].axis
                if not all(g.axis == axis for g in groups[1:]):
                    raise ValueError("group with different axes: %s"
                                     % str(key))
                return groups
            if isinstance(key, Group):
                return key
            elif isinstance(key, (int, basestring, list, slice)):
                return self._guess_axis(key)
            else:
                raise NotImplementedError("%s has invalid type (%s) for a "
                                          "group aggregate key"
                                          % (key, type(key).__name__))

        def standardise_arg(arg):
            if self.axes.isaxis(arg):
                return self.axes[arg]
            else:
                return to_labelgroup(to_keys(arg))

        operations = [standardise_arg(a) for a in args if a is not None] + \
                     [standardise_kw_arg(k, v) for k, v in sorted_kwargs]
        if not operations:
            # op() without args is equal to op(all_axes)
            operations = self.axes
        return operations

    def _aggregate(self, op, args, kwargs=None, keepaxes=False,
                   commutative=False, out=None, extra_kwargs={}):
        operations = self._prepare_aggregate(op, args, kwargs, commutative)
        res = self
        # group *consecutive* same-type (group vs axis aggregates) operations
        # we do not change the order of operations since we only group
        # consecutive operations.
        for are_axes, axes in groupby(operations, self.axes.isaxis):
            func = res._axis_aggregate if are_axes else res._group_aggregate
            res = func(op, axes, keepaxes=keepaxes, out=out, **extra_kwargs)
        return res

    def with_total(self, *args, **kwargs):
        """
        Parameters
        ----------
        args
        kwargs
        op : aggregate function
            Defaults to `sum()`.
        label : scalar value
            label to use for the total. Defaults to "total".

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = ndrange([xnat, xsex])
        >>> arr.with_total()
        nat\\sex | H | F | total
             BE | 0 | 1 |     1
             FO | 2 | 3 |     5
          total | 2 | 4 |     6
        >>> arr = ndrange([Axis('a', 2), Axis('b', 3)])
        >>> arr.with_total()
          a\\b | 0 | 1 | 2 | total
            0 | 0 | 1 | 2 |     3
            1 | 3 | 4 | 5 |    12
        total | 3 | 5 | 7 |    15
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
        operations = self._prepare_aggregate(op, args, kwargs, False)
        res = self
        # TODO: we should allocate the final result directly and fill it
        #       progressively, so that the original array is only copied once
        for axis in operations:
            # TODO: append/extend first with an empty array then
            #       _aggregate with out=
            if self.axes.isaxis(axis):
                value = res._axis_aggregate(npop[op], (axis,), keepaxes=label)
            else:
                # groups
                if not isinstance(axis, tuple):
                    # assume a single group
                    axis = (axis,)
                vgkey = axis
                axis = vgkey[0].axis
                value = res._aggregate(npop[op], (vgkey,))
            res = res.extend(axis, value)
        return res

    # TODO: make sure we can do
    # arr[x.sex.i[arr.posargmin(x.sex)]]
    # and
    # arr[arr.argmin(x.sex)]
    # should both be equal to arr.min(x.sex)
    def argmin(self, axis):
        """
        Return labels of the minimum values along the given axis of `a`.

        Parameters
        ----------
        axis : int or str or Axis
            Axis along which to work.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [xnat, xsex])
        >>> arr
        nat\\sex | H | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.argmin(x.sex)
        nat | BE | FR | IT
            |  H |  F |  H
        """
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        data = axis.labels[self.data.argmin(axis_idx)]
        return LArray(data, self.axes - axis)

    def posargmin(self, axis):
        """
        Return indices of the minimum values along the given axis of `a`.

        Parameters
        ----------
        axis : int or str or Axis
            Axis along which to work.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [xnat, xsex])
        >>> arr
        nat\\sex | H | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.posargmin(x.sex)
        nat | BE | FR | IT
            |  0 |  1 |  0
        """
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        return LArray(self.data.argmin(axis_idx), self.axes - axis)

    def argmax(self, axis):
        """
        Return labels of the maximum values along the given axis of `a`.

        Parameters
        ----------
        axis : int or str or Axis
            Axis along which to work.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [xnat, xsex])
        >>> arr
        nat\\sex | H | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.argmax(x.sex)
        nat | BE | FR | IT
            |  F |  H |  F
        """
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        data = axis.labels[self.data.argmax(axis_idx)]
        return LArray(data, self.axes - axis)

    def posargmax(self, axis):
        """
        Return indices of the maximum values along the given axis of `a`.

        Parameters
        ----------
        axis : int or str or Axis
            Axis along which to work.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [xnat, xsex])
        >>> arr
        nat\\sex | H | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.posargmax(x.sex)
        nat | BE | FR | IT
            |  1 |  0 |  1
        """
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        return LArray(self.data.argmax(axis_idx), self.axes - axis)

    def argsort(self, axis=None, kind='quicksort'):
        """
        Returns the labels that would sort this array.

        Perform an indirect sort along the given axis using the algorithm
        specified by the `kind` keyword. It returns an array of labels of the
        same shape as `a` that index data along the given axis in sorted order.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to sort.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Defaults to 'quicksort'.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [xnat, xsex])
        >>> arr
        nat\\sex | H | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.argsort(x.sex)
        nat\\sex | 0 | 1
             BE | H | F
             FR | F | H
             IT | H | F
        """
        if axis is None:
            if self.ndim > 1:
                raise ValueError("array has ndim > 1 and no axis specified for "
                                 "argsort")
            axis = self.axes[0]
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        data = axis.labels[self.data.argsort(axis_idx, kind=kind)]
        new_axis = Axis(axis.name, np.arange(len(axis)))
        return LArray(data, self.axes.replace(axis, new_axis))

    def posargsort(self, axis=None, kind='quicksort'):
        """
        Returns the indices that would sort this array.

        Perform an indirect sort along the given axis using the algorithm
        specified by the `kind` keyword. It returns an array of indices
        with the same axes as `a` that index data along the given axis in
        sorted order.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to sort.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Defaults to 'quicksort'.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [xnat, xsex])
        >>> arr
        nat\\sex | H | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.posargsort(x.sex)
        nat\\sex | H | F
             BE | 0 | 1
             FR | 1 | 0
             IT | 0 | 1
        """
        if axis is None:
            if self.ndim > 1:
                raise ValueError("array has ndim > 1 and no axis specified for "
                                 "posargsort")
            axis = self.axes[0]
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        return LArray(self.data.argsort(axis_idx, kind=kind), self.axes)

    def copy(self):
        return LArray(self.data.copy(), axes=self.axes[:])

    @property
    def info(self):
        """Describes a LArray (shape and labels for each axis).

        Returns
        -------
        str
            Description of the LArray (shape and labels for each axis).

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat0 = ones([xnat, xsex])
        >>> mat0.info
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'H' 'F'
        """
        return self.axes.info

    def ratio(self, *axes):
        """Returns a LArray with values LArray / LArray.sum(axes).

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            LArray = LArray / LArray.sum(axes).

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat = ndrange([xnat, xsex])
        >>> mat
        nat\\sex | H | F
             BE | 0 | 1
             FO | 2 | 3
        >>> mat.ratio()
        nat\\sex |              H |              F
             BE |            0.0 | 0.166666666667
             FO | 0.333333333333 |            0.5
        >>> mat.ratio(xsex)
        nat\\sex |   H |   F
             BE | 0.0 | 1.0
             FO | 0.4 | 0.6
        """
        return self / self.sum(*axes)

    def percent(self, *axes):
        """Returns a LArray with values LArray / LArray.sum(axes) * 100.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            LArray = LArray / LArray.sum(axes) * 100

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat = ndrange([xnat, xsex])
        >>> mat
        nat\\sex | H | F
             BE | 0 | 1
             FO | 2 | 3
        >>> mat.percent()
        nat\\sex |             H |             F
             BE |           0.0 | 16.6666666667
             FO | 33.3333333333 |          50.0
        >>> mat.percent(xsex)
        nat\\sex |    H |     F
             BE |  0.0 | 100.0
             FO | 40.0 |  60.0
        """
        # dividing by self.sum(*axes) * 0.01 would be faster in many cases but
        # I suspect it loose more precision.
        return self * 100 / self.sum(*axes)

    # aggregate method factory
    def _agg_method(npfunc, nanfunc=None, name=None, commutative=False):
        def method(self, *args, **kwargs):
            keepaxes = kwargs.pop('keepaxes', False)
            skipna = kwargs.pop('skipna', None)
            if skipna is None:
                skipna = nanfunc is not None
            if skipna and nanfunc is None:
                raise ValueError("skipna is not available for %s" % name)
            func = nanfunc if skipna else npfunc
            return self._aggregate(func, args, kwargs,
                                   keepaxes=keepaxes,
                                   commutative=commutative)
        if name is None:
            name = npfunc.__name__
        method.__name__ = name
        return method

    all = _agg_method(np.all, commutative=True)
    any = _agg_method(np.any, commutative=True)
    # commutative modulo float precision errors
    sum = _agg_method(np.sum, np.nansum, commutative=True)
    # nanprod needs numpy 1.10
    prod = _agg_method(np.prod, np_nanprod, commutative=True)
    min = _agg_method(np.min, np.nanmin, commutative=True)
    max = _agg_method(np.max, np.nanmax, commutative=True)
    mean = _agg_method(np.mean, np.nanmean, commutative=True)
    median = _agg_method(np.median, np.nanmedian, commutative=True)

    # percentile needs an explicit method because it has not the same
    # signature as other aggregate functions (extra argument)
    def percentile(self, q, *args, **kwargs):
        keepaxes = kwargs.pop('keepaxes', False)
        skipna = kwargs.pop('skipna', None)
        if skipna is None:
            skipna = True
        func = np.nanpercentile if skipna else np.percentile
        return self._aggregate(func, args, kwargs, keepaxes=keepaxes,
                               commutative=True, extra_kwargs={'q': q})

    # not commutative
    ptp = _agg_method(np.ptp)
    var = _agg_method(np.var, np.nanvar)
    std = _agg_method(np.std, np.nanstd)

    # cumulative aggregates
    def cumsum(self, axis):
        return self._cum_aggregate(np.cumsum, axis)

    def cumprod(self, axis):
        return self._cum_aggregate(np.cumprod, axis)

    # element-wise method factory
    def _binop(opname):
        fullname = '__%s__' % opname
        super_method = getattr(np.ndarray, fullname)

        def opmethod(self, other):
            res_axes = self.axes
            if isinstance(other, LArray):
                # TODO: first test if it is not already broadcastable
                (self, other), res_axes = \
                    make_numpy_broadcastable([self, other])
                other = other.data
            elif isinstance(other, np.ndarray):
                pass
            elif other is None:
                return False
            elif not np.isscalar(other):
                raise TypeError("unsupported operand type(s) for %s: '%s' "
                                "and '%s'" % (opname, type(self), type(other)))
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

    # XXX: rename/change to "add_axes" ?
    # TODO: add a flag copy=True to force a new array.
    def expand(self, target_axes=None, out=None, readonly=False):
        """expands array to target_axes

        target_axes will be added to array if not present. In most cases this
        function is not needed because LArray can do operations with arrays
        having different (compatible) axes.

        Parameters
        ----------
        target_axes : list of Axis or AxisCollection, optional
            self can contain axes not present in target_axes
        out : LArray, optional
            output array, must have the correct shape
        readonly : bool, optional
            whether returning a readonly view is acceptable or not (this is
            much faster)

        Returns
        -------
        LArray
            original array if possible (and out is None)

        Example
        -------
        >>> a = Axis('a', ['a1', 'a2'])
        >>> b = Axis('b', ['b1', 'b2'])
        >>> arr = ndrange([a, b])
        >>> arr
        a\\b | b1 | b2
         a1 |  0 |  1
         a2 |  2 |  3
        >>> c = Axis('c', ['c1', 'c2'])
        >>> arr.expand([a, c, b])
         a | c\\b | b1 | b2
        a1 |  c1 |  0 |  1
        a1 |  c2 |  0 |  1
        a2 |  c1 |  2 |  3
        a2 |  c2 |  2 |  3
        >>> arr.expand([b, c])
         a | b\\c | c1 | c2
        a1 |  b1 |  0 |  0
        a1 |  b2 |  1 |  1
        a2 |  b1 |  2 |  2
        a2 |  b2 |  3 |  3
        """
        if (target_axes is None and out is None or
                target_axes is not None and out is not None):
            raise ValueError("either target_axes or out must be defined "
                             "(not both)")
        if out is not None:
            target_axes = out.axes
        else:
            if not isinstance(target_axes, AxisCollection):
                target_axes = AxisCollection(target_axes)
            target_axes = (self.axes - target_axes) | target_axes
        if out is None and self.axes == target_axes:
            return self

        broadcasted = self.broadcast_with(target_axes)
        # this can only happen if only the order of axes differed
        if out is None and broadcasted.axes == target_axes:
            return broadcasted

        if out is None:
            if readonly:
                # requires numpy 1.10
                return LArray(np.broadcast_to(broadcasted, target_axes.shape),
                              target_axes)
            else:
                out = LArray(np.empty(target_axes.shape, dtype=self.dtype),
                             target_axes)
        out[:] = broadcasted
        return out

    def append(self, axis, value, label=None):
        """Adds a LArray to a LArray ('self') along an axis.

        Parameters
        ----------
        axis : axis
            the axis
        value : LArray
            LArray with compatible axes
        label : string
            optional
            label for the new item in axis

        Returns
        -------
        LArray
            LArray expanded with 'value' along 'axis'.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type', ['type1', 'type2'])
        >>> mat = ones([xnat, xsex])
        >>> mat
        nat\\sex |   H |   F
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
        >>> mat.append(x.sex, mat.sum(x.sex), 'H+F')
        nat\\sex |   H |   F | H+F
             BE | 1.0 | 1.0 | 2.0
             FO | 1.0 | 1.0 | 2.0
        >>> mat.append(x.nat, 2, 'Other')
        nat\\sex |   H |   F
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
          Other | 2.0 | 2.0
        >>> arr2 = zeros([xtype])
        >>> arr2
        type | type1 | type2
             |   0.0 |   0.0
        >>> mat.append(x.nat, arr2, 'Other')
          nat | sex\\type | type1 | type2
           BE |        H |   1.0 |   1.0
           BE |        F |   1.0 |   1.0
           FO |        H |   1.0 |   1.0
           FO |        F |   1.0 |   1.0
        Other |        H |   0.0 |   0.0
        Other |        F |   0.0 |   0.0
        """
        axis = self.axes[axis]
        if np.isscalar(value):
            value = LArray(np.asarray(value, self.dtype), [])
        # this does not prevent value to have more axes than self
        target_axes = self.axes.replace(axis, Axis(axis.name, [label]))
        value = value.broadcast_with(target_axes)
        return self.extend(axis, value)

    def prepend(self, axis, value, label=None):
        """Adds a LArray before 'self' along an axis.

        Parameters
        ----------
        axis : axis
            the axis
        value : LArray
            LArray with compatible axes
        label : string
            optional
            label for the new item in axis

        Returns
        -------
        LArray
            LArray expanded with 'value' at the start of 'axis'.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type', ['type1', 'type2', 'type3'])
        >>> mat = ones([xnat, xsex, xtype])
        >>> mat
        nat | sex\\type | type1 | type2 | type3
         BE |        H |   1.0 |   1.0 |   1.0
         BE |        F |   1.0 |   1.0 |   1.0
         FO |        H |   1.0 |   1.0 |   1.0
         FO |        F |   1.0 |   1.0 |   1.0
        >>> mat.prepend(x.type, mat.sum(x.type), 'type0')
        nat | sex\\type | type0 | type1 | type2 | type3
         BE |        H |   3.0 |   1.0 |   1.0 |   1.0
         BE |        F |   3.0 |   1.0 |   1.0 |   1.0
         FO |        H |   3.0 |   1.0 |   1.0 |   1.0
         FO |        F |   3.0 |   1.0 |   1.0 |   1.0
        >>> mat.prepend(x.type, 2, 'type0')
        nat | sex\\type | type0 | type1 | type2 | type3
         BE |        H |   2.0 |   1.0 |   1.0 |   1.0
         BE |        F |   2.0 |   1.0 |   1.0 |   1.0
         FO |        H |   2.0 |   1.0 |   1.0 |   1.0
         FO |        F |   2.0 |   1.0 |   1.0 |   1.0
        """
        axis = self.axes[axis]
        if np.isscalar(value):
            value = LArray(np.asarray(value, self.dtype), [])
        # this does not prevent value to have more axes than self
        target_axes = self.axes.replace(axis, Axis(axis.name, [label]))
        # we cannot simply add the "new" axis to value because in that case
        # the resulting axes would not be in the correct order
        value = value.broadcast_with(target_axes)
        return value.extend(axis, self)

    def extend(self, axis, other):
        """Adds a LArray to a LArray ('self') along an axis.

        Parameters
        ----------
        axis : axis
            the axis
        other : LArray
            LArray with compatible axes

        Returns
        -------
        LArray
            LArray expanded with 'other' along 'axis'.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xsex2 = Axis('sex', ['U'])
        >>> xtype = Axis('type', ['type1', 'type2'])
        >>> arr1 = ones([xsex, xtype])
        >>> arr1
        sex\\type | type1 | type2
               H |   1.0 |   1.0
               F |   1.0 |   1.0
        >>> arr2 = zeros([xsex2, xtype])
        >>> arr2
        sex\\type | type1 | type2
               U |   0.0 |   0.0
        >>> arr1.extend(x.sex, arr2)
        sex\\type | type1 | type2
               H |   1.0 |   1.0
               F |   1.0 |   1.0
               U |   0.0 |   0.0
        >>> arr3 = zeros([xsex2, xnat])
        >>> arr3
        sex\\nat |  BE |  FO
              U | 0.0 | 0.0
        >>> arr1.extend(x.sex, arr3)
        sex | type\\nat |  BE |  FO
          H |    type1 | 1.0 | 1.0
          H |    type2 | 1.0 | 1.0
          F |    type1 | 1.0 | 1.0
          F |    type2 | 1.0 | 1.0
          U |    type1 | 0.0 | 0.0
          U |    type2 | 0.0 | 0.0
        """
        result, self_target, other_target = \
            concat_empty(axis, self.axes, other.axes, self.dtype)
        self.expand(out=self_target)
        other.expand(out=other_target)
        return result

    def transpose(self, *args):
        """
        reorder axes

        accepts either a tuple of axes specs or axes specs as *args

        Parameters
        ----------
        *args
            accepts either a tuple of axes specs or axes specs as *args

        Returns
        -------
        LArray
            LArray with reordered axes.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat1 = ones([xnat, xsex, xtype])
        >>> mat1
        nat | sex\\type | type1 | type2 | type3
         BE |        H |   1.0 |   1.0 |   1.0
         BE |        F |   1.0 |   1.0 |   1.0
         FO |        H |   1.0 |   1.0 |   1.0
         FO |        F |   1.0 |   1.0 |   1.0
        >>> mat1.transpose(xtype, xsex, xnat)
         type | sex\\nat |  BE |  FO
        type1 |       H | 1.0 | 1.0
        type1 |       F | 1.0 | 1.0
        type2 |       H | 1.0 | 1.0
        type2 |       F | 1.0 | 1.0
        type3 |       H | 1.0 | 1.0
        type3 |       F | 1.0 | 1.0
        >>> mat1.transpose(xtype)
         type | nat\\sex |   H |   F
        type1 |      BE | 1.0 | 1.0
        type1 |      FO | 1.0 | 1.0
        type2 |      BE | 1.0 | 1.0
        type2 |      FO | 1.0 | 1.0
        type3 |      BE | 1.0 | 1.0
        type3 |      FO | 1.0 | 1.0
        """
        if len(args) == 1 and isinstance(args[0],
                                         (tuple, list, AxisCollection)):
            axes = args[0]
        elif len(args) == 0:
            axes = self.axes[::-1]
        else:
            axes = args
        axes = [self.axes[a] for a in axes]
        # FIXME: this is not unnamed axes friendly
        axes_names = set(axis.name for axis in axes)
        missing_axes = [axis for axis in self.axes
                        if axis.name not in axes_names]
        res_axes = axes + missing_axes
        axes_indices = [self.axes.index(axis) for axis in res_axes]
        src_data = np.asarray(self)
        res_data = src_data.transpose(axes_indices)
        return LArray(res_data, res_axes)
    T = property(transpose)

    def clip(self, a_min, a_max, out=None):
        from larray.ufuncs import clip
        return clip(self, a_min, a_max, out)

    def to_csv(self, filepath, sep=',', na_rep='', transpose=True,
               dropna=None, dialect='default', **kwargs):
        """
        write LArray to a csv file.

        Parameters
        ----------
        filepath : string
            path where the csv file has to be written.
        sep : string
            seperator for the csv file.
        na_rep : string
            replace NA values with na_rep.
        transpose : boolean
            transpose = True  => transpose over last axis.
            transpose = False => no transpose.
        dialect : 'default' | 'classic'
            Whether or not to write the last axis name (using '\' )
        dropna : None, 'all', 'any' or True, optional
            Drop lines if 'all' its values are NA, if 'any' value is NA or do
            not drop any line (default). True is equivalent to 'all'.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat = ndrange([xnat, xsex])
        >>> mat
        nat\\sex | H | F
             BE | 0 | 1
             FO | 2 | 3
        >>> mat.to_csv('test.csv')
        >>> with open('test.csv') as f:
        ...     print(f.read().strip())
        nat\\sex,H,F
        BE,0,1
        FO,2,3
        >>> mat.to_csv('test.csv', sep=';', transpose=False)
        >>> with open('test.csv') as f:
        ...     print(f.read().strip())
        nat;sex;0
        BE;H;0
        BE;F;1
        FO;H;2
        FO;F;3
        >>> mat.to_csv('test.csv', dialect='classic')
        >>> with open('test.csv') as f:
        ...     print(f.read().strip())
        nat,H,F
        BE,0,1
        FO,2,3
        """
        fold = dialect == 'default'
        if transpose:
            frame = self.to_frame(fold, dropna)
            frame.to_csv(filepath, sep=sep, na_rep=na_rep, **kwargs)
        else:
            series = self.to_series(dropna is not None)
            series.to_csv(filepath, sep=sep, na_rep=na_rep, header=True,
                          **kwargs)

    def to_hdf(self, filepath, key, *args, **kwargs):
        """
        write LArray to a HDF file

        a HDF file can contain multiple LArray's. The 'key' parameter
        is a unique identifier for the LArray.

        Parameters
        ----------
        filepath : string
            path where the hdf file has to be written.
        key : string
            name of the matrix within the HDF file.
        *args
        **kargs

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat = ndrange([xnat, xsex])
        >>> mat.to_hdf('test.h5', 'mat')
        """
        self.to_frame().to_hdf(filepath, key, *args, **kwargs)

    def to_excel(self, filepath=None, sheet_name=None, position='A1',
                 overwrite_file=False, clear_sheet=False, header=True,
                 transpose=False, engine=None, *args, **kwargs):
        """
        write LArray in the specified sheet of specified excel workbook

        Parameters
        ----------
        filepath : str or int or None, optional
            path where the excel file has to be written. If None (default),
            creates a new Excel Workbook in a live Excel instance
            (Windows only). Use -1 to use the currently active Excel
            Workbook. Use a name without extension (.xlsx) to use any
            *unsaved* workbook.
        sheet_name : str or int or None, optional
            sheet where the data has to be written. Defaults to None,
            Excel standard name if adding a sheet to an existing file,
            "Sheet1" otherwise. sheet_name can also refer to the position of
            the sheet (e.g. 0 for the first sheet, -1 for the last one).
        position : str or tuple of integers, optional
            Integer position (row, column) must be 1-based. Defaults to 'A1'.
        overwrite_file : bool, optional
            whether or not to overwrite the existing file (or just modify the
            specified sheet). Defaults to False.
        clear_sheet : bool, optional
            whether or not to clear the existing sheet (if any) before writing.
            Defaults to False.
        header : bool, optional
            whether or not to write a header (axes names and labels).
            Defaults to True.
        transpose : bool, optional
            whether or not to transpose the resulting array. This can be used,
            for example, for writing one dimensional arrays vertically.
            Defaults to False.
        engine : 'xlwings' | 'openpyxl' | 'xlsxwriter' | 'xlwt' | None, optional
            engine to use to make the output. If None (default), it will use
            'xlwings' by default if the module is installed and relies on
            Pandas default writer otherwise.
        *args
        **kargs

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> a = ndrange([xnat, xsex])
        >>> # write to a new (unnamed) sheet
        >>> a.to_excel('test.xlsx')  # doctest: +SKIP
        >>> # write to top-left corner of an existing sheet
        >>> a.to_excel('test.xlsx', 'Sheet1')  # doctest: +SKIP
        >>> # add to existing sheet starting at position A15
        >>> a.to_excel('test.xlsx', 'Sheet1', 'A15')  # doctest: +SKIP
        """
        df = self.to_frame(fold_last_axis_name=True)
        if engine is None:
            engine = 'xlwings' if xw is not None else None

        if engine == 'xlwings':
            save = False
            close = False
            new_workbook = False
            if filepath == -1:
                wb = xw.Workbook.active()
            elif filepath is None:
                # creates a new/blank Workbook
                wb = xw.Workbook(app_visible=True)
                new_workbook = True
            else:
                basename, ext = os.path.splitext(filepath)
                if ext:
                    # XXX: we might want to be more precise than .xl* because
                    #      I am unsure writing anything else than .xlsx will
                    #      work as intended
                    if not ext.startswith('.xl'):
                        raise ValueError("'%s' is not a supported file "
                                         "extension" % ext)

                    # This is necessary because otherwise we cannot open/save to
                    # a workbook in the current directory without giving the
                    # full/absolute path. By doing this, we basically loose the
                    # ability to target an already open workbook *of a saved
                    # file* not in the current directory without using its
                    # path. I can live with that restriction though because you
                    # usually either work with relative paths or with the
                    # currently active workbook.
                    filepath = os.path.abspath(filepath)
                    save = True
                    close = not xw.xlplatform.is_file_open(filepath)
                    if os.path.isfile(filepath) and overwrite_file:
                        os.remove(filepath)

                    # file already exists (and is a file)
                    if os.path.isfile(filepath):
                        # do not change Excel visibility
                        wb = xw.Workbook(filepath, app_visible=None)
                    else:
                        # open new workbook, do not change Excel visibility
                        wb = xw.Workbook(app_visible=None)
                        new_workbook = True
                else:
                    # try to open an existing unsaved workbook
                    # XXX: I wonder if app_visible=None wouldn't be better in this case?
                    wb = xw.Workbook(filepath, app_visible=True)

            def sheet_exists(wb, sheet):
                if isinstance(sheet, int):
                    length = xw.Sheet.count(wb)
                    return -length <= sheet < length
                else:
                    assert isinstance(sheet_name, str)
                    sheet_names = [i.name.lower() for i in xw.Sheet.all(wkb=wb)]
                    return sheet_name.lower() in sheet_names

            if new_workbook:
                sheet = xw.Sheet(1, wkb=wb)
                if sheet_name is not None:
                    sheet.name = sheet_name
            elif sheet_name is not None and sheet_exists(wb, sheet_name):
                if isinstance(sheet_name, int):
                    if sheet_name < 0:
                        sheet_name += xw.Sheet.count(wb)
                    sheet_name += 1
                sheet = xw.Sheet(sheet_name, wkb=wb)
                if clear_sheet:
                    sheet.clear()
            else:
                sheet = xw.Sheet.add(sheet_name, wkb=wb)

            options = dict(header=header, index=header, transpose=transpose)
            xw.Range(sheet, position).options(**options).value = df
            if save:
                wb.save(filepath)
                if close:
                    wb.close()
                # using app.quit() unconditionally is NOT a good idea because
                # it quits excel even if there are other workbooks open.
                # XXX: we might want to use:
                # app = xw.Application(wb)
                # wbs = xw.xlplatform.get_all_open_xl_workbooks(app.xl_app)
                # if not wbs:
                #    app.quit()
                # but it is not cross-platform (get_all_open_xl...)
        else:
            if sheet_name is None:
                sheet_name = 'Sheet1'
            # TODO: implement position in this case
            # startrow, startcol
            df.to_excel(filepath, sheet_name, *args, engine=engine, **kwargs)

    def to_clipboard(self, *args, **kwargs):
        """
        sends the content of a LArray to clipboard

        using to_clipboard() makes it possible to paste the content of LArray
        into a file (Excel, ascii file,...)

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat = ndrange([xnat, xsex])
        >>> mat.to_clipboard()  # doctest: +SKIP
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
        """
        plots the data of a LArray into a graph (window pop-up).

        the graph can be tweaked to achieve the desired formatting and can be
        saved to a .png file

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.plot()  # doctest: +SKIP
        """
        return self.to_frame().plot

    @property
    def shape(self):
        """
        returns string representation of current shape.

        Returns
        -------
            returns string representation of current shape.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.shape  # doctest: +SKIP
        (2, 2, 3)
        """
        return self.data.shape

    @property
    def ndim(self):
        """
        returns the number of dimensions of a LArray.

        Returns
        -------
            returns the number of dimensions of a LArray.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.ndim
        3
        """
        return self.data.ndim

    @property
    def size(self):
        """
        returns the number of cells in a LArray.

        Returns
        -------
        integer
            returns the number of cells in a LArray.

        Example
        -------
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type', ['type1', 'type2', 'type3'])
        >>> mat = ndrange([xsex, xtype])
        >>> mat.size
        6
        """
        return self.data.size

    @property
    def nbytes(self):
        """
        returns the number of bytes in a LArray.

        Returns
        -------
        integer
            returns the number of bytes in a LArray.

        Example
        -------
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type', ['type1', 'type2', 'type3'])
        >>> mat = ndrange([xsex, xtype], dtype=float)
        >>> mat.nbytes
        48
        """
        return self.data.nbytes

    @property
    def dtype(self):
        """
        returns the type of the data in the cells of LArray.

        Returns
        -------
        string
            returns the type of the data in the cells of LArray.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = zeros([xnat, xsex, xtype])
        >>> mat.dtype
        dtype('float64')
        """
        return self.data.dtype

    @property
    def item(self):
        return self.data.item

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        return self.data

    __array_priority__ = 100

    def set_labels(self, axis, labels, inplace=False):
        """
        replaces the labels of axis of a LArray

        Parameters
        ----------
        axis
            the axis for which we want to replace the labels.
        labels : list of axis labels
            the new labels.
        inplace : boolean
            whether or not to modify the original object or return a new
            LArray and leave the original intact.

        Returns
        -------
        LArray
            LArray with modified labels.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.set_labels(x.sex, ['Hommes', 'Femmes'])
        nat | sex\\type | type1 | type2 | type3
         BE |   Hommes |     0 |     1 |     2
         BE |   Femmes |     3 |     4 |     5
         FO |   Hommes |     6 |     7 |     8
         FO |   Femmes |     9 |    10 |    11
        """
        axis = self.axes[axis]
        if inplace:
            axis.labels = labels
            return self
        else:
            return LArray(self.data,
                          self.axes.replace(axis, Axis(axis.name, labels)))

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return LArray(self.data.astype(dtype, order, casting, subok, copy),
                      self.axes)
    astype.__doc__ = np.ndarray.astype.__doc__

    def shift(self, axis, n=1):
        """
        shifts the cells of a LArray n-times to the left along axis.

        Parameters
        ----------
        axis : int, str or Axis
            the axis for which we want to perform the shift.
        n : int
            the number of cells to shift.

        Returns
        -------
        LArray

        Example
        -------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([sex, xtype])
        >>> mat
        sex\\type | type1 | type2 | type3
               M |     0 |     1 |     2
               F |     3 |     4 |     5
        >>> mat.shift(x.type)
        sex\\type | type2 | type3
               M |     0 |     1
               F |     3 |     4
        >>> mat.shift(x.type, n=-1)
        sex\\type | type1 | type2
               M |     1 |     2
               F |     4 |     5
        """
        axis = self.axes[axis]
        if n > 0:
            return self[axis.i[:-n]].set_labels(axis, axis.labels[n:])
        elif n < 0:
            return self[axis.i[-n:]].set_labels(axis, axis.labels[:n])
        else:
            return self[:]

    def diff(self, axis=-1, d=1, n=1, label='upper'):
        """
        Calculate the n-th order discrete difference along a given axis.

        The first order difference is given by out[n] = a[n + 1] - a[n] along the
        given axis, higher order differences are calculated by using diff
        recursively.

        Parameters
        ----------
        axis : int, str or Axis, optional
            The axis along which the difference is taken. Defaults to the last
            axis.

        d : int, optional
            Periods to shift for forming difference. Defaults to 1.

        n : int, optional
            The number of times values are differenced. Defaults to 1.

        label : 'lower' or 'upper', optional
            The new labels in `axis` will have the labels of either
            the array being subtracted ('lower') or the array it is
            subtracted from ('upper'). Defaults to 'upper'.

        Returns
        -------
        LArray : The n-th order differences. The shape of the output is the same
        as `a` except for `axis` which is smaller by `n` * `d`.

        Example
        -------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> xtype = Axis('type', ['type1', 'type2', 'type3'])
        >>> a = ndrange([sex, xtype]).cumsum(x.type)
        >>> a
        sex\\type | type1 | type2 | type3
               M |     0 |     1 |     3
               F |     3 |     7 |    12
        >>> a.diff()
        sex\\type | type2 | type3
               M |     1 |     2
               F |     4 |     5
        >>> a.diff(n=2)
        sex\\type | type3
               M |     1
               F |     1
        >>> a.diff(x.sex)
        sex\\type | type1 | type2 | type3
               F |     3 |     6 |     9
        """
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

    # XXX: this is called pct_change in Pandas (but returns the same results,
    # not results * 100, which I find silly). Maybe change_rate would be
    # better (because growth is not always positive)?
    def growth_rate(self, axis=-1, d=1, label='upper'):
        """
        Calculate the growth along a given axis.

        roughly equivalent to a.diff(axis, d, label) / a[axis.i[:-d]]

        Parameters
        ----------
        axis : int, str or Axis, optional
            The axis along which the difference is taken. Defaults to the last
            axis.

        d : int, optional
            Periods to shift for forming difference. Defaults to 1.

        label : 'lower' or 'upper', optional
            The new labels in `axis` will have the labels of either
            the array being subtracted ('lower') or the array it is
            subtracted from ('upper'). Defaults to 'upper'.

        Returns
        -------
        LArray

        Example
        -------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> year = Axis('year', [2015, 2016, 2017])
        >>> a = ndrange([sex, year]).cumsum(x.year)
        >>> a
        sex\\year | 2015 | 2016 | 2017
               M |    0 |    1 |    3
               F |    3 |    7 |   12
        >>> a.growth_rate()
        sex\\year |          2016 |           2017
               M |           inf |            2.0
               F | 1.33333333333 | 0.714285714286
        >>> a.growth_rate(d=2)
        sex\\year | 2017
               M |  inf
               F |  3.0
        """
        diff = self.diff(axis=axis, d=d, label=label)
        axis_obj = self.axes[axis]
        return diff / self[axis_obj.i[:-d]].drop_labels(axis)


def parse(s):
    """
    used to parse the "folded" axis ticks (usually periods)
    """
    # parameters can be strings or numbers
    if isinstance(s, basestring):
        s = s.strip()
        low = s.lower()
        if low == 'true':
            return True
        elif low == 'false':
            return False
        elif s.isdigit():
            return int(s)
        else:
            try:
                return float(s)
            except ValueError:
                return s
    else:
        return s


def df_labels(df, sort=True):
    """
    returns unique labels for each dimension
    """
    idx = df.index
    if isinstance(idx, pd.core.index.MultiIndex):
        if sort:
            return list(idx.levels)
        else:
            return [list(unique(idx.get_level_values(l))) for l in idx.names]
    else:
        assert isinstance(idx, pd.core.index.Index)
        # use .values if needed
        return [idx]


def cartesian_product_df(df, sort_rows=False, sort_columns=False, **kwargs):
    labels = df_labels(df, sort=sort_rows)
    if sort_rows:
        new_index = pd.MultiIndex.from_product(labels)
    else:
        new_index = pd.MultiIndex.from_tuples(list(product(*labels)))
    columns = sorted(df.columns) if sort_columns else list(df.columns)
    # the prodlen test is meant to avoid the more expensive array_equal test
    prodlen = np.prod([len(axis_labels) for axis_labels in labels])
    if prodlen == len(df) and columns == list(df.columns) and \
            np.array_equal(df.index.values, new_index.values):
        return df, labels
    return df.reindex(new_index, columns, **kwargs), labels


def df_aslarray(df, sort_rows=False, sort_columns=False, **kwargs):
    axes_names = [decode(name, 'utf8') for name in df.index.names]
    if isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]:
        last_axes = [name.strip() for name in axes_names[-1].split('\\')]
        axes_names = axes_names[:-1] + last_axes
    elif len(df) == 1 and axes_names == [None]:
        axes_names = [df.columns.name]
    elif len(df) > 1:
        axes_names += [df.columns.name]

    if len(axes_names) > 1:
        df, axes_labels = cartesian_product_df(df, sort_rows=sort_rows,
                                               sort_columns=sort_columns,
                                               **kwargs)
    else:
        axes_labels = []

    # we could inline df_aslarray into the functions that use it, so that the
    # original (non-cartesian) df is freed from memory at this point, but it
    # would be much uglier and would not lower the peak memory usage which
    # happens during cartesian_product_df.reindex

    # pandas treats the "time" labels as column names (strings) so we need
    # to convert them to values
    axes_labels.append([parse(cell) for cell in df.columns.values])

    axes = [Axis(name, labels) for name, labels in zip(axes_names, axes_labels)]
    data = df.values.reshape([len(axis) for axis in axes])
    return LArray(data, axes)


def read_csv(filepath, nb_index=0, index_col=[], sep=',', headersep=None,
             na=np.nan, sort_rows=False, sort_columns=False,
             dialect='larray', **kwargs):
    """
    reads csv file and returns a Larray with the contents

    Note
    ----
    csv file format:
    arr,ages,sex,nat\time,1991,1992,1993
    A1,BI,H,BE,1,0,0
    A1,BI,H,FO,2,0,0
    A1,BI,F,BE,0,0,1
    A1,BI,F,FO,0,0,0
    A1,A0,H,BE,0,0,0

    Parameters
    ----------
    filepath : str
        path where the csv file has to be written.
    nb_index : int, optional
        number of leading index columns (ex. 4).
    index_col : list, optional
        list of columns for the index (ex. [0, 1, 2, 3]).
    sep : str, optional
        separator.
    headersep : str or None, optional
        ???.
    na : ???
        ???.
    sort_rows : bool, optional
        Whether or not to sort the row dimensions alphabetically (sorting is
        more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the column dimension alphabetically (sorting is
        more efficient than not sorting). Defaults to False.
    dialect : 'classic' | 'larray' | 'liam2', optional
        Name of dialect. Defaults to 'larray'.
    **kwargs

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> mat = ndrange([xnat, xsex])
    >>> mat.to_csv('test.csv')
    >>> read_csv('test.csv')
    nat\\sex | H | F
         BE | 0 | 1
         FO | 2 | 3
    >>> read_csv('test.csv', sort_columns=True)
    nat\\sex | F | H
         BE | 1 | 0
         FO | 3 | 2
    >>> mat.to_csv('no_axis_name.csv', dialect='classic')
    >>> read_csv('no_axis_name.csv', nb_index=1)
    nat\\axis1 | H | F
           BE | 0 | 1
           FO | 2 | 3
    """
    # read the first line to determine how many axes (time excluded) we have
    with csv_open(filepath) as f:
        reader = csv.reader(f, delimiter=sep)
        line_stream = skip_comment_cells(strip_rows(reader))
        header = next(line_stream)
        if headersep is not None and headersep != sep:
            combined_axes_names = header[0]
            header = combined_axes_names.split(headersep)
        try:
            # take the first cell which contains '\'
            pos_last = next(i for i, v in enumerate(header) if '\\' in v)
        except StopIteration:
            # if there isn't any, assume 1d array, unless "liam2 dialect"
            pos_last = 0 if dialect != 'liam2' else len(header) - 1
        axes_names = header[:pos_last + 1]

    if len(index_col) == 0 and nb_index == 0:
        nb_index = len(axes_names)
        if dialect == 'liam2':
            nb_index -= 1

    if len(index_col) > 0:
        nb_index = len(index_col)
    else:
        index_col = list(range(nb_index))

    if headersep is not None:
        # we will set the index after having split the tick values
        index_col = None

    if dialect == 'liam2':
        if len(axes_names) < 2:
            index_col = None
        # FIXME: add number of lines skipped by comments (or not, pandas
        # might skip them by default)
        kwargs['skiprows'] = 1
        kwargs['comment'] = '#'
    df = pd.read_csv(filepath, index_col=index_col, sep=sep, **kwargs)
    if dialect == 'liam2':
        if len(axes_names) > 1:
            df.index.names = axes_names[:-1]
        df.columns.name = axes_names[-1]
    if headersep is not None:
        labels_column = df[combined_axes_names]
        label_columns = unzip(label.split(headersep) for label in labels_column)
        for name, column in zip(axes_names, label_columns):
            df[name] = column
        del df[combined_axes_names]
        df.set_index(axes_names, inplace=True)

    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=na)


def read_tsv(filepath, **kwargs):
    return read_csv(filepath, sep='\t', **kwargs)


def read_eurostat(filepath, **kwargs):
    """Read EUROSTAT TSV (tab-separated) file into an LArray

    EUROSTAT TSV files are special because they use tabs as data
    separators but comas to separate headers.

    Parameters
    ----------
    filepath : str
        Path to the file
    kwargs
        Arbitrary keyword arguments are passed through to read_csv

    Returns
    -------
    LArray
    """
    return read_csv(filepath, sep='\t', headersep=',', **kwargs)


def read_hdf(filepath, key, na=np.nan, sort_rows=False, sort_columns=False,
             **kwargs):
    """Reads a LArray named key from a h5 file in filepath (path+name)

    Parameters
    ----------
    filepath : str
        the filepath and name where the h5 file is stored.
    key : str
        the name of the LArray

    Returns
    -------
    LArray
    """
    df = pd.read_hdf(filepath, key, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=na)


def read_excel(filepath, sheetname=0, nb_index=0, index_col=[],
               na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """
    reads excel file from sheet name and returns an LArray with the contents
        nb_index: number of leading index columns (e.g. 4)
    or
        index_col: list of columns for the index (e.g. [0, 1, 3])
    """
    if len(index_col) == 0:
        index_col = list(range(nb_index))
    df = pd.read_excel(filepath, sheetname, index_col=index_col, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=na)


def read_sas(filepath, nb_index=0, index_col=[],
             na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """
    reads sas file and returns an LArray with the contents
        nb_index: number of leading index columns (e.g. 4)
    or
        index_col: list of columns for the index (e.g. [0, 1, 3])
    """
    if len(index_col) == 0:
        index_col = list(range(nb_index))
    df = pd.read_sas(filepath, index=index_col, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=na)


def zeros(axes, dtype=float, order='C'):
    """Returns a LArray with the specified axes and filled with zeros.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        a collection of axes or a shape.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> zeros([xnat, xsex])
    nat\sex |   H |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0
    """
    axes = AxisCollection(axes)
    return LArray(np.zeros(axes.shape, dtype, order), axes)


def zeros_like(array, dtype=None, order='K'):
    """Returns a LArray with the same axes as array and filled with zeros.

    Parameters
    ----------
    array : LArray
         is an array object.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' (default) means match the layout of `a` as closely
        as possible.

    Returns
    -------
    LArray

    Example
    -------
    >>> a = ndrange((2, 3))
    >>> zeros_like(a)
    axis0\\axis1 | 0 | 1 | 2
              0 | 0 | 0 | 0
              1 | 0 | 0 | 0
    """
    axes = array.axes
    return LArray(np.zeros_like(array, dtype, order), axes)


def ones(axes, dtype=float, order='C'):
    """Returns a LArray with the specified axes and filled with ones.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        a collection of axes or a shape.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> ones([xnat, xsex])
    nat\\sex |   H |   F
         BE | 1.0 | 1.0
         FO | 1.0 | 1.0
    """
    axes = AxisCollection(axes)
    return LArray(np.ones(axes.shape, dtype, order), axes)


def ones_like(array, dtype=None, order='K'):
    """Returns a LArray with the same axes as array and filled with ones.

    Parameters
    ----------
    array : LArray
        is an array object.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' (default) means match the layout of `a` as closely
        as possible.

    Returns
    -------
    LArray

    Example
    -------
    >>> a = ndrange((2, 3))
    >>> ones_like(a)
    axis0\\axis1 | 0 | 1 | 2
              0 | 1 | 1 | 1
              1 | 1 | 1 | 1
    """
    axes = array.axes
    return LArray(np.ones_like(array, dtype, order), axes)


def empty(axes, dtype=float, order='C'):
    """Returns a LArray with the specified axes and uninitialized (arbitrary)
    data.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        a collection of axes or a shape.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> empty([xnat, xsex])  # doctest: +SKIP
    nat\\sex |                  H |                  F
         BE | 2.47311483356e-315 | 2.47498446195e-315
         FO |                0.0 | 6.07684618082e-31
    """
    axes = AxisCollection(axes)
    return LArray(np.empty(axes.shape, dtype, order), axes)


def empty_like(array, dtype=None, order='K'):
    """Returns a LArray with the same axes as array and uninitialized
    (arbitrary) data.

    Parameters
    ----------
    array : LArray
        is an array object.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' (default) means match the layout of `a` as closely
        as possible.

    Returns
    -------
    LArray

    Example
    -------
    >>> a = ndrange((3, 2))
    >>> empty_like(a)   # doctest: +SKIP
    -\- |                  0 |                  1
      0 | 2.12199579097e-314 | 6.36598737388e-314
      1 | 1.06099789568e-313 | 1.48539705397e-313
      2 | 1.90979621226e-313 | 2.33419537056e-313
    """
    # cannot use empty() because order == 'K' is not understood
    return LArray(np.empty_like(array.data, dtype, order), array.axes)


def full(axes, fill_value, dtype=None, order='C'):
    """Returns a LArray with the specified axes and filled with fill_value.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        a collection of axes or a shape.
    fill_value : scalar or LArray
        value to fill the array
    dtype : data-type, optional
        The desired data-type for the array. Default is the data type of
        fill_value.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Examples
    ========

    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> full([nat, sex], 42.0)
    nat\\sex |    M |    F
         BE | 42.0 | 42.0
         FO | 42.0 | 42.0
    >>> initial_value = ndrange([sex])
    >>> initial_value
    sex | M | F
        | 0 | 1
    >>> full([nat, sex], initial_value)
    nat\\sex | M | F
         BE | 0 | 1
         FO | 0 | 1
    """
    dtype = np.asarray(fill_value).dtype
    res = empty(axes, dtype, order)
    res[:] = fill_value
    return res


def full_like(array, fill_value, dtype=None, order='K'):
    """Returns a LArray with the same axes as array and filled with fill_value.

    Parameters
    ----------
    array : LArray
        is an array object.
    fill_value : scalar or LArray
        value to fill the array
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' (default) means match the layout of `a` as closely
        as possible.

    Returns
    -------
    LArray

    Example
    -------
    >>> a = ndrange((2, 3))
    >>> full_like(a, 5)
    axis0\\axis1 | 0 | 1 | 2
              0 | 5 | 5 | 5
              1 | 5 | 5 | 5
    """
    # cannot use full() because order == 'K' is not understood
    # cannot use np.full_like() because it would not handle LArray fill_value
    dtype = np.asarray(fill_value).dtype
    res = empty_like(array, dtype, order)
    res[:] = fill_value
    return res


# XXX: would it be possible to generalize to multiple axes and deprecate
#      ndrange (or rename this one to ndrange)? ndrange is only ever used to
#      create test data (except for 1d)
# see https://github.com/pydata/pandas/issues/4567
def create_sequential(axis, initial=0, inc=None, mult=1, func=None, axes=None):
    """
    Creates an array by sequentially applying modifications to the array along
    axis.

    The value for each label in axis will be given by sequentially transforming
    the value for the previous label. This transformation on the previous label
    value consists of applying the function "func" on that value if provided, or
    to multiply it by mult and increment it by inc otherwise.

    Parameters
    ----------
    axis : axis reference (Axis, string, int)
        axis along which to apply mod.
    initial : scalar or LArray, optional
        value for the first label of axis. Defaults to 0.
    inc : scalar, LArray, optional
        value to increment the previous value by. Defaults to 0 if mult is
        provided, 1 otherwise.
    mult : scalar, LArray, optional
        value to multiply the previous value by. Defaults to 1.
    func : function/callable, optional
        function to apply to the previous value. Defaults to None.
    axes : int, tuple of int or tuple/list/AxisCollection of Axis, optional
        axes of the result. Defaults to the union of axes present in other
        arguments.

    Example
    -------
    >>> year = Axis('year', range(2016, 2020))
    >>> sex = Axis('sex', ['M', 'F'])
    >>> create_sequential(year)
    year | 2016 | 2017 | 2018 | 2019
         |    0 |    1 |    2 |    3
    >>> create_sequential(year, 1.0, 0.1)
    year | 2016 | 2017 | 2018 | 2019
         |  1.0 |  1.1 |  1.2 |  1.3
    >>> create_sequential(year, 1.0, mult=1.1)
    year | 2016 | 2017 | 2018 |  2019
         |  1.0 |  1.1 | 1.21 | 1.331
    >>> inc = LArray([1, 2], [sex])
    >>> inc
    sex | M | F
        | 1 | 2
    >>> create_sequential(year, 1.0, inc)
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  3.0 |  4.0
           F |  1.0 |  3.0 |  5.0 |  7.0
    >>> mult = LArray([2, 3], [sex])
    >>> mult
    sex | M | F
        | 2 | 3
    >>> create_sequential(year, 1.0, mult=mult)
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  4.0 |  8.0
           F |  1.0 |  3.0 |  9.0 | 27.0
    >>> initial = LArray([3, 4], [sex])
    >>> initial
    sex | M | F
        | 3 | 4
    >>> create_sequential(year, initial, inc, mult)
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |    3 |    7 |   15 |   31
           F |    4 |   14 |   44 |  134
    >>> def modify(prev_value):
    ...     return prev_value / 2
    >>> create_sequential(year, 8, func=modify)
    year | 2016 | 2017 | 2018 | 2019
         |    8 |    4 |    2 |    1
    >>> create_sequential(3)
    axis0* | 0 | 1 | 2
           | 0 | 1 | 2
    >>> create_sequential(x.year, axes=(sex, year))
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |    0 |    1 |    2 |    3
           F |    0 |    1 |    2 |    3
    >>> a = ndrange([sex, year], start=1, dtype=float)
    >>> a
    sex\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  3.0 |  4.0
           F |  5.0 |  6.0 |  7.0 |  8.0
    >>> g = a.growth_rate() + 1
    >>> g
    sex\year | 2017 |          2018 |          2019
           M |  2.0 |           1.5 | 1.33333333333
           F |  1.2 | 1.16666666667 | 1.14285714286
    >>> create_sequential(a.axes.year, a[2016], mult=g)
    sex\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  3.0 |  4.0
           F |  5.0 |  6.0 |  7.0 |  8.0
    """
    if inc is None:
        inc = 1 if mult is 1 else 0
    if isinstance(axis, int):
        axis = Axis(None, axis)
    if axes is None:
        def strip_axes(col):
            return get_axes(col) - axis
        # we need to remove axis if present, because it might be incompatible
        axes = strip_axes(initial) | strip_axes(inc) | strip_axes(mult) | axis
    else:
        axes = AxisCollection(axes)
    axis = axes[axis]
    # FIXME: we should compute combined dtype for inital, inc, mult
    res = empty(axes, dtype=np.asarray(initial).dtype)
    res[axis.i[0]] = initial
    if func is not None:
        for i in range(1, len(axis)):
            res[axis.i[i]] = func(res[axis.i[i - 1]])
    else:
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
    return res


def ndrange(axes, start=0, dtype=int):
    """Returns a LArray with the specified axes and filled with increasing int.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        a collection of axes or a shape.
    start : number, optional
    dtype : dtype, optional
        The type of the output array.  Defaults to int.

    Returns
    -------
    LArray

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> ndrange([xnat, xsex])
    nat\\sex | H | F
         BE | 0 | 1
         FO | 2 | 3
    >>> ndrange([2, 3], dtype=float)
    axis0\\axis1 |   0 |   1 |   2
              0 | 0.0 | 1.0 | 2.0
              1 | 3.0 | 4.0 | 5.0
    >>> ndrange(3, start=2)
    axis0 | 0 | 1 | 2
          | 2 | 3 | 4

    potential alternate syntaxes:
    ndrange((2, 3), names=('a', 'b'))
    ndrange(2, 3, names=('a', 'b'))
    ndrange([('a', 2), ('b', 3)])
    ndrange(('a', 2), ('b', 3))
    ndrange((2, 3)).rename([0, 1], ['a', 'b'])
    ndrange((2, 3)).rename({0: 'a', 1: 'b'})
    # current syntaxes
    ndrange((2, 3)).rename(0, 'a').rename(1, 'b')
    ndrange([Axis('a', 2), Axis('b', 3)])
    """
    # XXX: try to come up with a syntax where start is before "end". For ndim
    #  > 1, I cannot think of anything nice.
    axes = AxisCollection(axes)
    data = np.arange(start, start + axes.size, dtype=dtype)
    return LArray(data.reshape(axes.shape), axes)


# TODO: I could generalize this to multiple axes
# >>> nat = Axis('nat', ['BE', 'FO'])
# >>> sex = Axis('sex', ['H', 'F'])
# >>> identity((nat, sex))
# nat\\sex |    H |    F
#      BE | BE,H | BE,F
#      FO | FO,H | FO,F


def identity(axis):
    """Returns a LArray with the value equal to Axis

    Parameters
    ----------
    axis : Axis object

    Returns
    -------
    LArray

    Example
    -------
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> identity(xsex)
    sex | H | F
        | H | F
    """
    axes = AxisCollection([axis])
    return LArray(axis.labels, axes)


def stack(arrays, axis):
    """
    stack([numbirths * HMASC,
           numbirths * (1 - HMASC)], Axis('sex', 'H,F'))
    potential alternate syntaxes
    stack(['H', numbirths * HMASC,
           'F', numbirths * (1 - HMASC)], 'sex')
    stack(('H', numbirths * HMASC),
          ('F', numbirths * (1 - HMASC)), name='sex')
    """
    # append an extra length 1 dimension
    data_arrays = [a.data.reshape(a.shape + (1,)) for a in arrays]
    axes = arrays[0].axes
    for a in arrays[1:]:
        a.axes.check_compatible(axes)
    return LArray(np.concatenate(data_arrays, axis=-1), axes + axis)


class AxisReference(Axis):
    def __init__(self, name):
        self.name = name
        self._labels = None
        self._iswildcard = False

    def translate(self, key):
        raise NotImplementedError("an AxisReference (x.) cannot translate "
                                  "labels")

    def __repr__(self):
        return 'AxisReference(%r)' % self.name


class AxisReferenceFactory(object):
    def __getattr__(self, key):
        return AxisReference(key)
x = AxisReferenceFactory()


def get_axes(value):
    return value.axes if isinstance(value, LArray) else AxisCollection([])


def make_numpy_broadcastable(values):
    """
    return values where LArrays are (numpy) broadcastable between them.
    For that to be possible, all common axes must be either 1 or the same
    length. Extra axes (in any array) can have any length.

    * the resulting arrays will have the combination of all axes found in the
      input arrays, the earlier arrays defining the order of axes. Axes with
      labels take priority over wildcard axes.
    * axes with a single '*' label will be added for axes not present in input
    """
    all_axes = AxisCollection.union(*[get_axes(v) for v in values])

    # 1) reorder axes
    values = [v.transpose(all_axes & v.axes) if isinstance(v, LArray) else v
              for v in values]

    # 2) add length one axes
    return [v.reshape([v.axes.get(axis, Axis(axis.name, 1))
                       for axis in all_axes]) if isinstance(v, LArray) else v
            for v in values], all_axes


# excel IO tools in Python
# - openpyxl, the slowest but most-complete package but still lags behind
#   PHPExcel from which it was ported. despite the drawbacks the API is very
#   complete.
#   biggest drawbacks:
#   * you can get either the "cached" value of cells OR their formulas but NOT
#     BOTH and this is a file-wide setting (data_only=True).
#     if you have an excel file and want to add a sheet to it, you either loose
#     all cached values (which is problematic in many cases since you do not
#     necessarily have linked files) or loose all formulas.
#   * it loose "charts" on read. => cannot append/update a sheet to a file with
#     charts, which is precisely what many users asked. => users need to
#     create their charts using code.
# - xlsxwriter: faster and slightly more feature-complete than openpyxl
#   regarding writing but does not read anything => cannot update an existing
#   file. API seems extremely complete.
# - pyexcelerate: yet faster but also write only. Didn't check whether API
#   is more featured than xlsxwriter or not.
# - xlwings: wraps win32com & equivalent on mac, so can potentially do
#   everything (I guess) but this is SLOW and needs a running excel instance,
#   etc.
