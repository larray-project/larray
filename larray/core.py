# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

__version__ = "0.7.1"

__all__ = [
    'LArray', 'Axis', 'AxisCollection', 'LGroup',
    'union', 'stack',
    'read_csv', 'read_eurostat', 'read_excel', 'read_hdf', 'read_tsv',
    'x',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like',
    'ndrange', 'identity', 'larray_equal',
    'all', 'any', 'sum', 'prod', 'cumsum', 'cumprod', 'min', 'max', 'mean',
    'ptp', 'var', 'std', 'median', 'percentile',
    '__version__'
]

"""
Matrix class
"""
# TODO
# * add check there is no duplicate label in axes!

# * for NDGroups, we have two options: cross product or intersection.
#   Technically, this is easy, we just need to store a boolean and in getitem
#   act accordingly (use ix_ or not), but what is the best API for users?
#   a different class or a flag? In fact, the same question applies to
#   positional vs label (in total, we got 4 different possibilities).

# ? how do you combine a product group with an intersection group?
#   a[pgroup, igroup]
#   -> no problem if they are on different dimensions: the igroup
#      dimension(s) are collapsed into one, pgroup dimension(s) stay.
#      The index need to be constructed carefully, but it can be done. See
#      np_indexing.py
#   -> if they are on the same dimensions, we have two options:
#      * apply one then the other (left to right)
#      * fail <-- I think this is safer at least to implement. One after the
#                 other can still be achieved by a[pgroup][igroup]

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

# ? keepdims=True instead of/in addition to group tuples

# ? implement newaxis

# * int labels vs indice-based indexing
#   one way to disambiguate is to use marker objects:
#   time[start + 5:]
#   time[end - 10:]
#   time[end(-10):]
#   time[stop - 10:]
#   another way is to use a special attribute:
#   time.ix[5:]
#   time.ix[-10:]

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
# * implement iloc
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

from itertools import product, chain, groupby, repeat, islice
import sys
import csv
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

import numpy as np
import pandas as pd

from larray.utils import (table2str, unique, csv_open, unzip, long,
                          decode, basestring, izip, rproduct, ReprString,
                          duplicates, array_lookup)


# TODO: return a generator, not a list
def srange(*args):
    return list(map(str, range(*args)))

def range_to_slice(seq):
    """
    seq is a sequence-like (list, tuple or ndarray) of integers
    returns a slice if possible (including for sequences of 1 element)
    otherwise returns the input sequence itself
    """
    if len(seq) < 1:
        return seq
    first = seq[0]
    if len(seq) == 1:
        return slice(first, first + 1)
    second = seq[1]
    step = second - first
    prev_value = second
    for value in seq[2:]:
        if value != prev_value + step:
            return seq
        prev_value = value
    return slice(first, prev_value + step, step)


def slice_to_str(key):
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
    start = key.start if key.start is not None else ''
    stop = key.stop if key.stop is not None else ''
    step = (":" + str(key.step)) if key.step is not None else ''
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
        raise ValueError("no stop bound provided in range: %s" % s)
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
    return (first.axes == other.axes and
            np.array_equal(np.asarray(first), np.asarray(other)))


def isnoneslice(v):
    return isinstance(v, slice) and v == slice(None)

class PGroupMaker(object):
    def __init__(self, axis):
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
        self._mapping = {}
        self._length = None
        self._iswildcard = False
        self.labels = labels

    @property
    def i(self):
        return PGroupMaker(self.name)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            raise TypeError("labels should be ndarray or int")
        if isinstance(labels, int):
            length = labels
            labels = np.arange(length)
            # TODO: this would be more efficient but does not work in all cases
            # mapping = labels
            mapping = {label: i for i, label in enumerate(labels)}
            iswildcard = True
        else:
            # TODO: move this to to_ticks????
            # we convert to an ndarray to save memory (for scalar ticks, for
            # LGroup ticks, it does not make a difference since a list of VG
            # and an ndarray of VG are both arrays of pointers)
            ticks = to_ticks(labels)
            if any(isinstance(tick, LGroup) for tick in ticks):
                # avoid getting a 2d array if all LGroup have the same length
                labels = np.empty(len(ticks), dtype=object)
                labels[:] = ticks
            else:
                labels = np.asarray(ticks)
            length = len(labels)
            mapping = {label: i for i, label in enumerate(labels)}
            # we have no choice but to do that!
            # otherwise we could not make geo['Brussels'] work efficiently
            # (we could have to traverse the whole mapping checking for each
            # name, which is not an option)
            mapping.update({label.name: i for i, label in enumerate(labels)
                            if isinstance(label, Group)})
            iswildcard = False
        self._length = length
        self._labels = labels
        self._mapping = mapping
        self._iswildcard = iswildcard

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
        return LGroup(key, name, self.name)

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
        if self._iswildcard:
            return len(self) == 1 or len(self) == len(other)
        elif other._iswildcard:
            return len(other) == 1 or len(self) == len(other)
        else:
            return np.array_equal(self.labels, other.labels)

    def __eq__(self, other):
        return (isinstance(other, Axis) and self.name == other.name and
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

    def translate(self, key):
        """
        translates a label key to its numerical index counterpart
        fancy index with boolean vectors are passed through unmodified
        """
        mapping = self._mapping

        # first, try the key as-is, so that we can target elements in aggregated
        # arrays (those are either strings containing comas or LGroups)
        try:
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
        elif isinstance(key, np.ndarray) and key.dtype.kind is 'b':
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
            # TODO: the result should be cached (or at least the sorted_keys
            # & sorted_values)
            # TODO: benchmark this against the tuple/list version above when
            # mapping is large
            # array_lookup is O(len(key) * log(len(mapping)))
            # vs
            # tuple/list version is O(len(key)) (dict.getitem is O(1))
            return array_lookup(key, mapping)
        elif isinstance(key, LArray):
            return LArray(array_lookup(key.data, mapping), key.axes)
        else:
            # the first mapping[key] above will cover most cases. This code
            # path is only used if the key was given in "non normalized form"
            assert np.isscalar(key), "%s (%s) is not scalar" % (key, type(key))
            # key is scalar (integer, float, string, ...)
            return mapping[key]

    @property
    def display_name(self):
        name = self.name if self.name is not None else '-'
        return (name + '*') if self._iswildcard else name

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return 'Axis(%r, %r)' % (self.name, list(self.labels))

    def short_labels(self):
        def shorten(l):
            return l if len(l) < 7 else l[:3] + ['...'] + list(l[-3:])
        return ' '.join(shorten([repr(l) for l in self.labels]))

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
        # XXX: I wonder if we should make a copy of the labels. There should
        # at least be an option.
        return Axis(self.name, self.labels)

    def _rename(self, name):
        return Axis(name, self.labels)


# We need a separate class for LGroup and cannot simply create a
# new Axis with a subset of values/ticks/labels: the subset of
# ticks/labels of the LGroup need to correspond to its *Axis*
# indices
class Group(object):
    def __init__(self, key, name, axis):
        raise NotImplementedError()


class LGroup(Group):
    def __init__(self, key, name=None, axis=None):
        """
        key should be either a sequence of labels, a slice with label bounds
        or a string
        axis, is only used to check the key and later to cache the translated
        key
        """
        self.key = key
        # we do NOT assign a name in all cases because that makes it
        # impossible to know whether a name was explicitly given or computed
        self.name = name

        # we store the Axis name, instead of the axis object itself so that
        # LGroups are more compatible between themselves.
        if isinstance(axis, Axis):
            axis = axis.name
        if axis is not None:
            assert isinstance(axis, basestring), \
                "axis is not an instance of str (%s)" % axis
            # check the key is valid
            # TODO: for performance reasons, we should cache the result. This will
            # need to be invalidated correctly
            # axis.translate(key)
        self.axis = axis

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
        other_key = other.key if isinstance(other, LGroup) else other
        return to_tick(to_key(self.key)) == to_tick(to_key(other_key))

    def __str__(self):
        return to_string(self.key) if self.name is None else self.name

    def __repr__(self):
        name = ", %r" % self.name if self.name is not None else ''
        return "LGroup(%r%s)" % (self.key, name)

    def __len__(self):
        return len(self.key)

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
    def __init__(self, key, name=None, axis=None):
        if isinstance(key, tuple):
            key = list(key)
        self.key = key
        self.name = name
        assert axis is None or isinstance(axis, basestring), \
            "invalid axis '%s' (%s)" % (axis, type(axis).__name__)
        self.axis = axis

    def __repr__(self):
        name = ", %r" % self.name if self.name is not None else ''
        return "PGroup(%r%s)" % (self.key, name)

    def __len__(self):
        return len(self.key)


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
                    else axis
                for axis in axes]
        assert all(isinstance(a, Axis) for a in axes)
        self._list = axes
        self._map = {axis.name: axis for axis in axes if axis.name is not None}

    def __getattr__(self, key):
        try:
            return self._map[key]
        except KeyError:
            return self.__getattribute__(key)

    def __getitem__(self, key):
        if isinstance(key, Axis):
            # match by object if name is None
            if key.name is None and key in self._list:
                return key
            # we should NOT check that the object is the same, so that we can
            # use AxisReference objects to target real axes
            key = key.name

        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, (tuple, list, AxisCollection)):
            return AxisCollection([self[k] for k in key])
        elif isinstance(key, slice):
            return AxisCollection(self._list[key])
        else:
            assert isinstance(key, basestring), type(key)
            if key in self._map:
                return self._map[key]
            else:
                raise KeyError("axis '%s' not found in %s" % (key, self))

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise NotImplementedError("slice set")
        if isinstance(key, int):
            axis = self._list[key]
            self._list[key] = value
            if axis.name is not None:
                del self._map[axis.name]
            if value.name is not None:
                self._map[value.name] = value
        elif isinstance(key, Axis):
            # XXX: check that it is the same object????
            self.__setitem__(key.name, value)
        else:
            assert isinstance(key, basestring), type(key)
            if key in self._map:
                axis = self._map[key]
            else:
                raise KeyError("axis '%s' not found in %s" % (key, self))
            idx = self._list.index(axis)
            self._list[idx] = value
            self._map[key] = value

    def __delitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError("slice delete")
        if isinstance(key, int):
            axis = self._list.pop(key)
            if axis.name is not None:
                del self._map[axis.name]
        elif isinstance(key, Axis):
            self._list.remove(key)
            if key.name is not None:
                del self._map[key.name]
        else:
            assert isinstance(key, basestring)
            axis = self._map.pop(key)
            self._list.remove(axis)

    def __add__(self, other):
        result = self[:]
        if isinstance(other, Axis):
            other = [other]
        # other should be a sequence
        assert len(other) >= 0
        result.extend(other)
        return result

    __or__ = __add__

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
        if isinstance(key, Axis):
            if key.name is None:
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
        return isinstance(value, Axis) or isinstance(value, basestring) and value in self
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
        return axis

    def append(self, axis):
        """
        append axis at the end of the collection
        """
        # when __setitem__(slice) will be implemented, we could simplify this
        self._list.append(axis)
        self._map[axis.name] = axis

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

    def extend(self, axes, validate=True):
        """
        extend the collection by appending the axes from axes
        """
        # check that common axes are the same
        if validate:
            self.check_compatible(axes)
        to_add = [axis for axis in axes if axis.name not in self._map]

        # when __setitem__(slice) will be implemented, we could simplify this
        self._list.extend(to_add)
        for axis in to_add:
            self._map[axis.name] = axis

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
        # first look by object
        if isinstance(axis, Axis) and axis.name is None:
            return self._list.index(axis)
        elif isinstance(axis, int):
            return axis
        name = axis.name if isinstance(axis, Axis) else axis
        return self.names.index(name)

    # XXX: we might want to return a new AxisCollection (same question for
    # other inplace operations: append, extend, pop, __delitem__, __setitem__)
    def insert(self, index, axis):
        """
        insert axis before index
        """
        # when __setitem__(slice) will be implemented, we could simplify this
        self._list.insert(index, axis)
        self._map[axis.name] = axis

    def copy(self):
        return self[:]

    def replace(self, old, new):
        res = self[:]
        if not isinstance(old, (tuple, list, AxisCollection)):
            old = [old]
        if not isinstance(new, (tuple, list, AxisCollection)):
            new = [new]
        if len(old) != len(new):
            raise ValueError('must have as many old axes as new axes')
        for o, n in zip(old, new):
            res[self.index(o)] = n
        return res

    def without(self, axes):
        """
        returns a new collection without some axes
        you can use a comma separated list
        axes must exist
        """
        res = self[:]
        if isinstance(axes, basestring):
            axes = axes.split(',')
        elif isinstance(axes, (int, Axis)):
            axes = [axes]
        # transform positional axis to axis objects
        axes = [self[axis] for axis in axes]
        for axis in axes:
            del res[axis]
        return res

    def __sub__(self, axes):
        """
        returns a new collection without some axes
        you can use a comma separated list
        set operations so axes can contain axes not present in self
        """
        if isinstance(axes, basestring):
            axes = axes.split(',')
        elif isinstance(axes, Axis):
            axes = [axes]

        # transform positional axis to axis objects
        axes = [self[axis] if isinstance(axis, int) else axis for axis in axes]
        to_remove = set(axis.name if isinstance(axis, Axis) else axis
                        for axis in axes)
        return AxisCollection([axis for axis in self
                               if axis.name not in to_remove])

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
        ['a', 'b*', '-']
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
                                   axis.short_labels())
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
    -\\- | 0 | 1 | 2
      0 | 0 | 1 | 2
      1 | 3 | 4 | 5
    >>> sum(a)
    15
    >>> sum(a, axis=0)
    - | 0 | 1 | 2
      | 3 | 5 | 7
    >>> sum(a, axis=1)
    - | 0 |  1
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
    -\\- |  0 | 1 | 2
      0 | 10 | 7 | 4
      1 |  3 | 2 | 1
    >>> median(a)
    3.5
    >>> median(a, axis=0)
    - |   0 |   1 |   2
      | 6.5 | 4.5 | 2.5
    >>> median(a, axis=1)
    - |   0 |   1
      | 7.0 | 2.0
    """
    return array.median(*args, **kwargs)


def percentile(array, *args, **kwargs):
    """
    Examples
    --------

    >>> a = LArray([[10, 7, 4], [3, 2, 1]])
    >>> a
    -\\- |  0 | 1 | 2
      0 | 10 | 7 | 4
      1 |  3 | 2 | 1
    >>> # this is a bug in numpy: np.nanpercentile(all axes) returns an ndarray,
    >>> # instead of a scalar.
    >>> percentile(a, 50)
    array(3.5)
    >>> percentile(a, 50, axis=0)
    - |   0 |   1 |   2
      | 6.5 | 4.5 | 2.5
    >>> percentile(a, 50, axis=1)
    - |   0 |   1
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
    array_axes.extend(other_axes)
    other_axes.extend(array_axes, validate=False)
    result_axes = AxisCollection([
        axis1 if len(axis2) <= len(axis1) else axis2
        for axis1, axis2 in zip(array_axes, other_axes[array_axes])])
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

    def __next__(self):
        array = self.array
        if self.position == len(self.array):
            raise StopIteration
        result = array[array.axes[0].i[self.position]]
        self.position += 1
        return result
    # Python 2
    next = __next__


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
        self.data = data
        self.axes = axes

    def to_frame(self, fold_last_axis_name=False):
        columns = pd.Index(self.axes[-1].labels)
        if not fold_last_axis_name:
            columns.name = self.axes[-1].name
        if self.ndim > 1:
            axes_names = self.axes.names[:-1]
            if fold_last_axis_name and axes_names[-1] is not None:
                axes_names[-1] = axes_names[-1] + '\\' + self.axes[-1].name

            index = pd.MultiIndex.from_product(self.axes.labels[:-1],
                                               names=axes_names)
        else:
            index = pd.Index([''])
            if fold_last_axis_name:
                index.name = self.axes.names[-1]
        data = np.asarray(self).reshape(len(index), len(columns))
        return pd.DataFrame(data, index, columns)

    @property
    def series(self):
        index = pd.MultiIndex.from_product([axis.labels for axis in self.axes],
                                           names=self.axes.names)
        return pd.Series(np.asarray(self).reshape(self.size), index)

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
        axes = [Axis(newname, a.labels) if a is axis else a
                for a in self.axes]
        return LArray(self.data, axes)

    def sort_axis(self, axis=None, reverse=False):
        """Sorts axes of the LArray.

        Parameters
        ----------
        axis : axis reference (Axis, string, int)
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
        >>> a.sort_axis(reverse=True)
        nat\\sex | H | F
             FO | 2 | 3
             EU | 0 | 1
             BE | 4 | 5
        """
        if axis is None:
            axes = self.axes
        else:
            axes = [self.axes[axis]]

        def sort_key(axis):
            key = np.argsort(axis.labels)
            if reverse:
                key = key[::-1]
            return PGroup(key, axis=axis.name)

        return self[tuple(sort_key(axis) for axis in axes)]

    def _translate_axis_key(self, axis_key):
        if isinstance(axis_key, Group):
            return axis_key

        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        for axis in self.axes:
            try:
                axis_pos_key = axis.translate(axis_key)
                valid_axes.append(axis.name)
            except KeyError:
                pass
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis"
                             % axis_key)
        elif len(valid_axes) > 1:
            raise ValueError('%s is ambiguous (valid in %s)' %
                             (axis_key, valid_axes))
        return PGroup(axis_pos_key, axis=valid_axes[0])

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
            except KeyError:
                continue
            valid_axes.append(axis.name)
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis"
                             % axis_key)
        elif len(valid_axes) > 1:
            raise ValueError('%s is ambiguous (valid in %s)' %
                             (axis_key, valid_axes))
        return LGroup(axis_key, axis=valid_axes[0])

    def translated_key(self, key):
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

        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, bool):
            return key.nonzero()
        if isinstance(key, LArray) and np.issubdtype(key.dtype, bool):
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
            num_ellipses = key.count(Ellipsis)
            if num_ellipses > 1:
                raise ValueError("cannot use more than one Ellipsis (...)")
            elif num_ellipses == 1:
                pos = key.index(Ellipsis)
                none_slices = (slice(None),) * (self.ndim - len(key) + 1)
                key = key[:pos] + none_slices + key[pos + 1:]

            # translate non LKey to PGroup and drop slice(None) since
            # they are meaningless at this point
            # XXX: we might want to raise an exception when we find (most)
            # slice(None) because except for a single slice(None) a[:], I don't
            # think there is any point.
            key = tuple(self._translate_axis_key(axis_key) for axis_key in key
                        if not isnoneslice(axis_key))

            assert all(isinstance(axis_key, Group) for axis_key in key)

            # handle keys containing LGroups (at potentially wrong places)

            # XXX: support LGroup without axis?
            # extract axis name from LGroup keys

            dupe_axes = list(duplicates(axis_key.axis for axis_key in key))
            if dupe_axes:
                raise ValueError("key with duplicate axis: %s" % dupe_axes)
            key = dict((axis_key.axis, axis_key) for axis_key in key)

        # dict -> tuple (complete and order key)
        assert isinstance(key, dict)
        axes_names = set(self.axes.names)
        for axis_name in key:
            if axis_name not in axes_names:
                raise KeyError("'{}' is not an axis name".format(axis_name))
        key = tuple(key[axis.name] if axis.name in key else slice(None)
                    for axis in self.axes)

        # label -> raw positional
        return tuple(axis.translate(axis_key)
                     for axis, axis_key in zip(self.axes, key))

    # XXX: we only need axes length, so we might want to move this out of the
    # class. to AxisCollection? but this backend/numpy-specific, so maybe not
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
            key = [range_to_slice(axis_key)
                   if isinstance(axis_key, sequence)
                   else axis_key
                   for axis_key in key]

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
        data = np.asarray(self)
        translated_key = self.translated_key(key)

        # TODO: make the combined keys should be objects which display as:
        # (axis1_label, axis2_label, ...) but should also store the axis (names)
        # Q: Should it be the same object as the NDLGroup?/NDKey?
        # A: yes, probably. On the Pandas backend, we could/should have
        #    separate axes. On the numpy backend we cannot.
        # FIXME: the issubdtype test is buggy, int dtypes return True
        # >>> key.dtype
        # dtype('int32')
        # >>> np.issubdtype(key.dtype, bool)
        # True
        if isinstance(key, (LArray, np.ndarray)) and \
                np.issubdtype(key.dtype, bool):
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

        axes = [axis.subaxis(axis_key)
                for axis, axis_key in zip(self.axes, translated_key)
                if not np.isscalar(axis_key)]

        cross_key = self.cross_key(translated_key, collapse_slices)
        data = data[cross_key]
        # drop length 1 dimensions created by scalar keys
        data = data.reshape(tuple(len(axis) for axis in axes))
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            return LArray(data, axes)

    def __setitem__(self, key, value, collapse_slices=True):
        data = np.asarray(self)
        translated_key = self.translated_key(key)

        if isinstance(key, (LArray, np.ndarray)) and \
                np.issubdtype(key.dtype, bool):
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
                         if not isnoneslice(axis_key)]
        other_axes = [axis for axis_key, axis in zip(key, self.axes)
                      if isnoneslice(axis_key)]
        assert len(combined_axes) > 0
        assert len(key) > 0
        axes_indices = [self.axes.index(axis) for axis in combined_axes]
        diff = np.diff(axes_indices)
        if np.any(diff > 1):
            # combined axes in front
            combined_axis_pos = 0
        else:
            combined_axis_pos = axes_indices[0]
        combined_name = ','.join(axis.name for axis in combined_axes)
        if wildcard_allowed:
            lengths = [len(axis_key) for axis_key in key
                       if not isnoneslice(axis_key)]
            combined_axis_len = lengths[0]
            assert all(l == combined_axis_len for l in lengths)
            combined_axis = Axis(combined_name, combined_axis_len)
        else:
            axes_labels = [axis.labels[axis_key]
                           for axis_key, axis in zip(key, self.axes)
                           if not isnoneslice(axis_key)]
            if len(combined_axes) == 1:
                combined_labels = axes_labels[0]
            else:
                combined_labels = list(zip(*axes_labels))

            combined_axis = Axis(combined_name, combined_labels)
        new_axes = other_axes
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
        >>> # TODO: use arr2.axes in this case
        >>> arr1.drop_labels() * arr2
        a*\\b* | 0 | 1
            0 | 0 | 1
            1 | 4 | 9
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

    def as_table(self, maxlines=80, edgeitems=5):
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
            res_axes = self.axes.without(axes_indices)
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
        return LArray(op(np.asarray(self), axis=self.axes.index(axis)),
                      self.axes)

    # TODO: now that items is never a (k, v), it should be renamed to
    # something else: args? (groups would be misleading because each "item"
    # can contain several groups)
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
            res_dtype = res.dtype if op is not np.mean else float
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
        def standardise_kw_arg(axis, key):
            if isinstance(key, str):
                key = to_keys(key)
            if isinstance(key, tuple):
                return tuple(standardise_kw_arg(axis, k) for k in key)
            if isinstance(key, LGroup):
                return key
            assert isinstance(key, (str, list, slice))
            return LGroup(key, axis=axis)

        def to_vg(key):
            if isinstance(key, str):
                key = to_keys(key)
            if isinstance(key, tuple):
                # a tuple is supposed to be several groups on the same axis
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
                return to_vg(to_keys(arg))

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
        return LArray(data, self.axes.without(axis))

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
        return LArray(self.data.argmin(axis_idx), self.axes.without(axis))

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
        return LArray(data, self.axes.without(axis))

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
        return LArray(self.data.argmax(axis_idx), self.axes.without(axis))

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
            if len(self.axes) > 1:
                raise ValueError("more than one axis in array and no axis "
                                 "specified for argsort")
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
            if len(self.axes) > 1:
                raise ValueError("more than one axis in array and no axis "
                                 "specified for argsort")
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
    prod = _agg_method(np.prod, np.nanprod, commutative=True)
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
            # so that we can do key.count(Ellipsis)
            elif other is Ellipsis or other is None:
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

    # XXX: rename/change to "add_axes" ?
    # TODO: add a flag copy=True to force a new array.
    def expand(self, target_axes=None, out=None):
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
               dialect='default', **kwargs):
        """
        write LArray to a csv file.

        Parameters
        ----------
        filepath : string
            path where the csv file has to be written.
        sep : string
            seperator for the csv file.
        na_rep : string
            replace na values with na_rep.
        transpose : boolean
            transpose = True  => transpose over last axis.
            transpose = False => no transpose.
        dialect : 'default' | 'classic'
            Whether or not to write the last axis name (using '\' )

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
            self.to_frame(fold).to_csv(filepath, sep=sep, na_rep=na_rep,
                                       **kwargs)
        else:
            self.series.to_csv(filepath, sep=sep, na_rep=na_rep, header=True,
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

    def to_excel(self, filepath, sheet_name='Sheet1', *args, **kwargs):
        """
        write LArray to an excel file in the specified sheet 'sheet_name'

        Parameters
        ----------
        filepath : string
            path where the excel file has to be written.
        sheet_name : string
            sheet where the data has to be written.
        *args
        **kargs

        Returns
        -------
        Excel file
            with LArray pasted in sheet_name.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> mat = ndrange([xnat, xsex])
        >>> mat.to_excel('test.xlsx', 'Sheet1')
        """
        self.to_frame().to_excel(filepath, sheet_name, *args, **kwargs)

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

    def plot(self, *args, **kwargs):
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
        self.to_frame().plot(*args, **kwargs)

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
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.size
        12
        """
        return self.data.size

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
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xsex, xtype])
        >>> mat
        sex\\type | type1 | type2 | type3
               H |     0 |     1 |     2
               F |     3 |     4 |     5
        >>> mat.shift(x.type)
        sex\\type | type2 | type3
               H |     0 |     1
               F |     3 |     4
        >>> mat.shift(x.type, n=-1)
        sex\\type | type1 | type2
               H |     1 |     2
               F |     4 |     5
        """
        axis = self.axes[axis]
        if n > 0:
            res = self[axis.i[:-n]]
            new_labels = axis.labels[n:]
        else:
            res = self[axis.i[-n:]]
            new_labels = axis.labels[:n]
        return res.set_labels(axis, new_labels)


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
             na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
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
    nat\\- | H | F
       BE | 0 | 1
       FO | 2 | 3
    """
    # read the first line to determine how many axes (time excluded) we have
    with csv_open(filepath) as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader)
        if headersep is not None and headersep != sep:
            combined_axes_names = header[0]
            header = combined_axes_names.split(headersep)
        try:
            # take the first cell which contains '\'
            pos_last = next(i for i, v in enumerate(header) if '\\' in v)
        except StopIteration:
            # if there isn't any, assume 1d array
            pos_last = 0
        axes_names = header[:pos_last + 1]

    if len(index_col) == 0 and nb_index == 0:
        nb_index = len(axes_names)

    if len(index_col) > 0:
        nb_index = len(index_col)
    else:
        index_col = list(range(nb_index))

    if headersep is not None:
        # we will set the index after having split the tick values
        index_col = None

    # force str for dimensions
    # because pandas autodetect failed (thought it was int when it was a string)
    dtype = {}
    for axis in axes_names[:nb_index]:
        dtype[axis] = np.str
    df = pd.read_csv(filepath, index_col=index_col, sep=sep, dtype=dtype,
                     **kwargs)
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
        index_col : list of columns for the index (e.g. [0, 1, 3])
    """
    if len(index_col) == 0:
        index_col = list(range(nb_index))
    df = pd.read_excel(filepath, sheetname, index_col=index_col, **kwargs)
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
    -\\- | 0 | 1 | 2
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
    -\\- | 0 | 1 | 2
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
    -\\- |   0 |   1 |   2
      0 | 0.0 | 1.0 | 2.0
      1 | 3.0 | 4.0 | 5.0
    >>> ndrange(3, start=2)
    - | 0 | 1 | 2
      | 2 | 3 | 4

    potential alternate syntaxes:
    ndrange((2, 3), names=('a', 'b'))
    ndrange(2, 3, names=('a', 'b'))
    ndrange([('a', 2), ('b', 3)])
    ndrange(('a', 2), ('b', 3))
    ndrange((2, 3)).rename([0, 1], ['a', 'b'])
    # current syntaxes
    ndrange((2, 3)).rename(0, 'a').rename(1, 'b')
    ndrange([Axis('a', 2), Axis('b', 3)])
    """
    # XXX: try to come up with a syntax where start is before "end". For ndim
    #  > 1, I cannot think of anything nice.
    axes = AxisCollection(axes)
    data = np.arange(start, start + np.prod(axes.shape), dtype=dtype)
    return LArray(data.reshape(axes.shape), axes)


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


def make_numpy_broadcastable(values):
    """
    return values where LArrays are (numpy) broadcastable between them.
    For that to be possible, all common axes must be either 1 or the same
    length. Extra axes (in any array) can have any length.

    * the resulting arrays will have the combination of all axes found in the
      input arrays, the earlier arrays defining the order.
    * axes with a single '*' label will be added for axes not present in input
    """
    # TODO: implement AxisCollection.union
    # all_axes = [v.axes for v in values if isinstance(v, LArray)]
    # all_axes = AxisCollection.union(all_axes)
    all_axes = AxisCollection()
    for v in values:
        if isinstance(v, LArray):
            all_axes.extend(v.axes)

    # 1) reorder axes
    values = [v.transpose(all_axes & v.axes) if isinstance(v, LArray) else v
              for v in values]

    # 2) add length one axes
    return [v.reshape([v.axes.get(axis.name, Axis(axis.name, 1))
                       for axis in all_axes]) if isinstance(v, LArray) else v
            for v in values], all_axes
