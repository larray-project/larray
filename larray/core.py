# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

__version__ = "0.2.6"

__all__ = [
    'LArray', 'Axis', 'AxisCollection', 'ValueGroup',
    'union', 'stack',
    'read_csv', 'read_eurostat', 'read_excel', 'read_hdf', 'read_tsv',
    'x',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like',
    'ndrange',
    '__version__'
]

"""
Matrix class
"""
# TODO
# * rename ValueGroup to LabelGroup

# * implement named groups in strings
#   eg "vla=A01,A02;bru=A21;wal=A55,A56"

# ? implement multi group in one axis getitem:
#   lipro['P01,P02;P05'] <=> (lipro.group('P01,P02'), lipro.group('P05'))
#                        <=> (lipro['P01,P02'], lipro['P05'])

# ? age, geo, sex, lipro = la.axes_names
#   => user only use axes strings and this allows them to not have to bother
#      about incompatible axes
#   => sadly, this prevents slicing axes (time[-10:])
#   => maybe la.axes should return another class (say GenericAxis) which only
#      contain a name, and can be "indexed"/sliced. No check that the key is
#      actually valid would be done until the valueGroup is actually used on
#      a specific LArray

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

# * avg on last 10 years
#     time = Axis('time', ...)
#     x = time[-10:]  # <- does not work (-10 is not a tick on the Axis)!
    # la.mean(time.i[-10:])
    # la[time[-10:]].mean(time)
    # la.append(la.mean(time[-10:]), axis=time)
    # la.append(time=la.avg(time[-10:]))
    # la.append(time=la.avg(time='-10:'))

# * drop last year
#   la = la[time[:-1]] # <- implement this !

# * split unit tests

# * easily add sum column for a dimension
#   - in all cases, we will need to define a new Axis object
#   - in the examples below, we suppose a.label is 'income'
#   - best candidates (IMO)
#       - a.append(a.sum(age), age, 'total')
#         * label can be optional (it could be set to either None or an
#           autoincrementing int/label) not all operations need a label anyway
#         * even axis can be optional in some (the usual) case: if value is
# only missing one dimension compared to a and the other dimensions are
# compatibles, we could assume it is the missing dimension. This would make
# this possible: a.append(a.sum(age)). This is DRY but possibly too magical
# and diverges from numpy where no axis => flattened result
#         it is probably better to have an alias: with_total/append_total
# which does it.
#       - a.append_total(axis, func=np.sum, label=None) # label = func.__name__?
#       - a.append_total(axis, func=np.mean, label=None)
#       - a.extend(values, axis)
#       - a.append(age, 'total', a.sum(age))
#       - a.append(age=a.sum(age))   # label is "income.sum(age)"
#                                    # ideally, it should be just "sum(age)"
#                                    # (the label on the array stays "income"
#                                    # after all, so it is redundant to add
#                                    # it here) but that is probably harder
#                                    # to get because a.sum(age).label should
#                                    # really be "income.sum(age)", it is just
#                                    # the label/tick on the new Axis that
#                                    # should not contain "income".
#       - a.append(age=a.sum(age).label('total'))  # label is "total"
#       - a.append(a.sum(age), axis=age)
#       - a.append_total(age)     # default aggregate is sum
#                                 # default label is "total"
#       - a.append_total(age=avg) # default aggregate is sum,
#       - a.append_total(age, sum) # default aggregate is sum,
#       - a.append_total(age, sex=avg) # default aggregate is sum,

# other candidates
#   - a.with_total(age=np.sum)
#   - a.with_total(age=np.sum,np.avg) # potentially several totals
#   - a.append(age=a.sum(age))
#   - a.append(age='sum')
#   - a.append(age=sum)

#   - a.append(total=a.sum(age), axis=age) # total = the name of the new label
#   - a.append(age='total=sum') # total = the name of the new label

#   - the following should work already (modulo the axis name -> axis num)
#   - all_ages = a.sum(age=(':',))
#   - np.concatenate((a, all_ages), axis=age)

#   - np.append(a, a.sum(age), axis=age)
#   - a.append(a.sum(age), axis=age)

# * check axes on arithmetics

# * but special case for length 1 (to be able to do: "H + F" or "vla / belgium")

# * reindex a dataset (ie make it conform to the index of another dataset)
#   so that you can do operations involving both (add, divide, ...)

# * reorder an axis labels
# * test to_csv: does it consume too much mem?
#   ---> test pandas (one dimension horizontally)
# * add labels in ValueGroups.__str__
# * xlsx export workbook without overwriting some sheets (charts)

# ? allow naming "one-shot" groups? e.g:
#   regsum = bel.sum(lipro='P01,P02 = P01P02; : = all')

# * review __getitem__ vs labels
#   o integer key on a non-integer label dimension is non-ambiguous:
#     => treat them like indices
#   o int key on in int label dimension is ambiguous:
#     => treat them like indices
#     OR
#     => treat them like values to lookup (len(key) has not relation with
#        len(dim) BUT if key is a tuple (nd-key), we have
#        len(dim0) == dim(dimX)
#   o bool key on a non-bool dimension is non-ambiguous:
#     - treat them as a filter (len(key) must be == len(dim))
#   o bool key on a bool dimension is ambiguous:
#     - treat them as a filter (len(key) must be == len(dim) == 2)
#       eg [False, True], [True, False], [True, True], [False, False]
#       >>> I think this usage is unlikely to be used by users directly but...
#     - treat them like a subset of values to include in the cartesian product
#       eg, supposing we have a array of shape (bool[2], int[110], bool[2])
#       the key ([False], [1, 5, 9], [False, True]) would return an array
#       of shape [1, 3, 2]
#     OR
#     - treat them like values to lookup (len(key) has not relation with
#       len(dim) BUT if key is a tuple (nd-key), we have len(dim0) == dim(dimX)
# * evaluate the impact of label-only __getitem__: numpy/matplotlib/...
#   functions probably rely on __getitem__ with indices

# * docstring for all methods
# * choose between subset and group. Having both is just confusing, I think.

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
# * data alignment in arithmetic methods (or at least check that axes are
#   compatible and raise an exception if they are not)
# * test structured arrays
# * review all method & argument names
# * implement ValueGroup.__getitem__
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

import numpy as np
import pandas as pd

from larray.utils import (prod, table2str, unique, array_equal, csv_open, unzip,
                          decode, basestring, izip, rproduct, ReprString,
                          duplicates)


# TODO: return a generator, not a list
def srange(*args):
    return list(map(str, range(*args)))

def range_to_slice(seq):
    """
    seq is a sequence-like (list, tuple or ndarray (*)) of integers
    returns a slice if possible (including for sequences of 1 element)
    otherwise returns the input sequence itself

    (*) isinstance(ndarray, Sequence) is False but it behaves like one
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
    # the fact that an "aggregated tick" is passed as a ValueGroup or as a
    # string should be as irrelevant as possible. The thing is that we cannot
    # (currently) use the more elegant to_tick(e.key) that means the
    # ValueGroup is not available in Axis.__init__ after to_ticks, and we
    # need it to update the mapping if it was named. Effectively,
    # this creates two entries in the mapping for a single tick. Besides,
    # I like having the ValueGroup as the tick, as it provides extra info as
    # to where it comes from.
    if np.isscalar(e) or isinstance(e, ValueGroup):
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
    if isinstance(s, LKey):
        # a single ValueGroup used for all ticks of an Axis
        raise NotImplemented("not sure what to do with it yet")
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
    elif not isinstance(v, basestring):
        return v

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
            # a single group => collapse dimension
            return to_key(value)
    elif isinstance(value, LKey):
        return value
    elif isinstance(value, list):
        return to_key(value)
    else:
        assert isinstance(value, tuple), "%s is not a tuple" % value
        return tuple([to_key(group) for group in value])


def union(*args):
    # TODO: add support for ValueGroup and lists
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


class PositionalKeyMaker(object):
    def __init__(self, axis):
        self.axis = axis

    def __getitem__(self, key):
        return PositionalKey(key, None, self.axis)


class Axis(object):
    # ticks instead of labels?
    # XXX: make name and labels optional?
    def __init__(self, name, labels):
        """
        labels should be an array-like (convertible to an ndarray)
        """
        if isinstance(name, Axis):
            name = name.name
        self.name = name
        labels = to_ticks(labels)

        # TODO: move this to to_ticks????
        # we convert to an ndarray to save memory (for scalar ticks, for
        # ValueGroup ticks, it does not make a difference since a list of VG
        # and an ndarray of VG are both arrays of pointers)
        self._labels = None
        self._mapping = {}
        self.labels = np.asarray(labels)

    @property
    def i(self):
        return PositionalKeyMaker(self.name)

    def get_labels(self):
        return self._labels
    def set_labels(self, new_labels):
        self._labels = new_labels
        self._update_mapping()
    labels = property(get_labels, set_labels)

    def _update_mapping(self):
        labels = self._labels
        self._mapping = {label: i for i, label in enumerate(labels)}
        # we have no choice but to do that!
        # otherwise we could not make geo['Brussels'] work efficiently
        # (we could have to traverse the whole mapping checking for each name,
        # which is not an option)
        self._mapping.update({label.name: i for i, label in enumerate(labels)
                              if isinstance(label, ValueGroup)})

    # XXX: not sure I should offer an *args version
    def group(self, *args, **kwargs):
        """
        key is label-based (slice and fancy indexing are supported)
        returns a ValueGroup usable in .sum or .filter
        """
        name = kwargs.pop('name', None)
        if kwargs:
            raise ValueError("invalid keyword argument(s): %s"
                             % list(kwargs.keys()))
        key = args[0] if len(args) == 1 else args
        if isinstance(key, ValueGroup):
            # XXX: I am not sure this test even makes sense. eg if we have two
            # axes arr_from and arr_to, we might want to reuse groups
            if key.axis != self.name:
                raise ValueError("cannot subset an axis with a ValueGroup of "
                                 "an incompatible axis")
            # FIXME: we should respect the given name (overrides key.name)
            return key
        return ValueGroup(key, name, self.name)

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
        return Axis(name, self.labels[key])

    def __eq__(self, other):
        return (isinstance(other, Axis) and self.name == other.name and
                array_equal(self.labels, other.labels))

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return len(self.labels)

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
        # arrays (those are either strings containing comas or ValueGroups)
        try:
            return mapping[key]
        # we must catch TypeError because key might not be hashable (eg slice)
        except (KeyError, TypeError):
            pass

        if isinstance(key, PositionalKey):
            return key.key

        if isinstance(key, ValueGroup):
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
        elif isinstance(key, (tuple, list, np.ndarray)):
            # handle fancy indexing with a sequence of labels
            # TODO: the result should be cached
            res = np.empty(len(key), int)
            for i, label in enumerate(key):
                res[i] = mapping[label]
            return res
        else:
            # the first mapping[key] above will cover most cases. This code
            # path is only used if the key was given in "non normalized form"
            assert np.isscalar(key), "%s (%s) is not scalar" % (key, type(key))
            # key is scalar (integer, float, string, ...)
            return mapping[key]

    def __str__(self):
        return self.name if self.name is not None else 'Unnamed axis'

    def __repr__(self):
        return 'Axis(%r, %r)' % (self.name, list(self.labels))

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
        # XXX: I wonder if we should make a copy of the labels
        return Axis(self.name, self.labels)

    def sorted(self):
        res = self.copy()
        # FIXME: this probably also sorts the original axis !
        res.labels.sort()
        res._update_mapping()
        return res


# We need a separate class for ValueGroup and cannot simply create a
# new Axis with a subset of values/ticks/labels: the subset of
# ticks/labels of the ValueGroup need to correspond to its *Axis*
# indices
class LKey(object):
    def __init__(self, key, name, axis):
        raise NotImplementedError()


class ValueGroup(LKey):
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
        # ValueGroups are more compatible between themselves.
        if isinstance(axis, Axis):
            axis = axis.name
        if axis is not None:
            assert isinstance(axis, str)
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
        # to_tick directly, instead of using to_key explicitly here
        return hash(to_tick(to_key(self.key)))

    def __eq__(self, other):
        # different name or axis compare equal !
        other_key = other.key if isinstance(other, ValueGroup) else other
        return to_tick(to_key(self.key)) == to_tick(to_key(other_key))

    def __str__(self):
        return to_string(self.key) if self.name is None else self.name

    def __repr__(self):
        name = ", %r" % self.name if self.name is not None else ''
        return "ValueGroup(%r%s)" % (self.key, name)

    def __len__(self):
        return len(self.key)

    def __lt__(self, other):
        return self.key.__lt__(other.key)

    def __gt__(self, other):
        return self.key.__gt__(other.key)


class PositionalKey(LKey):
    """
    Positional Key
    """
    def __init__(self, key, name=None, axis=None):
        if isinstance(key, tuple):
            key = list(key)
        self.key = key
        self.name = name
        self.axis = axis

    def __repr__(self):
        name = ", %r" % self.name if self.name is not None else ''
        return "PositionalKey(%r%s)" % (self.key, name)

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
        axes = [Axis(None, range(axis)) if isinstance(axis, int) else axis
                for axis in axes]
        assert all(isinstance(a, Axis) for a in axes)

        if not isinstance(axes, list):
            axes = list(axes)
        self._list = axes
        self._map = {axis.name: axis for axis in axes}

    def __getattr__(self, key):
        try:
            return self._map[key]
        except KeyError:
            return self.__getattribute__(key)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, Axis):
            # XXX: check that it is the same object????
            return self._map[key.name]
        elif isinstance(key, slice):
            return AxisCollection(self._list[key])
        else:
            return self._map[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise NotImplementedError("slice set")
        if isinstance(key, int):
            axis = self._list[key]
            self._list[key] = value
            del self._map[axis.name]
            self._map[value.name] = value
        else:
            assert isinstance(key, basestring)
            try:
                axis = self._map[key]
            except KeyError:
                raise ValueError("inserting a new axis by name is not possible")
            idx = self._list.index(axis)
            self._list[idx] = value
            self._map[key] = value

    def __delitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError("slice delete")
        if isinstance(key, int):
            axis = self._list.pop(key)
            del self._map[axis.name]
        elif isinstance(key, Axis):
            self._list.remove(key)
            del self._map[key.name]
        else:
            assert isinstance(key, basestring)
            axis = self._map.pop(key)
            self._list.remove(axis)

    def __add__(self, other):
        result = self[:]
        if isinstance(other, Axis):
            result.append(other)
        else:
            # other should be a sequence
            assert len(other) >= 0
            result.extend(other)
        return result

    def __eq__(self, other):
        """
        other collection compares equal if all axes compare equal and in the
        same order. Works with a list.
        """
        if not isinstance(other, list):
            other = list(other)
        return self._list == other

    def __contains__(self, key):
        if isinstance(key, Axis):
            key = key.name
        return key in self._map

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return "{%s}" % ', '.join([axis.name if axis.name is not None else '-'
                                   for axis in self._list])

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
        if isinstance(key, Axis):
            # XXX: check that it is the same object????
            key = key.name
        return self._map.get(key, default)

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
        for axis in axes:
            local_axis = self._map.get(axis.name)
            if local_axis is not None:
                if axis != local_axis:
                    raise ValueError("incompatible axes:\n%r\nvs\n%r"
                                     % (axis, local_axis))

    def extend(self, axes):
        """
        extend the collection by appending the axes from axes
        """
        # check that common axes are the same
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
        if the Axis object is from another LArray, index() will return the
        index of the local axis with the same name, whether it is compatible
        (has the same ticks) or not.

        Raises ValueError if the axis is not present.
        """
        name_or_idx = axis.name if isinstance(axis, Axis) else axis
        return self.names.index(name_or_idx) \
            if isinstance(name_or_idx, basestring) \
            else name_or_idx

    def insert(self, index, axis):
        """
        insert axis before index
        """
        # when __setitem__(slice) will be implemented, we could simplify this
        self._list.insert(index, axis)
        self._map[axis.name] = axis

    def copy(self):
        return self[:]

    def replace(self, oldaxis, newaxis):
        res = self[:]
        idx = self.index(oldaxis)
        res[idx] = newaxis
        return res

    def without(self, axes):
        """
        returns a new collection without some axes
        you can use a comma separated list
        """
        res = self[:]
        if isinstance(axes, basestring):
            axes = axes.split(',')
        elif isinstance(axes, Axis):
            axes = [axes]
        # transform positional axis to axis objects
        axes = [self[axis] for axis in axes]
        for axis in axes:
            del res[axis]
        return res

    @property
    def names(self):
        return [axis.name for axis in self._list]

    @property
    def shape(self):
        return tuple(len(axis) for axis in self._list)


class LArray(object):
    """
    LArray class
    """
    def __init__(self, data, axes=None):
        ndim = data.ndim
        if axes is not None:
            if len(axes) != ndim:
                raise ValueError("number of axes (%d) does not match "
                                 "number of dimensions of data (%d)"
                                 % (len(axes), ndim))
            shape = tuple(len(axis) for axis in axes)
            if shape != data.shape:
                raise ValueError("length of axes %s does not match "
                                 "data shape %s" % (shape, data.shape))

        if axes is not None and not isinstance(axes, AxisCollection):
            axes = AxisCollection(axes)
        self.data = np.asarray(data)
        self.axes = axes

    @property
    def df(self):
        axes_names = self.axes_names[:-1]
        if axes_names[-1] is not None:
            axes_names[-1] = axes_names[-1] + '\\' + self.axes[-1].name

        columns = self.axes[-1].labels
        index = pd.MultiIndex.from_product(self.axes_labels[:-1],
                                           names=axes_names)
        data = np.asarray(self).reshape(len(index), len(columns))
        return pd.DataFrame(data, index, columns)

    @property
    def series(self):
        index = pd.MultiIndex.from_product([axis.labels for axis in self.axes],
                                           names=self.axes_names)
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

    @property
    def axes_labels(self):
        return [axis.labels for axis in self.axes]

    @property
    def axes_names(self):
        """Returns a list of names of the axes of a LArray.

        Returns
        -------
        List
            List of names of the axes of a LArray.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> a = zeros([xnat, xsex])
        >>> a.axes_names
        ['nat', 'sex']
        """
        return [axis.name for axis in self.axes]

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

    def _translate_axis_key(self, axis_key):
        if isinstance(axis_key, LKey):
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
        return PositionalKey(axis_pos_key, axis=valid_axes[0])

    def translated_key(self, key):
        """
        Complete and translate key

        Parameters
        ----------
        key can have any of the following forms
        a) a single value
        b) a tuple of values (possibly including ValueGroups)
        c) an {axis_name: value} dict

        Returns
        -------
        Returns a full N dimensional positional key
        """

        # convert scalar keys to 1D keys
        if not isinstance(key, (tuple, dict)):
            key = (key,)

        # handle keys containing an Ellipsis
        if isinstance(key, tuple):
            num_ellipses = key.count(Ellipsis)
            if num_ellipses > 1:
                raise ValueError("cannot use more than one Ellipsis (...)")
            elif num_ellipses == 1:
                pos = key.index(Ellipsis)
                none_slices = (slice(None),) * (self.ndim - len(key) + 1)
                key = key[:pos] + none_slices + key[pos + 1:]

            # translate non LKey to PositionalKey and drop slice(None) since
            # they are meaningless at this point
            # XXX: we might want to raise an exception when we find (most)
            # slice(None) because except for a single slice(None) a[:], I don't
            # think there is any point.
            key = tuple(self._translate_axis_key(axis_key) for axis_key in key
                        if axis_key != slice(None))

            assert all(isinstance(axis_key, LKey) for axis_key in key)

            # handle keys containing ValueGroups (at potentially wrong places)

            # XXX: support ValueGroup without axis?
            # extract axis name from ValueGroup keys

            dupe_axes = list(duplicates(axis_key.axis for axis_key in key))
            if dupe_axes:
                raise ValueError("key with duplicate axis: %s" % dupe_axes)
            key = dict((axis_key.axis, axis_key) for axis_key in key)

        # dict -> tuple (complete and order key)
        assert isinstance(key, dict)
        axes_names = set(self.axes_names)
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
            # TODO: cache the range in the axis?
            listkey = tuple(np.arange(*axis_key.indices(len(axis)))
                            if isinstance(axis_key, slice)
                            else axis_key
                            for axis_key, axis in zip(noscalar_key, self.axes))
            # np.ix_ computes the cross product of all lists
            return np.ix_(*listkey)
        else:
            return key

    def __getitem__(self, key, collapse_slices=False):
        data = np.asarray(self)

        if isinstance(key, (np.ndarray, LArray)) and \
                np.issubdtype(key.dtype, bool):
            # TODO: return an LArray with Axis labels = combined keys
            # these combined keys should be objects which display as:
            # (axis1_label, axis2_label, ...) but should also store the axis
            # (names).
            # Q: Should it be the same object as the NDValueGroup?/NDKey?
            # A: yes, probably
            return data[np.asarray(key)]

        # translated_key = self.translated_key(self.full_key(key))
        translated_key = self.translated_key(key)

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

        if (isinstance(key, (np.ndarray, LArray)) and
                np.issubdtype(key.dtype, bool)):
            if isinstance(key, LArray):
                key = key.broadcast_with(self.axes)
            data[np.asarray(key)] = value
            return

        translated_key = self.translated_key(key)

        # XXX: we might want to create fakes axes in this case, as we only
        # use axes names and axes length, not the ticks, and those could
        # theoretically take a significant time to compute

        # FIXME: this breaks when using a boolean fancy index. eg
        # a[isnan(a)] = 0 (which breaks np.nan_to_num(a), which was used in
        # LArray.ratio())
        axes = [axis.subaxis(axis_key)
                for axis, axis_key in zip(self.axes, translated_key)
                if not np.isscalar(axis_key)]

        cross_key = self.cross_key(translated_key, collapse_slices)

        # if value is a "raw" ndarray we rely on numpy broadcasting
        data[cross_key] = value.broadcast_with(axes) \
            if isinstance(value, LArray) else value

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
        #            -> 4, 6 is potentially ok (merging dimensions)
        #            -> 24 is potentially ok (merging dimensions)
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

    def broadcast_with(self, target):
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
        if isinstance(target, LArray):
            target_axes = target.axes
        else:
            target_axes = target
            if not isinstance(target, AxisCollection):
                target_axes = AxisCollection(target_axes)
        target_names = [a.name for a in target_axes]

        # self.axes.check_compatible(target_axes)
        # this breaks la['1,5,9'] = la['2,7,3']
        # solution?
        # a) explicitly ask to drop labels
        # la['1,5,9'] = la['2,7,3'].data
        # la['1,5,9'] = la['2,7,3'].raw()
        # what if there is another dimension we want to broadcast?
        # b) ask to set correct labels explicitly
        # la['1,5,9'] = la['2,7,3'].set_labels(x.ages, [1, 5, 9])

        # 1) append length-1 axes for axes in target but not in source (I do not
        #    think their position matters).
        # TODO: factorize with make_numpy_broadcastable
        array = self.reshape(self.axes +
                             [Axis(name, ['*']) for name in target_names
                              if name not in self.axes])
        # 2) reorder axes to target order (move source only axes to the front)
        sourceonly_axes = [axis for axis in self.axes
                           if axis.name not in target_axes]
        other_axes = [self.axes.get(name, Axis(name, ['*']))
                      for name in target_names]
        return array.transpose(sourceonly_axes + other_axes)

    def __str__(self):
        if not self.ndim:
            return str(np.asscalar(self))
        elif not len(self):
            return 'LArray([])'
        else:
            return table2str(list(self.as_table()), 'nan', True,
                             keepcols=self.ndim - 1)
    __repr__ = __str__

    def as_table(self, maxlines=80, edgeitems=5):
        if not self.ndim:
            return

        # ert    | unit | geo\time | 2012   | 2011   | 2010
        # NEER27 | I05  | AT       | 101.41 | 101.63 | 101.63
        # NEER27 | I05  | AU       | 134.86 | 125.29 | 117.08
        width = self.shape[-1]
        height = prod(self.shape[:-1])
        data = np.asarray(self).reshape(height, width)

        if self.axes is not None:
            axes_names = [name if name is not None else '-'
                          for name in self.axes_names]
            if len(axes_names) > 1:
                axes_names[-2] = '\\'.join(axes_names[-2:])
                axes_names.pop()
            labels = self.axes_labels[:-1]
            if self.ndim == 1:
                # There is no vertical axis, so the axis name should not have
                # any "tick" below it and we add an empty "tick".
                ticks = [['']]
            else:
                ticks = product(*labels)

            yield axes_names + list(self.axes_labels[-1])
        else:
            # endlessly repeat empty list
            ticks = repeat([])

        # summary if needed
        if height > maxlines:
            data = chain(data[:edgeitems], [["..."] * width], data[-edgeitems:])
            if self.axes is not None:
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

    def _axis_aggregate(self, op, axes=()):
        """
        op is a numpy aggregate function: func(arr, axis=(0, 1))
        axes is a tuple of axes (each axis can be an Axis object, str or int)
        """
        src_data = np.asarray(self)
        if not axes:
            axes = self.axes

        axes_indices = tuple(self.axes.index(a) for a in axes)
        res_data = op(src_data, axis=axes_indices)
        axes_tokill = set(axes_indices)
        res_axes = [axis for axis_num, axis in enumerate(self.axes)
                    if axis_num not in axes_tokill]
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

    def _group_aggregate(self, op, items):
        res = self
        # TODO: when working with several "axes" at the same times, we should
        # not produce the intermediary result at all. It should be faster and
        # consume a bit less memory.
        for item in items:
            if isinstance(item, LKey):
                axis, groups = item.axis, item
            else:
                axis, groups = item
            groups = to_keys(groups)

            axis, axis_idx = res.axes[axis], res.axes.index(axis)
            res_axes = res.axes[:]
            res_shape = list(res.shape)

            if not isinstance(groups, tuple):
                # groups is in fact a single group
                assert isinstance(groups, (basestring, slice, list, LKey)), \
                       type(groups)
                if isinstance(groups, list):
                    assert len(groups) > 0

                    # Make sure this is actually a single group, not multiple
                    # mistakenly given as a list instead of a tuple
                    assert all(not isinstance(g, (tuple, list)) for g in groups)

                groups = (groups,)
                del res_axes[axis_idx]

                # it is easier to kill the axis after the fact
                killaxis = True
            else:
                # convert all value groups to strings
                # groups = tuple(str(g) if isinstance(g, ValueGroup) else g
                #                for g in groups)
                # grx = tuple(g.key if isinstance(g, ValueGroup) else g
                #             for g in groups)

                # We do NOT modify the axis name (eg append "_agg" or "*") even
                # though this creates a new axis that is independent from the
                # original one because the original name is what users will
                # want to use to access that axis (eg in .filter kwargs)
                res_axes[axis_idx] = Axis(axis.name, groups)
                killaxis = False

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
                group = [group] if group in axis else group

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
                op(arr, axis=axis_idx, out=out)
                del arr
            if killaxis:
                assert group_idx[axis_idx] == 0
                res_data = res_data[idx]
            if isinstance(res_data, np.ndarray):
                res = LArray(res_data, res_axes)
            else:
                res = res_data
        return res

    def _aggregate(self, op, args, kwargs, commutative=False):
        if not commutative and len(kwargs) > 1:
            raise ValueError("grouping aggregates on multiple axes at the same "
                             "time using keyword arguments is not supported "
                             "for '%s' (because it is not a commutative"
                             "operation and keyword arguments are *not* "
                             "ordered in Python)" % op.__name__)

        # Sort kwargs by axis name so that we have consistent results
        # between runs because otherwise rounding errors could lead to
        # slightly different results even for commutative operations.

        # XXX: transform kwargs to ValueGroups? ("geo", [1, 2]) -> geo[[1, 2]]
        operations = list(args) + sorted(kwargs.items())
        if not operations:
            # op() without args is equal to op(all_axes)
            return self._axis_aggregate(op)

        def isaxis(a):
            return isinstance(a, (int, basestring, Axis))

        res = self
        # group *consecutive* same-type (group vs axis aggregates) operations
        # we do not change the order of operations since we only group
        # consecutive operations.
        for are_axes, axes in groupby(operations, isaxis):
            func = res._axis_aggregate if are_axes else res._group_aggregate
            res = func(op, axes)
        return res

    def copy(self):
        return LArray(self.data.copy(), axes=self.axes[:])

    @property
    def info(self):
        """Describes a LArray (shape and labels for each axis).

        Returns
        -------
        String
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
        def shorten(l):
            return l if len(l) < 7 else l[:3] + ['...'] + list(l[-3:])
        axes_labels = [' '.join(shorten([repr(l) for l in axis.labels]))
                       for axis in self.axes]
        lines = [" %s [%d]: %s" % (axis.name, len(axis), labels)
                 for axis, labels in zip(self.axes, axes_labels)]
        shape = " x ".join(str(s) for s in self.shape)
        return ReprString('\n'.join([shape] + lines))

    def ratio(self, *axes):
        """Returns a LArray with values LArray/LArray.sum(axes).

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            LArray = LArray/LArray.sum(axes).

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ones([xnat, xsex, xtype])
        >>> # 0.0833 == 1 / mat.sum()
        >>> mat.ratio()
        nat | sex\\type |           type1 |           type2 |           type3
         BE |        H | 0.0833333333333 | 0.0833333333333 | 0.0833333333333
         BE |        F | 0.0833333333333 | 0.0833333333333 | 0.0833333333333
         FO |        H | 0.0833333333333 | 0.0833333333333 | 0.0833333333333
         FO |        F | 0.0833333333333 | 0.0833333333333 | 0.0833333333333
        >>> # 0.16666 == 1 / mat.sum(xsex, xtype)
        >>> mat.ratio(xsex, xtype)
        nat | sex\\type |          type1 |          type2 |          type3
         BE |        H | 0.166666666667 | 0.166666666667 | 0.166666666667
         BE |        F | 0.166666666667 | 0.166666666667 | 0.166666666667
         FO |        H | 0.166666666667 | 0.166666666667 | 0.166666666667
         FO |        F | 0.166666666667 | 0.166666666667 | 0.166666666667
       """
        if not axes:
            axes = self.axes
        return self / self.sum(*axes)

    # aggregate method factory
    def _agg_method(npfunc, name=None, commutative=False):
        def method(self, *args, **kwargs):
            return self._aggregate(npfunc, args, kwargs,
                                   commutative=commutative)
        if name is None:
            name = npfunc.__name__
        method.__name__ = name
        return method

    all = _agg_method(np.all, commutative=True)
    any = _agg_method(np.any, commutative=True)
    # commutative modulo float precision errors
    sum = _agg_method(np.sum, commutative=True)
    prod = _agg_method(np.prod, commutative=True)
    min = _agg_method(np.min, commutative=True)
    max = _agg_method(np.max, commutative=True)
    mean = _agg_method(np.mean, commutative=True)
    # not commutative
    ptp = _agg_method(np.ptp)
    var = _agg_method(np.var)
    std = _agg_method(np.std)

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

    def append(self, axis, value, label=None):
        """Adds a LArray to a LArray ('self') along an axis.

        Parameters
        ----------
        axis : axis
            the axis
        value : LArray
            LArray of the same shape as self
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
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ones([xnat, xsex, xtype])
        >>> mat
        nat | sex\\type | type1 | type2 | type3
         BE |        H |   1.0 |   1.0 |   1.0
         BE |        F |   1.0 |   1.0 |   1.0
         FO |        H |   1.0 |   1.0 |   1.0
         FO |        F |   1.0 |   1.0 |   1.0
        >>> mat.append(x.type, mat.sum(x.type), 'Type4')
        nat | sex\\type | type1 | type2 | type3 | Type4
         BE |        H |   1.0 |   1.0 |   1.0 |   3.0
         BE |        F |   1.0 |   1.0 |   1.0 |   3.0
         FO |        H |   1.0 |   1.0 |   1.0 |   3.0
         FO |        F |   1.0 |   1.0 |   1.0 |   3.0
        """
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        shape = self.shape
        value = np.asarray(value)
        if value.shape == shape[:axis_idx] + shape[axis_idx+1:]:
            # adding a dimension of size one if it is missing
            new_shape = shape[:axis_idx] + (1,) + shape[axis_idx+1:]
            value = value.reshape(new_shape)
        data = np.append(np.asarray(self), value, axis=axis_idx)
        new_axes = self.axes[:]
        new_axes[axis_idx] = Axis(axis.name, np.append(axis.labels, label))
        return LArray(data, axes=new_axes)

    def extend(self, axis, other):
        """Adds a LArray to a LArray ('self') along an axis.

        Parameters
        ----------
        axis : axis
            the axis
        other : LArray
            LArray of the same shape as self

        Returns
        -------
        LArray
            LArray expanded with 'other' along 'axis'.

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xsex2 = Axis('sex', ['U'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat1 = ones([xnat, xsex, xtype])
        >>> mat1
        nat | sex\\type | type1 | type2 | type3
         BE |        H |   1.0 |   1.0 |   1.0
         BE |        F |   1.0 |   1.0 |   1.0
         FO |        H |   1.0 |   1.0 |   1.0
         FO |        F |   1.0 |   1.0 |   1.0
        >>> mat2 = zeros([xnat, xsex2, xtype])
        >>> mat2
        nat | sex\\type | type1 | type2 | type3
         BE |        U |   0.0 |   0.0 |   0.0
         FO |        U |   0.0 |   0.0 |   0.0
        >>> mat1.extend(x.sex, mat2)
        nat | sex\\type | type1 | type2 | type3
         BE |        H |   1.0 |   1.0 |   1.0
         BE |        F |   1.0 |   1.0 |   1.0
         BE |        U |   0.0 |   0.0 |   0.0
         FO |        H |   1.0 |   1.0 |   1.0
         FO |        F |   1.0 |   1.0 |   1.0
         FO |        U |   0.0 |   0.0 |   0.0
        """
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        # Get axis by name, so that we do *NOT* check they are "compatible",
        # because it makes sense to append axes of different length
        other_axis = other.axes[axis]

        data = np.append(np.asarray(self), np.asarray(other), axis=axis_idx)
        new_axes = self.axes[:]
        new_axes[axis_idx] = Axis(axis.name,
                                  np.append(axis.labels, other_axis.labels))
        return LArray(data, axes=new_axes)

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

    def to_csv(self, filepath, sep=',', na_rep='', transpose=True, **kwargs):
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

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat
        nat | sex\\type | type1 | type2 | type3
         BE |        H |     0 |     1 |     2
         BE |        F |     3 |     4 |     5
         FO |        H |     6 |     7 |     8
         FO |        F |     9 |    10 |    11
        >>> mat.to_csv('test.csv', ';', transpose=True)
        >>> # nat;sex\type;type1;type2;type3
        >>> # BE;H;0;1;2tra
        >>> # BE;F;3;4;5
        >>> # FO;H;6;7;8
        >>> # FO;F;9;10;11
        >>> mat.to_csv('test.csv', ';', transpose=False)
        >>> # nat;sex;type;0
        >>> # BE;H;type1;0
        >>> # BE;H;type2;1
        >>> # BE;H;type3;2
        >>> # BE;F;type1;3
        >>> # BE;F;type2;4
        >>> # BE;F;type3;5
        >>> # FO;H;type1;6
        >>> # FO;H;type2;7
        >>> # FO;H;type3;8
        >>> # FO;F;type1;9
        >>> # FO;F;type2;10
        >>> # FO;F;type3;11
        """
        if transpose:
            self.df.to_csv(filepath, sep=sep, na_rep=na_rep, **kwargs)
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
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.to_hdf('test.h5', 'mat')
        """
        self.df.to_hdf(filepath, key, *args, **kwargs)

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
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.to_excel('test.xlsx', 'Sheet1')
        """
        self.df.to_excel(filepath, sheet_name, *args, **kwargs)

    def to_clipboard(self, *args, **kwargs):
        """
        sends the content of a LArray to clipboard

        using to_clipboard() makes it possible to paste the content of LArray
        into a file (Excel, ascii file,...)

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat.to_clipboard()  # doctest: +SKIP
        """
        self.df.to_clipboard(*args, **kwargs)

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
        self.df.plot(*args, **kwargs)

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
        lables : list of axis labels
            the new labels.
        inplace : boolean
            ???

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
        axis
            the axis for which we want to perform the shift.
        n : intger
            the numer of cells to shift.

        Returns
        -------
        LArray

        Example
        -------
        >>> xnat = Axis('nat', ['BE', 'FO'])
        >>> xsex = Axis('sex', ['H', 'F'])
        >>> xtype = Axis('type',['type1', 'type2', 'type3'])
        >>> mat = ndrange([xnat, xsex, xtype])
        >>> mat
        nat | sex\\type | type1 | type2 | type3
         BE |        H |     0 |     1 |     2
         BE |        F |     3 |     4 |     5
         FO |        H |     6 |     7 |     8
         FO |        F |     9 |    10 |    11
        >>> mat.shift(x.type, n=-1)
        nat | sex\\type | type1 | type2
         BE |        H |     1 |     2
         BE |        F |     4 |     5
         FO |        H |     7 |     8
         FO |        F |    10 |    11
        >>> mat.shift(x.sex, n=1)
        nat | sex\\type | type1 | type2 | type3
         BE |        F |     0 |     1 |     2
         FO |        F |     6 |     7 |     8
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
        if s.lower() in ('0', '1', 'false', 'true'):
            return s in ('1', 'true')
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


def cartesian_product_df(df, sort_rows=True, sort_columns=False, **kwargs):
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


def df_aslarray(df, sort_rows=True, sort_columns=True, **kwargs):
    axes_names = [decode(name, 'utf8') for name in df.index.names]
    if axes_names == [None]:
        last_axis = None, None
    else:
        last_axis = axes_names[-1].split('\\')
    axes_names[-1] = last_axis[0]
    # FIXME: hardcoded "time"
    axes_names.append(last_axis[1] if len(last_axis) > 1 else 'time')
    df, axes_labels = cartesian_product_df(df, sort_rows=sort_rows,
                                           sort_columns=sort_columns, **kwargs)

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
             na=np.nan, sort_rows=True, sort_columns=True, **kwargs):
    """
    reads csv file and returns a Larray with the contents

    Note
    ----
    format csv file:
    arr,ages,sex,nat\time,1991,1992,1993
    A1,BI,H,BE,1,0,0
    A1,BI,H,FO,2,0,0
    A1,BI,F,BE,0,0,1
    A1,BI,F,FO,0,0,0
    A1,A0,H,BE,0,0,0

    Parameters
    ----------
    filepath : string
        path where the csv file has to be written.
    nb_index : integer
        number of leading index columns (ex. 4).
    index_col : list
        list of columns for the index (ex. [0, 1, 2, 3]).
    sep : string
        seperator.
    headersep : ???
        ???.
    na : ???
        ???.
    sort_rows : boolean
        True (default) => ???.
        False => ???.
    sort_columns : boolean
        True (default) => ???.
        False => ???.
    **kwargs

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> xtype = Axis('type',['type1', 'type2', 'type3'])
    >>> mat = ndrange([xnat, xsex, xtype])
    >>> mat.to_csv('test.csv', ';')
    >>> read_csv('test.csv', sep=';')
    nat | sex\\type | type1 | type2 | type3
     BE |        F |     3 |     4 |     5
     BE |        H |     0 |     1 |     2
     FO |        F |     9 |    10 |    11
     FO |        H |     6 |     7 |     8
    >>> read_csv('test.csv', sep=';', sort_rows=False,
    ...          sort_columns=False)
    nat | sex\\type | type1 | type2 | type3
     BE |        H |     0 |     1 |     2
     BE |        F |     3 |     4 |     5
     FO |        H |     6 |     7 |     8
     FO |        F |     9 |    10 |    11
    """
    # read the first line to determine how many axes (time excluded) we have
    with csv_open(filepath) as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader)
        if headersep is not None and headersep != sep:
            combined_axes_names = header[0]
            header = combined_axes_names.split(headersep)
        pos_last = next(i for i, v in enumerate(header) if '\\' in v)
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
    result : LArray

    Examples
    --------

    >>>

    """
    return read_csv(filepath, sep='\t', headersep=',', **kwargs)


def read_hdf(filepath, key, na=np.nan, sort_rows=True, sort_columns=True,
             **kwargs):
    """
    read an LArray from a h5 file with the specified name
    """
    df = pd.read_hdf(filepath, key, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=na)


def read_excel(filepath, sheetname=0, nb_index=0, index_col=[],
               na=np.nan, sort_rows=True, sort_columns=True, **kwargs):
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


def zeros(axes):
    """Returns a LArray with the shape defined by axes and filled with zeros.

    Parameters
    ----------
    axes
        either a collection of axes or a shape.

    Returns
    -------
    LArray
        LArray with a shape defined by axes and filled with zeros.

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
    return LArray(np.zeros(axes.shape), axes)


def zeros_like(array):
    """Returns a LArray with the same shape as array and filled with zeros.

    Parameters
    ----------
    array
         is an array object.

    Returns
    -------
    LArray
        LArray with the same shape as array and filled with zeros.

    Example
    -------
    >>> a = ndrange((2, 3))
    >>> zeros_like(a)
    -\\- |   0 |   1 |   2
      0 | 0.0 | 0.0 | 0.0
      1 | 0.0 | 0.0 | 0.0
    """
    return zeros(array.axes)


def ones(axes):
    """Returns a LArray with the shape defined by axes and filled with ones.

    Parameters
    ----------
    axes
        either a collection of axes or a shape.

    Returns
    -------
    LArray
        LArray with the shape defined by axes and filled with ones.

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
    return LArray(np.ones(axes.shape), axes)


def ones_like(array):
    """Returns a LArray with the same shape as array and filled with ones.
    zeros

    Parameters
    ----------
    array
        is an array object.

    Returns
    -------
    LArray
        LArray with the same shape as array and filled with ones.

    Example
    -------
    >>> a = ndrange((2, 3))
    >>> ones_like(a)
    -\\- |   0 |   1 |   2
      0 | 1.0 | 1.0 | 1.0
      1 | 1.0 | 1.0 | 1.0
    """
    return ones(array.axes)


def empty(axes):
    """Returns a LArray with the shape defined by axes without initializing
    entries.

    Parameters
    ----------
    axes
        either a collection of axes or a shape.

    Returns
    -------
    larray
        LArray with a shape defined by axes and values are uninitialized
        (arbitrary) data.

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
    return LArray(np.empty(axes.shape), axes)


def empty_like(array):
    """Returns a LArray with the shape defined by axes without initial. entries.

    Parameters
    ----------
    array
        is an array object.

    Returns
    -------
    LArray
        LArray with a shape defined by axes and values are uninitialized
        (arbitrary) data.

    Example
    -------
    >>> a = ndrange((3, 2))
    >>> empty_like(a)   # doctest: +SKIP
    -\- |                  0 |                  1
      0 | 2.12199579097e-314 | 6.36598737388e-314
      1 | 1.06099789568e-313 | 1.48539705397e-313
      2 | 1.90979621226e-313 | 2.33419537056e-313
    """
    return empty(array.axes)


def ndrange(axes):
    """Returns a LArray with the shape defined and filled with increasing int.

    Parameters
    ----------
    axes
        either a collection of axes or a shape.

    Returns
    -------
    LArray
        LArray with a shape defined by axes and filled with increasing int.

    Example
    -------
    >>> xnat = Axis('nat', ['BE', 'FO'])
    >>> xsex = Axis('sex', ['H', 'F'])
    >>> ndrange([xnat, xsex])
    nat\\sex | H | F
         BE | 0 | 1
         FO | 2 | 3
    """
    axes = AxisCollection(axes)
    return LArray(np.arange(prod(axes.shape)).reshape(axes.shape), axes)


def stack(arrays, axis):
    """
    stack([numbirths * HMASC,
           numbirths * (1 - HMASC)], Axis('sex', 'H,F'))
    potential alternate syntaxes
    stack(['H', numbirths * HMASC,
           'F', numbirths * (1 - HMASC)], 'sex')
    stack(('H', numbirths * HMASC),
          ('F', numbirths * (1 - HMASC)), 'sex')
    """
    # append an extra length 1 dimension
    data_arrays = [a.data.reshape(a.shape + (1,)) for a in arrays]
    axes = arrays[0].axes
    for a in arrays[1:]:
        a.axes.check_compatible(axes)
    return LArray(np.concatenate(data_arrays, axis=-1), axes + axis)


class AxisRef(Axis):
    def __init__(self, name):
        self.name = name
        self._labels = None

    def translate(self, key):
        raise NotImplementedError("an Axis reference (x.) cannot translate "
                                  "labels")

    def __repr__(self):
        return 'AxisRef(%r)' % self.name


class AxisFactory(object):
    def __getattr__(self, key):
        return AxisRef(key)
x = AxisFactory()


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

    # 1) add length one axes
    # v.axes.extend([Axis(name, ['*']) for name in all_axes - v.axes])
    # values = [v.reshape([v.axes.get(axis.name, Axis(axis, ['*']))
    #                      for axis in all_axes]) if isinstance(v, LArray) else v
    #           for v in values]
    # 1) reorder axes
    values = [v.transpose(all_axes & v.axes) if isinstance(v, LArray) else v
              for v in values]
    # print("transposed")
    # print(values)

    # 2) add length one axes
    # v.axes.extend([Axis(name, ['*']) for name in all_axes - v.axes])
    return [v.reshape([v.axes.get(axis.name, Axis(axis, ['*']))
                       for axis in all_axes]) if isinstance(v, LArray) else v
            for v in values], all_axes
