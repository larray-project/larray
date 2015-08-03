# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function


# this branch tries to implement the following structure:
# class LArray(object):  # abstract class (or possibly ndarray API)
#     pass
#
#
# class DataFrameLArray(LArray):
#     def __init__(self, data):
#         # data is a pd.DataFrame
#         self.data = data

__version__ = "0.2dev"

"""
Matrix class
"""
#TODO
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
# * implement newaxis
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
    # la.avg(time[-10:])
    # la[time[-10:]].avg(time)
    # la.append(la.avg(time[-10:]), axis=time)
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

from larray.utils import (prod, unique, array_equal, csv_open, unzip,
                          decode, basestring, izip, rproduct, ReprString,
                          duplicates, _sort_level_inplace,
                          _pandas_insert_index_level, _pandas_transpose_any,
                          _pandas_transpose_any_like, _pandas_align,
                          _pandas_broadcast_to, multi_index_from_product,
                          _index_level_unique_labels)
from larray.sorting import set_topological_index


#TODO: return a generator, not a list
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
            return str(v[0]) + ','
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

    #XXX: we might want to return real int instead, because if we ever
    # want to have more complex queries, such as:
    # arr.filter(age > 10 and age < 20)
    # this would break for string values (because '10' < '2')
    >>> to_ticks(':3')
    ['0', '1', '2', '3']
    """
    if isinstance(s, ValueGroup):
        # a single ValueGroup used for all ticks of an Axis
        raise NotImplemented("not sure what to do with it yet")
    elif isinstance(s, pd.Index):
        return s.values
    elif isinstance(s, np.ndarray):
        #XXX: we assume it has already been translated. Is it a safe assumption?
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
    elif isinstance(value, ValueGroup):
        return value
    elif isinstance(value, list):
        return to_key(value)
    else:
        assert isinstance(value, tuple), "%s is not a tuple" % value
        return tuple([to_key(group) for group in value])


def union(*args):
    #TODO: add support for ValueGroup and lists
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


class Axis(object):
    # ticks instead of labels?
    #XXX: make name and labels optional?
    def __init__(self, name, labels):
        """
        labels should be an array-like (convertible to an ndarray)
        """
        self.name = name
        labels = to_ticks(labels)

        #TODO: move this to to_ticks????
        # we convert to an ndarray to save memory (for scalar ticks, for
        # ValueGroup ticks, it does not make a difference since a list of VG
        # and an ndarray of VG are both arrays of pointers)
        self.labels = np.asarray(labels)
        self._mapping = {}
        self._update_mapping()

    def _update_mapping(self):
        labels = self.labels
        self._mapping = {label: i for i, label in enumerate(labels)}
        # we have no choice but to do that!
        # otherwise we could not make geo['Brussels'] work efficiently
        # (we could have to traverse the whole mapping checking for each name,
        # which is not an option)
        self._mapping.update({label.name: i for i, label in enumerate(labels)
                              if isinstance(label, ValueGroup)})

    #XXX: not sure I should offer an *args version
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
            if key.axis != self:
                raise ValueError("cannot subset an axis with a ValueGroup of "
                                 "an incompatible axis")
            return key
        return ValueGroup(key, name, self)

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
            #TODO: the result should be cached
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
        return 'Axis(%r, %r)' % (self.name, self.labels.tolist())

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
            return Axis(self.name, [l for l in self.labels if l not in other.labels])
        else:
            try:
                return Axis(self.name, self.labels - other)
            except Exception:
                raise ValueError

    def copy(self):
        #XXX: I wonder if we should make a copy of the labels
        return Axis(self.name, self.labels)

    def sorted(self):
        res = self.copy()
        #FIXME: this probably also sorts the original axis !
        res.labels.sort()
        res._update_mapping()
        return res


# We need a separate class for ValueGroup and cannot simply create a
# new Axis with a subset of values/ticks/labels: the subset of
# ticks/labels of the ValueGroup need to correspond to its *Axis*
# indices
class ValueGroup(object):
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

        if axis is not None:
            # check the key is valid
            #TODO: for performance reasons, we should cache the result. This will
            # need to be invalidated correctly
            axis.translate(key)
        self.axis = axis

    def __hash__(self):
        # to_tick & to_key are partially opposite operations but this
        # standardize on a single notation so that they can all target each
        # other. eg, this removes spaces in "list strings", instead of
        # hashing them directly
        #XXX: but we might want to include that normalization feature in
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
            #XXX: check that it is the same object????
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
        return key in self._map

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return "{%s}" % ', '.join([axis.name if axis.name is not None else '-'
                                   for axis in self._list])

    def __repr__(self):
        axes_repr = (repr(axis) for axis in self._list)
        return "AxisCollection([\n    %s\n])" % ',\n    '.join(axes_repr)

    def get(self, key, default=None):
        return self._map.get(key, default)

    def keys(self):
        return [a.name for a in self._list]

    def pop(self, index=-1):
        axis = self._list.pop(index)
        del self._map[axis.name]
        return axis

    def append(self, axis):
        """
        append axis at the end of the collection
        """
        # when __setitem__(slice) will be implemented, we could simplify this
        self._list.append(axis)
        self._map[axis.name] = axis

    def extend(self, axes):
        """
        extend the collection by appending the axes from axes
        """
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
            # if len(axes) != ndim:
            #     raise ValueError("number of axes (%d) does not match "
            #                      "number of dimensions of data (%d)"
            #                      % (len(axes), ndim))
            shape = tuple(len(axis) for axis in axes)
            # if prod(data.shape) != prod(shape):
            #     raise ValueError("bad shape: %s vs %s" % (data.shape, shape))
            # if shape != data.shape:
            #     raise ValueError("length of axes %s does not match "
            #                      "data shape %s" % (shape, data.shape))

        if axes is not None and not isinstance(axes, AxisCollection):
            axes = AxisCollection(axes)
        self.data = data
        self.axes = axes

    def __array_finalize__(self, obj):
        raise Exception("does this happen?")

    @property
    def axes_labels(self):
        return [axis.labels for axis in self.axes]

    @property
    def axes_names(self):
        return [axis.name for axis in self.axes]

    @property
    def shape(self):
        return tuple(len(axis) for axis in self.axes)

    @property
    def ndim(self):
        return len(self.axes)

    def axes_rename(self, **kwargs):
        for k in kwargs.keys():
            if k not in self.axes:
                raise KeyError("'%s' axis not found in array")
        axes = [Axis(kwargs[a.name] if a.name in kwargs else a.name, a.labels)
                for a in self.axes]
        self.axes = AxisCollection(axes)
        return self

    def rename(self, axis, newname):
        axis = self.get_axis(axis)
        axes = [Axis(newname, a.labels) if a is axis else a
                for a in self.axes]
        return LArray(self, axes)

    def full_key(self, key):
        """
        Returns a full nd-key from a key in any of the following forms:
        a) a single value b) a tuple of values (possibly including ValueGroups)
        c) an {axis_name: value} dict
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

        # handle keys containing ValueGroups (at potentially wrong places)
        if any(isinstance(axis_key, ValueGroup) for axis_key in key):
            #XXX: support ValueGroup without axis?
            listkey = [(axis_key.axis.name
                        if isinstance(axis_key, ValueGroup)
                        else axis_name, axis_key)
                       for axis_key, axis_name in zip(key, self.axes_names)]
            dupe_axes = list(duplicates(k for k, v in listkey))
            if dupe_axes:
                raise ValueError("key with duplicate axis: %s" % dupe_axes)
            key = dict(listkey)

        if isinstance(key, dict):
            axes_names = set(self.axes_names)
            for axis_name in key:
                if axis_name not in axes_names:
                    raise KeyError("'{}' is not an axis name".format(axis_name))
            key = tuple(key[axis.name] if axis.name in key else slice(None)
                        for axis in self.axes)

        # convert xD keys to ND keys
        if len(key) < self.ndim:
            key += (slice(None),) * (self.ndim - len(key))

        return key

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
            #TODO: cache the range in the axis?
            listkey = tuple(np.arange(*axis_key.indices(len(axis)))
                            if isinstance(axis_key, slice)
                            else axis_key
                            for axis_key, axis in zip(noscalar_key, self.axes))
            # np.ix_ computes the cross product of all lists
            return np.ix_(*listkey)
        else:
            return key

    def reshape(self, target_axes):
        """
        self.size must be equal to prod([len(axis) for axis in target_axes])
        """
        data = np.asarray(self).reshape([len(axis) for axis in target_axes])
        return LArray(data, target_axes)

    def reshape_like(self, target):
        """
        target is an LArray, total size must be compatible
        """
        return self.reshape(target.axes)

    # deprecated since Python 2.0 but we need to define it to catch "simple"
    # slices (with integer bounds !) because ndarray is a "builtin" type
    def __getslice__(self, i, j):
        # sadly LArray[:] translates to LArray.__getslice__(0, sys.maxsize)
        return self[slice(i, j) if i != 0 or j != sys.maxsize else slice(None)]

    def __setslice__(self, i, j, value):
        self[slice(i, j) if i != 0 or j != sys.maxsize else slice(None)] = value

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
            axes_names = self.axes_names[:]
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

    def set(self, value, **kwargs):
        """
        sets a subset of LArray to value

        * all common axes must be either 1 or the same length
        * extra axes in value must be of length 1
        * extra axes in self can have any length
        """
        self.__setitem__(kwargs, value)

    def get_axis(self, axis, idx=False):
        """
        axis can be an index, a name or an Axis object
        if the Axis object is from another LArray, get_axis will return the
        local axis with the same name, **whether it is compatible (has the
        same ticks) or not**.
        """
        axis_idx = self.axes.index(axis)
        axis = self.axes[axis_idx]
        return (axis, axis_idx) if idx else axis

    def _aggregate(self, op_name, args, kwargs, commutative=False):
        if not commutative and len(kwargs) > 1:
            raise ValueError("grouping aggregates on multiple axes at the same "
                             "time using keyword arguments is not supported "
                             "for '%s' (because it is not a commutative"
                             "operation and keyword arguments are *not* "
                             "ordered in Python)" % op_name.__name__)

        # Sort kwargs by axis name so that we have consistent results
        # between runs because otherwise rounding errors could lead to
        # slightly different results even for commutative operations.

        #XXX: transform kwargs to ValueGroups? ("geo", [1, 2]) -> geo[[1, 2]]
        operations = list(args) + sorted(kwargs.items())
        if not operations:
            # op() without args is equal to op(all_axes)
            return self._axis_aggregate(op_name)

        def isaxis(a):
            return isinstance(a, (int, basestring, Axis))

        res = self
        # group *consecutive* same-type (group vs axis aggregates) operations
        for are_axes, axes in groupby(operations, isaxis):
            func = res._axis_aggregate if are_axes else res._group_aggregate
            res = func(op_name, axes)
        return res

    # aggregate method factory
    def _agg_method(name, commutative=False):
        def method(self, *args, **kwargs):
            return self._aggregate(name, args, kwargs,
                                   commutative=commutative)
        method.__name__ = name
        return method

    all = _agg_method('all', commutative=True)
    any = _agg_method('any', commutative=True)
    # commutative modulo float precision errors
    sum = _agg_method('sum', commutative=True)
    prod = _agg_method('prod', commutative=True)

    # no level argument
    # cumsum = _agg_method('cumsum', commutative=True)
    # cumprod = _agg_method('cumprod', commutative=True)
    min = _agg_method('min', commutative=True)
    max = _agg_method('max', commutative=True)
    mean = _agg_method('mean', commutative=True)

    # not commutative
    # N/A in pd.DataFrame
    # ptp = _agg_method('ptp')
    var = _agg_method('var')
    std = _agg_method('std')

    def ratio(self, *axes):
        if not axes:
            axes = self.axes
        return self / self.sum(*axes)

    @property
    def info(self):
        def shorten(l):
            return l if len(l) < 7 else l[:3] + ['...'] + list(l[-3:])
        axes_labels = [' '.join(shorten([repr(l) for l in axis.labels]))
                       for axis in self.axes]
        lines = [" %s [%d]: %s" % (axis.name, len(axis), labels)
                 for axis, labels in zip(self.axes, axes_labels)]
        shape = " x ".join(str(s) for s in self.shape)
        return ReprString('\n'.join([shape] + lines))

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        return np.asarray(self.data)

    def to_csv(self, filepath, sep=',', na_rep='', transpose=True, **kwargs):
        """
        write LArray to a csv file
        """
        if transpose:
            self.df.to_csv(filepath, sep=sep, na_rep=na_rep, **kwargs)
        else:
            self.series.to_csv(filepath, sep=sep, na_rep=na_rep, header=True,
                               **kwargs)

    def to_hdf(self, filepath, key, *args, **kwargs):
        """
        write LArray to an HDF file at the specified name
        """
        self.df.to_hdf(filepath, key, *args, **kwargs)

    def to_excel(self, filepath, sheet_name='Sheet1', *args, **kwargs):
        """
        write LArray to an excel file in the specified sheet
        """
        self.df.to_excel(filepath, sheet_name, *args, **kwargs)

    #XXX: sep argument does not seem very useful
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
    #             #XXX: this will NOT work for unicode strings !
    #             sheetname = sheetname.translate(string.maketrans('[:]', '(-)'),
    #                                             r'\/?*') # chars to delete
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

    def to_clipboard(self, *args, **kwargs):
        self.df.to_clipboard(*args, **kwargs)

    def plot(self, *args, **kwargs):
        self.df.plot(*args, **kwargs)


class NumpyLArray(LArray):
    def reshape(self, target_axes):
        """
        self.size must be equal to prod([len(axis) for axis in target_axes])
        """
        data = np.asarray(self).reshape([len(axis) for axis in target_axes])
        return LArray(data, target_axes)


class PandasLArray(LArray):
    def _wrap_pandas(self, res_data):
        if isinstance(res_data, pd.DataFrame):
            res_type = DataFrameLArray
        elif isinstance(res_data, pd.Series):
            res_type = SeriesLArray
        else:
            assert np.isscalar(res_data)
            return res_data
        return res_type(res_data)

    @property
    def size(self):
        return self.data.size

    @property
    def item(self):
        return self.data.item

    def copy(self):
        return self._wrap_pandas(self.data.copy())

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        return np.asarray(self.data)

    def _translate_axis_key(self, axis, key):
        # we do not use axis.translate because we have to let Pandas do the
        # label -> position conversion
        if key in axis:
            return key

        if isinstance(key, ValueGroup):
            key = key.key

        return to_key(key)

    #XXX: we only need axes length, so we might want to move this out of the
    # class
    # def translated_key(self, key):
    #     return tuple(axis.translate(axis_key)
    #                  for axis, axis_key in zip(self.axes, key))
    def translated_key(self, key):
        """
        translate ValueGroups to lists
        """
        return tuple(self._translate_axis_key(axis, k)
                     for axis, k in zip(self.axes, key))

    def _df_axis_level(self, axis):
        axis_idx = self.axes.index(axis)
        index_ndim = self._df_index_ndim
        if axis_idx < index_ndim:
            return 0, axis_idx
        else:
            return 1, axis_idx - index_ndim

    @property
    def _df_index_ndim(self):
        return len(self.data.index.names)

    def _group_aggregate(self, op_name, items):
        res = self

        # we cannot use Pandas groupby functionality because it is only meant
        # for disjoint groups, and we need to support a "row" being in several
        # groups.

        #TODO: when working with several "axes" at the same times, we should
        # not produce the intermediary result at all. It should be faster and
        # consume a bit less memory.
        for item in items:
            if isinstance(item, ValueGroup):
                axis, groups = item.axis, item
            else:
                axis, groups = item
            groups = to_keys(groups)
            axis, axis_idx = res.get_axis(axis, idx=True)

            if not isinstance(groups, tuple):
                # groups is in fact a single group
                assert isinstance(groups, (basestring, slice, list,
                                           ValueGroup)), type(groups)
                if isinstance(groups, list):
                    assert len(groups) > 0

                    # Make sure this is actually a single group, not multiple
                    # mistakenly given as a list instead of a tuple
                    assert all(not isinstance(g, (tuple, list)) for g in groups)

                groups = (groups,)

                # it is easier to kill the axis after the fact
                killaxis = True
            else:
                killaxis = False

            results = []
            for group in groups:
                # we need only lists of ticks, not single ticks, otherwise the
                # dimension is discarded too early (in __getitem__ instead of in
                # the aggregate func)
                group = [group] if group in axis else group

                # We do NOT modify the axis name (eg append "_agg" or "*") even
                # though this creates a new axis that is independent from the
                # original one because the original name is what users will
                # want to use to access that axis (eg in .filter kwargs)
                #TODO: we should bypass wrapping the result in DataFrameLArray
                arr = res.__getitem__({axis.name: group}, collapse_slices=True)
                result = arr._axis_aggregate(op_name, [axis])
                del arr
                results.append(result.data)

            if killaxis:
                assert len(results) == 1
                res_data = results[0]
            else:
                groups = to_ticks(groups)
                df_axis, df_level = self._df_axis_level(axis)
                res_data = pd.concat(results, axis=df_axis, keys=groups,
                                     names=[axis.name])
                # workaround a bug in Pandas (names ignored when one result)
                if len(results) == 1 and df_axis == 1:
                    res_data.columns.name = axis.name

                if df_level != 0:
                    # move the new axis to the correct place
                    levels = list(range(1, self._df_axis_nlevels(df_axis)))
                    levels.insert(df_level, 0)
                    # Series.reorder_levels does not support axis argument
                    kwargs = {'axis': df_axis} if df_axis else {}

                    # reordering levels is quite cheap (it creates a new
                    # index but the data itself is not copied)
                    res_data = res_data.reorder_levels(levels, **kwargs)

                    # sort using index levels order (to make index lexsorted)
                    #XXX: this is expensive, but I am not sure it can be
                    # avoided. Maybe only reorder_levels + sortlevel() after
                    # the loop? Not sure whether we can afford to temporarily
                    # loose sync between axes order and level orders?
                    res_data = _sort_level_inplace(res_data)

            res = self._wrap_pandas(res_data)
        return res

    def __str__(self):
        return str(self.data)
        # if not self.ndim:
        #     return str(np.asscalar(self))
        # elif not len(self):
        #     return 'LArray([])'
        # else:
        #     s = table2str(list(self.as_table()), 'nan', True,
        #                   keepcols=self.ndim - 1)
        #     return '\n' + s + '\n'

    __repr__ = __str__

    # element-wise method factory
    def _binop(opname):
        # fill_values = {
        #     'add': 0, 'radd': 0, 'sub': 0, 'rsub': 0,
        #     'mul': 1, 'rmul': 1, 'div': 1, 'rdiv': 1
        # }
        # fill_value = fill_values.get(opname)
        def opmethod(self, other):
            pandas_method = getattr(self.data.__class__, opname)
            if isinstance(other, PandasLArray):
                axis, level, (self_al, other_al) = _pandas_align(self.data,
                                                                 other.data,
                                                                 join='left')
                res_data = pandas_method(self_al, other_al, axis=axis,
                                         level=level)
                return self._wrap_pandas(res_data)
            elif isinstance(other, LArray):
                raise NotImplementedError("mixed LArrays")
            elif isinstance(other, np.ndarray):
                # XXX: not sure how clever Pandas is. We should be able to
                # handle extra/missing axes of length 1 (that is why I
                # separated the ndarray and scalar cases)
                res_data = pandas_method(self.data, other)
                return self._wrap_pandas(res_data)
            elif np.isscalar(other):
                res_data = pandas_method(self.data, other)
                return self._wrap_pandas(res_data)
            else:
                raise TypeError("unsupported operand type(s) for %s: '%s' "
                                "and '%s'" % (opname, type(self), type(other)))

        opmethod.__name__ = '__%s__' % opname
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
    # __divmod__ = _binop('divmod')
    # __rdivmod__ = _binop('rdivmod')
    __pow__ = _binop('pow')
    __rpow__ = _binop('rpow')
    # __lshift__ = _binop('lshift')
    # __rlshift__ = _binop('rlshift')
    # __rshift__ = _binop('rshift')
    # __rrshift__ = _binop('rrshift')
    # __and__ = _binop('and')
    # __rand__ = _binop('rand')
    # __xor__ = _binop('xor')
    # __rxor__ = _binop('rxor')
    # __or__ = _binop('or')
    # __ror__ = _binop('ror')

    # element-wise method factory
    def _unaryop(opname):
        def opmethod(self):
            pandas_method = getattr(self.data.__class__, opname)
            return self._wrap_pandas(pandas_method(self.data))
        opmethod.__name__ = '__%s__' % opname
        return opmethod

    # unary ops do not need broadcasting so do not need to be overridden
    # __neg__ = _unaryop('neg')
    # __pos__ = _unaryop('pos')
    __abs__ = _unaryop('abs')
    # __invert__ = _unaryop('invert')

    def _transpose(self, ncoldims, *args):
        """
        reorder axes
        accepts either a tuple of axes specs or axes specs as *args
        produces a copy if axes are not exactly the same (on Pandas)
        """
        assert 0 <= ncoldims <= len(self.axes)
        # all in columns is equivalent to none (we get a Series)
        ncoldims = ncoldims if ncoldims != len(self.axes) else 0
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            axes = args[0]
        else:
            axes = args

        if len(axes) == 0:
            axes = self.axes[::-1]

        axes = [self.get_axis(a) for a in axes]
        axes_specified = set(axis.name for axis in axes)
        missing_axes = [axis for axis in self.axes
                        if axis.name not in axes_specified]
        res_axes = axes + missing_axes
        res_axes = [a.name for a in res_axes]

        nrowdims = len(res_axes) - ncoldims
        res_data = _pandas_transpose_any(self.data, res_axes[:nrowdims],
                                         res_axes[nrowdims:])
        return self._wrap_pandas(res_data)

    def append(self, **kwargs):
        label = kwargs.pop('label', None)
        # It does not make sense to accept multiple axes at once, as "values"
        # will not have the correct shape for all axes after the first one.
        #XXX: Knowing that, it might be better to use a required (non kw) axis
        # argument, but it would be inconsistent with filter and sum.
        # It would look like: la.append(lipro, la.sum(lipro), label='sum')
        if len(kwargs) > 1:
            raise ValueError("Cannot append to several axes at the same time")
        axis_name, values = list(kwargs.items())[0]
        axis, axis_idx = self.get_axis(axis_name, idx=True)

        #TODO: add support for "raw" ndarrays (of the correct shape or
        # missing length-one dimensions)
        pd_values = values.data
        if axis_idx < self._df_index_ndim:
            expanded_value = _pandas_insert_index_level(pd_values, axis_name,
                                                        label, axis_idx)
        else:
            #FIXME: this is likely bogus (same code than other if branch)
            expanded_value = _pandas_insert_index_level(pd_values, axis_name,
                                                        label, axis_idx)
        expanded_value = self._wrap_pandas(expanded_value)
        return self.extend(axis, expanded_value)

    def extend(self, axis, other):
        axis, axis_idx = self.get_axis(axis, idx=True)

        # Get axis by name, so that we do *NOT* check they are "compatible",
        # because it makes sense to append axes of different length
        other_axis = other.get_axis(axis)

        # TODO: also "broadcast" (handle missing dimensions) other to self
        transposed_value = _pandas_transpose_any_like(other.data, self.data,
                                                      sort=False)
        # do we append on an index level?
        pd_axis = 0 if axis_idx < self._df_index_ndim else 1

        # using concat is a bit faster than combine_first (and we need
        # to reindex/sort anyway because combine_first does not always
        # give use the ordering we want).
        # when appending on columns, this is slower for 1 column than
        # data.copy(); data[label] = values
        # it fails (forget some level names) when transposed_value has not
        # the same index order
        result = pd.concat((self.data, transposed_value), axis=pd_axis)

        if axis_idx < self._df_index_ndim:
            idx = self.data.index

            #TODO: assert value has not already a "level" level
            if isinstance(idx, pd.MultiIndex):
                # Index.append() only works with a single value or an Index
                newlabels = pd.Index(other_axis.labels)
                neworders = [level if i != axis_idx
                             else level.append(newlabels)
                             for i, level in enumerate(idx.levels)]
                for i, neworder in enumerate(neworders):
                    result = result.reindex(neworder, level=i)

        return self._wrap_pandas(result)

    def _axis_aggregate(self, op_name, axes=()):
        """
        op is an aggregate function: func(arr, axis=(0, 1))
        axes is a tuple of axes (Axis objects or integers)
        """
        data = self.data
        if not axes:
            axes = self.axes
        else:
            # axes can be an iterator
            axes = tuple(axes)

        # first x second x third \ fourth
        # sum(first) -> x.sum(axis=0, level=[1, 2])
        # sum(second) -> x.sum(axis=0, level=[0, 2])
        # sum(third) -> x.sum(axis=0, level=[0, 1])
        # sum(fourth) -> x.sum(axis=1)

        # sum(first, second) -> x.sum(axis=0, level=2)
        # sum(second, third) -> x.sum(axis=0, level=0)
        # sum(first, third) -> x.sum(axis=0, level=1)

        # sum(first, second, third) -> x.sum(axis=0)

        # sum(third, fourth) -> x.sum(axis=0, level=[0, 1]).sum(axis=1)
        # axis=1 first is faster
        # sum(first, second, fourth) -> x.sum(axis=1).sum(level=2)

        # sum(first, second, third, fourth) -> x.sum(axis=0).sum()
        # axis=0 first is faster
        # sum(first, second, third, fourth) -> x.sum(axis=1).sum()

        dfaxes = [self._df_axis_level(axis) for axis in axes]
        all_axis0_levels = list(range(self._df_index_ndim))
        colnames = data.columns.names if isinstance(data, pd.DataFrame) else ()
        all_axis1_levels = list(range(len(colnames)))
        axis0_levels = [level for dfaxis, level in dfaxes if dfaxis == 0]
        axis1_levels = [level for dfaxis, level in dfaxes if dfaxis == 1]

        shift_axis1 = False
        res_data = data
        if axis0_levels:
            levels_left = set(all_axis0_levels) - set(axis0_levels)
            kwargs = {'level': sorted(levels_left)} if levels_left else {}
            res_data = getattr(res_data, op_name)(axis=0, **kwargs)
            if not levels_left:
                assert isinstance(res_data, pd.Series) or np.isscalar(res_data)
                shift_axis1 = True

        if axis1_levels:
            if shift_axis1:
                axis_num = 0
            else:
                axis_num = 1
            levels_left = set(all_axis1_levels) - set(axis1_levels)
            kwargs = {'level': sorted(levels_left)} if levels_left else {}
            res_data = getattr(res_data, op_name)(axis=axis_num, **kwargs)

        return self._wrap_pandas(res_data)

    def split_tuple(self, full_tuple):
        """
        splits a tuple with one value per axis to two tuples corresponding to
        the DataFrame axes
        """
        index_ndim = self._df_index_ndim
        return full_tuple[:index_ndim], full_tuple[index_ndim:]

    def split_key(self, full_key):
        """
        splits an LArray key with all axes to a key with two axes
        """
        a0_key, a1_key = self.split_tuple(full_key)
        # avoid producing length-1 tuples (it confuses Pandas)
        a0_key = a0_key[0] if len(a0_key) == 1 else a0_key
        a1_key = a1_key[0] if len(a1_key) == 1 else a1_key
        return a0_key, a1_key

    def __getitem__(self, key, collapse_slices=False):
        data = self.data
        if isinstance(key, (np.ndarray, LArray)) and \
                np.issubdtype(key.dtype, bool):
            # XXX: would it be better to return an LArray with Axis labels =
            # combined ticks where the "filter" (key) is True
            # these combined ticks should be objects which display as:
            # (axis1_label, axis2_label, ...) but should also store the axis
            # (names). Should it be the same object as the NDValueGroup?/NDKey?
            if isinstance(key, PandasLArray):
                key = key.data
            return self._wrap_pandas(data[key])

        translated_key = self.translated_key(self.full_key(key))
        a0_key, a1_key = self.split_key(translated_key)
        if isinstance(data, pd.DataFrame):
            res_data = data.loc[a0_key, a1_key]
        else:
            assert not a1_key
            res_data = data.loc[a0_key]

        #XXX: I wish I could avoid doing this manually. For some reason,
        # df.loc['a'] kills the level but both df.loc[('a', slice(None)), :]
        # and (for other levels) df.loc(axis=0)[:, 'b'] leave the level
        def mishandled_by_pandas(key):
            return isinstance(key, tuple) and any(isinstance(k, slice)
                                                  for k in key)

        a0_axes, a1_axes = self.split_tuple(self.axes)
        if mishandled_by_pandas(a0_key):
            a0_tokill = [axis.name for axis, k in zip(a0_axes, a0_key)
                         if k in axis]
            res_data.index = res_data.index.droplevel(a0_tokill)

        if a1_key and mishandled_by_pandas(a1_key):
            a1_tokill = [axis.name for axis, k in zip(a1_axes, a1_key)
                         if k in axis]
            res_data.columns = res_data.columns.droplevel(a1_tokill)

        return self._wrap_pandas(res_data)

    def __setitem__(self, key, value, collapse_slices=True):
        data = self.data

        if isinstance(key, (np.ndarray, LArray)) and \
                np.issubdtype(key.dtype, bool):
            if isinstance(key, PandasLArray):
                #TODO: broadcast/transpose key
                # key = key.broadcast_with(self.axes)
                key = key.data
            data[key] = value
            return

        translated_key = self.translated_key(self.full_key(key))
        a0_key, a1_key = self.split_key(translated_key)
        if isinstance(value, PandasLArray):
            value = value.data

        #FIXME: only do this if we *need* to broadcast
        if isinstance(data.index, pd.MultiIndex) and \
                isinstance(value, (pd.Series, pd.DataFrame)):
            # this is how Pandas works internally. Ugly (locs are bool arrays.
            # Ugh!)
            a0_locs = data.index.get_locs(a0_key)
            if isinstance(data, pd.DataFrame):
                a1_locs = a1_key if a1_key == slice(None) \
                    else data.columns.get_locs(a1_key)
                target_columns = data.columns[a1_locs]

            # data.iloc[(a0_locs, a1_locs)] = ...
            target_index = data.index[a0_locs]

            # broadcast to the index so that we do not need to create the target
            # slice
            #TODO: also broadcast columns
            value = _pandas_broadcast_to(value, target_index)
        elif isinstance(value, (np.ndarray, list)):
            a0size = data.index.get_locs(a0_key).sum()
            if isinstance(data, pd.DataFrame):
                a1size = len(data.columns) if a1_key == slice(None) \
                    else data.columns.get_locs(a1_key).sum()
                target_shape = (a0size, a1size)
            else:
                target_shape = (a0size,)
            vsize = value.size if isinstance(value, np.ndarray) else len(value)
            if vsize == np.prod(target_shape):
                value = np.asarray(value).reshape(target_shape)

        if isinstance(data, pd.DataFrame):
            data.loc[a0_key, a1_key] = value
        else:
            assert not a1_key
            data.loc[a0_key] = value



class SeriesLArray(PandasLArray):
    def __init__(self, data, axes=None):
        if isinstance(data, np.ndarray):
            axes = AxisCollection(axes)
            #XXX: add a property "labels" on AxisCollection?
            if len(axes) > 1:
                idx = multi_index_from_product([axis.labels for axis in axes],
                                               names=axes.names,
                                               sortvalues=False)
            else:
                idx = pd.Index(axes[0].labels, name=axes[0].name)
            array = data.reshape(prod(axes.shape))
            data = pd.Series(array, idx)
        elif isinstance(data, pd.Series):
            if isinstance(data.index, pd.MultiIndex) and \
                    not data.index.is_lexsorted():
                data = data.sortlevel()
            #TODO: accept axes argument and check that it is consistent
            # or possibly even override data in Series?
            assert axes is None
            assert all(name is not None for name in data.index.names)
            axes = [Axis(name, labels) for name, labels in _df_levels(data, 0)]
        else:
            raise TypeError("data must be an numpy ndarray or pandas.Series")

        LArray.__init__(self, data, axes)

    @property
    def dtype(self):
        return self.data.dtype

    def _df_axis_nlevels(self, df_axis):
        assert df_axis == 0
        return len(self.data.index.names)

    def transpose(self, *args):
        """
        reorder axes
        accepts either a tuple of axes specs or axes specs as *args
        produces a copy if axes are not exactly the same (on Pandas)
        """
        return self._transpose(0, *args)


#TODO: factorize with df_labels
def _df_levels(df, axis):
    idx = df.index if axis == 0 else df.columns
    if isinstance(idx, pd.MultiIndex):
        return [(name, _index_level_unique_labels(idx, name))
                for name in idx.names]
    else:
        assert isinstance(idx, pd.Index)
        # not sure the unique() is really useful here
        return [(idx.name, idx.unique())]


class MixedDtype(dict):
    def __init__(self, dtypes):
        dict.__init__(self, dtypes)


class DataFrameLArray(PandasLArray):
    def __init__(self, data, axes=None):
        """
        data should be a DataFrame with a (potentially)MultiIndex set for rows
        """
        if isinstance(data, np.ndarray):
            axes = AxisCollection(axes)
            #XXX: add a property "labels" on AxisCollection?
            if len(axes) > 2:
                idx = multi_index_from_product([axis.labels for axis in axes[:-1]],
                                               names=axes.names[:-1],
                                               sortvalues=False)
            elif len(axes) == 2:
                idx = pd.Index(axes[0].labels, name=axes[0].name)
            else:
                raise ValueError("need at least 2 axes")
            array = data.reshape(prod(axes.shape[:-1]), axes.shape[-1])
            columns = pd.Index(axes[-1].labels, name=axes[-1].name)
            data = pd.DataFrame(array, idx, columns)
        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex) and \
                    not data.index.is_lexsorted():
                # let us be well behaved and not do it inplace even though that
                # would be more efficient
                data = data.sortlevel()
            #TODO: accept axes argument and check that it is consistent
            # or possibly even override data in DataFrame?
            assert axes is None
            axes = [Axis(name, labels)
                    for name, labels in _df_levels(data, 0) + _df_levels(data, 1)]
        else:
            raise TypeError("data must be an numpy ndarray or pandas.DataFrame")

        LArray.__init__(self, data, axes)

    @property
    def df(self):
        idx = self.data.index.copy()
        names = idx.names
        idx.names = names[:-1] + [names[-1] + '\\' + self.data.columns.name]
        return pd.DataFrame(self.data, idx)

    @property
    def series(self):
        return self.data.stack()

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

        # 1) append length-1 axes for axes in target but not in source (I do not
        #    think their position matters).
        array = self.reshape(list(self.axes) +
                             [Axis(name, ['*']) for name in target_names
                              if name not in self.axes])
        # 2) reorder axes to target order (move source only axes to the front)
        sourceonly_axes = [axis for axis in self.axes
                           if axis.name not in target_axes]
        other_axes = [self.axes.get(name, Axis(name, ['*']))
                      for name in target_names]
        return array.transpose(sourceonly_axes + other_axes)

    def _df_axis_nlevels(self, df_axis):
        idx = self.data.index if df_axis == 0 else self.data.columns
        return len(idx.names)

    # def transpose(self, *args, ncoldims=1):
    def transpose(self, *args, **kwargs):
        """
        reorder axes
        accepts either a tuple of axes specs or axes specs as *args
        ncoldims: number of trailing dimensions to use as columns (default 1)
        produces a copy if axes are not exactly the same (on Pandas)
        """
        ncoldims = kwargs.pop('ncoldims', 1)
        return self._transpose(ncoldims, *args)

    @property
    def dtype(self):
        dtypes = self.data.dtypes
        if all(dtypes == dtypes[0]):
            return dtypes[0]
        else:
            return MixedDtype(dtypes.to_dict())

    __array_priority__ = 100


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
        return [_index_level_unique_labels(idx, l) for l in idx.names]
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


#TODO: implement sort_columns
def df_aslarray(df, sort_rows=True, sort_columns=True, **kwargs):
    axes_names = [decode(name, 'utf8') for name in df.index.names]
    if axes_names == [None]:
        last_axis = None, None
    else:
        last_axis = axes_names[-1].split('\\')
    axes_names[-1] = last_axis[0]
    #FIXME: hardcoded "time"
    axes_names.append(last_axis[1] if len(last_axis) > 1 else 'time')

    # pandas treats the "time" labels as column names (strings) so we need
    # to convert them to values
    column_labels = [parse(cell) for cell in df.columns.values]

    #FIXME: do not modify original DataFrame !
    df.index.names = axes_names[:-1]
    df.columns = column_labels
    df.columns.name = axes_names[-1]

    return DataFrameLArray(df)


def read_csv(filepath, nb_index=0, index_col=[], sep=',', headersep=None,
             na=np.nan, sort_rows=False, sort_columns=True, **kwargs):
    """
    reads csv file and returns an Larray with the contents
        nb_index: number of leading index columns (ex. 4)
    or
        index_col : list of columns for the index (ex. [0, 1, 2, 3])

    when sort_rows is False, LArray tries to produce a global order of labels
    from all partial orders.

    format csv file:
    arr,ages,sex,nat\time,1991,1992,1993
    A1,BI,H,BE,1,0,0
    A1,BI,H,FO,2,0,0
    A1,BI,F,BE,0,0,1
    A1,BI,F,FO,0,0,0
    A1,A0,H,BE,0,0,0

    """
    # TODO
    # * make sure sort_rows=True works
    # * implement sort_rows='firstseen' (this is what index.factorize does)
    # * for "dense" arrays, this should result in the same thing as
    #   sort_rows=True/"partial"

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

    if not sort_rows or headersep is not None:
        # we will set the index later
        orig_index_col = index_col
        index_col = None

    # force str for dimensions
    # because pandas autodetect failed (thought it was int when it was a string)
    dtype = {}
    for axis in axes_names[:nb_index]:
        dtype[axis] = np.str
    df = pd.read_csv(filepath, index_col=index_col, sep=sep, dtype=dtype,
                     **kwargs)
    if not sort_rows:
        set_topological_index(df, orig_index_col, inplace=True)
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
    """
    read an LArray from a tsv file
    """
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
    """
    return read_csv(filepath, sep='\t', headersep=',', **kwargs)


def read_hdf(filepath, key, sort_rows=True, sort_columns=True, **kwargs):
    """
    read an LArray from a h5 file with the specified name
    """
    df = pd.read_hdf(filepath, key, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns)


def read_excel(filepath, sheetname=0, nb_index=0, index_col=[],
               na=np.nan, sort_rows=True, sort_columns=True, **kwargs):
    """
    reads excel file from sheet name and returns an Larray with the contents
        nb_index: number of leading index columns (ex. 4)
    or
        index_col : list of columns for the index (ex. [0, 1, 2, 3])
    """
    if len(index_col) == 0:
        index_col = list(range(nb_index))
    df = pd.read_excel(filepath, sheetname, index_col=index_col, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=na)


def zeros(axes, cls=LArray):
    axes = AxisCollection(axes)
    return cls(np.zeros(axes.shape), axes)


def zeros_like(array, cls=None):
    """
    :param cls: use same than source by default
    """
    return zeros(array.axes, cls=array.__class__ if cls is None else cls)


def empty(axes, cls=LArray):
    axes = AxisCollection(axes)
    return cls(np.empty(axes.shape), axes)


def ndrange(axes, cls=LArray):
    """
    :param axes: either a collection of axes or a shape
    """
    axes = AxisCollection(axes)
    return cls(np.arange(prod(axes.shape)).reshape(axes.shape), axes)
