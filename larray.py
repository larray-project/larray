from __future__ import division, print_function
# -*- coding: utf8 -*-
"""
Matrix class
"""
#TODO
# * implement named groups in strings
#   eg "vla=A01,A02;bru=A21;wal=A55,A56"

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
# * smarter str() for large arrays
# * fix str() for 1D LArray
# * int labels
# * avg on last 10 years
#     time = Axis('time', ...)
#     x = time[-10:]  # <- does not work!
    # la[time[-10:]].avg(time)
    # la.append(la.avg(time[-10:]), axis=time)
    # la.append(time=la.avg(time[-10:]))
    # la.append(time=la.avg(time='-10:'))

# * reshape

# la.append(la.avg(time[-10:]), axis=time)

# la.avg(time[-10:])

# * drop last year
#   la = la[:,:,:,:,time[:-1]]
#   la = la.filter(time[:-1]) # <- implement this !
#   (equal to "la = la.filter(time=time[:-1])")
#   la = la.filter(geo='A25,A11')
#   la = la.filter(geo['A25,A11'])

# also for __getitem__
#   la = la[time[:-1]] # <- implement this !
#
# * split unit tests


# * easily add sum column for a dimension
#   - in all cases, we will need to define a new Axis object
#   - in the examples below, we suppose a.label is 'income'
#   - best candidates (IMO)
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
# * modify read_csv format (last_column / time)
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
#       >>> I think this usage is unlikely to be used by users directly but might
#     - treat them like a subset of values to include in the cartesian product
#       eg, supposing we have a array of shape (bool[2], int[110], bool[2])
#       the key ([False], [1, 5, 9], [False, True]) would return an array
#       of shape [1, 3, 2]
#     OR
#     - treat them like values to lookup (len(key) has not relation with len(dim)
#       BUT if key is a tuple (nd-key), we have len(dim0) == dim(dimX)
# * evaluate the impact of label-only __getitem__: numpy/matplotlib/...
#   functions probably rely on __getitem__ with indices

# * docstring for all methods
# * choose between subset and group. Having both is just confusing, I think.
# * check whether we could use np.array_repr/array_str (and
#   np.set_printoptions) instead of our own as_table/table2str
# * IO functions: csv/hdf/excel?/...?
#   >> needs discussion of the formats (users involved in the discussion?)
#      + check pandas dialects
# * better info()
#   ? make info a property?
#   * only display X label ticks by default (with an argument to display all)
#     eg 'A11' ... 'A93'
# * __setitem__
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
# * check Collapse: is this needed? can't we generalize it?
# * data alignment in arithmetic methods (or at least check that axes are
#   compatible and raise an exception if they are not)
# * test structured arrays
# * review all method & argument names
# * implement ValueGroup.__getitem__
# ? allow __getitem__ with ValueGroups at any position since we know
#   which axis the ValueGroup correspond to. ie: allow bel[vla] even though
#   geo is not the first dimension of bel.
# ? move "utils" to its own project (so that it is not duplicated between
#   larray and liam2)
#   OR
#   include utils only in larray project and make larray a dependency of liam2
#   (and potentially rename it to reflect the broader scope)
# ? move "excelcom" to its own project (so that it is not duplicated between
#   potential projects using it)

# ? make pywin32 optional?
# ? implement dict-like behavior for LArray.axes (to be able to do stuff like
#   la.axes['sex'].labels
#

from itertools import product, chain
import string
import sys

import numpy as np
import pandas as pd

import tables

from utils import (prod, table2str, table2csv, table2iode, timed, unique,
                   array_equal)


def srange(*args):
    return map(str, range(*args))


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


def to_label(e):
    """
    make it hashable
    """
    if isinstance(e, list):
        return tuple(e)
    elif isinstance(e, basestring):
        return e
    elif isinstance(e, slice):
        return slice_to_str(e)
    else:
        print('to_label not implemented for', e)
        raise NotImplementedError


def to_labels(s):
    """
    converts a string to a list of values, usable in Axis
    """
    if isinstance(s, (list, tuple)):
        return [to_label(e) for e in s]

    numcolons = s.count(':')
    if numcolons:
        assert numcolons <= 2
        fullstr = s + ':1' if numcolons == 1 else s
        start, stop, step = [int(a) if a else None for a in fullstr.split(':')]
        if start is None:
            start = 0
        if stop is None:
            raise ValueError("no stop bound provided in range: %s" % s)
        stop += 1
        return srange(start, stop, step)
    else:
        return [v.strip() for v in s.split(',')]


def to_key(s):
    #FIXME: it does not accept only strings
    """
    converts a string to a structure usable as a key (slice objects, etc.)
    This is used, for example in .filter: arr.filter(axis=key) or in .sum:
    arr.sum(axis=key) or arr.sum(axis=(key, key, key))
    colons (:) are translated to slice objects, but "int strings" are not
    converted to int.
    leading and trailing commas are stripped.
    >>> to_key('a:c')
    slice('a', 'c', None)
    >>> to_key('a,b,c')
    ['a', 'b', 'c']
    >>> to_key('a,')
    ['a']
    >>> to_key('a')
    'a'
    >>> to_key(10)
    10
    """
    if isinstance(s, tuple):
        return list(s)
    elif not isinstance(s, basestring):
        return s

    numcolons = s.count(':')
    if numcolons:
        assert numcolons <= 2
        # can be of len 2 or 3 (if step is provided)
        bounds = [a if a else None for a in s.split(':')]
        return slice(*bounds)
    else:
        if ',' in s:
            # strip extremity commas to avoid empty string keys
            s = s.strip(',')
            return [v.strip() for v in s.split(',')]
        else:
            return s.strip()


def to_keys(s):
    # FIXME: fix doc: it does not accept only strings
    """
    converts a "family string" to its corresponding structure.
    It is only used for .sum(axis=xxx)
    'label' or ['l1', 'l2'] or [['l1', 'l2'], ['l3']]
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
    if isinstance(s, basestring):
        if ';' in s:
            return tuple([to_key(group) for group in s.split(';')])
        else:
            # a single group => collapse dimension
            return to_key(s)
    elif isinstance(s, ValueGroup):
        return s
    elif isinstance(s, list):
        return to_key(s)
    else:
        assert isinstance(s, tuple)
        return tuple([to_key(group) for group in s])


def union(*args):
    #TODO: add support for ValueGroup and lists
    """
    returns the union of several "value strings" as a list
    """
    if args:
        return list(unique(chain(*(to_labels(arg) for arg in args))))
    else:
        return []


def strip_chars(s, chars):
    if isinstance(s, unicode):
        return s.translate({ord(c): u'' for c in chars})
    else:
        return s.translate(None, ''.join(chars))


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
        if isinstance(labels, basestring):
            labels = to_labels(labels)
        self.labels = np.asarray(labels)
        self._mapping = {label: i for i, label in enumerate(labels)}

    @property
    def parent_axis(self):
        label0 = self.labels[0]
        return label0.axis if isinstance(label0, ValueGroup) else None

    @property
    def is_aggregated(self):
        return self.parent_axis is not None

    def group(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        if kwargs:
            raise ValueError("invalid keyword argument(s): %s" % kwargs.keys())
        key = args[0] if len(args) == 1 else args
        return ValueGroup(self, key, name)

    def subset(self, key, name=None):
        """
        key is a label-based key (slice and fancy indexing are supported)
        returns a ValueGroup usable in .sum or .filter
        """
        if isinstance(key, ValueGroup):
            if key.axis != self:
                raise ValueError("cannot subset an axis with a ValueGroup of "
                                 "an incompatible axis")
            return key
        return ValueGroup(self, key, name)

    def all(self, name=None):
        return self.subset(slice(None),
                           name=name if name is not None else "all")

    def subaxis(self, key, name=None):
        """
        key is an integer-based key (slice and fancy indexing are supported)
        returns an Axis for a sub-array
        """
        if (isinstance(key, slice) and
                key.start is None and key.stop is None and key.step is None):
            return self
        # we must NOT modify the axis name, even though this creates a new axis
        # that is independent from the original one because the original
        # name is probably what users will want to use to filter
        return Axis(self.name, self.labels[key])

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
        return self.subset(key)

    def __contains__(self, key):
        try:
            self.translate(key)
            return True
        except Exception:
            return False

    def translate(self, key):
        """
        translates a label key to its numerical index counterpart
        fancy index with boolean vectors are passed through unmodified
        """
        mapping = self._mapping

        if isinstance(key, basestring):
            key = to_key(key)
        elif isinstance(key, ValueGroup):
            if self.is_aggregated:
                # the array is an aggregate (it has ValueGroup keys in its
                # mapping) => return the index of the group
                return mapping[key]
            else:
                # the array is not an aggregate (it has normal label keys in its
                # mapping) => return the index of all the elements in the group
                if key.axis == self:
                    key = key.key
                else:
                    raise ValueError("group %s cannot be used on axis %s"
                                     % (key, self))

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
            assert np.isscalar(key), "%s (%s) is not scalar" % (key, type(key))
            # key is scalar (integer, float, string, ...)
            return mapping[key]

    def __str__(self):
        return self.name if self.name is not None else 'Unnamed axis'

    def __repr__(self):
        return 'Axis(%r, %r)' % (self.name, self.labels.tolist())


# We need a separate class for ValueGroup and cannot simply create a
# new Axis with a subset of values/ticks/labels: the subset of
# ticks/labels of the ValueGroup need to correspond to its *Axis*
# indices
class ValueGroup(object):
    def __init__(self, axis, key, name=None):
        """
        key should be either a sequence of labels, a slice with label bounds
        or a string
        """
        self.axis = axis
        if isinstance(key, basestring):
            key = to_key(key)
        elif isinstance(key, slice):
            pass
        elif isinstance(key, ValueGroup):
            pass
        else:
            # transform tuples and the like
            key = list(key)
        #TODO: valueGroups will very likely be used as "groups" so they should
        # cache the indices of their labels
        self.key = key

        # this is only meant the check the key is valid, later we might want
        # to cache the result to check that it does not change over time
        self.axis.translate(key)
        if name is None:
            if isinstance(key, slice):
                name = slice_to_str(key)
            elif isinstance(key, list):
                name = ','.join(str(k) for k in key)
            else:
                # key can be a ValueGroup or a string
                # assert isinstance(key, basestring)
                name = str(key)
        self.name = name

    def __hash__(self):
        key = self.key
        if isinstance(key, list):
            key = tuple(key)
        elif isinstance(key, slice):
            key = slice_to_str(key)
        return hash((self.axis, key))

    def __eq__(self, other):
        # two VG with different names but the same key compare equal
        return self.axis == other.axis and self.key == other.key

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s[%s]" % (self.axis.name, self.name)


class LArray(np.ndarray):
    def __new__(cls, data, axes=None):
        obj = np.asarray(data).view(cls)
        ndim = obj.ndim
        if axes is not None:
            if len(axes) != ndim:
                raise ValueError("number of axes (%d) does not match "
                                 "number of dimensions of data (%d)"
                                 % (len(axes), ndim))
            shape = tuple(len(axis) for axis in axes)
            if shape != obj.shape:
                raise ValueError("length of axes %s does not match "
                                 "data shape %s" % (shape, obj.shape))

        if axes is not None and not isinstance(axes, list):
            axes = list(axes)
        obj.axes = axes
        return obj
    
    def as_dataframe(self):
        axes_labels = [a.labels.tolist() for a in self.axes[:-1]]
        axes_names = [a.name for a in self.axes[:-1]]
        axes_names[-1] = axes_names[-1] + '\\' + self.axes[-1].name
        columns = self.axes[-1].labels.tolist()
        full_index=[i for i in product(*axes_labels)] 
        index = pd.MultiIndex.from_tuples(full_index, names=axes_names)
        df = pd.DataFrame(self.reshape(len(full_index), len(columns)), index, columns)
        return df

    #noinspection PyAttributeOutsideInit
    def __array_finalize__(self, obj):
        # We are in the middle of the LabeledArray.__new__ constructor,
        # and our special attributes will be set when we return to that
        # constructor, so we do not need to set them here.
        if obj is None:
            return

        # obj is our "template" object (on which we have asked a view on).
        if isinstance(obj, LArray) and self.shape == obj.shape:
            # obj.view(LArray)
            # larr[:3]
            self.axes = obj.axes
        else:
            self.axes = None
            #self.row_totals = None
            #self.col_totals = None

    @property
    def axes_names(self):
        return [axis.name for axis in self.axes]

    @property
    def is_aggregated(self):
        return any(axis.is_aggregated for axis in self.axes)

    def __getitem__(self, key, collapse_slices=False):
        data = np.asarray(self)

        # convert scalar keys to 1D keys
        if not isinstance(key, tuple):
            key = (key,)

        # expand string keys with commas
        #XXX: is it the right place to do this?
        key = tuple(to_key(axis_key) for axis_key in key)

        # convert xD keys to ND keys
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

        if self.is_aggregated:
            # convert values on aggregated axes to (value)groups on the
            # *parent* axis. The goal is to allow targeting a ValueGroup
            # label by a string. eg.
            # reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
            # we want all the following to work:
            #   reg[geo.group('A21', name='bru')]
            #   reg['A21']
            #   reg[:] -> all lines, and not the "belgium" line. It is not
            # ideal but it is the lesser evil, because
            # reg.filter(lipro='PO1,PO2') maps to reg[:, 'PO1,PO2'] and
            # it should return the whole "aggregated" geo dimension,
            # not one line only
            def convert(axis, values):
                if (axis.is_aggregated and not isinstance(values, ValueGroup)):
                    vg = axis.parent_axis.group(values)
                    if vg in axis:
                        return vg
                    else:
                        return values
                else:
                    return values
            key = tuple(convert(axis, axis_key)
                        for axis, axis_key in zip(self.axes, key))

        # translate labels to integers
        translated_key = tuple(axis.translate(axis_key)
                               for axis, axis_key in zip(self.axes, key))

        # isinstance(ndarray, collections.Sequence) is False but it
        # behaves like one
        sequence = (tuple, list, np.ndarray)
        if collapse_slices:
            translated_key = [range_to_slice(axis_key)
                                  if isinstance(axis_key, sequence)
                                  else axis_key
                              for axis_key in translated_key]

        # count number of indexing arrays (ie non scalar/slices) in tuple
        num_ix_arrays = sum(isinstance(axis_key, sequence)
                            for axis_key in translated_key)
        num_scalars = sum(np.isscalar(axis_key) for axis_key in translated_key)

        # handle advanced indexing with more than one indexing array:
        # basic indexing (only integer and slices) and advanced indexing
        # with only one indexing array are handled fine by numpy
        if num_ix_arrays > 1 or (num_ix_arrays > 0 and num_scalars):
            # np.ix_ wants only lists so:

            # 1) kill scalar-key axes (if any) by indexing them (we cannot
            #    simply transform the scalars into lists of 1 element because
            #    in that case those dimensions are not dropped by
            #    ndarray.__getitem__)
            keyandaxes = zip(translated_key, self.axes)
            if any(np.isscalar(axis_key) for axis_key in translated_key):
                killscalarskey = tuple(axis_key
                                           if np.isscalar(axis_key)
                                           else slice(None)
                                       for axis_key in translated_key)
                data = data[killscalarskey]
                noscalar_keyandaxes = [(axis_key, axis)
                                        for axis_key, axis in keyandaxes
                                        if not np.isscalar(axis_key)]
            else:
                noscalar_keyandaxes = keyandaxes

            # 2) expand slices to lists (ranges)
            #TODO: cache the range in the axis?
            listkey = tuple(np.arange(*axis_key.indices(len(axis)))
                            if isinstance(axis_key, slice) else axis_key
                            for axis_key, axis in noscalar_keyandaxes)
            # np.ix_ computes the cross product of all lists
            full_key = np.ix_(*listkey)
        else:
            full_key = translated_key

        # it might be tempting to make subaxis take a label-based key but this
        # is more complicated as np.isscalar works better after translate
        # (eg for "aggregate tables" where ValueGroups are used as keys)
        axes = [axis.subaxis(axis_key)
                for axis, axis_key in zip(self.axes, translated_key)
                if not np.isscalar(axis_key)]
        return LArray(data[full_key], axes)

    # deprecated since Python 2.0 but we need to define it to catch "simple"
    # slices (with integer bounds !) because ndarray is a "builtin" type
    def __getslice__(self, i, j):
        return self[slice(i, j)]

    def __str__(self):
        # return str(self.shape)
        if not self.ndim:
            return str(np.asscalar(self))
        else:
            return '\n' + table2str(self.as_table(), 'nan', True) + '\n'
    __repr__ = __str__

    def as_table(self):
        if not self.ndim:
            return []
    
        #ert	| unit	| geo\time	| 2012 	| 2011 	| 2010 	
        #NEER27	| I05	| AT	| 101.41 	| 101.63 	| 101.63 	
        #NEER27	| I05	| AU	| 134.86 	| 125.29 	| 117.08 	

        width = self.shape[-1]
        height = prod(self.shape[:-1])
        if self.axes is not None:
            axes_names = self.axes_names
            if len(axes_names) > 1:
                axes_names[-2] = '\\'.join(axes_names[-2:])
                axes_names.pop()
                
            axes_labels = [axis.labels for axis in self.axes]
        else:
            axes_names = None
            axes_labels = None

        if axes_names is not None:
            result = [axes_names + list(axes_labels[-1])]
            #if self.row_totals is not None:
            #    result[0].append('')
            #    result[1].append('total')
        else:
            result = []
        data = np.asarray(self).ravel()
        if axes_labels is not None:
            categ_values = list(product(*axes_labels[:-1]))
        else:
            categ_values = [[] for y in range(height)]
        #row_totals = self.row_totals
        for y in range(height):
            line = list(categ_values[y]) + \
                   list(data[y * width:(y + 1) * width])
            #if row_totals is not None:
            #    line.append(row_totals[y])
            result.append(line)
        #if self.col_totals is not None and self.ndim > 1:
        #    result.append([''] * (self.ndim - 2) + ['total'] + self.col_totals)
        return result

    #XXX: should filter(geo=['W']) return a view by default? (collapse=True)
    # I think it would be dangerous to make it the default
    # behavior, because that would introduce a subtle difference between
    # filter(dim=[a, b]) and filter(dim=[a]) even though it would be faster
    # and uses less memory. Maybe I should have a "view" argument which
    # defaults to 'auto' (ie collapse by default), can be set to False to
    # force a copy and to True to raise an exception if a view is not possible.
    def filter(self, collapse=False, **kwargs):
        """
        filters the array along the axes given as keyword arguments.
        It is similar to np.take but works with several axes at once.
        """
        axes_names = set(self.axes_names)
        for kwarg in kwargs:
            if kwarg not in axes_names:
                raise KeyError("{} is not an axis name".format(kwarg))
        full_idx = tuple(kwargs[ax.name] if ax.name in kwargs else slice(None)
                         for ax in self.axes)
        return self.__getitem__(full_idx, collapse)

    def _axis_aggregate(self, op, axes):
        """
        op is an aggregate function: func(arr, axis=(0, 1))
        axes is a tuple of axes (Axis objects or integers)
        """
        src_data = np.asarray(self)
        if not axes:
            # scalars don't need to be wrapped in LArray
            return op(src_data)

        axes_indices = tuple(self.get_axis_idx(a) for a in axes)
        res_data = op(src_data, axis=axes_indices)
        axes_tokill = set(axes_indices)
        res_axes = [axis for axis_num, axis in enumerate(self.axes)
                    if axis_num not in axes_tokill]
        return LArray(res_data, res_axes)

    def get_axis_idx(self, axis):
        """
        axis can be an index, a name or an Axis object
        if the Axis object is from another LArray, get_axis_idx will return the
        index of the local axis with the same name, whether it is compatible
        (has the same ticks) or not.
        """
        name_or_idx = axis.name if isinstance(axis, Axis) else axis
        axis_names = [a.name for a in self.axes]
        return axis_names.index(name_or_idx) \
            if isinstance(name_or_idx, basestring) \
            else name_or_idx

    def get_axis(self, axis, idx=False):
        """
        axis can be an index, a name or an Axis object
        if the Axis object is from another LArray, get_axis will return the
        local axis with the same name, whether it is compatible (has the
        same ticks) or not.
        """
        axis_idx = self.get_axis_idx(axis)
        axis = self.axes[axis_idx]
        return (axis, axis_idx) if idx else axis

    def _group_aggregate(self, op, kwargs, commutative=False):
        if not commutative and len(kwargs) > 1:
            raise ValueError("grouping aggregates on multiple axes at the same "
                             "time is not supported for '%s' (because it is "
                             "not a commutative operation)" % op.func_name)

        res = self
        for agg_axis_name, groups in kwargs.iteritems():
            groups = to_keys(groups)

            agg_axis, agg_axis_idx = res.get_axis(agg_axis_name, idx=True)
            res_axes = res.axes[:]
            res_shape = list(res.shape)

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
                del res_axes[agg_axis_idx]

                # it is easier to kill the axis after the fact
                killaxis = True
            else:
                # make sure all groups are ValueGroup and use that as the axis
                # ticks
                #TODO: assert that if isinstance(g, ValueGroup):
                # g.axis == agg_axis (no conversion needed)
                # or g.axis == agg_axis.parent_axis (we are grouping groups)
                groups = tuple(agg_axis.group(g)
                                   if (not isinstance(g, ValueGroup) or
                                       g.axis != agg_axis)
                                   else g
                               for g in groups)
                assert all(vg.axis == agg_axis for vg in groups)

                # Make sure each (value)group is not a single-value group.
                # Groups with a list of one value are fine, we just want to
                # avoid the axis being discarded by the .filter() operation.
                groups = [ValueGroup(g.axis, [g.key], g.name)
                            if np.isscalar(g.key) else g
                          for g in groups]

                # We do NOT modify the axis name (eg append "_agg" or "*") even
                # though this creates a new axis that is independent from the
                # original one because the original name is what users will
                # want to use to access that axis (eg in .filter kwargs)
                res_axes[agg_axis_idx] = Axis(agg_axis.name, groups)
                killaxis = False

            res_shape[agg_axis_idx] = len(groups)
            res_data = np.empty(res_shape, dtype=res.dtype)

            group_idx = [slice(None) for _ in res_shape]
            for i, group in enumerate(groups):
                group_idx[agg_axis_idx] = i

                # we need only lists not single labels, otherwise the dimension
                # is discarded too early (in filter instead of in the
                # aggregate func)
                #XXX: shouldn't this be handled in to_keys?
                group = [group] if np.isscalar(group) else group

                # we don't reuse kwargs because we might have modified "groups"
                arr = res.filter(collapse=True, **{agg_axis_name: group})
                arr = np.asarray(arr)
                op(arr, axis=agg_axis_idx, out=res_data[group_idx])
                del arr
            if killaxis:
                assert group_idx[agg_axis_idx] == 0
                res_data = res_data[group_idx]
            res = LArray(res_data, res_axes)

        return res

    def _aggregate(self, op, args, kwargs, commutative=False):
        # op() without args is equal to op(all_axes)
        if args and kwargs:
            intermediate = self._axis_aggregate(op, axes=args)
            return intermediate._group_aggregate(op, kwargs, commutative)
        elif kwargs:
            return self._group_aggregate(op, kwargs, commutative)
        else:
            return self._axis_aggregate(op, axes=args)

    def copy(self):
        return LArray(np.ndarray.copy(self), axes=self.axes[:])
    
    def zeros_like(self):
        return LArray(np.zeros_like(np.asarray(self)), axes=self.axes[:])
    
    def info(self):
        axes_labels = [' '.join(repr(label) for label in axis.labels)
                       for axis in self.axes]
        lines = [" %s [%d]: %s" % (axis.name, len(axis), labels)
                 for axis, labels in zip(self.axes, axes_labels)]
        return ("%s\n" % str(self.shape)) + '\n'.join(lines)

    def ratio(self, *axes):
        if not axes:
            axes = self.axes
        return np.nan_to_num(self / self.sum(*axes))

    # aggregate method factory
    def agg_method(npfunc, name=None, commutative=False):
        def method(self, *args, **kwargs):
            return self._aggregate(npfunc, args, kwargs,
                                   commutative=commutative)
        if name is None:
            name = npfunc.__name__
        method.__name__ = name
        return method

    all = agg_method(np.all, commutative=True)
    any = agg_method(np.any, commutative=True)
    # commutative modulo float precision errors
    sum = agg_method(np.sum, commutative=True)
    prod = agg_method(np.prod, commutative=True)
    cumsum = agg_method(np.cumsum, commutative=True)
    cumprod = agg_method(np.cumprod, commutative=True)
    min = agg_method(np.min, commutative=True)
    max = agg_method(np.max, commutative=True)
    mean = agg_method(np.mean, commutative=True)
    # not commutative
    ptp = agg_method(np.ptp)
    var = agg_method(np.var)
    std = agg_method(np.std)

    def append(self, **kwargs):
        label = kwargs.pop('label', None)
        # It does not make sense to accept multiple axes at once, as "values"
        # will not have the correct shape for all axes after the first one.
        #XXX: Knowing that, it might be better to use a required (non kw) axis
        # argument, but it would be inconsistent with filter and sum.
        # It would look like: la.append(lipro, la.sum(lipro), label='sum')
        if len(kwargs) > 1:
            raise ValueError("Cannot append to several axes at the same time")
        axis_name, values = kwargs.items()[0]
        axis, axis_idx = self.get_axis(axis_name, idx=True)
        shape = self.shape
        values = np.asarray(values)
        if values.shape == shape[:axis_idx] + shape[axis_idx+1:]:
            # adding a dimension of size one if it is missing
            new_shape = shape[:axis_idx] + (1,) + shape[axis_idx+1:]
            values = values.reshape(new_shape)
        data = np.append(np.asarray(self), values, axis=axis_idx)
        new_axes = self.axes[:]
        new_axes[axis_idx] = Axis(axis.name, np.append(axis.labels, label))
        return LArray(data, axes=new_axes)

    def extend(self, axis, other):
        axis, axis_idx = self.get_axis(axis, idx=True)
        # Get axis by name, so that we do *NOT* check they are "compatible",
        # because it makes sense to append axes of different length
        other_axis = other.get_axis(axis)

        data = np.append(np.asarray(self), np.asarray(other), axis=axis_idx)
        new_axes = self.axes[:]
        new_axes[axis_idx] = Axis(axis.name,
                                  np.append(axis.labels, other_axis.labels))
        return LArray(data, axes=new_axes)

    #XXX: sep argument does not seem very useful
    #XXX: use pandas function instead?
    def to_excel(self, filename, sep=None):
        # Why xlsxwriter? Because it is faster than openpyxl and xlwt
        # currently does not .xlsx (only .xls).
        # PyExcelerate seem like a decent alternative too
        import xlsxwriter as xl

        if sep is None:
            sep = '_'
            #sep = self.sep
        workbook = xl.Workbook(filename)
        if self.ndim > 2:
            for key in product(*[axis.labels for axis in self.axes[:-2]]):
                sheetname = sep.join(str(k) for k in key)
                # sheet names must not:
                # * contain any of the following characters: : \ / ? * [ ]
                #XXX: this will NOT work for unicode strings !
                sheetname = sheetname.translate(string.maketrans('[:]', '(-)'),
                                                r'\/?*') # chars to delete
                # * exceed 31 characters
                # sheetname = sheetname[:31]
                # * be blank
                assert sheetname, "sheet name cannot be blank"
                worksheet = workbook.add_worksheet(sheetname)
                worksheet.write_row(0, 1, self.axes[-1].labels) 
                worksheet.write_column(1, 0, self.axes[-2].labels)                    
                for row, data in enumerate(np.asarray(self[key])):
                    worksheet.write_row(1+row, 1, data)                    
                     
        else:
            worksheet = workbook.add_worksheet('Sheet1')
            worksheet.write_row(0, 1, self.axes[-1].labels) 
            if self.ndim == 2:
                 worksheet.write_column(1, 0, self.axes[-2].labels)
            for row, data in enumerate(np.asarray(self)):
                worksheet.write_row(1+row, 1, data)                    

    def transpose(self, *args):
        axes = [self.get_axis(a) for a in args]
        axes_names = set(axis.name for axis in axes)
        missing_axes = [axis for axis in self.axes
                        if axis.name not in axes_names]
        res_axes = axes + missing_axes
        axes_indices = [self.get_axis_idx(axis) for axis in res_axes]
        src_data = np.asarray(self)
        res_data = src_data.transpose(axes_indices)
        return LArray(res_data, res_axes)
    #XXX: is this necessary?
    reorder = transpose

    def ToCsv(self, filename):
        res = table2csv(self.as_table(), ',', 'nan')
        f = open(filename, "w")
        f.write(res)

    def Collapse(self, filename):
        res = table2csv(self.as_table(), ',', 'nan', self.dimcount)
        f = open(filename, "w")
        f.write(res)

    def ToAv(self, filename):
        res = table2iode(self.as_table(), self.samplestr, self.dimcount, '_',
                         'nan')
        f = open(filename, "w")
        f.write(res)


def parse(s):
    #parameters can be strings or numbers
    if isinstance(s, str):
        s = s.lower()
        if s in ('0', '1', 'false', 'true'):
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


def df_aslarray(df, na=np.nan):
    if isinstance(df.index, pd.core.index.MultiIndex):
        axes_labels = [list(unique(level[labels]))
                   for level, labels in zip(df.index.levels, df.index.labels)]
        axes_names = list(df.index.names)
        laxis = axes_names[-1].split('\\')                                                       
        if len(laxis) > 0:
            axes_names[-1] = laxis[0]
        axes = [Axis(name, labels) for name, labels in zip(axes_names, axes_labels)]
        # pandas treats the "time" labels as column names (strings) so we need to
        # convert them to values
        if len(laxis) > 0:
            axes_names[-1] = laxis[0]
            axes.append(Axis(laxis[1], [parse(cell) for cell in df.columns.values]))
        else:
            axes.append(Axis('time', [parse(cell) for cell in df.columns.values]))
        sdf = df.reindex([i for i in product(*axes_labels)], df.columns.values)
        if na != np.nan:
            sdf.fillna(na,inplace=True)
        data = sdf.values.reshape([len(axis.labels) for axis in axes])    
        return LArray(data, axes) 
    elif isinstance(df.index, pd.core.index.Index):        
        labels = [l for l in df.index]
        axes_names = list(df.index.names)
        laxis = axes_names[-1].split('\\')                                                       
        if len(laxis) > 0:
            axes_names[-1] = laxis[0]        
        axes = [Axis(axes_names[0], labels)]
        # pandas treats the "time" labels as column names (strings) so we need to
        # convert them to values
        if len(laxis) > 0:
            axes.append(Axis(laxis[1], [parse(cell) for cell in df.columns.values]))
        else:
            axes.append(Axis('time', [parse(cell) for cell in df.columns.values]))
#        sdf = df.reindex([i for i in product(*axes_labels)], df.columns.values)
#        if na != np.nan:
#            sdf.fillna(na,inplace=True)
#        data = sdf.values.reshape([len(axis.labels) for axis in axes])    
        data = df.values
        return LArray(data, axes) 

        
    else:
        return None
        


# CSV functions
def read_csv(filepath, nb_index=0, index_col=[], sep=',', na=np.nan):  
    import csv
    """
    reads csv file and returns an Larray with the contents
        nb_index: number of leading index columns (ex. 4)
    or 
        index_col : list of columns for the index (ex. [0, 1, 2, 3])
    
    format csv file:
    arr,ages,sex,nat\time,1991,1992,1993
    A1,BI,H,BE,1,0,0
    A1,BI,H,FO,2,0,0
    A1,BI,F,BE,0,0,1
    A1,BI,F,FO,0,0,0
    A1,A0,H,BE,0,0,0

    """    

    if(len(index_col) == 0 and nb_index == 0):
        # read the first line to determine how many axes (time excluded) we have
        with open(filepath, 'rb') as f:
            reader = csv.reader(f, delimiter=sep)
            header = [parse(cell) for cell in reader.next()]
            axes_names = [cell for cell in header if isinstance(cell, basestring)]
            nb_index = len(axes_names)
        
    if len(index_col) > 0:
        df = pd.read_csv(filepath, index_col=index_col, sep=sep)
    else:
        df = pd.read_csv(filepath, index_col=range(nb_index), sep=sep)
        
    return df_aslarray(df.reindex_axis(sorted(df.columns), axis=1), na)


def save_csv(l_array, filepath, sep=',', na=np.nan):
    """
    saves an LArray to a csv file
    """    
    df = l_array.as_dataframe()
    df.to_csv(filepath, sep=sep)


# HDF5 functions
def save_h5(l_array, name, filepath):
    """
    save a l_array to a h5-store using the specified name
    """
    df = l_array.as_dataframe()
    store = pd.HDFStore(filepath)
    store.put(name, df)
    store.close()    
    

def read_h5(name, filepath):
    """
    read a l_array from a h5-store with the specified name
    """
    
    store = pd.HDFStore(filepath)
    df = store.get(name)
    store.close()
    return df_aslarray(df)


def SaveMatrices(h5_filename):
    try:
        h5file = tables.openFile(h5_filename, mode="w", title="IodeMatrix")
        matnode = h5file.createGroup("/", "matrices", "IodeMatrices")
        d = sys._getframe(1).f_locals
        for k, v in d.iteritems():
            if isinstance(v, LArray):
                # print "storing %s %s" % (k, v.info())
                disk_array = h5file.createArray(matnode, k, v.matdata, k)
                attrs = disk_array.attrs
                attrs._dimensions = np.array(v.dimnames)
                attrs._sep = v.sep
                attrs._sample = np.array(v.samplestr)
                attrs._t = np.array(v.samplelist)
                attrs.shape = np.array(v.matrixshape())
                for i, dimlist in enumerate(v.dimlist):
                    setattr(attrs, '%s' %v.dimnames[i], np.array(v.dimlist[i]))
    finally:
        h5file.close()


def ListMatrices(h5_filename):
    try:
        h5file = tables.openFile(h5_filename, mode="r")
        h5root = h5file.root
        if 'matrices' not in h5root:
            raise Exception('could not find any matrices in the input data file')
        matnames = [mat.name for mat in h5root.matrices]
    finally:
        h5file.close()
        return matnames


def LoadMatrix(h5_filename, matname):
    try:
        h5file = tables.openFile(h5_filename, mode="r")
        h5root = h5file.root
        if 'matrices' not in h5root:
            #raise Exception('could not find any matrices in the input data file')
            # print 'could not find any matrices in the input data file'
            return None
        if matname not in [mat.name for mat in h5root.matrices]:
            #raise Exception('could not find %s in the input data file' % matname)
            # print 'could not find %s in the input data file' % matname
            return None
        mat = getattr(h5root.matrices, matname)
        dimnames = list(mat.attrs._dimensions)
        dimlist = [list(mat.getAttr('%s' % name)) for name in dimnames]
        axes = [Axis(name, labels) for name, labels in zip(dimnames, dimlist)]
        axes.append(Axis('time', list(mat.attrs._t)))
        data = timed(mat.read)
        return LArray(data, axes)
    finally:
        h5file.close()


# EXCEL functions
def save_excel(l_array, name, filepath):
    """
    saves an LArray to the sheet name in the file: filepath
    """
    df = l_array.as_dataframe()
    writer = pd.ExcelWriter(filepath)
    df.to_excel(writer, name)
    writer.save()
    
def read_excel(name, filepath, nb_index=0, index_col=[]):
    """
    reads excel file from sheet name and returns an Larray with the contents
        nb_index: number of leading index columns (ex. 4)
    or 
        index_col : list of columns for the index (ex. [0, 1, 2, 3])    
    """    
    if len(index_col) > 0:
        df=pd.read_excel(filepath, name, index_col=index_col)
    else:
        df=pd.read_excel(filepath, name, index_col=range(nb_index))    
    return df_aslarray(df.reindex_axis(sorted(df.columns), axis=1))     
    
def zeros(axes):
    s = tuple(len(axis) for axis in axes)
    return LArray(np.zeros(s), axes)  
    
    
def ArrayAssign(larray, larray_new, **kwargs):
    axes_names = set(larray.axes_names)
    for kwarg in kwargs:
        if kwarg not in axes_names:
            raise KeyError("{} is not an axis name".format(kwarg))
    full_idx = tuple(kwargs[ax.name] if ax.name in kwargs else slice(None)
                for ax in larray.axes)
    def fullkey(larray, key, collapse_slices=False):
        '''
        based in __getitem__
        '''
        data = np.asarray(larray)

        # convert scalar keys to 1D keys
        if not isinstance(key, tuple):
            key = (key,)

        # expand string keys with commas
        #XXX: is it the right place to do this?
        key = tuple(to_key(axis_key) for axis_key in key)

        # convert xD keys to ND keys
        if len(key) < larray.ndim:
            key = key + (slice(None),) * (larray.ndim - len(key))

        if larray.is_aggregated:
            # convert values on aggregated axes to (value)groups on the
            # *parent* axis. The goal is to allow targeting a ValueGroup
            # label by a string. eg.
            # reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
            # we want all the following to work:
            #   reg[geo.group('A21', name='bru')]
            #   reg['A21']
            #   reg[:] -> all lines, and not the "belgium" line. It is not
            # ideal but it is the lesser evil, because
            # reg.filter(lipro='PO1,PO2') maps to reg[:, 'PO1,PO2'] and
            # it should return the whole "aggregated" geo dimension,
            # not one line only
            def convert(axis, values):
                if (axis.is_aggregated and not isinstance(values, ValueGroup)):
                    vg = axis.parent_axis.group(values)
                    if vg in axis:
                        return vg
                    else:
                        return values
                else:
                    return values
            key = tuple(convert(axis, axis_key)
                        for axis, axis_key in zip(larray.axes, key))

        # translate labels to integers
        translated_key = tuple(axis.translate(axis_key)
                               for axis, axis_key in zip(larray.axes, key))

        # isinstance(ndarray, collections.Sequence) is False but it
        # behaves like one
        sequence = (tuple, list, np.ndarray)
        if collapse_slices:
            translated_key = [range_to_slice(axis_key)
                                  if isinstance(axis_key, sequence)
                                  else axis_key
                              for axis_key in translated_key]

        # count number of indexing arrays (ie non scalar/slices) in tuple
        num_ix_arrays = sum(isinstance(axis_key, sequence)
                            for axis_key in translated_key)
        num_scalars = sum(np.isscalar(axis_key) for axis_key in translated_key)

        # handle advanced indexing with more than one indexing array:
        # basic indexing (only integer and slices) and advanced indexing
        # with only one indexing array are handled fine by numpy
        if num_ix_arrays > 1 or (num_ix_arrays > 0 and num_scalars):
            # np.ix_ wants only lists so:

            # 1) kill scalar-key axes (if any) by indexing them (we cannot
            #    simply transform the scalars into lists of 1 element because
            #    in that case those dimensions are not dropped by
            #    ndarray.__getitem__)
            keyandaxes = zip(translated_key, larray.axes)
            if any(np.isscalar(axis_key) for axis_key in translated_key):
                killscalarskey = tuple(axis_key
                                           if np.isscalar(axis_key)
                                           else slice(None)
                                       for axis_key in translated_key)
                data = data[killscalarskey]
                noscalar_keyandaxes = [(axis_key, axis)
                                        for axis_key, axis in keyandaxes
                                        if not np.isscalar(axis_key)]
            else:
                noscalar_keyandaxes = keyandaxes

            # 2) expand slices to lists (ranges)
            #TODO: cache the range in the axis?
            listkey = tuple(np.arange(*axis_key.indices(len(axis)))
                            if isinstance(axis_key, slice) else axis_key
                            for axis_key, axis in noscalar_keyandaxes)
            # np.ix_ computes the cross product of all lists
            full_key = np.ix_(*listkey)
        else:
            full_key = translated_key

        return data, full_key
        
    data, full_key = fullkey(larray, full_idx)
    #DIFFERENT SHAPE BUT SAME SIZE
    if(data[full_key].shape != larray_new.shape) and (data[full_key].size == larray_new.size):
        data[full_key] = np.asarray(larray_new).reshape(data[full_key].shape) 
        return
            
    #DIFFERENT SHAPE BUT ONLY ONE OR MORE MISSING DIMENSION(S)
    if(len(data[full_key].shape) != len(larray_new.shape)):
        bshape = broadcastshape(larray_new.shape, data[full_key].shape)
        if bshape is not None:
            data[full_key] = np.asarray(larray_new).reshape(bshape) 
        return   
            
    # SAME DIMENSIONS
    data[full_key] = np.asarray(larray_new)
        
def broadcastshape(oshape, nshape):
    bshape = list(nshape)
    dshape = set(nshape).difference(set(oshape))
    if len(dshape) == len(nshape)-len(oshape):
        for i in range(len(bshape)):
            if bshape[i] in dshape:
                bshape[i] = 1        
        return tuple(bshape)
    else:
        return None    

#if __name__ == '__main__':
#    #reg.Collapse('c:/tmp/reg.csv')
#    #reg.ToAv('reg.av')
#    bel = read_csv('bel.csv', index_col=[0,1,2,3]) 
#    test = read_csv('ert_eff_ic_a.tsv', index_col=[0,1,2], sep='\t', na=0)
#    test.ToCsv('brol.csv')
#    save_csv(test, 'brolpd.csv')
#    test_csv = read_csv('brolpd.csv', index_col=[0,1,2])
#    save_excel(test, "TEST", "test.xls")
#    test_xls = read_excel("TEST", "test.xls", index_col=[0,1,2])
#    save_h5(test, 'test', 'store.h5')
#    test_h5 = read_h5('test', 'store.h5')
#    save_h5(bel, 'bel', 'store.h5')