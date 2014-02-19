"""
Matrix class
"""
#TODO
# implement new syntax

# move to github
# easily add sum column for a dimension
# reorder an axis labels
# modify read_csv format (last_column / time)
# test to_csv: does it consume too much mem?
# ---> test pandas (one dimension horizontally)
# add labels in ValueGroups.__str__
# xlsx export workbook without overwriting some sheets (charts)
# implement x = bel.filter(age='0:10')
# implement y = bel.sum(sex='H,F')


#TODO:

# * unit TESTS !!!!
# * evaluate the impact of label-only __getitem__: numpy/matplotlib/...
#   functions probably rely on __getitem__ with indices
# * docstring for all methods
# * choose between subset and group. Having both is just confusing.
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
# * aggregates other than sum: all, any, sum, prod, cumsum, cumprod, min, max,
#   ptp, mean, var, std
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
# ? allow __getitem__ with ValueGroups at any position since we usually know
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
# ? improve Labeler I don't know how though :)
# ? implement dict-like behavior for LArray.axes (to be able to do stuff like
#   la.axes['sex'].labels
#

import csv
from itertools import izip, product
import string
import sys

import numpy as np
import tables

from utils import prod, table2str, table2csv, table2iode, timed, unique


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


class Axis(object):
    # ticks instead of labels?
    #XXX: make name and labels optional?
    def __init__(self, name, labels):
        """
        labels should be an array-like (convertible to an ndarray)
        """
        self.name = name
        self.labels = np.asarray(labels)
        self._mapping = {label: i for i, label in enumerate(labels)}

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
            if key.axis is not self:
                raise ValueError("cannot subset an axis with a ValueGroup of "
                                 "another axis")
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
        if (isinstance(key, slice)
            and key.start is None and key.stop is None and key.step is None):
            return self
        # we must NOT modify the axis name, even though this creates a new axis
        # that is independent from the original one because the original
        # name is probably what users will want to use to filter
        return Axis(self.name, self.labels[key])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        """
        key is a label-based key (slice and fancy indexing are supported)
        """
        return self.subset(key)

    def translate(self, key):
        """
        translates a label key to its numerical index counterpart
        fancy index with boolean vectors are passed through unmodified
        """
        mapping = self._mapping
        if isinstance(key, basestring):
            #XXX: not sure where to put this
            if ',' in key:
                key = key.split(',')
            # otherwise do nothing
        elif isinstance(key, ValueGroup):
            if key in mapping:
                # the array is an aggregate (it has ValueGroup keys in its
                # mapping) => return the index of the group
                return mapping[key]
            else:
                # the array is not an aggregate (it has normal label keys in its
                # mapping) => return the index of all the elements in the group
                key = key.key

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
            assert np.isscalar(key)
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
            key = key.split(',')

        #TODO: valueGroups will very likely be used as "groups" so they should
        # cache the indices of their labels
        self.key = key

        # this is only meant the check the key is valid, later we might want
        # to cache the result to check that it does not change over time
        self.axis.translate(key)

        if name is None:
            if isinstance(key, slice):
                # examples of result: [:24] [25:] [:]
                start = key.start if key.start is not None else ''
                stop = key.stop if key.stop is not None else ''
                step = (":" + key.step) if key.step is not None else ''
                name = '%s:%s%s' % (start, stop, step)
            else:
                name = str(key)
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s[%s]" % (self.axis.name, self.name)


class Labeler(object):
    def __init__(self):
        self.count = 0

    def get(self, group):
        #XXX: add support for slices here?
        if isinstance(group, ValueGroup):
            return group
        elif np.isscalar(group):
            return group
        elif len(group) == 1:
            return group[0]
        else:
            concat = '+'.join(str(v) for v in group)
            if len(concat) < 40:
                return concat
            else:
                self.count += 1
                return "group_{}".format(self.count)


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

    def __getitem__(self, key, collapse_slices=False):
        data = np.asarray(self)

        # convert scalar keys to 1D keys
        if not isinstance(key, tuple):
            key = (key,)

        # expand string keys with commas
        #XXX: is it the right place to do this?
        key = tuple(axis_key.split(',')
                        if isinstance(axis_key, basestring) and ',' in axis_key
                        else axis_key
                    for axis_key in key)

        # convert xD keys to ND keys
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

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

        # handle advanced indexing with more than one indexing array:
        # basic indexing (only integer and slices) and advanced indexing
        # with only one indexing array are handled fine by numpy
        if num_ix_arrays > 1:
            # np.ix_ wants only lists so:

            # 1) kill scalar-key axes (if any) by indexing them (we cannot
            #    simply transform the scalars into lists of 1 element because
            #    in that case those dimensions are not dropped by
            #    ndarray.__getitem__)
            if any(np.isscalar(axis_key) for axis_key in translated_key):
                killscalarskey = tuple(axis_key
                                           if np.isscalar(axis_key)
                                           else slice(None)
                                       for axis_key in translated_key)
                data = data[killscalarskey]
                noscalarkey = tuple(axis_key for axis_key in translated_key
                                    if not np.isscalar(axis_key))
            else:
                noscalarkey = translated_key

            # 2) expand slices to lists (ranges)
            #TODO: cache the range in the axis
            listkey = tuple(np.arange(*axis_key.indices(len(axis)))
                            if isinstance(axis_key, slice) else axis_key
                            for axis_key, axis
                            in zip(noscalarkey, self.axes))
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
        if not self.ndim:
            return str(np.asscalar(self))
        else:
            return '\n' + table2str(self.as_table(), 'nan', True) + '\n'
    __repr__ = __str__

    def as_table(self):
        if not self.ndim:
            return []

        # gender |      |
        #  False | True | total
        #     20 |   16 |    35

        #   dead | gender |      |
        #        |  False | True | total
        #  False |     20 |   15 |    35
        #   True |      0 |    1 |     1
        #  total |     20 |   16 |    36

        # agegroup | gender |  dead |      |
        #          |        | False | True | total
        #        5 |  False |    20 |   15 |    xx
        #        5 |   True |     0 |    1 |    xx
        #       10 |  False |    25 |   10 |    xx
        #       10 |   True |     1 |    1 |    xx
        #          |  total |    xx |   xx |    xx
        width = self.shape[-1]
        height = prod(self.shape[:-1])
        if self.axes is not None:
            axes_names = [axis.name for axis in self.axes]
            axes_labels = [axis.labels for axis in self.axes]
        else:
            axes_names = None
            axes_labels = None

        if axes_names is not None:
            result = [axes_names +
                      [''] * (width - 1),
                      # 2nd line
                      [''] * (self.ndim - 1) +
                      list(axes_labels[-1])]
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
    # [a, b] and [a] even though it is faster and uses less memory.
    def filter(self, collapse=False, **kwargs):
        """
        filters the array along the axes given as keyword arguments.
        It is similar to np.take but works with several axes at once.
        """
        axes_names = set(axis.name for axis in self.axes)
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

        # we need to search for the axis by name, instead of the axis object
        # itself because we need to support axes subsets (ValueGroup)
        axes_indices = [self._get_axis_idx(a.name) if isinstance(a, Axis) else a
                        for a in axes]
        res_data = op(src_data, axis=tuple(axes_indices))
        axes_tokill = set(axes_indices)
        res_axes = [axis for axis_num, axis in enumerate(self.axes)
                    if axis_num not in axes_tokill]
        return LArray(res_data, res_axes)

    def _get_axis(self, name):
        return self.axes[self._get_axis_idx(name)]

    def _get_axis_idx(self, name):
        axis_names = [a.name for a in self.axes]
        return axis_names.index(name)

    def _group_aggregate(self, op, kwargs, commutative=False):
        if not commutative and len(kwargs) > 1:
            raise ValueError("grouping aggregates on multiple dimensions"
                             "is not supported for '%s'" % op.func_name)

        # allow multiple-dimensions aggregates for commutative operations
        # (all except var, std and ptp I think)
        res = self
        for agg_axis_name, groups in kwargs.iteritems():
            if not isinstance(groups, (tuple, list)):
                groups = (groups,)

            agg_axis_idx = res._get_axis_idx(agg_axis_name)
            agg_axis = res.axes[agg_axis_idx]

            labeler = Labeler()
            group_labels = [labeler.get(group) for group in groups]
            res_axes = res.axes[:]

            # I don't think it is a good idea to modify the axis name (eg
            # append "_agg" or "*") even though this creates a new axis
            # that is independent from the original one because the original
            # name is probably what users will want to use to filter
            res_axes[agg_axis_idx] = Axis(agg_axis.name, group_labels)

            res_shape = list(res.shape)
            res_shape[agg_axis_idx] = len(groups)

            res_data = np.empty(res_shape, dtype=res.dtype)

            group_idx = [slice(None) for _ in res_shape]
            for i, group in enumerate(groups):
                group_idx[agg_axis_idx] = i

                # we need only lists not single labels, otherwise the dimension
                # is discarded
                group = [group] if np.isscalar(group) else group

                # we don't reuse kwargs because we might have modified "groups"
                arr = self.filter(collapse=True, **{agg_axis_name: group})
                arr = np.asarray(arr)
                op(arr, axis=agg_axis_idx, out=res_data[group_idx])
                del arr
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

    def sum(self, *args, **kwargs):
        return self._aggregate(np.sum, args, kwargs, commutative=True)

    #XXX: sep argument does not seem very useful
    def to_excel(self, filename, sep=None):
        import ExcelCom as ec

        if sep is None:
            sep = '_'
            #sep = self.sep
        xl = ec.comExcel()
        xl.load(filename)
        if self.ndim > 2:
            for key in product(*[axis.labels for axis in self.axes[:-2]]):
                sheetname = sep.join(str(k) for k in key)

                # sheet names must not:
                # * contain any of the following characters: : \ / ? * [ ]
                #XXX: this will NOT work for unicode strings !
                sheetname = sheetname.translate(string.maketrans('[:]', '(-)'),
                                                r'\/?*') # chars to delete
                # * exceed 31 characters
                sheetname = sheetname[:31]
                # * be blank
                assert sheetname, "sheet name cannot be blank"

                sheetdata = np.asarray(self[key])

                xl.addworksheets(sheetname)

                #TODO: reuse as_table, possibly adding an argument to
                # as_table to determine how many dimensions should be "folded"

                # last axis (time) as columns headers (ie the first row)
                xl.setRange(sheetname, 2, 1,
                            (tuple(str(l) for l in self.axes[-1].labels),))

                # next to last axis as rows headers (ie the first column)
                xl.setRange(sheetname, 1, 2,
                            tuple((x,) for x in self.axes[-2].labels))
                xl.setRange(sheetname, 2, 2, sheetdata)
        else:
            xl.addworksheets('Sheet1')

            # last axis (time) as columns headers (ie the first row)
            xl.setRange('Sheet1', 2, 1,
                        (tuple(str(l) for l in self.axes[-1].labels),))
            if self.ndim == 2:
                # next to last axis as rows headers (ie the first column)
                xl.setRange('Sheet1', 1, 2,
                            tuple((str(x),) for x in self.axes[-2].labels))
            xl.setRange('Sheet1', 2, 2, np.asarray(self))

        xl.save(filename)
        xl.close()
        xl.end
        del xl

    def transpose(self, *args):
        axes_names = set(axis.name for axis in args)
        missing_axes = [axis for axis in self.axes
                        if axis.name not in axes_names]
        res_axes = list(args) + missing_axes
        axes_indices = [self._get_axis_idx(axis.name) for axis in res_axes]
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


def read_csv(filepath):
    import pandas as pd

    # read the first line to determine how many axes (time excluded) we have
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        header = [parse(cell) for cell in reader.next()]
        axes_names = [cell for cell in header if isinstance(cell, basestring)]
    df = pd.read_csv(filepath, index_col=range(len(axes_names)))
    assert df.index.names == axes_names, "%s != %s" % (df.index.names,
                                                       axes_names)

    # labels in index.levels are sorted, but the data is not, so we need to
    # compute the "unsorted" labels !
    # alternatives are to either use "df = df.sort_index()", or
    # "df.index.get_level_values(level)" but they are both slower.
    axes_labels = [list(unique(level[labels]))
                   for level, labels in zip(df.index.levels, df.index.labels)]
    axes = [Axis(name, labels) for name, labels in zip(axes_names, axes_labels)]
    # pandas treats the "time" labels as column names (strings) so we need to
    # convert them to values
    axes.append(Axis('time', [parse(cell) for cell in df.columns.values]))
    data = df.values.reshape([len(axis.labels) for axis in axes])
    return LArray(data, axes)


def SaveMatrices(h5_filename):
    try:
        h5file = tables.openFile(h5_filename, mode="w", title="IodeMatrix")
        matnode = h5file.createGroup("/", "matrices", "IodeMatrices")
        d = sys._getframe(1).f_locals
        for k, v in d.iteritems():
            if isinstance(v, LArray):
                print "storing %s %s" % (k, v.info())
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
            print 'could not find any matrices in the input data file'
            return None
        if matname not in [mat.name for mat in h5root.matrices]:
            #raise Exception('could not find %s in the input data file' % matname)
            print 'could not find %s in the input data file' % matname
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

if __name__ == '__main__':
    #reg.Collapse('c:/tmp/reg.csv')
    #reg.ToAv('reg.av')
    pass