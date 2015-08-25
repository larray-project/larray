"""
Misc tools
"""
from __future__ import absolute_import, division, print_function

import sys
import operator
from textwrap import wrap
from functools import reduce
from itertools import product
from collections import defaultdict

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

from pandas import Index, MultiIndex
import pandas as pd


if sys.version < '3':
    basestring = basestring
    bytes = str
else:
    basestring = str
    unicode = str


def csv_open(filename, mode='r'):
    assert 'b' not in mode and 't' not in mode
    if sys.version < '3':
        return open(filename, mode + 'b')
    else:
        return open(filename, mode, newline='', encoding='utf8')


def decode(s, encoding='utf-8', errors='strict'):
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    else:
        assert s is None or isinstance(s, unicode), "unexpected " + str(type(s))
        return s


def prod(values):
    return reduce(operator.mul, values, 1)


def format_value(value, missing, fullinfo=False):
    if isinstance(value, float) and not fullinfo:
        # nans print as "-1.#J", let's use something nicer
        if value != value:
            return missing
        else:
            return '%2.f' % value
    elif isinstance(value, np.ndarray) and value.shape:
        # prevent numpy's default wrapping
        return str(list(value)).replace(',', '')
    else:
        return str(value)


def get_col_width(table, index):
    return max(len(row[index]) for row in table)


def longest_word(s):
    return max(len(w) for w in s.split()) if s else 0


def get_min_width(table, index):
    return max(longest_word(row[index]) for row in table)


def table2str(table, missing, fullinfo=False, summarize=True,
              maxwidth=80, numedges='auto', sep=' | ', cont='...', keepcols=0):
    """
    table is a list of lists
    :type table: list of list
    """
    if not table:
        return ''
    numcol = max(len(row) for row in table)
    # pad rows that have too few columns
    for row in table:
        if len(row) < numcol:
            row.extend([''] * (numcol - len(row)))
    formatted = [[format_value(value, missing, fullinfo) for value in row]
                 for row in table]
    maxwidths = [get_col_width(formatted, i) for i in range(numcol)]

    total_colwidth = sum(maxwidths)
    sep_width = (numcol - 1) * len(sep)
    if total_colwidth + sep_width > maxwidth:
        minwidths = [get_min_width(formatted, i) for i in range(numcol)]
        available_width = maxwidth - sep_width - sum(minwidths)
        if available_width >= 0:
            ratio = available_width / total_colwidth
            colwidths = [minw + int(maxw * ratio)
                         for minw, maxw in zip(minwidths, maxwidths)]
        else:
            # need to exceed maxwidth or hide some data
            if summarize:
                if numedges == 'auto':
                    w = sum(minwidths[:keepcols]) + len(cont)
                    maxedges = (numcol - keepcols) // 2
                    if maxedges:
                        maxi = 0
                        for i in range(1, maxedges + 1):
                            w += minwidths[i] + minwidths[-i]
                            # + 1 for the "continuation" column
                            ncol = keepcols + i * 2 + 1
                            sepw = (ncol - 1) * len(sep)
                            maxi = i
                            if w + sepw > maxwidth:
                                break
                        numedges = maxi - 1
                    else:
                        numedges = 0
                head = keepcols+numedges
                tail = -numedges if numedges else numcol
                formatted = [row[:head] + [cont] + row[tail:]
                             for row in formatted]
                colwidths = minwidths[:head] + [len(cont)] + minwidths[tail:]
            else:
                colwidths = minwidths
    else:
        colwidths = maxwidths

    lines = []
    for row in formatted:
        wrapped_row = [wrap(value, width)
                       for value, width in zip(row, colwidths)]
        maxlines = max(len(value) for value in wrapped_row)
        newlines = [[] for _ in range(maxlines)]
        for value, width in zip(wrapped_row, colwidths):
            for i in range(maxlines):
                chunk = value[i] if i < len(value) else ''
                newlines[i].append(chunk.rjust(width))
        lines.extend(newlines)
    return '\n'.join(sep.join(row) for row in lines)


# copied from itertools recipes
def unique(iterable):
    """
    Yields all elements once, preserving order. Remember all elements ever
    seen.
    >>> list(unique('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    for element in iterable:
        if element not in seen:
            seen_add(element)
            yield element


# inspired from SQLAlchemy util/_collection
def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [e for e in seq if e not in seen and not seen_add(e)]


def duplicates(iterable):
    """
    List duplicated elements once, preserving order. Remember all elements ever
    seen.
    """
    # duplicates('AAAABBBCCDAABBB') --> A B C
    counts = defaultdict(int)
    for element in iterable:
        counts[element] += 1
        if counts[element] == 2:
            yield element


def rproduct(*i):
    return product(*[x[::-1] for x in i])


def array_equal(a, b):
    # np.array_equal is not implemented on strings in numpy < 1.9
    if np.issubdtype(a.dtype, np.str) and np.issubdtype(b.dtype, np.str):
        try:
            return (a == b).all()
        except ValueError:
            return False
    else:
        return np.array_equal(a, b)


def array_nan_equal(a, b):
    # np.array_equal is not implemented on strings in numpy < 1.9
    if np.issubdtype(a.dtype, np.str) and np.issubdtype(b.dtype, np.str):
        try:
            return (a == b).all()
        except ValueError:
            return False
    else:
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)))


def unzip(iterable):
    return list(zip(*iterable))


class ReprString(str):
    def __repr__(self):
        return self


#TODO: this function should really be upstreamed in some way to Pandas
def multi_index_from_arrays(arrays, sortorder=None, names=None,
                            categories=None):
    from pandas.core.categorical import Categorical

    if len(arrays) == 1:
        name = None if names is None else names[0]
        return Index(arrays[0], name=name)

    if categories is None:
        cats = [Categorical(levelarr, ordered=True) for levelarr in arrays]
    else:
        cats = [Categorical(levelarr, levelcat, ordered=True)
                for levelarr, levelcat in zip(arrays, categories)]
    levels = [c.categories for c in cats]
    labels = [c.codes for c in cats]
    if names is None:
        names = [c.name for c in cats]
    return MultiIndex(levels=levels, labels=labels,
                      sortorder=sortorder, names=names,
                      verify_integrity=False)


# TODO: this function should really be upstreamed in some way to Pandas
def multi_index_from_product(iterables, sortorder=None, names=None,
                             sortvalues=True):
    """
    Make a MultiIndex from the cartesian product of multiple iterables

    Parameters
    ----------
    iterables : list / sequence of iterables
        Each iterable has unique labels for each level of the index.
    sortorder : int or None
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : list / sequence of strings or None
        Names for the levels in the index.
    sortvalues : bool
        Whether each level values should be sorted alphabetically.

    Returns
    -------
    index : MultiIndex

    Examples
    --------
    >>> numbers = [0, 1]
    >>> colors = [u'red', u'green', u'blue']
    >>> MultiIndex.from_product([numbers, colors], names=['number', 'color'])
    MultiIndex(levels=[[0, 1], ['blue', 'green', 'red']],
               labels=[[0, 0, 0, 1, 1, 1], [2, 1, 0, 2, 1, 0]],
               names=['number', 'color'])
    >>> multi_index_from_product([numbers, colors], names=['number', 'color'],
    ...                          sortvalues=False)
    MultiIndex(levels=[[0, 1], ['red', 'green', 'blue']],
               labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
               names=['number', 'color'],
               sortorder=0)

    See Also
    --------
    MultiIndex.from_arrays : Convert list of arrays to MultiIndex
    MultiIndex.from_tuples : Convert list of tuples to MultiIndex
    """
    from pandas.core.categorical import Categorical
    from pandas.tools.util import cartesian_product

    if sortvalues:
        categoricals = [Categorical(it, ordered=True) for it in iterables]
    else:
        categoricals = [Categorical(it, it, ordered=True) for it in iterables]
        sortorder = 0
    labels = cartesian_product([c.codes for c in categoricals])
    return MultiIndex(levels=[c.categories for c in categoricals],
                      labels=labels, sortorder=sortorder, names=names)


def _sort_level_inplace(data):
    if isinstance(data, pd.Series):
        # as of Pandas 0.16 inplace not implemented for Series
        data = data.sortlevel()
    else:
        data.sortlevel(inplace=True)
    return data


def _pandas_index_as_df(index):
    for labels in index.labels:
        # I do not know when this can even happen
        assert not np.any(labels == -1)
    names = [name if name is not None else 'level_%d' % i
             for i, name in enumerate(index.names)]
    columns = [level.values[labels]
               for level, labels in zip(index.levels, index.labels)]
    return pd.DataFrame(dict(zip(names, columns)))


def _pandas_rename_axis(obj, axis, level, newname):
    """inplace rename"""
    idx = obj.index if axis == 0 else obj.columns
    names = idx.names
    idx.names = names[:level] + [newname] + names[level + 1:]


def _pandas_broadcast_to_index(left, right_index, right_columns=None):
    orig_left = left
    li_names = oset(left.index.names)
    lc_names = oset(left.columns.names if isinstance(left, pd.DataFrame)
                    else ())
    ri_names = oset(right_index.names)
    rc_names = oset(right_columns.names if isinstance(right_columns, pd.Index)
                    else ())
    if li_names == ri_names and lc_names == rc_names:
        # we do not need to do anything
        return left

    # drop index levels if needed
    if li_names > ri_names:
        left_extra = li_names - ri_names
        # this assertion is expensive to compute
        assert all(len(_index_level_unique_labels(left.index, level)) == 1
                   for level in left_extra)
        left = left.copy(deep=False)
        left.index = left.index.droplevel(list(left_extra))

    # drop column levels if needed
    if lc_names > rc_names:
        left_extra = lc_names - rc_names
        # this assertion is expensive to compute
        assert all(len(_index_level_unique_labels(left.columns, level)) == 1
                   for level in left_extra)
        left = left.copy(deep=False)
        left.columns = left.columns.droplevel(list(left_extra))

    li_names = oset(left.index.names)
    lc_names = oset(left.columns.names if isinstance(left, pd.DataFrame)
                    else ())
    if li_names == ri_names and lc_names == rc_names:
        # we do not need to do anything else
        return left

    common_names = li_names & ri_names
    if not common_names:
        raise NotImplementedError("Cannot broadcast to an array with no common "
                                  "axis")
    # assuming left has a subset of right levels
    if li_names < ri_names:
        if isinstance(left, pd.Series):
            left = left.to_frame('__left__')
        rightdf = _pandas_index_as_df(right_index)
        # left join because we use the levels of right but the labels of left
        # XXX: use left.join() instead?
        merged = left.merge(rightdf, how='left', right_on=list(common_names),
                            left_index=True, sort=False)
        merged.set_index(right_index.names, inplace=True)
        # TODO: index probably needs to be sorted!
        if isinstance(orig_left, pd.Series):
            assert merged.columns == ['__left__']
            merged = merged['__left__']
    else:
        merged = left

    if lc_names == rc_names:
        return merged
    else:
        assert lc_names < rc_names
        if not lc_names:
            return pd.DataFrame({c: merged for c in right_columns},
                                index=merged.index,
                                columns=right_columns)
        else:
            raise NotImplementedError("Cannot broadcast existing columns")


def _pandas_broadcast_to(left, right):
    columns = right.columns if isinstance(right, pd.DataFrame) else None
    return _pandas_broadcast_to_index(left, right.index, columns)


# We need this function because
# 1) set_index does not exist on Series
# 2) set_index can only append at the end (not insert)
# 3) set_index uses MultiIndex.from_arrays which loose "levels" inherent
#    ordering (it sorts values), even though it keeps "apparent" ordering (if
#    you print the df it seems in the same order)
def _pandas_insert_index_level(obj, name, value, position=-1,
                               axis=0, inplace=False):
    assert axis in (0, 1)
    assert np.isscalar(value)

    if not inplace:
        obj = obj.copy()

    if axis == 0:
        idx = obj.index
    else:
        idx = obj.columns

    if isinstance(idx, MultiIndex):
        levels = list(idx.levels)
        labels = list(idx.labels)
    else:
        assert isinstance(idx, pd.Index)
        levels = [idx]
        labels = [np.arange(len(idx))]
    names = [x for x in idx.names]

    dtype = object if isinstance(value, str) else type(value)
    newlevel = np.empty(len(idx), dtype=dtype)
    newlevel.fill(value)
    newlabels = np.zeros(len(idx), dtype=np.int8)

    levels.insert(position, newlevel)
    labels.insert(position, newlabels)
    names.insert(position, name)

    sortorder = 0 if isinstance(idx, pd.Index) or idx.is_lexsorted() else None
    newidx = MultiIndex(levels=levels, labels=labels,
                        sortorder=sortorder, names=names,
                        verify_integrity=False)
    assert newidx.is_lexsorted()
    if axis == 0:
        obj.index = newidx
    else:
        obj.columns = newidx
    return obj


def _pandas_transpose_any(obj, index_levels, column_levels=None, sort=True,
                          copy=False):
    if column_levels and not index_levels:
        # we asked for a Series by asking for only column levels
        index_levels = tuple(column_levels)
        column_levels = ()
    else:
        index_levels = tuple(index_levels)
        column_levels = tuple(column_levels) if column_levels is not None else ()

    idxnames = obj.index.names
    colnames = obj.columns.names if isinstance(obj, pd.DataFrame) else ()

    # if idxnames == index_levels and colnames == column_levels:
    #     return obj.copy()

    idxnames_set = set(idxnames)
    colnames_set = set(colnames)

    if idxnames_set == set(column_levels) and colnames_set == set(index_levels):
        obj = obj.transpose()
    else:
        # levels that are in columns but should be in index
        tostack = [l for l in index_levels if l in colnames_set]
        # levels that are in index but should be in columns
        tounstack = [l for l in column_levels if l in idxnames_set]

        # TODO: it is usually faster to go via the path which minimize
        # max(len(axis0), len(axis1))
        # eg 100x10 \ 100 to 100x100 \ 10
        # will be faster via 100 \ 100x10 than via 100x10x100
        if tostack:
            obj = obj.stack(tostack)

        if tounstack:
            obj = obj.unstack(tounstack)

        if not tounstack and not tostack and copy:
            obj = obj.copy()

    idxnames = tuple(obj.index.names)
    colnames = tuple(obj.columns.names) if isinstance(obj, pd.DataFrame) else ()
    if idxnames != index_levels:
        obj = _pandas_reorder_levels(obj, index_levels, inplace=True)
        if sort:
            obj = _sort_level_inplace(obj)
    if colnames != column_levels:
        _pandas_reorder_levels(obj, column_levels, axis=1, inplace=True)
        if sort:
            obj.sortlevel(axis=1, inplace=True)
    return obj


def _pandas_transpose_any_like(obj, other, sort=True):
    idxnames = other.index.names
    colnames = other.columns.names if isinstance(other, pd.DataFrame) else ()
    return _pandas_transpose_any(obj, idxnames, colnames, sort)


# workaround for no inplace arg.
def _pandas_reorder_levels(self, order, axis=0, inplace=False):
    """
    Rearrange index levels using input order.
    May not drop or duplicate levels

    Parameters
    ----------
    order : list of int or list of str
        List representing new level order. Reference level by number
        (position) or by key (label).
    axis : int
        Where to reorder levels.

    Returns
    -------
    type of caller (new object)
    """
    axis = self._get_axis_number(axis)
    if not isinstance(self._get_axis(axis), MultiIndex):
        raise TypeError('Can only reorder levels on a hierarchical axis.')

    result = self if inplace else self.copy()
    if axis == 0:
        result.index = result.index.reorder_levels(order)
    else:
        assert axis == 1
        result.columns = result.columns.reorder_levels(order)
    return result


class oset(object):
    def __init__(self, data):
        self.l = []
        self.s = set()
        for e in data:
            self.add(e)

    def add(self, e):
        if e not in self.s:
            self.s.add(e)
            self.l.append(e)

    def __and__(self, other):
        i = self.s & other.s
        return oset([e for e in self.l if e in i])

    def __or__(self, other):
        # duplicates will be discarded automatically
        if isinstance(other, oset):
            other_l = other.l
        else:
            other_l = list(other)
        return oset(self.l + other_l)

    def __sub__(self, other):
        if isinstance(other, oset):
            other_s = other.s
        else:
            other_s = set(other)
        return oset([e for e in self.l if e not in other_s])

    def __eq__(self, other):
        return self.s == other.s

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return len(self.l)

    def __getitem__(self, key):
        return self.l[key]

    def issubset(self, other):
        return self.s.issubset(other.s)
    __le__ = issubset

    def __lt__(self, other):
        return self.s < other.s

    def issuperset(self, other):
        return self.s.issuperset(other.s)
    __ge__ = issuperset

    def __gt__(self, other):
        return self.s > other.s

    def __repr__(self):
        return "oset([" + ', '.join(repr(e) for e in self.l) + "])"


def _pandas_align_viamerge(left, right, on=None, join='left',
                           left_index=False, right_index=False):
    orig_left, orig_right = left, right
    if isinstance(left, pd.Series):
        left = left.to_frame('__left__')
    if isinstance(right, pd.Series):
        right = right.to_frame('__right__')
    else:
        # make sure we can differentiate which column comes from where
        colmap = {c: '__right__' + str(c) for c in right.columns}
        right = right.rename(columns=colmap, copy=False)
    if not left_index:
        left = left.reset_index()
    if not right_index:
        right = right.reset_index()

    if left_index and right_index:
        kwargs = {}
    elif left_index:
        kwargs = {'right_on': on}
    elif right_index:
        kwargs = {'left_on': on}
    else:
        kwargs = {'on': on}

    # FIXME: the columns are not aligned, so it does not work correctly if
    # columns are not the same on both sides. If there are more columns on one
    # side than the other, the side with less columns is not "expanded".
    # XXX: would .stack() solve this problem?
    merged = left.merge(right, how=join, sort=False, right_index=right_index,
                        left_index=left_index, **kwargs)
    # right_index True means right_index is a subset of left_index
    if right_index and join == 'left':
        merged.drop(orig_left.index.names, axis=1, inplace=True)
        # we can reuse left index as is
        merged.index = orig_left.index
    elif left_index and join == 'right':
        merged.drop(orig_right.index.names, axis=1, inplace=True)
        # we can reuse right index as is
        merged.index = orig_right.index
    else:
        lnames = oset(orig_left.index.names)
        rnames = oset(orig_right.index.names)
        # priority to left order for all join methods except "right"
        merged_names = rnames | lnames if join == 'right' else lnames | rnames
        merged.set_index(list(merged_names), inplace=True)
        # FIXME: does not work if the "priority side" (eg left side on a left
        # join) contains more values. There will be NaN in the index for the
        # combination of the new dimension of the right side and those extra
        # left side indexes.
        # FIXME: at the minimum, we should detect this case and raise
    left = merged[[c for c in merged.columns
                   if not isinstance(c, str) or not c.startswith('__right__')]]
    right = merged[[c for c in merged.columns
                    if isinstance(c, str) and c.startswith('__right__')]]

    if isinstance(orig_right, pd.DataFrame):
        # not inplace to avoid warning
        right = right.rename(columns={c: c[9:] for c in right.columns},
                             copy=False)
        # if there was a type conversion, convert them back
        right.columns = right.columns.astype(orig_right.columns.dtype)
    else:
        assert right.columns == ['__right__']
        right = right['__right__']
    if isinstance(orig_left, pd.Series):
        assert left.columns == ['__left__']
        left = left['__left__']
    return left, right


def _pandas_align(left, right, join='left'):
    li_names = oset(left.index.names)
    lc_names = oset(left.columns.names if isinstance(left, pd.DataFrame)
                    else ())
    ri_names = oset(right.index.names)
    rc_names = oset(right.columns.names if isinstance(right, pd.DataFrame)
                    else ())

    left_names = li_names | lc_names
    right_names = ri_names | rc_names
    common_names = left_names & right_names

    if not common_names:
        raise NotImplementedError("Cannot do binary operations between arrays "
                                  "with no common axis")

    # rules imposed by Pandas (found empirically)
    # -------------------------------------------
    # a) there must be at least one common level on the index (unless right is
    #    a Series)
    # b) each common level need to be on the same "axis" for both operands
    #    (eg level "a" need to be either on index for both operands or
    #    on columns for both operands)
    # c) there may only be common levels in columns
    # d) common levels need to be in the same order
    # e) cannot merge Series (with anything) and cannot join Series to Series
    #    => must have at least one DataFrame if we need join
    #    => must have 2 DataFrames for merge

    # algorithm
    # ---------

    # 1) left

    if isinstance(right, pd.DataFrame):
        # a) if no common level on left index (there is implicitly at least
        #    one in columns) move first common level in columns to index
        #    (transposing left is a bad idea because there would be uncommon on
        #    columns which we would need to move again)
        to_stack = []
        if isinstance(right, pd.DataFrame) and not (li_names & common_names):
            to_stack.append(common_names[0])

        # b) move all uncommon levels from columns to index
        to_stack.extend(lc_names - common_names)

        # c) transpose
        new_li = li_names | to_stack
        new_lc = lc_names - to_stack
        #FIXME: (un)stacked levels are sorted!!!
        left = _pandas_transpose_any(left, new_li, new_lc, sort=False)
    else:
        new_li = li_names
        new_lc = lc_names

    # 2) right

    # a) right index should be (left index & right both) (left order) + right
    #    uncommon (from both index & columns), right columns should be
    #    (left columns)
    if len(right_names) > 1:
        new_ri = (new_li & right_names) | (right_names - new_lc)
        new_rc = new_lc & right_names
    else:
        # do not modify Series with a single level/dimension
        new_ri = ri_names
        new_rc = rc_names

    # b) transpose
    right = _pandas_transpose_any(right, new_ri, new_rc, sort=False)

    # 3) (after binop) unstack all the levels stacked in "left" step in result
    # -------
    if right_names == left_names:
        axis = None if isinstance(left, pd.DataFrame) else 0
        return axis, None, left.align(right, join=join)

    # DF + Series (rc == [])
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.Series):
        # Series levels match DF index levels
        if new_ri == new_li:
            return 0, None, left.align(right, join=join, axis=0)
        # Series levels match DF columns levels
        elif new_ri == new_lc:
            return 1, None, left.align(right, join=join, axis=1)
        # Series level match one DF columns levels
        elif len(new_ri) == 1:
            # it MUST be in either index or columns
            level = new_ri[0]
            axis = 0 if level in new_li else 1
            return axis, level, left.align(right, join=join, axis=axis,
                                           level=level)
    elif isinstance(right, pd.DataFrame) and isinstance(left, pd.Series):
        raise NotImplementedError("do not know how to handle S + DF yet")
    elif isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        if len(new_li) == 1 or len(new_ri) == 1:
            return None, None, left.align(right, join=join)
    elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
        if len(new_li) == 1 or len(new_ri) == 1:
            return 0, None, left.align(right, join=join)

    # multi-index on both sides
    assert len(new_li) > 1 and len(new_ri) > 1

    right_index = new_ri.issubset(new_li)
    left_index = new_li.issubset(new_ri)
    merged = _pandas_align_viamerge(left, right,
                                    on=list(new_ri & new_li),
                                    join=join, right_index=right_index,
                                    left_index=left_index)
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        axis = None
    else:
        axis = 0
    return axis, None, merged


#TODO: this function should really be upstreamed in some way to Pandas
def _index_level_unique_labels(idx, level):
    """
    returns the unique values for one level, respecting the parent ordering.
    :param idx: pd.MultiIndex
    :param level: num or name
    :return: list of values
    """
    # * using idx.levels[level_num] as is does not work for DataFrame subsets
    #   (it contains all the parent values even if not all of them are used in
    #   the subset).
    # * using idx.get_level_values(level).unique() is both slower and does not
    #   respect the index order (unique() use a first-seen order)
    # * if using .labels[level].values() gets unsupported at one point,
    #   simply use "unique_values = set(idx.get_level_values(level))" instead

    level_num = idx._get_level_number(level)
    # .values() to get a straight ndarray from the FrozenNDArray that .labels[]
    # gives us, which is slower to iterate on
    # .astype(object) because set() needs python objects and it is faster to
    # convert all ints in bulk than having them converted in the array iterator
    # (it only pays for itself with len(unique) > ~100)
    unique_labels = set(np.unique(idx.labels[level_num].values())
                        .astype(object))
    order = idx.levels[level_num]
    return [v for i, v in enumerate(order) if i in unique_labels]