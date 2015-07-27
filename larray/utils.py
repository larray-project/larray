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


#TODO: this function should really be upstreamed in some way to Pandas
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

        #TODO: it is usually faster to go via the path which minimize
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
