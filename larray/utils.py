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
                        for i in range(1, maxedges + 1):
                            w += minwidths[i] + minwidths[-i]
                            # + 1 for the "continuation" column
                            ncol = keepcols + i * 2 + 1
                            sepw = (ncol - 1) * len(sep)
                            if w + sepw > maxwidth:
                                break
                        numedges = i - 1
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
