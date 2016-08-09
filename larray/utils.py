"""
Misc tools
"""
from __future__ import absolute_import, division, print_function

import itertools
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

if sys.version < '3':
    basestring = basestring
    bytes = str
    long = long
    PY3 = False
else:
    basestring = str
    unicode = str
    long = int
    PY3 = True


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


def unique_list(iterable):
    """
    Returns a list of all unique elements, preserving order. Remember all
    elements ever seen.
    >>> unique_list('AAAABBBCCDAABBB')
    ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    res = []
    res_append = res.append
    for element in iterable:
        if element not in seen:
            seen_add(element)
            res_append(element)
    return res


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


def array_nan_equal(a, b):
    if np.issubdtype(a.dtype, np.str) and np.issubdtype(b.dtype, np.str):
        return np.array_equal(a, b)
    else:
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)))


def unzip(iterable):
    return list(zip(*iterable))


class ReprString(str):
    def __repr__(self):
        return self


def array_lookup(array, mapping):
    """pass all elements of an np.ndarray through a mapping"""
    array = np.asarray(array)
    # TODO: this must be cached in the Axis
    # TODO: range axes should be optimized (reuse Pandas 0.18 indexes)
    sorted_keys, sorted_values = tuple(zip(*sorted(mapping.items())))
    sorted_keys = np.array(sorted_keys)
    # prevent an array of booleans from matching a integer axis (sorted_keys)
    # XXX: we might want to allow signed and unsigned integers to match
    #      against each other
    if array.dtype.kind != sorted_keys.dtype.kind:
        raise KeyError('key has not the same dtype than axis')
    # TODO: it is very important to fail quickly, so guess_axis should try
    # this in chunks (first test first element of key, if several axes match,
    #  try [1:11] elements, [12:112], [113:1113], ...
    if not np.all(np.in1d(array, sorted_keys)):
        raise KeyError('all keys not in array')

    sorted_values = np.array(sorted_values)
    if not len(array):
        return np.empty(0, dtype=sorted_values.dtype)
    indices = np.searchsorted(sorted_keys, array)
    return sorted_values[indices]


def array_lookup2(array, sorted_keys, sorted_values):
    """pass all elements of an np.ndarray through a "mapping" """
    if not len(array):
        return np.empty(0, dtype=sorted_values.dtype)

    array = np.asarray(array)
    # TODO: this must be cached in the Axis
    # TODO: range axes should be optimized (reuse Pandas 0.18 indexes)

    # prevent an array of booleans from matching a integer axis (sorted_keys)
    # XXX: we might want to allow signed and unsigned integers to match
    #      against each other
    if array.dtype.kind != sorted_keys.dtype.kind:
        raise KeyError('key has not the same dtype than axis')
    # TODO: it is very important to fail quickly, so guess_axis should try
    # this in chunks (first test first element of key, if several axes match,
    #  try [1:11] elements, [12:112], [113:1113], ...
    if not np.all(np.in1d(array, sorted_keys)):
        raise KeyError('all keys not in array')

    indices = np.searchsorted(sorted_keys, array)
    return sorted_values[indices]


def split_on_condition(seq, condition):
    """splits an iterable into two lists depending on a condition

    Parameters
    ----------
    seq : iterable
    condition : function(e) -> True or False

    Returns
    -------
    a, b: list

    Notes
    -----
    If the condition can be inlined into a list comprehension, a double list
    comprehension is faster than this function. So if performance is crucial,
    you should inline this function with the condition itself inlined.
    """
    a, b = [], []
    append_a, append_b = a.append, b.append
    for e in seq:
        append_a(e) if condition(e) else append_b(e)
    return a, b


def split_on_values(seq, values):
    """splits an iterable into two lists depending on a list of values

    Parameters
    ----------
    seq : iterable
    values : iterable
        set of values which must go to the first list

    Returns
    -------
    a, b: list
    """
    values = set(values)
    a, b = [], []
    append_a, append_b = a.append, b.append
    for e in seq:
        append_a(e) if e in values else append_b(e)
    return a, b


def skip_comment_cells(lines):
    def notacomment(v):
        return not v.startswith('#')
    for line in lines:
        stripped_line = list(itertools.takewhile(notacomment, line))
        if stripped_line:
            yield stripped_line


def strip_rows(lines):
    """
    returns an iterator of lines with trailing blank (empty or
    which contain only space) cells.
    """
    def isblank(s):
        return s == '' or s.isspace()
    for line in lines:
        rev_line = list(itertools.dropwhile(isblank, reversed(line)))
        yield list(reversed(rev_line))
