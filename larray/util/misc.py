"""
Misc tools.
"""

import __main__
import math
import itertools
import os
import sys
import operator
import warnings
from textwrap import wrap
from functools import reduce, wraps
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd

from larray.util.types import R
from typing import Callable

try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass


def is_interactive_interpreter():
    try:
        # When running using IPython, sys.ps1 is always defined, so we cannot use the standard "hasattr(sys, 'ps1')"
        # Additionally, an InProcessInteractiveShell can have a __main__ module with a file
        main_lacks_file = not hasattr(__main__, '__file__')
        return main_lacks_file or get_ipython().__class__.__name__ == 'InProcessInteractiveShell'
    except NameError:
        return hasattr(sys, 'ps1')


def prod(values):
    return reduce(operator.mul, values, 1)


def format_value(value, missing, precision=None):
    if isinstance(value, float):
        # nans print as "-1.#J", let's use something nicer
        if value != value:
            return missing
        elif precision is not None:
            return f'{value:.{precision}f}'
        else:
            return str(value)
    elif isinstance(value, np.ndarray) and value.shape:
        # prevent numpy's default wrapping
        return str(list(value)).replace(',', '')
    else:
        return str(value)


def get_col_width(table, index):
    return max(len(row[index]) for row in table)


def longest_word(s):
    """Return length of the longest word in the given string.

    Parameters
    ----------
    s : str
        string to check

    Returns
    -------
    int
        length of longest word

    Examples
    --------
    >>> longest_word('12 123 1234')
    4
    >>> longest_word('12 1234 123')
    4
    >>> longest_word('123 12 123')
    3
    >>> longest_word('')
    0
    >>> longest_word(' ')
    0
    """
    return max(len(w) for w in s.split()) if s and not s.isspace() else 0


def get_min_width(table, index):
    return max(longest_word(row[index]) for row in table)


def table2str(table, missing, summarize=True, maxwidth=200, numedges='auto', sep='  ', cont='...',
              keepcols=0, precision=None):
    """
    Convert list of list to string.

    Parameters
    ----------
    table : list of list
        input to convert.
    """
    if not table:
        return ''
    numcol = max(len(row) for row in table)
    # pad rows that have too few columns
    for row in table:
        if len(row) < numcol:
            row.extend([''] * (numcol - len(row)))
    formatted = [[format_value(value, missing, precision) for value in row]
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
                head = keepcols + numedges
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
        wrapped_row = [wrap(value, width) if width > 0 else value
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
    Yield all elements once, preserving order. Remember all elements ever seen.

    >>> list(unique('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    for element in iterable:
        if element not in seen:
            seen_add(element)
            yield element


def unique_list(iterable, res=None, seen=None):
    """
    Return a list of all unique elements, preserving order. Remember all elements ever seen.

    >>> unique_list('AAAABBBCCDAABBB')
    ['A', 'B', 'C', 'D']
    """
    if res is None:
        res = []
    res_append = res.append
    if seen is None:
        seen = set()
    seen_add = seen.add
    for element in iterable:
        if element not in seen:
            seen_add(element)
            res_append(element)
    return res


def unique_multi(iterable_of_iterables):
    """
    Return a list of all unique elements across multiple iterables.

    Elements of earlier iterables will come first.
    """
    seen = set()
    res = []
    for iterable in iterable_of_iterables:
        unique_list(iterable, res, seen)
    return res


def has_duplicates(iterable):
    """
    Return whether iterable contains any duplicated element.
    """
    # using a dict is faster than using a set (at least for Python <= 3.9)
    seen = {}
    for element in iterable:
        if element in seen:
            return True
        else:
            seen[element] = True
    return False


def duplicates(iterable):
    """
    List duplicated elements once, preserving order. Remember all elements ever seen.
    """
    # duplicates('AAAABBBCCDAABBB') --> A B C
    counts = defaultdict(int)
    for element in iterable:
        counts[element] += 1
        if counts[element] == 2:
            yield element


def rproduct(*i):
    return product(*[x[::-1] for x in i])


def light_product(*iterables, repeat=1):
    """Cartesian product of input iterables, replacing repeated values by empty strings.

    Parameters
    ----------
    *iterables : iterable
        Input iterables
    repeat : int, optional
        Number of times to repeat (reuse) input iterables. Defaults to 1.

    Returns
    -------
    Generator

    Examples
    --------
    >>> list(light_product('ab', range(3)))
    [('a', 0), ('', 1), ('', 2), ('b', 0), ('', 1), ('', 2)]
    >>> list(light_product('ab', repeat=2))
    [('a', 'a'), ('', 'b'), ('b', 'a'), ('', 'b')]
    """
    p = product(*iterables, repeat=repeat)
    prev_t = (None,) * len(iterables) * repeat
    for t in p:
        yield tuple(e if e != prev_e else ''
                    for e, prev_e in zip(t, prev_t))
        prev_t = t


def array_nan_equal(a, b):
    if np.issubdtype(a.dtype, str) and np.issubdtype(b.dtype, str):
        return np.array_equal(a, b)
    else:
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)))


def unzip(iterable):
    return list(zip(*iterable))


class ReprString(str):
    def __repr__(self):
        return self


def array_lookup(array, mapping):
    """Pass all elements of an np.ndarray through a mapping."""
    array = np.asarray(array)
    # TODO: this must be cached in the Axis
    # TODO: range axes should be optimized (reuse Pandas 0.18 indexes)
    sorted_keys, sorted_values = tuple(zip(*sorted(mapping.items())))
    sorted_keys = np.array(sorted_keys)
    # prevent an array of booleans from matching a integer axis (sorted_keys)
    # XXX: we might want to allow signed and unsigned integers to match against each other
    if array.dtype.kind != sorted_keys.dtype.kind:
        raise KeyError('key has not the same dtype than axis')
    # TODO: it is very important to fail quickly, so guess_axis should try this in chunks
    # (first test first element of key, if several axes match, try [1:11] elements, [12:112], [113:1113], ...
    if not np.all(np.isin(array, sorted_keys)):
        raise KeyError('all keys not in array')

    sorted_values = np.array(sorted_values)
    if not len(array):
        return np.empty(0, dtype=sorted_values.dtype)
    indices = np.searchsorted(sorted_keys, array)
    return sorted_values[indices]


def array_lookup2(array, sorted_keys, sorted_values):
    """Pass all elements of an np.ndarray through a "mapping"."""
    if not len(array):
        return np.empty(0, dtype=sorted_values.dtype)

    array = np.asarray(array)
    # TODO: range axes should be optimized (reuse Pandas 0.18 indexes)

    # prevent an array of booleans from matching a integer axis (sorted_keys)
    # XXX: we might want to allow signed and unsigned integers to match against each other
    if array.dtype.kind != sorted_keys.dtype.kind:
        raise KeyError('key has not the same dtype than axis')
    # TODO: it is very important to fail quickly, so guess_axis should try this in chunks
    # (first test first element of key, if several axes match, try [1:11] elements, [12:112], [113:1113], ...
    if not np.all(np.isin(array, sorted_keys)):
        raise KeyError('all keys not in array')

    indices = np.searchsorted(sorted_keys, array)
    return sorted_values[indices]


def split_on_condition(seq, condition):
    """Split an iterable into two lists depending on a condition.

    Parameters
    ----------
    seq : iterable
    condition : function(e) -> True or False

    Returns
    -------
    a, b: list

    Notes
    -----
    If the condition can be inlined into a list comprehension, a double list comprehension is faster than this function.
    So if performance is crucial, you should inline this function with the condition itself inlined.
    """
    a, b = [], []
    append_a, append_b = a.append, b.append
    for e in seq:
        append_a(e) if condition(e) else append_b(e)
    return a, b


def split_on_values(seq, values):
    """Split an iterable into two lists depending on a list of values.

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
    Return an iterator of lines without trailing blank cells.

    Empty cells or cells containing only space are considered blank.
    """
    def isblank(s):
        return s == '' or s.isspace()
    for line in lines:
        rev_line = list(itertools.dropwhile(isblank, reversed(line)))
        yield list(reversed(rev_line))


def size2str(value):
    """
    Convert number of bytes to a size string.

    Examples
    --------
    >>> size2str(0)
    '0 bytes'
    >>> size2str(100)
    '100 bytes'
    >>> size2str(1023)
    '1023 bytes'
    >>> size2str(1024)
    '1.00 Kb'
    >>> size2str(2000)
    '1.95 Kb'
    >>> size2str(10000000)
    '9.54 Mb'
    >>> size2str(1.27 * 1024 ** 3)
    '1.27 Gb'
    """
    units = ["bytes", "Kb", "Mb", "Gb", "Tb", "Pb"]
    scale = int(math.log(value, 1024)) if value else 0
    scaled_value = value / 1024.0 ** scale
    return f"{scaled_value:.2f} {units[scale]}" if scale > 0 else f"{int(scaled_value)} bytes"


def find_closing_chr(s, start=0):
    """
    Find the position of character which matches/closes to first character of string s.

    Parameters
    ----------
    s : str
        string to search the characters. s[start] must be in '({['
    start : int, optional
        position in the string from which to start searching

    Returns
    -------
    position of matching brace

    Examples
    --------
    >>> find_closing_chr('(a) + (b)')
    2
    >>> find_closing_chr('(a) + (b)', 6)
    8
    >>> find_closing_chr('(a{b[c(d)e]f}g)')
    14
    >>> find_closing_chr('(a{b[c(d)e]f}g)', 2)
    12
    >>> find_closing_chr('(a{b[c(d)e]f}g)', 4)
    10
    >>> find_closing_chr('(a{b[c(d)e]f}g)', 6)
    8
    >>> find_closing_chr('((a) + (b))')
    10
    >>> find_closing_chr('((a) + (b))')
    10
    >>> find_closing_chr('((a) + (b))')
    10
    >>> find_closing_chr('({)}')
    Traceback (most recent call last):
      ...
    ValueError: malformed expression: expected '}' but found ')'
    >>> find_closing_chr('({}})')
    Traceback (most recent call last):
      ...
    ValueError: malformed expression: expected ')' but found '}'
    >>> find_closing_chr('}()')
    Traceback (most recent call last):
      ...
    ValueError: malformed expression: found '}' before '{'
    >>> find_closing_chr('(()')
    Traceback (most recent call last):
      ...
    ValueError: malformed expression: reached end of string without finding the expected ')'
    """
    opening, closing = '({[', ')}]'
    match = {o: c for o, c in zip(opening, closing)}
    match.update({c: o for o, c in zip(opening, closing)})
    opening_set, closing_set = set(opening), set(closing)

    needle = s[start]
    assert needle in match
    last_open = []
    for pos in range(start, len(s)):
        c = s[pos]
        if c in match:
            if c in opening_set:
                last_open.append(c)
            if c in closing_set:
                if not last_open:
                    raise ValueError(f"malformed expression: found '{c}' before '{match[c]}'")
                expected = match[last_open.pop()]
                if c != expected:
                    raise ValueError(f"malformed expression: expected '{expected}' but found '{c}'")
                if not last_open:
                    assert c == match[needle]
                    return pos
    raise ValueError(f"malformed expression: reached end of string without finding the expected '{match[needle]}'")


def float_error_handler_factory(stacklevel):
    def error_handler(error, flag):
        if error == 'invalid value':
            error = 'invalid value (NaN)'
            extra = ' (this is typically caused by a 0 / 0)'
        else:
            # for division by 0, we use a specific error handler *just* to set the correct stacklevel
            extra = ''
        warnings.warn(f"{error} encountered during operation{extra}", RuntimeWarning, stacklevel=stacklevel)
    return error_handler


def _isintstring(s):
    """
    Return True if the passed string represents an integer.

    Zero padded integers are considered as strings and not integers.

    Parameters
    ----------
    s : str
        string to test if representing an integer.

    Examples
    --------
    >>> _isintstring('12')
    True
    >>> _isintstring('-12')
    True
    >>> _isintstring('a1')
    False
    >>> _isintstring('01')
    False
    """
    def isposint(s):
        # exclude zero padded strings
        return s.isdigit() and not (len(s) > 1 and s[0] == '0')
    return isposint(s) or (len(s) > 1 and s[0] == '-' and isposint(s[1:]))


def _parse_bound(s, stack_depth=1, parse_int=True):
    """Parse a string representing a single value.

    It converts int-like strings to integers and evaluates expressions within {}.

    Parameters
    ----------
    s : str
        string to evaluate
    stack_depth : int
        how deep to go in the stack to get local variables for evaluating {expressions}.

    Returns
    -------
    any

    Examples
    --------
    >>> _parse_bound(' a ')
    'a'
    >>> # returns None
    >>> _parse_bound(' ')
    >>> ext = 1
    >>> _parse_bound(' {ext + 1} ')
    2
    >>> _parse_bound('42')
    42
    >>> _parse_bound('01')
    '01'
    """
    s = s.strip()
    if s == '':
        return None
    elif s[0] == '{':
        expr = s[1:find_closing_chr(s)]
        return eval(expr, sys._getframe(stack_depth).f_locals)
    elif parse_int and _isintstring(s):
        return int(s)
    else:
        return s


def _isnoneslice(v):
    """
    Check if input is slice(None) object.
    """
    return isinstance(v, slice) and v.start is None and v.stop is None and v.step is None


def _seq_summary(seq, n=3, repr_func=repr, sep=' '):
    """
    Return a string representing a sequence by showing only the n first and last elements.

    Examples
    --------
    >>> _seq_summary(range(10), 2)
    '0 1 ... 8 9'
    """
    if len(seq) <= 2 * n:
        short_seq = [repr_func(v) for v in seq]
    else:
        short_seq = [repr_func(v) for v in seq[:n]] + ['...'] + [repr_func(v) for v in seq[-n:]]
    return sep.join(short_seq)


def index_by_id(seq, value):
    """
    Return position of an object in a sequence.

    Raises an error if the object is not in the list.

    Parameters
    ----------
    seq : sequence
        Any sequence (list, tuple, str, unicode).

    value : object
        Object for which you want to retrieve its position in the sequence.

    Raises
    ------
    ValueError
        If `value` object is not contained in the sequence.

    Examples
    --------
    >>> from larray import Axis
    >>> age = Axis('age=0..9')
    >>> sex = Axis('sex=M,F')
    >>> time = Axis('time=2007..2010')
    >>> index_by_id([age, sex, time], sex)
    1
    >>> gender = Axis('sex=M,F')
    >>> index_by_id([age, sex, time], gender)
    Traceback (most recent call last):
        ...
    ValueError: sex is not in list
    >>> gender = sex
    >>> index_by_id([age, sex, time], gender)
    1
    """
    for i, item in enumerate(seq):
        if item is value:
            return i
    raise ValueError(f"{value} is not in list")


def renamed_to(newfunc, old_name, stacklevel=2, raise_error=False):
    if not raise_error:
        def wrapper(*args, **kwargs):
            warnings.warn(f"{old_name}() is deprecated. Use {newfunc.__name__}() instead.",
                          FutureWarning, stacklevel=stacklevel)
            return newfunc(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            raise TypeError(f"{old_name}() is deprecated. Use {newfunc.__name__}() instead.")
    return wrapper


# deprecate_kwarg is derived from pandas.util._decorators (0.21)
def deprecate_kwarg(old_arg_name: str, new_arg_name: str, mapping=None, arg_converter=None, stacklevel=2):
    if mapping is not None and not isinstance(mapping, dict):
        raise TypeError("mapping from old to new argument values must be dict!")

    def _deprecate_kwarg(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> R:
            old_arg_value = kwargs.pop(old_arg_name, None)
            if old_arg_value is not None:
                if mapping is not None:
                    new_arg_value = mapping.get(old_arg_value, old_arg_value)
                elif arg_converter is not None:
                    new_arg_value = arg_converter(old_arg_value)
                else:
                    new_arg_value = old_arg_value
                warnings.warn(f"The {old_arg_name}={old_arg_value!r} keyword is deprecated, "
                              f"use {new_arg_name}={new_arg_value!r} instead", FutureWarning, stacklevel=stacklevel)
                if new_arg_name in kwargs:
                    raise ValueError(f"Can only specify '{old_arg_name}' or '{new_arg_name}', not both")
                else:
                    kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)
        return wrapper
    return _deprecate_kwarg


class lazy_attribute:
    """
    Decorate a method of a class to turn it into an instance attribute when first called.

    Should obviously only be used when the result of the method is constant.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    # descriptor protocol
    # see https://docs.python.org/3/reference/datamodel.html#implementing-descriptors
    def __get__(self, instance, owner):
        if instance is None:
            return self

        func = self.func
        value = func(instance)
        # do not use setattr() if instance is of a class with overridden __setattr__
        object.__setattr__(instance, func.__name__, value)
        return value


def inverseop(opname):
    comparison_ops = {
        'lt': 'gt',
        'gt': 'lt',
        'le': 'ge',
        'ge': 'le',
        'eq': 'eq',
        'ne': 'ne'
    }
    if opname in comparison_ops:
        return comparison_ops[opname]
    else:
        return 'r' + opname


_numeric_kinds = 'buifc'    # Boolean, Unsigned integer, Integer, Float, Complex
_string_kinds = 'SU'        # String, Unicode
_meta_kind = {k: 'str' for k in _string_kinds}
_meta_kind.update({k: 'numeric' for k in _numeric_kinds})


def np_array_common_dtype(arrays) -> np.dtype:
    """
    Return a dtype that all input numpy arrays can be safely cast to.

    Parameters
    ----------
    arrays : Iterable of np.ndarray
        Arrays to inspect.

    Returns
    -------
    np.dtype
        Data type which can hold the data from any input array.

    Notes
    -----
    If the arrays mixes 'numeric' and 'string' types, the function returns 'object' as common type.
    """
    dtypes = [a.dtype for a in arrays]
    meta_kinds = [_meta_kind.get(dt.kind, 'other') for dt in dtypes]
    # mixing string and numeric => object
    if any(mk != meta_kinds[0] for mk in meta_kinds[1:]):
        return np.dtype(object)
    elif meta_kinds[0] == 'numeric':
        return np.result_type(*dtypes)
    elif meta_kinds[0] == 'str':
        need_unicode = any(dt.kind == 'U' for dt in dtypes)
        # unicode are coded with 4 bytes
        max_size = max(dt.itemsize // 4 if dt.kind == 'U' else dt.itemsize
                       for dt in dtypes)
        return np.dtype(('U' if need_unicode else 'S', max_size))
    else:
        return np.dtype(object)


def common_dtype(arrays) -> np.dtype:
    """
    Return a dtype that all input arrays can be safely cast to.

    Parameters
    ----------
    arrays : Iterable of array-like
        Arrays to inspect. Any type convertible to np.ndarray (Array, list, ...)

    Returns
    -------
    np.dtype
        Data type which can hold the data from any input array.

    Notes
    -----
    If the arrays mixes 'numeric' and 'string' types, the function returns 'object' as common type.
    """
    arrays = [np.asarray(a) for a in arrays]
    return np_array_common_dtype(arrays)


class LHDFStore:
    """Context manager for pandas HDFStore."""

    def __init__(self, filepath_or_buffer, **kwargs):
        if isinstance(filepath_or_buffer, pd.HDFStore):
            if not filepath_or_buffer.is_open:
                raise IOError('The HDFStore must be open for reading.')
            self.store = filepath_or_buffer
            self.close_store = False
        else:
            self.store = pd.HDFStore(filepath_or_buffer, **kwargs)
            self.close_store = True

    def __enter__(self):
        return self.store

    def __exit__(self, type_, value, traceback):
        if self.close_store:
            self.store.close()


class SequenceZip:
    """
    Represents the "combination" of several sequences.

    This is very similar to python's builtin zip but only accepts sequences and acts as a Sequence (it can be
    indexed and has a len).

    Parameters
    ----------
    sequences : Iterable of Sequence
        Sequences to combine.

    Examples
    --------
    >>> z = SequenceZip([['a', 'b', 'c'], [1, 2, 3]])
    >>> for i in range(len(z)):
    ...     print(z[i])
    ('a', 1)
    ('b', 2)
    ('c', 3)
    >>> for v in z:
    ...     print(v)
    ('a', 1)
    ('b', 2)
    ('c', 3)
    >>> list(z[1:4])
    [('b', 2), ('c', 3)]
    """

    def __init__(self, sequences):
        self.sequences = sequences
        length = len(sequences[0])
        bad_length_seqs = [i for i, s in enumerate(sequences[1:], start=1) if len(s) != length]
        if bad_length_seqs:
            first_bad = bad_length_seqs[0]
            raise ValueError(f"sequence {first_bad} has a length of {len(sequences[first_bad])} which is different "
                             f"from the length of the first sequence ({length})")
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return tuple(seq[key] for seq in self.sequences)
        else:
            assert isinstance(key, slice), f"key ({key}) has invalid type ({type(key)})"
            return SequenceZip([seq[key] for seq in self.sequences])

    def __iter__(self):
        return zip(*self.sequences)

    def __repr__(self):
        return f'SequenceZip({self.sequences})'


class Repeater:
    """
    Return a virtual sequence with value repeated n times.

    The sequence is never actually created in memory.

    Parameters
    ----------
    value : any
        Value to repeat.
    n : int
        Number of times to repeat value.

    Notes
    -----
    This is very similar to itertools.repeat except this version returns a Sequence instead of an iterator,
    meaning it has a length and can be indexed.

    Examples
    --------
    >>> r = Repeater('a', 3)
    >>> list(r)
    ['a', 'a', 'a']
    >>> r[0]
    'a'
    >>> r[2]
    'a'
    >>> r[3]
    Traceback (most recent call last):
    ...
    IndexError: index out of range
    >>> r[-1]
    'a'
    >>> r[-3]
    'a'
    >>> r[-4]
    Traceback (most recent call last):
    ...
    IndexError: index out of range
    >>> len(r)
    3
    >>> list(r[1:])
    ['a', 'a']
    >>> list(r[:2])
    ['a', 'a']
    >>> list(r[10:])
    []
    """

    def __init__(self, value, n):
        self.value = value
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key >= self.n or key < -self.n:
                raise IndexError('index out of range')
            return self.value
        else:
            assert isinstance(key, slice), f"key ({key}) has invalid type ({type(key)})"
            start, stop, step = key.indices(self.n)
            # XXX: unsure // step is correct
            return Repeater(self.value, (stop - start) // step)

    def __iter__(self):
        return itertools.repeat(self.value, self.n)

    def __repr__(self):
        return f'Repeater({self.value}, {self.n})'


# TODO: remove Product from larray_editor.utils (it is almost identical)
class Product:
    """
    Represents the `cartesian product` of several sequences.

    This is very similar to itertools.product but only accepts sequences and acts as a sequence (it can be
    indexed and has a len).

    Parameters
    ----------
    sequences : Iterable of Sequence
        Sequences on which to apply the cartesian product.

    Examples
    --------
    >>> p = Product([['a', 'b', 'c'], [1, 2]])
    >>> for i in range(len(p)):
    ...     print(p[i])
    ('a', 1)
    ('a', 2)
    ('b', 1)
    ('b', 2)
    ('c', 1)
    ('c', 2)
    >>> p[1:4]
    [('a', 2), ('b', 1), ('b', 2)]
    >>> p[-3:]
    [('b', 2), ('c', 1), ('c', 2)]
    >>> list(p)
    [('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]
    >>> list(Product([['a', 'b', 'c']]))
    [('a',), ('b',), ('c',)]
    >>> list(Product([]))
    [()]
    """

    def __init__(self, sequences):
        self.sequences = sequences
        shape = [len(a) for a in self.sequences]
        self._div_mod = [(int(np.prod(shape[i + 1:])), shape[i])
                         for i in range(len(shape))]
        self._length = np.prod(shape, dtype=int)

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key >= self._length:
                raise IndexError(f"index {key} out of range for Product of length {self._length}")
            # this is similar to np.unravel_index but a tad faster for scalars
            return tuple(array[key // div % mod]
                         for array, (div, mod) in zip(self.sequences, self._div_mod))
        else:
            assert isinstance(key, slice), f"key ({key}) has invalid type ({type(key)})"
            start, stop, step = key.indices(self._length)
            div_mod = self._div_mod
            arrays = self.sequences
            # XXX: we probably want to return another Product object with an updated start/stop to stay
            #      lazy in that case too.
            return [tuple(array[idx // div % mod]
                          for array, (div, mod) in zip(arrays, div_mod))
                    for idx in range(start, stop, step)]

    def __iter__(self):
        return product(*self.sequences)

    def __repr__(self):
        return f'Product({self.sequences})'


_np_generic = np.generic


def _kill_np_type(value):
    return value.item() if isinstance(value, _np_generic) else value


_kill_np_types = np.vectorize(_kill_np_type, otypes=[object])


def ensure_no_numpy_type(array):
    """
    Convert array to a (potentially nested) list of builtin Python values (i.e. using no numpy-specific types).

    Parameters
    ----------
    array : np.ndarray
        array to convert.

    Returns
    -------
    list
        a (potentially nested) list with the same "shape" as `array` but any numpy type converted to the closest
        Python builtin type
    """
    assert isinstance(array, np.ndarray)
    if array.dtype.kind == 'O':
        array = _kill_np_types(array)
    return array.tolist()


# ################# #
#  validator funcs  #
# ################# #

def _positive_integer(value):
    if not (isinstance(value, int) and value > 0):
        raise ValueError("Expected positive integer")


def _validate_dir(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} could not be found")


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def exactly_one(a: bool, b: bool, c: bool = False) -> bool:
    """Return True if exactly one of a, b or c boolean arguments is True, False otherwise."""
    return (a or b) and not (a and b) if not c else not (a or b)


def concatenate_ndarrays(arrays) -> np.ndarray:
    """Concatenate Sequence of np.ndarray, converting to object dtype if needed."""
    dtype = np_array_common_dtype(arrays)
    if dtype.kind == 'O':
        # astype always copies, while asarray only copies if necessary
        arrays = [np.asarray(labels, dtype=object) for labels in arrays]
    # TODO: try using the new dtype argument to concatenate instead of converting labels explicitly as above
    return np.concatenate(arrays)


def first(iterable, default=None):
    return next(iter(iterable), default)


try:
    # Python 3.14+
    import annotationlib

    def get_annotations(namespace):
        # should not happen in Python3.14+ unless
        # "from __future__ import annotations" is used
        if "__annotations__" in namespace:
            return namespace["__annotations__"]
        elif annotate := annotationlib.get_annotate_from_class_namespace(namespace):
            return annotationlib.call_annotate_function(
                annotate, format=annotationlib.Format.FORWARDREF
            )
        else:
            return {}
except ImportError:
    # Python <3.14
    def get_annotations(namespace):
        # any type hints defined in the class body will land in a
        # __annotations__ key in its namespace (this is not pydantic-specific)
        # but __annotations__ is only defined if there are type hints
        return namespace.get('__annotations__', {})


def find_names(obj, depth=0):
    """Return all names an object is bound to.

    Parameters
    ----------
    obj : object
        the object to find names for.
    depth : int
        depth of call frame to inspect. 0 is where find_names was called,
        1 the caller of find_names, etc.

    Returns
    -------
    list of str
        all names obj is bound to, sorted alphabetically. Can be [] if we
        computed an array just to view it.
    """
    # noinspection PyProtectedMember
    local_vars = sys._getframe(depth + 1).f_locals
    names = [k for k, v in local_vars.items() if v is obj]
    if any(not name.startswith('_') for name in names):
        names = [name for name in names if not name.startswith('_')]
    return sorted(names)


PY312_OR_LATER = sys.version_info[:2] >= (3, 12)
