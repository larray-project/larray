# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

__version__ = "0.21"

__all__ = [
    'LArray', 'Axis', 'AxisCollection', 'LGroup', 'LSet', 'PGroup',
    'union', 'stack',
    'read_csv', 'read_eurostat', 'read_excel', 'read_hdf', 'read_tsv',
    'read_sas',
    'x',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like',
    'full', 'full_like', 'create_sequential', 'ndrange', 'labels_array',
    'ndtest', 'from_lists', 'from_string',
    'identity', 'diag', 'eye',
    'larray_equal', 'aslarray',
    'all', 'any', 'sum', 'prod', 'cumsum', 'cumprod', 'min', 'max', 'mean',
    'ptp', 'var', 'std', 'median', 'percentile',
    '__version__'
]

"""
Matrix class
"""

# * when trying to aggregate on an non existing Axis (using x.blabla),
#   the error message is awful

# ? implement multi group in one axis getitem:
#   lipro['P01,P02;P05'] <=> (lipro.group('P01,P02'), lipro.group('P05'))
#                        <=> (lipro['P01,P02'], lipro['P05'])

# discuss VG with Geert:
# I do not "expand" key (eg :) upon group creation for perf reason
# VG[:] is much faster than [A01,A02,...,A99]
# I could make that all "contiguous" ranges are conv to slices (return views)
# but that might introduce confusing differences if they update/setitem their
# arrays

# * we need an API to get to the "next" label. Sometimes, we want to use
#   label+1, but when label is not numeric, or has not a step of 1, that's
#   problematic. x.agegroup[x.agegroup.after(25):]

# * implement keepaxes=True for _group_aggregate instead of/in addition to
#   group tuples

# ? implement newaxis

# * split unit tests

# * reindex array (ie make it conform to another index, eg of another
#   array). This can be used both for doing operations (add, divide, ...)
#   involving arrays with incompatible axes and to (manually) reorder one axis
#   labels

# * test to_csv: does it consume too much mem?
#   ---> test pandas (one dimension horizontally)

# * add labels in LGroups.__str__

# * docstring for all methods

# * IO functions: csv/hdf/excel?/...?
#   >> needs discussion of the formats (users involved in the discussion?)
#      + check pandas dialects
# * plotting (see plot.py)
#   >> check pandas API
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

import csv
from itertools import product, chain, groupby, islice
import os
import re
import sys
import warnings

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

import numpy as np
import pandas as pd

try:
    import xlwings as xw
except ImportError:
    xw = None

try:
    from numpy import nanprod as np_nanprod
except ImportError:
    np_nanprod = None

from larray.oset import *
from larray.utils import (table2str, size2str, unique, csv_open, unzip, long,
                          decode, basestring, unicode, bytes, izip, rproduct,
                          ReprString, duplicates, array_lookup2, strip_rows,
                          skip_comment_cells, find_closing_chr, StringIO, PY3)


def _range_to_slice(seq, length=None):
    """
    Returns a slice if possible (including for sequences of 1 element)
    otherwise returns the input sequence itself

    Parameters
    ----------
    seq : sequence-like of int
        List, tuple or ndarray of integers representing the range.
        It should be something like [start, start+step, start+2*step, ...]
    length : int, optional
        length of sequence of positions.
        This is only useful when you must be able to transform decreasing
        sequences which can stop at 0.

    Returns
    -------
    slice or sequence-like
        return the input sequence if a slice cannot be defined

    Examples
    --------
    >>> _range_to_slice([3, 4, 5])
    slice(3, 6, None)
    >>> _range_to_slice([3, 5, 7])
    slice(3, 9, 2)
    >>> _range_to_slice([-3, -2])
    slice(-3, -1, None)
    >>> _range_to_slice([-1, -2])
    slice(-1, -3, -1)
    >>> _range_to_slice([2, 1])
    slice(2, 0, -1)
    >>> _range_to_slice([1, 0], 4)
    slice(-3, -5, -1)
    >>> _range_to_slice([1, 0])
    [1, 0]
    >>> _range_to_slice([1])
    slice(1, 2, None)
    >>> _range_to_slice([])
    []
    """
    if len(seq) < 1:
        return seq
    start = seq[0]
    if len(seq) == 1:
        return slice(start, start + 1)
    second = seq[1]
    step = second - start
    prev_value = second
    for value in seq[2:]:
        if value != prev_value + step:
            return seq
        prev_value = value
    stop = prev_value + step
    if prev_value == 0 and step < 0:
        if length is None:
            return seq
        else:
            stop -= length
            start -= length
    if step == 1:
        step = None
    return slice(start, stop, step)


def _slice_to_str(key, repr_func=str):
    """
    Converts a slice to a string

    Examples
    --------
    >>> _slice_to_str(slice(None))
    ':'
    >>> _slice_to_str(slice(24))
    ':24'
    >>> _slice_to_str(slice(25, None))
    '25:'
    >>> _slice_to_str(slice(5, 10))
    '5:10'
    >>> _slice_to_str(slice(None, 5, 2))
    ':5:2'
    """
    # examples of result: ":24" "25:" ":" ":5:2"
    start = repr_func(key.start) if key.start is not None else ''
    stop = repr_func(key.stop) if key.stop is not None else ''
    step = (":" + repr_func(key.step)) if key.step is not None else ''
    return '%s:%s%s' % (start, stop, step)


def irange(start, stop, step=None):
    """Create a range, with inclusive stop bound and automatic sign for step.

    Parameters
    ----------
    start : int
        Start bound
    stop : int
        Inclusive stop bound
    step : int, optional
        Distance between two generated numbers. If provided this *must* be a positive integer.

    Returns
    -------
    range

    Examples
    --------
    >>> list(irange(1, 3))
    [1, 2, 3]
    >>> list(irange(2, 0))
    [2, 1, 0]
    >>> list(irange(1, 6, 2))
    [1, 3, 5]
    >>> list(irange(6, 1, 2))
    [6, 4, 2]
    >>> list(irange(-1, 1))
    [-1, 0, 1]
    """
    if step is None:
        step = 1
    else:
        assert step > 0
    step = step if start <= stop else -step
    stop = stop + 1 if start <= stop else stop - 1
    return range(start, stop, step)


_range_bound_pattern = re.compile('([0-9]+|[a-zA-Z]+)')

def generalized_range(start, stop, step=1):
    """Create a range, with inclusive stop bound and automatic sign for step. Bounds can be strings.

    Parameters
    ----------
    start : int or str
        Start bound
    stop : int or str
        Inclusive stop bound
    step : int, optional
        Distance between two generated numbers. If provided this *must* be a positive integer.

    Returns
    -------
    range

    Examples
    --------
    works with both number and letter bounds

    >>> list(generalized_range(-1, 2))
    [-1, 0, 1, 2]
    >>> generalized_range('a', 'c')
    ['a', 'b', 'c']

    can generate in reverse

    >>> list(generalized_range(2, 0))
    [2, 1, 0]
    >>> generalized_range('c', 'a')
    ['c', 'b', 'a']

    can combine letters and numbers

    >>> generalized_range('a0', 'c1')
    ['a0', 'a1', 'b0', 'b1', 'c0', 'c1']

    any special character is left intact

    >>> generalized_range('a_0', 'c_1')
    ['a_0', 'a_1', 'b_0', 'b_1', 'c_0', 'c_1']

    consecutive digits are treated like numbers

    >>> generalized_range('P01', 'P12')
    ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12']

    consecutive letters create all combinations

    >>> generalized_range('AA', 'CC')
    ['AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC']
    """
    if isinstance(start, str):
        assert isinstance(stop, str)
        start_parts = _range_bound_pattern.split(start)
        stop_parts = _range_bound_pattern.split(stop)
        assert len(start_parts) == len(stop_parts)
        ranges = []
        for start_part, stop_part in zip(start_parts, stop_parts):
            # we only handle non-negative int-like strings on purpose. Int-only bounds should already be converted to
            # real integers by now, and mixing negative int-like strings and letters yields some strange results.
            if start_part.isdigit():
                assert stop_part.isdigit()
                numchr = max(len(start_part), len(stop_part))
                r = ['%0*d' % (numchr, num) for num in irange(int(start_part), int(stop_part))]
            elif start_part.isalpha():
                assert stop_part.isalpha()
                int_start = [ord(c) for c in start_part]
                int_stop = [ord(c) for c in stop_part]
                sranges = [[chr(c) for c in irange(r_start, r_stop) if chr(c).isalnum()]
                           for r_start, r_stop in zip(int_start, int_stop)]
                r = [''.join(p) for p in product(*sranges)]
            else:
                # special characters
                assert start_part == stop_part
                r = [start_part]
            ranges.append(r)
        res = [''.join(p) for p in product(*ranges)]
        return res if step == 1 else res[::step]
    else:
        return irange(start, stop, step)


_range_str_pattern = re.compile('(?P<start>[^\s.]+)?\s*\.\.\s*(?P<stop>[^\s.]+)?(\s+step\s+(?P<step>\d+))?')


def _range_str_to_range(s):
    """
    Converts a range string to a range (of values).
    The end point is included.

    Parameters
    ----------
    s : str
        String representing a range of values

    Returns
    -------
    range
        range of int or list of str.

    Examples
    --------
    >>> list(_range_str_to_range('-1..2'))
    [-1, 0, 1, 2]
    >>> _range_str_to_range('a..c')
    ['a', 'b', 'c']
    >>> list(_range_str_to_range('2..6 step 2'))
    [2, 4, 6]

    any special character except . and spaces should work
    >>> _range_str_to_range('a|+*@-b .. a|+*@-d')
    ['a|+*@-b', 'a|+*@-c', 'a|+*@-d']
    """
    m = _range_str_pattern.match(s)

    groups = m.groupdict()
    start, stop, step = groups['start'], groups['stop'], groups['step']
    start = _parse_bound(start) if start is not None else 0
    if stop is None:
        raise ValueError("no stop bound provided in range: %r" % s)
    stop = _parse_bound(stop)
    step = int(step) if step is not None else 1
    return generalized_range(start, stop, step)


def _to_tick(v):
    """
    Converts any value to a tick (ie makes it hashable, and acceptable as an ndarray element)

    scalar -> not modified
    slice -> 'start:stop'
    list|tuple -> 'v1,v2,v3'
    Group with name -> v.name
    Group without name -> _to_tick(v.key)
    other -> str(v)

    Parameters
    ----------
    v : any
        value to be converted.

    Returns
    -------
    any scalar
        scalar representing the tick
    """
    # the fact that an "aggregated tick" is passed as a LGroup or as a
    # string should be as irrelevant as possible. The thing is that we cannot
    # (currently) use the more elegant _to_tick(e.key) that means the
    # LGroup is not available in Axis.__init__ after to_ticks, and we
    # need it to update the mapping if it was named. Effectively,
    # this creates two entries in the mapping for a single tick. Besides,
    # I like having the LGroup as the tick, as it provides extra info as
    # to where it comes from.
    if np.isscalar(v):
        return v
    elif isinstance(v, Group):
        return v.name if v.name is not None else _to_tick(v.key)
    elif isinstance(v, slice):
        return _slice_to_str(v)
    elif isinstance(v, (tuple, list)):
        if len(v) == 1:
            return str(v) + ','
        else:
            # TODO: it would be nicer/saner to use n=1, sep='' but this currently breaks at lot of tests
            return _seq_summary(v, n=1000, repr_func=str, sep=',')
    else:
        return str(v)


def _to_ticks(s):
    """
    Makes a (list of) value(s) usable as the collection of labels for an
    Axis (ie hashable). Strip strings, split them on ',' and translate
    "range strings" to list of values **including the end point** !

    Parameters
    ----------
    s : iterable
        List of values usable as the collection of labels for an Axis.

    Returns
    -------
    collection of labels

    Notes
    -----
    This function is only used in Axis.__init__ and union().

    Examples
    --------
    >>> _to_ticks('M , F')
    ['M', 'F']

    >>> list(_to_ticks('..3'))
    [0, 1, 2, 3]
    """
    if isinstance(s, Group):
        # a single LGroup used for all ticks of an Axis
        return _to_ticks(s.eval())
    elif isinstance(s, pd.Index):
        return s.values
    elif isinstance(s, np.ndarray):
        # we assume it has already been translated
        # XXX: Is it a safe assumption?
        return s
    elif isinstance(s, (list, tuple)):
        return [_to_tick(e) for e in s]
    elif sys.version >= '3' and isinstance(s, range):
        return list(s)
    elif isinstance(s, basestring):
        if ':' in s:
            raise ValueError("using : to define axes is deprecated, please use .. instead")
        elif '..' in s:
            return _range_str_to_range(s)
        else:
            return [v.strip() for v in s.split(',')]
    elif hasattr(s, '__array__'):
        return s.__array__()
    else:
        try:
            return list(s)
        except TypeError:
            raise TypeError("ticks must be iterable (%s is not)" % type(s))


def _isintstring(s):
    return s.isdigit() or (len(s) > 1 and s[0] == '-' and s[1:].isdigit())


def _parse_bound(s, stack_depth=1, parse_int=True):
    """Parse a string representing a single value, converting int-like
    strings to integers and evaluating expressions within {}.

    Parameters
    ----------
    s : str
        string to evaluate
    stack_depth : int
        how deep to go in the stack to get local variables for evaluating
        {expressions}.

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


_axis_name_pattern = re.compile('\s*(([A-Za-z]\w*)(\.i)?\s*\[)?(.*)')


def _to_key(v, stack_depth=1, parse_single_int=False):
    """
    Converts a value to a key usable for indexing (slice object, list of values,...).
    Strings are split on ',' and stripped. Colons (:) are interpreted as slices.

    Parameters
    ----------
    v : int or basestring or tuple or list or slice or LArray or Group
        value to convert into a key usable for indexing

    Returns
    -------
    key
        a key represents any object that can be used for indexing

    Examples
    --------
    >>> _to_key('a:c')
    slice('a', 'c', None)
    >>> _to_key('a, b,c ,')
    ['a', 'b', 'c']
    >>> _to_key('a,')
    ['a']
    >>> _to_key(' a ')
    'a'
    >>> _to_key(10)
    10
    >>> _to_key('10')
    '10'
    >>> _to_key('10:20')
    slice(10, 20, None)
    >>> _to_key(slice('10', '20'))
    slice('10', '20', None)
    >>> _to_key('year.i[-1]')
    year.i[-1]
    >>> _to_key('age[10:19]>>teens')
    age[10:19] >> 'teens'
    >>> _to_key('a,b,c >> abc')
    LGroup(['a', 'b', 'c']) >> 'abc'
    >>> _to_key('a:c >> abc')
    LGroup(slice('a', 'c', None)) >> 'abc'

    # evaluated variables do not work on Python 2, probably because the stackdepth is different
    # >>> ext = [1, 2, 3]
    # >>> _to_key('{ext} >> ext')
    # LGroup([1, 2, 3], name='ext')
    # >>> answer = 42
    # >>> _to_key('{answer}')
    # 42
    # >>> _to_key('{answer} >> answer')
    # LGroup(42, name='answer')
    # >>> _to_key('10:{answer} >> answer')
    # LGroup(slice(10, 42, None), name='answer')
    # >>> _to_key('4,{answer},2 >> answer')
    # LGroup([4, 42, 2], name='answer')
    """
    if isinstance(v, tuple):
        return list(v)
    elif isinstance(v, Group):
        return v.__class__(_to_key(v.key, stack_depth + 1), v.name, v.axis)
    elif isinstance(v, basestring):
        # axis name
        m = _axis_name_pattern.match(v)
        _, axis, positional, key = m.groups()
        # group name. using rfind in the unlikely case there is another >>
        name_pos = key.rfind('>>')
        name = None
        if name_pos != -1:
            key, name = key[:name_pos].strip(), key[name_pos + 2:].strip()
        if axis is not None:
            axis = axis.strip()
            axis_bracket_open = m.end(1) - 1
            # check that the string parentheses are correctly balanced
            _ = find_closing_chr(v, axis_bracket_open)
            # strip closing bracket (it should be at the end because we took
            # care of the name earlier)
            assert key[-1] == ']'
            key = key[:-1]
        cls = PGroup if positional else LGroup
        if name is not None or axis is not None:
            key = _to_key(key, stack_depth + 1, parse_single_int=positional)
            return cls(key, name=name, axis=axis)
        else:
            numcolons = v.count(':')
            if numcolons:
                assert numcolons <= 2
                # bounds can be of len 2 or 3 (if step is provided)
                # stack_depth + 2 because the list comp has its own stack
                bounds = [_parse_bound(b, stack_depth + 2)
                          for b in v.split(':')]
                return slice(*bounds)
            else:
                if ',' in v:
                    # strip extremity commas to avoid empty string keys
                    v = v.strip(',')
                    # stack_depth + 2 because the list comp has its own stack
                    return [_parse_bound(b, stack_depth + 2)
                            for b in v.split(',')]
                else:
                    return _parse_bound(v, stack_depth + 1, parse_int=parse_single_int)
    elif v is Ellipsis or np.isscalar(v) or isinstance(v, (slice, list, np.ndarray, LArray, OrderedSet)):
        return v
    else:
        raise TypeError("%s has an invalid type (%s) for a key"
                        % (v, type(v).__name__))


def to_keys(value, stack_depth=1):
    """
    Converts a (collection of) group(s) to a structure usable for indexing.
    'label' or ['l1', 'l2'] or [['l1', 'l2'], ['l3']]

    Parameters
    ----------
    value : int or basestring or tuple or list or slice or LArray or Group
        (collection of) value(s) to convert into key(s) usable for indexing

    Returns
    -------
    list of keys

    Examples
    --------
    It is only used for .sum(axis=xxx)
    >>> to_keys('P01,P02')  # <-- one group => collapse dimension
    ['P01', 'P02']
    >>> to_keys(('P01,P02',))  # <-- do not collapse dimension
    (['P01', 'P02'],)
    >>> to_keys('P01;P02,P03;:')
    ('P01', ['P02', 'P03'], slice(None, None, None))

    # evaluated variables do not work on Python 2, probably because the stack depth is different
    # >>> ext = 'P03'
    # >>> to_keys('P01,P02,{ext}')
    # ['P01', 'P02', 'P03']
    # >>> to_keys('P01;P02;{ext}')
    # ('P01', 'P02', 'P03')

    >>> to_keys('age[10:19] >> teens ; year.i[-1]')
    (age[10:19] >> 'teens', year.i[-1])

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
    if isinstance(value, basestring) and ';' in value:
        value = tuple(value.split(';'))

    if isinstance(value, tuple):
        # stack_depth + 2 because the list comp has its own stack
        return tuple([_to_key(group, stack_depth + 2) for group in value])
    else:
        return _to_key(value, stack_depth + 1)


def union(*args):
    # TODO: add support for LGroup and lists
    """
    Returns the union of several "value strings" as a list.

    Parameters
    ----------
    *args
        (collection of) value(s) to be converted into label(s).
        Repeated values are taken only once.

    Returns
    -------
    list of labels

    Examples
    --------
    >>> union('a', 'a, b, c, d', ['d', 'e', 'f'], '..2')
    ['a', 'b', 'c', 'd', 'e', 'f', 0, 1, 2]
    """
    if args:
        return list(unique(chain(*(_to_ticks(arg) for arg in args))))
    else:
        return []


def larray_equal(first, other):
    """
    Compares two arrays and returns True if they have the
    same axes and elements, False otherwise.

    Parameters
    ----------
    first, other : LArray
        Input arrays.

    Returns
    -------
    bool
        Returns True if the arrays are equal.

    Examples
    --------
    >>> age = Axis('age', range(0, 100, 10))
    >>> sex = Axis('sex', ['M', 'F'])
    >>> a = ndrange([age, sex])
    >>> b = a.copy()
    >>> larray_equal(a, b)
    True
    >>> b['F'] += 1
    >>> larray_equal(a, b)
    False
    >>> b = a.set_labels(x.sex, ['Men', 'Women'])
    >>> larray_equal(a, b)
    False
    """
    if not isinstance(first, LArray) or not isinstance(other, LArray):
        return False
    return (first.axes == other.axes and
            np.array_equal(np.asarray(first), np.asarray(other)))


def _isnoneslice(v):
    """
    Checks if input is slice(None) object.
    """
    return isinstance(v, slice) and v.start is None and v.stop is None and v.step is None


def _seq_summary(seq, n=3, repr_func=repr, sep=' '):
    """
    Returns a string representing a sequence by showing only the n first and last elements.

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


class PGroupMaker(object):
    """
    Generates a new instance of PGroup for a given axis and key.

    Attributes
    ----------
    axis : Axis
        an axis.

    Notes
    -----
    This class is used by the method `Axis.i`
    """
    def __init__(self, axis):
        assert isinstance(axis, Axis)
        self.axis = axis

    def __getitem__(self, key):
        return PGroup(key, None, self.axis)


def _is_object_array(array):
    return isinstance(array, np.ndarray) and array.dtype.type == np.object_


def _can_have_groups(seq):
    return _is_object_array(seq) or isinstance(seq, (tuple, list))


def _contain_group_ticks(ticks):
    return _can_have_groups(ticks) and any(isinstance(tick, Group) for tick in ticks)


def _seq_group_to_name(seq):
    if _can_have_groups(seq):
        return [v.name if isinstance(v, Group) else v for v in seq]
    else:
        return seq


class Axis(object):
    """
    Represents an axis. It consists of a name and a list of labels.

    Parameters
    ----------
    name : str or Axis
        name of the axis or another instance of Axis.
        In the second case, the name of the other axis is simply copied.
    labels : array-like or int
        collection of values usable as labels, i.e. numbers or strings or the size of the axis.
        In the last case, a wildcard axis is created.

    Attributes
    ----------
    name : str
        name of the axis.
    labels : array-like or int
        collection of values usable as labels, i.e. numbers or strings

    Examples
    --------
    >>> age = Axis('age', 10)
    >>> age
    Axis('age', 10)
    >>> age.name
    'age'
    >>> age.labels
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> sex
    Axis('sex', ['M', 'F'])
    """
    # ticks instead of labels?
    # XXX: make name and labels optional?
    def __init__(self, name, labels):
        if isinstance(name, Axis):
            name = name.name
        # make sure we do not have np.str_ as it causes problems down the
        # line with xlwings. Cannot use isinstance to check that though.
        is_python_str = type(name) is unicode or type(name) is bytes
        assert name is None or isinstance(name, int) or is_python_str, \
            type(name)
        self.name = name
        self._labels = None
        self.__mapping = None
        self.__sorted_keys = None
        self.__sorted_values = None
        self._length = None
        self._iswildcard = False
        self.labels = labels

    @property
    def _mapping(self):
        # To map labels with their positions
        mapping = self.__mapping
        if mapping is None:
            labels = self._labels
            # TODO: this would be more efficient for wildcard axes but
            # does not work in all cases
            # mapping = labels
            mapping = {label: i for i, label in enumerate(labels)}
            if not self._iswildcard:
                # we have no choice but to do that!
                # otherwise we could not make geo['Brussels'] work efficiently
                # (we could have to traverse the whole mapping checking for each
                # name, which is not an option)
                # TODO: only do this if labels.dtype is object, or add
                # "contains_lgroup" flag in above code (if any(...))
                # 0.179
                mapping.update({label.name: i for i, label in enumerate(labels)
                                if isinstance(label, Group)})
            self.__mapping = mapping
        return mapping

    def _update_key_values(self):
        mapping = self._mapping
        if mapping:
            sorted_keys, sorted_values = tuple(zip(*sorted(mapping.items())))
        else:
            sorted_keys, sorted_values = (), ()
        keys, values = np.array(sorted_keys), np.array(sorted_values)
        self.__sorted_keys = keys
        self.__sorted_values = values
        return keys, values

    @property
    def _sorted_keys(self):
        if self.__sorted_keys is None:
            keys, _ = self._update_key_values()
        return self.__sorted_keys

    @property
    def _sorted_values(self):
        values = self.__sorted_values
        if values is None:
            _, values = self._update_key_values()
        return values

    @property
    def i(self):
        """
        Allows to define a subset using positions along the axis
        instead of labels.

        Examples
        --------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> arr = ndrange([sex, time])
        >>> arr
        sex\\time | 2007 | 2008 | 2009 | 2010
               M |    0 |    1 |    2 |    3
               F |    4 |    5 |    6 |    7
        >>> arr[time.i[0, -1]]
        sex\\time | 2007 | 2010
               M |    0 |    3
               F |    4 |    7
        """
        return PGroupMaker(self)

    @property
    def labels(self):
        """
        List of labels.
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            raise TypeError("labels should be a sequence or a single int")
        if isinstance(labels, (int, long)):
            length = labels
            labels = np.arange(length)
            iswildcard = True
        else:
            # TODO: move this to _to_ticks????
            # we convert to an ndarray to save memory for scalar ticks (for
            # LGroup ticks, it does not make a difference since a list of VG
            # and an ndarray of VG are both arrays of pointers)
            ticks = _to_ticks(labels)
            if _contain_group_ticks(ticks):
                # avoid getting a 2d array if all LGroup have the same length
                labels = np.empty(len(ticks), dtype=object)
                # this does not work if some values have a length (with a valid __len__) and others not
                # labels[:] = ticks
                for i, tick in enumerate(ticks):
                    labels[i] = tick
            else:
                labels = np.asarray(ticks)
            length = len(labels)
            iswildcard = False

        self._length = length
        self._labels = labels
        self._iswildcard = iswildcard

    def by(self, length, step=None):
        """Split axis into several groups of specified length.

        Parameters
        ----------
        length : int
            length of groups
        step : int, optional
            step between groups. Defaults to length.

        Notes
        -----
        step can be smaller than length, in which case, this will produce overlapping groups.

        Returns
        -------
        list of Group

        Examples
        --------
        >>> age = Axis('age', range(10))
        >>> age.by(3)
        (age.i[0:3], age.i[3:6], age.i[6:9], age.i[9:10])
        >>> age.by(3, 4)
        (age.i[0:3], age.i[4:7], age.i[8:10])
        >>> age.by(5, 3)
        (age.i[0:5], age.i[3:8], age.i[6:10], age.i[9:10])
        """
        return self[:].by(length, step)

    def extend(self, labels):
        """
        Append new labels to an axis or increase its length
        in case of wildcard axis.
        Note that `extend` does not occur in-place: a new axis
        object is allocated, filled and returned.

        Parameters
        ----------
        labels : int, iterable or Axis
            New labels to append to the axis.
            Passing directly another Axis is also possible.
            If the current axis is a wildcard axis, passing a length is enough.

        Returns
        -------
        Axis
            A copy of the axis with new labels appended to it or
            with increased length (if wildcard).

        Examples
        --------
        >>> time = Axis('time', [2007, 2008])
        >>> time
        Axis('time', [2007, 2008])
        >>> time.extend([2009, 2010])
        Axis('time', [2007, 2008, 2009, 2010])
        >>> waxis = Axis('wildcard_axis', 10)
        >>> waxis
        Axis('wildcard_axis', 10)
        >>> waxis.extend(5)
        Axis('wildcard_axis', 15)
        >>> waxis.extend([11, 12, 13, 14])
        Traceback (most recent call last):
        ...
        ValueError: Axis to append must (not) be wildcard if self is (not) wildcard
        """
        other = labels if isinstance(labels, Axis) else Axis(None, labels)
        if self.iswildcard != other.iswildcard:
            raise ValueError ("Axis to append must (not) be wildcard if " +
                              "self is (not) wildcard")
        labels = self._length + other._length if self.iswildcard else np.append(self.labels, other.labels)
        return Axis(self.name, labels)

    @property
    def iswildcard(self):
        return self._iswildcard

    def _group(self, *args, **kwargs):
        """
        Deprecated.

        Parameters
        ----------
        *args
            (collection of) selected label(s) to form a group.
        **kwargs
            name of the group. There is no other accepted keywords.

        Examples
        --------
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> odd_years = time._group([2007, 2009], name='odd_years')
        >>> odd_years
        time[2007, 2009] >> 'odd_years'
        """
        name = kwargs.pop('name', None)
        if kwargs:
            raise ValueError("invalid keyword argument(s): %s" % list(kwargs.keys()))
        key = args[0] if len(args) == 1 else args
        return self[key] >> name if name else self[key]

    def group(self,  *args, **kwargs):
        group_name = kwargs.pop('name', None)
        key = args[0] if len(args) == 1 else args
        syntax = '{}[{}]'.format(self.name if self.name else 'axis', key)
        if group_name is not None:
            syntax += ' >> {}'.format(repr(group_name))
        raise NotImplementedError('Axis.group is deprecated. Use {} instead.'.format(syntax))

    def all(self, name=None):
        """
        (Deprecated) Returns a group containing all labels.

        Parameters
        ----------
        name : str, optional
            Name of the group. If not provided, name is set to 'all'.
        """
        axis_name = self.name if self.name else 'axis'
        group_name = name if name else 'all'
        raise NotImplementedError('Axis.all is deprecated. '
                                  'Use {}[:] >> {} instead.'.format(axis_name, repr(group_name)))

    def subaxis(self, key, name=None):
        """
        Returns an axis for a sub-array.

        Parameters
        ----------
        key : int, or collection (list, slice, array, LArray) of them
            Position(s) of labels to use for the new axis.
        name : str, optional
            Name of the subaxis. Defaults to the name of the parent axis.

        Returns
        -------
        Axis
            Subaxis.
            If key is a None slice and name is None, the original Axis is returned.
            If key is a LArray, the list of axes is returned.

        Examples
        --------
        >>> age = Axis('age', range(100))
        >>> age.subaxis(range(10, 19), name='teenagers')
        Axis('teenagers', [10, 11, 12, 13, 14, 15, 16, 17, 18])
        """
        if (name is None and isinstance(key, slice) and
                key.start is None and key.stop is None and key.step is None):
            return self
        # we must NOT modify the axis name, even though this creates a new axis
        # that is independent from the original one because the original
        # name is probably what users will want to use to filter
        if name is None:
            name = self.name
        if isinstance(key, LArray):
            return tuple(key.axes)
        # TODO: compute length for wildcard axes more efficiently
        labels = len(self.labels[key]) if self.iswildcard else self.labels[key]
        return Axis(name, labels)

    def iscompatible(self, other):
        """
        Checks if self is compatible with another axis.

        * Two non-wildcard axes are compatible if they have the same name and labels.
        * A wildcard axis of length 1 is compatible with any other axis sharing the same name.
        * A wildcard axis of length > 1 is compatible with any axis of the same length or length 1 and sharing the
          same name.

        Parameters
        ----------
        other : Axis
            Axis to compare with.

        Returns
        -------
        bool
            True if input axis is compatible with self, False otherwise.

        Examples
        --------
        >>> a10  = Axis('a', range(10))
        >>> wa10 = Axis('a', 10)
        >>> wa1  = Axis('a', 1)
        >>> b10  = Axis('b', range(10))
        >>> a10.iscompatible(b10)
        False
        >>> a10.iscompatible(wa10)
        True
        >>> a10.iscompatible(wa1)
        True
        >>> wa1.iscompatible(b10)
        False
        """
        if self is other:
            return True
        if not isinstance(other, Axis):
            return False
        if self.name is not None and other.name is not None and self.name != other.name:
            return False
        if self.iswildcard or other.iswildcard:
            # wildcard axes of length 1 match with anything
            # wildcard axes of length > 1 match with equal len or len 1
            return len(self) == 1 or len(other) == 1 or len(self) == len(other)
        else:
            return np.array_equal(self.labels, other.labels)

    def equals(self, other):
        """
        Checks if self is equal to another axis.
        Two axes are equal if the have the same name and label(s).

        Parameters
        ----------
        other : Axis
            Axis to compare with.

        Returns
        -------
        bool
            True if input axis is equal to self, False otherwise.

        Examples
        --------
        >>> age = Axis('age', range(5))
        >>> age_2 = Axis('age', 5)
        >>> age_3 = Axis('young children', range(5))
        >>> age_4 = Axis('age', [0, 1, 2, 3, 4])
        >>> age.equals(age_2)
        False
        >>> age.equals(age_3)
        False
        >>> age.equals(age_4)
        True
        """
        if self is other:
            return True

        # this might need to change if we ever support wildcard axes with
        # real labels
        return isinstance(other, Axis) and self.name == other.name and \
               self.iswildcard == other.iswildcard and \
               (len(self) == len(other) if self.iswildcard else
                    np.array_equal(self.labels, other.labels))

    def matches(self, pattern):
        """
        Returns a group with all the labels matching the specified pattern (regular expression).

        Parameters
        ----------
        pattern : str or Group
            Regular expression (regex).

        Returns
        -------
        LGroup
            Group containing all the labels matching the pattern.

        Notes
        -----
        See `Regular Expression <https://docs.python.org/3/library/re.html>`_
        for more details about how to build a pattern.

        Examples
        --------
        >>> people = Axis('people', ['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'])

        All labels starting with "W" and ending with "o" are given by

        >>> people.matches('W.*o')
        people['Waldo']

        All labels not containing character "a"

        >>> people.matches('[^a]*$')
        people['Bruce Willis', 'Arthur Dent']
        """
        if isinstance(pattern, Group):
            pattern = pattern.eval()
        rx = re.compile(pattern)
        return LGroup([v for v in self.labels if rx.match(v)], axis=self)

    def startswith(self, prefix):
        """
        Returns a group with the labels starting with the specified string.

        Parameters
        ----------
        prefix : str or Group
            The prefix to search for.

        Returns
        -------
        LGroup
            Group containing all the labels starting with the given string.

        Examples
        --------
        >>> people = Axis('people', ['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'])
        >>> people.startswith('Bru')
        people['Bruce Wayne', 'Bruce Willis']
        """
        if isinstance(prefix, Group):
            prefix = prefix.eval()
        return LGroup([v for v in self.labels if v.startswith(prefix)], axis=self)

    def endswith(self, suffix):
        """
        Returns a LGroup with the labels ending with the specified string

        Parameters
        ----------
        suffix : str or Group
            The suffix to search for.

        Returns
        -------
        LGroup
            Group containing all the labels ending with the given string.

        Examples
        --------
        >>> people = Axis('people', ['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'])
        >>> people.endswith('Dent')
        people['Arthur Dent', 'Harvey Dent']
        """
        if isinstance(suffix, Group):
            suffix = suffix.eval()
        return LGroup([v for v in self.labels if v.endswith(suffix)], axis=self)

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter([self.i[i] for i in range(self._length)])

    def __getitem__(self, key):
        """
        Returns a group (list or unique element) of label(s) usable in .sum or .filter

        key is a label-based key (slice and fancy indexing are supported)

        Returns
        -------
        Group
            group containing selected label(s)/position(s).

        Notes
        -----
        key is label-based (slice and fancy indexing are supported)
        """
        if isinstance(key, basestring):
            key = to_keys(key)

        if isinstance(key, (tuple, list)):
            if any(isinstance(k, Group) for k in key):
                k0 = key[0]
                assert isinstance(k0, Group)
                cls_ = k0.__class__
                assert all(isinstance(k, cls_) for k in key[1:])
                res = [k.with_axis(self) for k in key]
                res = tuple(res) if isinstance(key, tuple) else res
                return res

        if isinstance(key, Group):
            return key.__class__(key.key, key.name, self)
        return LGroup(key, axis=self)

    def __contains__(self, key):
        return _to_tick(key) in self._mapping

    def __hash__(self):
        return id(self)

    def _is_key_type_compatible(self, key):
        key_kind = np.dtype(type(key)).kind
        label_kind = self.labels.dtype.kind
        # on Python2, ascii-only unicode string can match byte strings (and
        # vice versa), so we shouldn't be more picky here than dict hashing
        str_key = key_kind in ('S', 'U')
        str_label = label_kind in ('S', 'U')
        py2_str_match = not PY3 and str_key and str_label
        # object kind can match anything
        return key_kind == label_kind or \
               key_kind == 'O' or label_kind == 'O' or \
               py2_str_match

    def translate(self, key, bool_passthrough=True):
        """
        Translates a label key to its numerical index counterpart.

        Parameters
        ----------
        key : key
            Everything usable as a key.
        bool_passthrough : bool, optional
            If set to True and key is a boolean vector, it is returned as it.

        Returns
        -------
        (array of) int
            Numerical index(ices) of (all) label(s) represented by the key

        Notes
        -----
        Fancy index with boolean vectors are passed through unmodified

        Examples
        --------
        >>> people = Axis('people', ['John Doe', 'Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'])
        >>> people.translate('Waldo')
        3
        >>> people.translate(people.matches('Bruce'))
        array([1, 2])
        """
        mapping = self._mapping

        # first, for Group instances, try their name
        if isinstance(key, Group):
            # XXX: we should probably use _to_tick(key) instead of key.name and do it for all keys instead of only
            # for groups
            try:
                # avoid matching 0 against False or 0.0
                if self._is_key_type_compatible(key.name):
                    return mapping[key.name]
            # we must catch TypeError because key might not be hashable (eg slice)
            # IndexError is for when mapping is an ndarray
            except (KeyError, TypeError, IndexError):
                pass

        # then try the key as-is:
        # * for strings, this is useful to allow ticks with special characters
        # * for groups, it means trying if the string representation of the whole group is in the mapping
        #   e.g. mapping['v1,v2,v3']
        try:
            # avoid matching 0 against False or 0.0
            if self._is_key_type_compatible(key):
                return mapping[key]
        # we must catch TypeError because key might not be hashable (eg slice)
        # IndexError is for when mapping is an ndarray
        except (KeyError, TypeError, IndexError):
            pass

        if isinstance(key, basestring):
            # transform "specially formatted strings" for slices, lists, LGroup and PGroup to actual objects
            key = _to_key(key)

        if isinstance(key, PGroup):
            return key.key

        if isinstance(key, LGroup):
            # at this point we do not care about the axis nor the name
            key = key.key

        if isinstance(key, slice):
            start = mapping[key.start] if key.start is not None else None
            # stop is inclusive in the input key and exclusive in the output !
            stop = mapping[key.stop] + 1 if key.stop is not None else None
            return slice(start, stop, key.step)
        # XXX: bool LArray do not pass through???
        elif isinstance(key, np.ndarray) and key.dtype.kind is 'b' and \
                bool_passthrough:
            return key
        elif isinstance(key, (tuple, list, OrderedSet)):
            # TODO: the result should be cached
            # Note that this is faster than array_lookup(np.array(key), mapping)
            res = np.empty(len(key), int)
            try:
                for i, label in enumerate(_seq_group_to_name(key)):
                    res[i] = mapping[label]
            except KeyError:
                for i, label in enumerate(key):
                    res[i] = mapping[label]
            return res
        elif isinstance(key, np.ndarray):
            # handle fancy indexing with a ndarray of labels
            # TODO: the result should be cached
            # TODO: benchmark this against the tuple/list version above when
            # mapping is large
            # array_lookup is O(len(key) * log(len(mapping)))
            # vs
            # tuple/list version is O(len(key)) (dict.getitem is O(1))
            # XXX: we might want to special case dtype bool, because in that
            # case the mapping will in most case be {False: 0, True: 1} or
            # {False: 1, True: 0} and in those case key.astype(int) and
            # (~key).astype(int) are MUCH faster
            # see C:\Users\gdm\devel\lookup_methods.py and
            #     C:\Users\gdm\Desktop\lookup_methods.html
            try:
                return array_lookup2(_seq_group_to_name(key), self._sorted_keys, self._sorted_values)
            except KeyError:
                return array_lookup2(key, self._sorted_keys, self._sorted_values)
        elif isinstance(key, LArray):
            return LArray(self.translate(key.data), key.axes)
        else:
            # the first mapping[key] above will cover most cases. This code
            # path is only used if the key was given in "non normalized form"
            assert np.isscalar(key), "%s (%s) is not scalar" % (key, type(key))
            # key is scalar (integer, float, string, ...)
            if np.dtype(type(key)).kind == self.labels.dtype.kind:
                return mapping[key]
            else:
                # print("diff dtype", )
                raise KeyError(key)

    # FIXME: remove id
    @property
    def id(self):
        if self.name is not None:
            return self.name
        else:
            raise ValueError('Axis has no name, so no id')

    def __str__(self):
        name = str(self.name) if self.name is not None else '{?}'
        return (name + '*') if self.iswildcard else name

    def __repr__(self):
        labels = len(self) if self.iswildcard else list(self.labels)
        return 'Axis(%r, %r)' % (self.name, labels)

    def labels_summary(self):
        """
        Returns a short representation of the labels.

        Examples
        --------
        >>> Axis('age', 100).labels_summary()
        '0 1 2 ... 97 98 99'
        """
        def repr_on_strings(v):
            return repr(v) if isinstance(v, str) else str(v)
        return _seq_summary(self.labels, repr_func=repr_on_strings)

    # method factory
    def _binop(opname):
        """
        Method factory to create binary operators special methods.
        """
        fullname = '__%s__' % opname

        def opmethod(self, other):
            # give a chance to AxisCollection.__rXXX__ ops to trigger
            if isinstance(other, AxisCollection):
                # in this case it is indeed return NotImplemented, not raise
                # NotImplementedError!
                return NotImplemented

            self_array = labels_array(self)
            if isinstance(other, Axis):
                other = labels_array(other)
            return getattr(self_array, fullname)(other)
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
    __matmul__ = _binop('matmul')

    def __larray__(self):
        """
        Returns axis as LArray.
        """
        return labels_array(self)

    def copy(self):
        """
        Returns a copy of the axis.
        """
        new_axis = Axis(self.name, [])
        # XXX: I wonder if we should make a copy of the labels + mapping.
        # There should at least be an option.
        new_axis._labels = self._labels
        new_axis.__mapping = self.__mapping
        new_axis._length = self._length
        new_axis._iswildcard = self._iswildcard
        new_axis.__sorted_keys = self.__sorted_keys
        new_axis.__sorted_values = self.__sorted_values
        return new_axis

    # XXX: rename to named like Group?
    def rename(self, name):
        """
        Renames the axis.

        Parameters
        ----------
        name : str
            the new name for the axis.

        Returns
        -------
        Axis
            a new Axis with the same labels but a different name.

        Examples
        --------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> sex
        Axis('sex', ['M', 'F'])
        >>> sex.rename('gender')
        Axis('gender', ['M', 'F'])
        """
        res = self.copy()
        if isinstance(name, Axis):
            name = name.name
        res.name = name
        return res

    def _rename(self, name):
        raise TypeError("Axis._rename is deprecated, use Axis.rename instead")


# We need a separate class for LGroup and cannot simply create a
# new Axis with a subset of values/ticks/labels: the subset of
# ticks/labels of the LGroup need to correspond to its *Axis*
# indices
class Group(object):
    """Abstract Group.
    """
    format_string = None

    def __init__(self, key, name=None, axis=None):
        if isinstance(key, tuple):
            key = list(key)
        self.key = key

        # we do NOT assign a name automatically when missing because that
        # makes it impossible to know whether a name was explicitly given or
        # not
        self.name = name
        assert axis is None or isinstance(axis, (basestring, int, Axis)), \
            "invalid axis '%s' (%s)" % (axis, type(axis).__name__)

        # we could check the key is valid but this can be slow and could be
        # useless
        # TODO: for performance reasons, we should cache the result. This will
        # need to be invalidated correctly
        # axis.translate(key)

        # we store the Axis object and not its name like we did previously
        # so that groups on anonymous axes are more meaningful and that we
        # can iterate on a slice of an axis (an LGroup). The reason to store
        # the name instead of the object was to make sure that a Group from an
        # axis (or without axis) could be used on another axis with the same
        # name. See test_la.py:test_...
        self.axis = axis

    def __repr__(self):
        key = self.key

        # eval only returns a slice for groups without an Axis object
        if isinstance(key, slice):
            key_repr = _slice_to_str(key, repr_func=repr)
        elif isinstance(key, (tuple, list, np.ndarray, OrderedSet)):
            key_repr = _seq_summary(key, n=1000, repr_func=repr, sep=', ')
        else:
            key_repr = repr(key)

        axis_name = self.axis.name if isinstance(self.axis, Axis) else self.axis
        if axis_name is not None:
            axis_name = 'x.{}'.format(axis_name) if isinstance(self.axis, AxisReference) else axis_name
            s = self.format_string.format(axis=axis_name, key=key_repr)
        else:
            if self.axis is not None:
                # anonymous axis
                axis_ref = ', axis={}'.format(repr(self.axis))
            else:
                axis_ref = ''
            if isinstance(key, slice):
                key_repr = repr(key)
            elif isinstance(key, list):
                key_repr = '[{}]'.format(key_repr)
            s = '{}({}{})'.format(self.__class__.__name__, key_repr, axis_ref)
        return "{} >> {}".format(s, repr(self.name)) if self.name is not None else s

    def __str__(self):
        return str(self.eval())

    def translate(self, bound=None, stop=False):
        """

        Parameters
        ----------
        bound : any

        Returns
        -------
        int
        """
        raise NotImplementedError()

    def __len__(self):
        value = self.eval()
        # for some reason this breaks having LGroup ticks/labels on an axis
        if hasattr(value, '__len__'):
        # if isinstance(value, (tuple, list, LArray, np.ndarray, str)):
            return len(value)
        elif isinstance(value, slice):
            start, stop, key_step = value.start, value.stop, value.step
            # not using stop - start because that does not work for string
            # bounds (and it is different for LGroup & PGroup)
            start_pos = self.translate(start)
            stop_pos = self.translate(stop)
            return stop_pos - start_pos
        else:
            raise TypeError('len() of unsized object ({})'.format(value))

    def __iter__(self):
        # XXX: use translate/PGroup instead, so that it works even in the presence of duplicate labels
        #      possibly, only if axis is set?
        return iter([LGroup(v, axis=self.axis) for v in self.eval()])

    def named(self, name):
        """Returns group with a different name.

        Parameters
        ----------
        name : str
            new name for group

        Returns
        -------
        Group
        """
        return self.__class__(self.key, name, self.axis)
    __rshift__ = named

    def with_axis(self, axis):
        """Returns group with a different axis.

        Parameters
        ----------
        axis : int, str, Axis
            new axis for group

        Returns
        -------
        Group
        """
        return self.__class__(self.key, self.name, axis)

    def by(self, length, step=None):
        """Split group into several groups of specified length.

        Parameters
        ----------
        length : int
            length of new groups
        step : int, optional
            step between groups. Defaults to length.

        Notes
        -----
        step can be smaller than length, in which case, this will produce
        overlapping groups.

        Returns
        -------
        list of Group

        Examples
        --------
        >>> age = Axis('age', range(10))
        >>> age[[1, 2, 3, 4, 5]].by(2)
        (age[1, 2], age[3, 4], age[5])
        >>> age[1:5].by(2)
        (age.i[1:3], age.i[3:5], age.i[5:6])
        >>> age[1:5].by(2, 4)
        (age.i[1:3], age.i[5:6])
        >>> age[1:5].by(3, 2)
        (age.i[1:4], age.i[3:6], age.i[5:6])
        >>> x.age[[0, 1, 2, 3, 4]].by(2)
        (x.age[0, 1], x.age[2, 3], x.age[4])
        """
        if step is None:
            step = length
        return tuple(self[start:start + length]
                     for start in range(0, len(self), step))

    # TODO: __getitem__ should work by label and .i[] should work by
    # position. I guess it would be more consistent this way even if the
    # usefulness of subsetting a group with labels is dubious (but
    # it is sometimes practical to treat the group as if it was an axis).
    # >>> vla = geo['...']
    # >>> # first 10 regions of flanders (this could have some use)
    # >>> vla.i[:10]  # => PGroup on geo
    # >>> vla["antwerp", "gent"]  # => LGroup on geo

    # LGroup[] => LGroup
    # PGroup[] => LGroup
    # PGroup.i[] => PGroup
    # LGroup.i[] => PGroup
    def __getitem__(self, key):
        """

        Parameters
        ----------
        key : int, slice of int or list of int
            position-based key (even for LGroup)

        Returns
        -------
        Group
        """
        cls = self.__class__
        orig_key = self.key
        # XXX: unsure we should support tuple
        if isinstance(orig_key, (tuple, list)):
            return cls(orig_key[key], None, self.axis)
        elif isinstance(orig_key, slice):
            orig_start, orig_stop, orig_step = \
                orig_key.start, orig_key.stop, orig_key.step
            if orig_step is None:
                orig_step = 1

            orig_start_pos = self.translate(orig_start) if orig_start is not None else 0
            if isinstance(key, slice):
                key_start, key_stop, key_step = key.start, key.stop, key.step
                if key_step is None:
                    key_step = 1

                orig_stop_pos = self.translate(orig_stop, stop=True) if orig_stop is not None else len(self)
                new_start = orig_start_pos + key_start * orig_step
                new_stop = min(orig_start_pos + key_stop * orig_step,
                               orig_stop_pos)
                new_step = orig_step * key_step
                if new_step == 1:
                    new_step = None
                return PGroup(slice(new_start, new_stop, new_step), None,
                              self.axis)
            elif isinstance(key, int):
                return PGroup(orig_start_pos + key * orig_step, None, self.axis)
            elif isinstance(key, (tuple, list)):
                return PGroup([orig_start_pos + k * orig_step for k in key],
                              None, self.axis)
        elif isinstance(orig_key, LArray):
            return cls(orig_key.i[key], None, self.axis)
        elif isinstance(orig_key, int):
            # give the opportunity to subset the label/key itself (for example for string keys)
            value = self.eval()
            return value[key]
        else:
            raise TypeError("cannot take a subset of {} because it has a "
                            "'{}' key".format(self.key, type(self.key)))

    def __eq__(self, other):
        # different name or axis compare equal !
        # XXX: we might want to compare "expanded" keys using self.eval(), so that slices
        # can match lists and vice-versa. This might be too slow though.
        other_key = other.key if isinstance(other, Group) else _to_key(other)
        return _to_tick(self.key) == _to_tick(other_key)

    # method factory
    def _binop(opname):
        op_fullname = '__%s__' % opname

        # TODO: implement this in a delayed fashion for reference axes
        def opmethod(self, other):
            other_value = other.eval() if isinstance(other, Group) else other
            return getattr(self.eval(), op_fullname)(other_value)
        opmethod.__name__ = op_fullname
        return opmethod

    __matmul__ = _binop('matmul')
    __ror__ = _binop('ror')
    __or__ = _binop('or')
    __rxor__ = _binop('rxor')
    __xor__ = _binop('xor')
    __rand__ = _binop('rand')
    __and__ = _binop('and')
    __rpow__ = _binop('rpow')
    __pow__ = _binop('pow')
    __rdivmod__ = _binop('rdivmod')
    __divmod__ = _binop('divmod')
    __rmod__ = _binop('rmod')
    __mod__ = _binop('mod')
    __rfloordiv__ = _binop('rfloordiv')
    __floordiv__ = _binop('floordiv')
    __rtruediv__ = _binop('rtruediv')
    __truediv__ = _binop('truediv')
    if sys.version < '3':
        __div__ = _binop('div')
        __rdiv__ = _binop('rdiv')
    __rmul__ = _binop('rmul')
    __mul__ = _binop('mul')
    __rsub__ = _binop('rsub')
    __sub__ = _binop('sub')
    __radd__ = _binop('radd')
    __add__ = _binop('add')

    __ge__ = _binop('ge')
    __gt__ = _binop('gt')
    __le__ = _binop('le')
    __lt__ = _binop('lt')
    # __ne__ = _binop('ne')
    # __eq__ = _binop('eq')

    def __contains__(self, item):
        # XXX: ideally, we shouldn't need to test for Group (hash should hash to the same as item.eval())
        if isinstance(item, Group):
            item = item.eval()
        return item in self.eval()

    # this makes range(LGroup(int)) possible
    def __index__(self):
        return self.eval().__index__()

    def __int__(self):
        return self.eval().__int__()

    def __float__(self):
        return self.eval().__float__()

    def __hash__(self):
        # to_tick & to_key are partially opposite operations but this
        # standardize on a single notation so that they can all target each
        # other. eg, this removes spaces in "list strings", instead of
        # hashing them directly
        # XXX: but we might want to include that normalization feature in
        #      to_tick directly, instead of using to_key explicitly here
        # XXX: we might want to make hash use the position along the axis instead of the labels so that if an
        #      axis has ambiguous labels, they do not hash to the same thing.
        # XXX: for performance reasons, I think hash should not evaluate slices. It should only translate pos bounds to
        #      labels or vice versa. We would loose equality between list Groups and equivalent slice groups but that
        #      is a small price to pay if the performance impact is large.
        # the problem with using self.translate() is that we cannot compare groups without axis
        # return hash(_to_tick(self.translate()))
        return hash(_to_tick(self.key))


# TODO: factorize as much as possible between LGroup & PGroup (move stuff to
#       Group)
class LGroup(Group):
    """Label group.

    Represents a subset of labels of an axis.

    Parameters
    ----------
    key : key
        Anything usable for indexing.
        A key should be either sequence of labels, a slice with label bounds or a string.
    name : str, optional
        Name of the group.
    axis : int, str, Axis, optional
        Axis for group.

    Examples
    --------
    >>> age = Axis('age', '0..100')
    >>> teens = x.age[10:19].named('teens')
    >>> teens
    x.age[10:19] >> 'teens'
    >>> teens = x.age[10:19] >> 'teens'
    >>> teens
    x.age[10:19] >> 'teens'
    """
    format_string = "{axis}[{key}]"

    def __init__(self, key, name=None, axis=None):
        key = _to_key(key)
        Group.__init__(self, key, name, axis)

    def set(self):
        return LSet(self.eval(), self.name, self.axis)

    #XXX: return PGroup instead?
    def translate(self, bound=None, stop=False):
        """
        compute position(s) of group
        """
        if bound is None:
            bound = self.key
        if isinstance(self.axis, Axis):
            pos = self.axis.translate(bound)
            return pos + int(stop) if np.isscalar(pos) else pos
        else:
            raise ValueError("Cannot translate an LGroup without axis")

    def eval(self):
        if isinstance(self.key, slice):
            if isinstance(self.axis, Axis):
                # expand slices
                return self.axis.labels[self.translate()]
            else:
                return self.key
                # raise ValueError("Cannot evaluate a slice group without axis")
        else:
            # we do not check the group labels are actually valid on Axis
            return self.key


class LSet(LGroup):
    """Label set.

    Represents a set of (unique) labels of an axis.

    Parameters
    ----------
    key : key
        Anything usable for indexing.
        A key should be either sequence of labels, a slice with label bounds or a string.
    name : str, optional
        Name of the set.
    axis : int, str, Axis, optional
        Axis for set.

    Examples
    --------
    >>> letters = Axis('letters', 'a..z')
    >>> abc = letters[':c'].set() >> 'abc'
    >>> abc
    letters['a', 'b', 'c'].set() >> 'abc'
    >>> abc & letters['b:d']
    letters['b', 'c'].set()
    """
    format_string = "{axis}[{key}].set()"

    def __init__(self, key, name=None, axis=None):
        key = _to_key(key)
        if isinstance(key, LGroup):
            if name is None:
                name = key.name
            if axis is None:
                axis = key.axis
            if not isinstance(key, LSet):
                key = key.eval()
        if np.isscalar(key):
            key = [key]
        key = OrderedSet(key)
        LGroup.__init__(self, key, name, axis)

    # method factory
    def _binop(opname, c):
        op_fullname = '__%s__' % opname

        # TODO: implement this in a delayed fashion for reference axes
        def opmethod(self, other):
            if not isinstance(other, LSet):
                other = LSet(other)
            axis = self.axis if self.axis is not None else other.axis

            # setting a meaningful name is hard when either one has no name
            if self.name is not None and other.name is not None:
                name = '%s %s %s' % (self.name, c, other.name)
            else:
                name = None
            # TODO: implement this in a more efficient way for ndarray keys
            #       which can be large
            result_set = getattr(self.key, op_fullname)(other.key)
            return LSet(result_set, name=name, axis=axis)
        opmethod.__name__ = op_fullname
        return opmethod

    union = _binop('or', '|')
    __or__ = union

    intersection = _binop('and', '&')
    __and__ = intersection

    difference = _binop('sub', '-')
    __sub__ = difference


class PGroup(Group):
    """Positional Group.

    Represents a subset of indices of an axis.

    Parameters
    ----------
    key : key
        Anything usable for indexing.
        A key should be either a single position, a sequence of positions, or a slice with
        integer bounds.
    name : str, optional
        Name of the group.
    axis : int, str, Axis, optional
        Axis for group.
    """
    format_string = "{axis}.i[{key}]"

    def translate(self, bound=None, stop=False):
        """
        compute position(s) of group
        """
        if bound is not None:
            return bound
        else:
            return self.key

    def eval(self):
        if isinstance(self.axis, Axis):
            return self.axis.labels[self.key]
        else:
            raise ValueError("Cannot evaluate a positional group without axis")


def index_by_id(seq, value):
    """
    Returns position of an object in a sequence.
    Raises an error if the object is not in the list.

    Parameters
    ----------
    seq : sequence
        Any sequence (list, tuple, str, unicode).

    value : object
        Object for which you want to retrieve its position
        in the sequence.

    Raises
    ------
    ValueError
        If `value` object is not contained in the sequence.

    Examples
    --------
    >>> age = Axis('age', range(10))
    >>> sex = Axis('sex', ['M','F'])
    >>> time = Axis('time', ['2007','2008','2009','2010'])
    >>> index_by_id([age, sex, time], sex)
    1
    >>> gender = Axis('sex', ['M', 'F'])
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
    raise ValueError("%s is not in list" % value)


# not using OrderedDict because it does not support indices-based getitem
# not using namedtuple because we have to know the fields in advance (it is a
# one-off class) and we need more functionality than just a named tuple
class AxisCollection(object):
    """
    Represents a collection of axes.

    Parameters
    ----------
    axes : sequence of Axis or int or tuple or str, optional
        An axis can be given as an Axis object, an int or a
        tuple (name, labels) or a string of the kind
        'name=label_1,label_2,label_3'.

    Raises
    ------
    ValueError
        Cannot have multiple occurrences of the same axis object in a collection.

    Notes
    -----
    Multiple occurrences of the same axis object is not allowed.
    However, several axes with the same name are allowed but this is not recommended.

    Examples
    --------
    >>> age = Axis('age', range(10))
    >>> AxisCollection([3, age, ('sex', ['M', 'F']), 'time = 2007, 2008, 2009, 2010'])
    AxisCollection([
        Axis(None, 3),
        Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        Axis('sex', ['M', 'F']),
        Axis('time', ['2007', '2008', '2009', '2010'])
    ])
    """
    def __init__(self, axes=None):
        if axes is None:
            axes = []
        elif isinstance(axes, (int, long, Axis)):
            axes = [axes]
        elif isinstance(axes, str):
            axes = [axis.strip() for axis in axes.split(';')]

        def make_axis(obj):
            if isinstance(obj, Axis):
                return obj
            elif isinstance(obj, tuple):
                assert len(obj) == 2
                name, labels = obj
                return Axis(name, labels)
            elif isinstance(obj, Group):
                return Axis(obj.axis, obj.eval())
            else:
                if isinstance(obj, str) and '=' in obj:
                    name, labels = [o.strip() for o in obj.split('=')]
                    return Axis(name, labels)
                else:
                    return Axis(None, obj)

        axes = [make_axis(axis) for axis in axes]
        assert all(isinstance(a, Axis) for a in axes)
        # check for duplicate axes
        dupe_axes = list(duplicates(axes))
        if dupe_axes:
            axis = dupe_axes[0]
            raise ValueError("Cannot have multiple occurrences of the same "
                             "axis object in a collection "
                             "('%s' -- %s with id %d). "
                             "Several axes with the same name are allowed "
                             "though (but not recommended)."
                             % (axis.name, axis.labels_summary(), id(axis)))
        self._list = axes
        self._map = {axis.name: axis for axis in axes if axis.name is not None}

        # # check dupes on each axis
        # for axis in axes:
        #     axis_dupes = list(duplicates(axis.labels))
        #     if axis_dupes:
        #         dupe_labels = ', '.join(str(l) for l in axis_dupes)
        #         warnings.warn("duplicate labels found for axis %s: %s"
        #                       % (axis.name, dupe_labels),
        #                       category=UserWarning, stacklevel=2)
        #
        # # check dupes between axes. Using unique to not spot the dupes
        # # within the same axis that we just displayed.
        # all_labels = chain(*[np.unique(axis.labels) for axis in axes])
        # dupe_labels = list(duplicates(all_labels))
        # if dupe_labels:
        #     label_axes = [(label, ', '.join(display_name
        #                                     for axis, display_name
        #                                     in zip(axes, self.display_names)
        #                                     if label in axis))
        #                   for label in dupe_labels]
        #     dupes = '\n'.join("{} is valid in {{{}}}".format(label, axes)
        #                       for label, axes in label_axes)
        #     warnings.warn("ambiguous labels found:\n%s" % dupes,
        #                   category=UserWarning, stacklevel=5)

    def __dir__(self):
        # called by dir() and tab-completion at the interactive prompt,
        # should return a list of all valid attributes, ie all normal
        # attributes plus anything valid in getattr (string keys only).
        # make sure we return unique results because dir() does not ensure that
        # (ipython tab-completion does though).
        # order does not matter though (dir() sorts the results)
        names = set(axis.name for axis in self._list if axis.name is not None)
        return list(set(dir(self.__class__)) | names)

    def __iter__(self):
        return iter(self._list)

    def __getattr__(self, key):
        try:
            return self._map[key]
        except KeyError:
            return self.__getattribute__(key)

    def __getitem__(self, key):
        if isinstance(key, Axis):
            try:
                key = self.index(key)
            # transform ValueError to KeyError
            except ValueError:
                if key.name is None:
                    raise KeyError("axis '%s' not found in %s" % (key, self))
                else:
                    # we should NOT check that the object is the same, so that we can
                    # use AxisReference objects to target real axes
                    key = key.name

        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, (tuple, list)):
            # XXX: also use get_by_pos if tuple/list of Axis?
            return AxisCollection([self[k] for k in key])
        elif isinstance(key, AxisCollection):
            return AxisCollection([self.get_by_pos(k, i)
                                   for i, k in enumerate(key)])
        elif isinstance(key, slice):
            return AxisCollection(self._list[key])
        elif key is None:
            raise KeyError("axis '%s' not found in %s" % (key, self))
        else:
            assert isinstance(key, basestring), type(key)
            if key in self._map:
                return self._map[key]
            else:
                raise KeyError("axis '%s' not found in %s" % (key, self))

    # XXX: I wonder if this whole positional crap should really be part of
    # AxisCollection or the default behavior. It could either be moved to
    # make_numpy_broadcastable or made non default
    def get_by_pos(self, key, i):
        """
        Returns axis corresponding to a key, or to position i if the key
        has no name and key object not found.

        Parameters
        ----------
        key : key
            Key corresponding to an axis.
        i : int
            Position of the axis (used only if search by key failed).

        Returns
        -------
        Axis
            Axis corresponding to the key or the position i.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection([age, sex, time])
        >>> col.get_by_pos('sex', 1)
        Axis('sex', ['M', 'F'])
        """
        if isinstance(key, Axis) and key.name is None:
            try:
                # try by object
                return self[key]
            except KeyError:
                if i in self:
                    res = self[i]
                    if res.iscompatible(key):
                        return res
                    else:
                        raise ValueError("axis %s is not compatible with %s"
                                         % (res, key))
                # XXX: KeyError instead?
                raise ValueError("axis %s not found in %s"
                                 % (key, self))
        else:
            return self[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            assert isinstance(value, (tuple, list, AxisCollection))
            def slice_bound(bound):
                if bound is None or isinstance(bound, int):
                    # out of bounds integer bounds are allowed in slice setitem
                    # so we cannot use .index
                    return bound
                else:
                    return self.index(bound)
            start_idx = slice_bound(key.start)
            # XXX: we might want to make the stop bound inclusive, which makes
            # more sense for label bounds (but prevents inserts via setitem)
            stop_idx = slice_bound(key.stop)
            old = self._list[start_idx:stop_idx:key.step]
            for axis in old:
                if axis.name is not None:
                    del self._map[axis.name]
            for axis in value:
                if axis.name is not None:
                    self._map[axis.name] = axis
            self._list[start_idx:stop_idx:key.step] = value
        elif isinstance(key, (tuple, list, AxisCollection)):
            assert isinstance(value, (tuple, list, AxisCollection))
            if len(key) != len(value):
                raise ValueError('must have as many old axes as new axes')
            for k, v in zip(key, value):
                self[k] = v
        else:
            assert isinstance(value, Axis)
            idx = self.index(key)
            step = 1 if idx >= 0 else -1
            self[idx:idx + step:step] = [value]

    def __delitem__(self, key):
        if isinstance(key, slice):
            self[key] = []
        else:
            idx = self.index(key)
            axis = self._list.pop(idx)
            if axis.name is not None:
                del self._map[axis.name]

    def union(self, *args, **kwargs):
        validate = kwargs.pop('validate', True)
        replace_wildcards = kwargs.pop('replace_wildcards', True)
        result = self[:]
        for a in args:
            if not isinstance(a, AxisCollection):
                a = AxisCollection(a)
            result.extend(a, validate=validate, replace_wildcards=replace_wildcards)
        return result
    __or__ = union
    __add__ = union

    def __radd__(self, other):
        result = AxisCollection(other)
        result.extend(self)
        return result

    def __and__(self, other):
        """
        Returns the intersection of this collection and other.
        """
        if not isinstance(other, AxisCollection):
            other = AxisCollection(other)

        # XXX: add iscompatible when matching by position?
        # TODO: move this to a class method (possibly private) so that
        # we make sure we use same heuristic than in .extend
        def contains(col, i, axis):
            return axis in col or (axis.name is None and i in col)

        return AxisCollection([axis for i, axis in enumerate(self)
                               if contains(other, i, axis)])

    def __eq__(self, other):
        """
        Other collection compares equal if all axes compare equal and in the
        same order. Works with a list.
        """
        if self is other:
            return True
        if not isinstance(other, list):
            other = list(other)
        return len(self._list) == len(other) and \
               all(a.equals(b) for a, b in zip(self._list, other))

    # for python2, we need to define it explicitly
    def __ne__(self, other):
        return not self == other

    def __contains__(self, key):
        if isinstance(key, int):
            return -len(self) <= key < len(self)
        elif isinstance(key, Axis):
            if key.name is None:
                # XXX: use only this in all cases?
                try:
                    self.index(key)
                    return True
                except ValueError:
                    return False
            else:
                key = key.name
        return key in self._map

    def isaxis(self, value):
        """
        Tests if input is an Axis object or
        the name of an axis contained in self.

        Parameters
        ----------
        value : Axis or str
            Input axis or string

        Returns
        -------
        bool
            True if input is an Axis object or the name of an axis contained in
            the current AxisCollection instance, False otherwise.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> col = AxisCollection([age, sex])
        >>> col.isaxis(age)
        True
        >>> col.isaxis('sex')
        True
        >>> col.isaxis('city')
        False
        """
        # this is tricky. 0 and 1 can be both axes indices and axes ticks.
        # not sure what's worse:
        # 1) disallow aggregates(axis_num)
        #    users could still use arr.sum(arr.axes[0])
        #    we could also provide an explicit kwarg (ie this would
        #    effectively forbid having an axis named "axis").
        #    arr.sum(axis=0). I think this is the sanest option. The
        #    error message in case we use it without the keyword needs to
        #    be clearer though.
        return isinstance(value, Axis) or (isinstance(value, basestring) and
                                           value in self)
        # 2) slightly inconsistent API: allow aggregate over single labels
        #    if they are string, but not int
        #    arr.sum(0) would sum on the first axis, but arr.sum('M') would
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
    ndim = property(__len__)

    def __str__(self):
        return "{%s}" % ', '.join(self.display_names)

    def __repr__(self):
        axes_repr = (repr(axis) for axis in self._list)
        return "AxisCollection([\n    %s\n])" % ',\n    '.join(axes_repr)

    def get(self, key, default=None, name=None):
        """
        Returns axis corresponding to key. If not found,
        the argument `name` is used to create a new Axis.
        If `name` is None, the `default` axis is then returned.

        Parameters
        ----------
        key : key
            Key corresponding to an axis of the current AxisCollection.
        default : axis, optional
            Default axis to return if key doesn't correspond to any axis of
            the collection and argument `name` is None.
        name : str, optional
            If key doesn't correspond to any axis of the collection,
            a new Axis with this name is created and returned.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection([age, time])
        >>> col.get('time')
        Axis('time', [2007, 2008, 2009, 2010])
        >>> col.get('sex', sex)
        Axis('sex', ['M', 'F'])
        >>> col.get('nb_children', None, 'nb_children')
        Axis('nb_children', 1)
        """
        # XXX: use if key in self?
        try:
            return self[key]
        except KeyError:
            if name is None:
                return default
            else:
                return Axis(name, 1)

    def get_all(self, key):
        """
        Returns all axes from key if present and length 1 wildcard axes
        otherwise.

        Parameters
        ----------
        key : AxisCollection

        Returns
        -------
        AxisCollection

        Raises
        ------
        AssertionError
            Raised if the input key is not an AxisCollection object.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> city = Axis('city', ['London', 'Paris', 'Rome'])
        >>> col = AxisCollection([age, sex, time])
        >>> col2 = AxisCollection([age, city, time])
        >>> col.get_all(col2)
        AxisCollection([
            Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            Axis('city', 1),
            Axis('time', [2007, 2008, 2009, 2010])
        ])
        """
        assert isinstance(key, AxisCollection)
        def get_pos_default(k, i):
            try:
                return self.get_by_pos(k, i)
            except (ValueError, KeyError):
                # XXX: is having i as name really helps?
                if len(k) == 1:
                    return k
                else:
                    return Axis(k.name if k.name is not None else i, 1)

        return AxisCollection([get_pos_default(k, i)
                               for i, k in enumerate(key)])

    def keys(self):
        """
        Returns list of all axis names.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age, sex, time]).keys()
        ['age', 'sex', 'time']
        """
        # XXX: include id/num for anonymous axes? I think I should
        return [a.name for a in self._list]

    def pop(self, axis=-1):
        """
        Removes and returns an axis.

        Parameters
        ----------
        axis : key, optional
            Axis to remove and return.
            Default value is -1 (last axis).

        Returns
        -------
        Axis
            If no argument is provided, the last axis is removed and returned.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection([age, sex, time])
        >>> col.pop('age')
        Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        >>> col
        AxisCollection([
            Axis('sex', ['M', 'F']),
            Axis('time', [2007, 2008, 2009, 2010])
        ])
        >>> col.pop()
        Axis('time', [2007, 2008, 2009, 2010])
        """
        axis = self[axis]
        del self[axis]
        return axis

    def append(self, axis):
        """
        Appends axis at the end of the collection.

        Parameters
        ----------
        axis : Axis
            Axis to append.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection([age, sex])
        >>> col.append(time)
        >>> col
        AxisCollection([
            Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            Axis('sex', ['M', 'F']),
            Axis('time', [2007, 2008, 2009, 2010])
        ])
        """
        self[len(self):len(self)] = [axis]

    def check_compatible(self, axes):
        """
        Checks if axes passed as argument are compatible with those
        contained in the collection. Raises a ValueError if not.

        See Also
        --------
        Axis.iscompatible
        """
        for i, axis in enumerate(axes):
            local_axis = self.get_by_pos(axis, i)
            if not local_axis.iscompatible(axis):
                raise ValueError("incompatible axes:\n%r\nvs\n%r"
                                 % (axis, local_axis))

    def extend(self, axes, validate=True, replace_wildcards=False):
        """
        Extends the collection by appending the axes from `axes`.

        Parameters
        ----------
        axes : sequence of Axis (list, tuple, AxisCollection)
        validate : bool, optional
        replace_wildcards : bool, optional

        Raises
        ------
        TypeError
            Raised if `axes` is not a sequence of Axis (list, tuple or AxisCollection)

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection(age)
        >>> col.extend([sex, time])
        >>> col
        AxisCollection([
            Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            Axis('sex', ['M', 'F']),
            Axis('time', [2007, 2008, 2009, 2010])
        ])
        """
        # axes should be a sequence
        if not isinstance(axes, (tuple, list, AxisCollection)):
            raise TypeError("AxisCollection can only be extended by a "
                            "sequence of Axis, not %s" % type(axes).__name__)
        # check that common axes are the same
        # if validate:
        #     self.check_compatible(axes)

        # TODO: factorize with get_by_pos
        def get_axis(col, i, axis):
            if axis in col:
                return col[axis]
            elif axis.name is None and i in col:
                return col[i]
            else:
                return None

        for i, axis in enumerate(axes):
            old_axis = get_axis(self, i, axis)
            if old_axis is None:
                # append axis
                self[len(self):len(self)] = [axis]
            # elif replace_wildcards and old_axis.iswildcard:
            #     self[old_axis] = axis
            else:
                # check that common axes are the same
                if validate and not old_axis.iscompatible(axis):
                    raise ValueError("incompatible axes:\n%r\nvs\n%r"
                                     % (axis, old_axis))
                if replace_wildcards and old_axis.iswildcard:
                    self[old_axis] = axis

    def index(self, axis):
        """
        Returns the index of axis.

        `axis` can be a name or an Axis object (or an index).
        If the Axis object itself exists in the list, index() will return it.
        Otherwise, it will return the index of the local axis with the same
        name than the key (whether it is compatible or not).

        Parameters
        ----------
        axis : Axis or int or str
            Can be the axis itself or its position (returned if represents a valid index)
            or its name.

        Returns
        -------
        int
            Index of the axis.

        Raises
        ------
        ValueError
            Raised if the axis is not present.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection([age, sex, time])
        >>> col.index(time)
        2
        >>> col.index('sex')
        1
        """
        if isinstance(axis, int):
            if -len(self) <= axis < len(self):
                return axis
            else:
                raise ValueError("axis %d is not in collection" % axis)
        elif isinstance(axis, Axis):
            try:
                # first look by id. This avoids testing labels of each axis
                # and makes sure the result is correct even if there are
                # several axes with no name and the same labels.
                return index_by_id(self._list, axis)
            except ValueError:
                name = axis.name
        else:
            name = axis
        if name is None:
            raise ValueError("%r is not in collection" % axis)
        return self.names.index(name)

    # XXX: we might want to return a new AxisCollection (same question for
    # other inplace operations: append, extend, pop, __delitem__, __setitem__)
    def insert(self, index, axis):
        """
        Inserts axis before index.

        Parameters
        ----------
        index : int
            position of the inserted axis.
        axis : Axis
            axis to insert.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> col = AxisCollection([age, time])
        >>> col.insert(1, sex)
        >>> col
        AxisCollection([
            Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            Axis('sex', ['M', 'F']),
            Axis('time', [2007, 2008, 2009, 2010])
        ])
        """
        self[index:index] = [axis]

    def copy(self):
        """
        Returns a copy.
        """
        return self[:]

    def replace(self, old, new):
        """
        Replaces an axis.

        Parameters
        ----------
        old : Axis
            Axis to be replaced
        new : Axis
            Axis to be put in place of the `old` axis.

        Returns
        -------
        AxisCollection
            New collection with old axis replaced by the new one.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> age_new = Axis('age', range(10))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> col = AxisCollection([age, sex])
        >>> col.replace(age, age_new)
        AxisCollection([
            Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            Axis('sex', ['M', 'F'])
        ])
        """
        res = self[:]
        res[old] = new
        return res

    # XXX: kill this method?
    def without(self, axes):
        """
        Returns a new collection without some axes.

        You can use a comma separated list of names.

        Parameters
        ----------
        axes : int, str, Axis or sequence of those
            Axes to not include in the returned AxisCollection.
            In case of string, axes are separated by a comma and no whitespace is accepted.

        Returns
        -------
        AxisCollection
            New collection without some axes.

        Notes
        -----
        Set operation so axes can contain axes not present in self

        Examples
        --------
        >>> age = Axis('age', '0..5')
        >>> sex = Axis('sex', 'M,F')
        >>> time = Axis('time', '2015..2017')
        >>> col = AxisCollection([age, sex, time])
        >>> col.without([age, sex])
        AxisCollection([
            Axis('time', [2015, 2016, 2017])
        ])
        >>> col.without(0)
        AxisCollection([
            Axis('sex', ['M', 'F']),
            Axis('time', [2015, 2016, 2017])
        ])
        >>> col.without('sex,time')
        AxisCollection([
            Axis('age', [0, 1, 2, 3, 4, 5])
        ])
        """
        return self - axes

    def __sub__(self, axes):
        """
        See Also
        --------
        without
        """
        if isinstance(axes, basestring):
            axes = axes.split(',')
        elif isinstance(axes, (int, Axis)):
            axes = [axes]

        # only keep indices (as this works for unnamed axes too)
        to_remove = set(self.index(axis) for axis in axes if axis in self)
        return AxisCollection([axis for i, axis in enumerate(self)
                               if i not in to_remove])

    def translate_full_key(self, key):
        """
        Translates a label-based key to a positional key.

        Parameters
        ----------
        key : tuple
            A full label-based key.
            All dimensions must be present and in the correct order.

        Returns
        -------
        tuple
            A full positional key.

        See Also
        --------
        Axis.translate

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age,sex,time]).translate_full_key((':', 'F', 2009))
        (slice(None, None, None), 1, 2)
        """
        assert len(key) == len(self)
        return tuple(axis.translate(axis_key)
                     for axis_key, axis in zip(key, self))

    @property
    def labels(self):
        """
        Returns the list of labels of the axes.

        Returns
        -------
        list
            List of labels of the axes.

        Examples
        --------
        >>> age = Axis('age', range(10))
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age, time]).labels  # doctest: +NORMALIZE_WHITESPACE
        [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
         array([2007, 2008, 2009, 2010])]
        """
        return [axis.labels for axis in self._list]

    @property
    def names(self):
        """
        Returns the list of (raw) names of the axes.

        Returns
        -------
        list
            List of names of the axes.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age, sex, time]).names
        ['age', 'sex', 'time']
        """
        return [axis.name for axis in self._list]

    @property
    def display_names(self):
        """
        Returns the list of (display) names of the axes.

        Returns
        -------
        list
            List of names of the axes.
            Wildcard axes are displayed with an attached *.
            Anonymous axes (name = None) are replaced by their position in braces.

        Examples
        --------
        >>> a = Axis('a', ['a1', 'a2'])
        >>> b = Axis('b', 2)
        >>> c = Axis(None, ['c1', 'c2'])
        >>> d = Axis(None, 3)
        >>> AxisCollection([a, b, c, d]).display_names
        ['a', 'b*', '{2}', '{3}*']
        """
        def display_name(i, axis):
            name = axis.name if axis.name is not None else '{%d}' % i
            return (name + '*') if axis.iswildcard else name

        return [display_name(i, axis) for i, axis in enumerate(self._list)]

    @property
    def ids(self):
        """
        Returns the list of ids of the axes.

        Returns
        -------
        list
            List of ids of the axes.

        See Also
        --------
        axis_id

        Examples
        --------
        >>> a = Axis('a', 2)
        >>> b = Axis(None, 2)
        >>> c = Axis('c', 2)
        >>> AxisCollection([a, b, c]).ids
        ['a', 1, 'c']
        """
        return [axis.name if axis.name is not None else i
                for i, axis in enumerate(self._list)]

    def axis_id(self, axis):
        """
        Returns the id of an axis.

        Returns
        -------
        str or int
            Id of axis, which is its name if defined and its position otherwise.

        Examples
        --------
        >>> a = Axis('a', 2)
        >>> b = Axis(None, 2)
        >>> c = Axis('c', 2)
        >>> col = AxisCollection([a, b, c])
        >>> col.axis_id(a)
        'a'
        >>> col.axis_id(b)
        1
        >>> col.axis_id(c)
        'c'
        """
        axis = self[axis]
        return axis.name if axis.name is not None else self.index(axis)

    @property
    def shape(self):
        """
        Returns the shape of the collection.

        Returns
        -------
        tuple
            Tuple of lengths of axes.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age, sex, time]).shape
        (20, 2, 4)
        """
        return tuple(len(axis) for axis in self._list)

    @property
    def size(self):
        """
        Returns the size of the collection, i.e.
        the number of elements of the array.

        Returns
        -------
        int
            Number of elements of the array.

        Examples
        --------
        >>> age = Axis('age', range(20))
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age, sex, time]).size
        160
        """
        return np.prod(self.shape)

    @property
    def info(self):
        """
        Describes the collection (shape and labels for each axis).

        Returns
        -------
        str
            Description of the AxisCollection (shape and labels for each axis).

        Examples
        --------
        >>> age = Axis('age', 20)
        >>> sex = Axis('sex', ['M', 'F'])
        >>> time = Axis('time', [2007, 2008, 2009, 2010])
        >>> AxisCollection([age, sex, time]).info
        20 x 2 x 4
         age* [20]: 0 1 2 ... 17 18 19
         sex [2]: 'M' 'F'
         time [4]: 2007 2008 2009 2010
        """
        lines = [" %s [%d]: %s" % (name, len(axis), axis.labels_summary())
                 for name, axis in zip(self.display_names, self._list)]
        shape = " x ".join(str(s) for s in self.shape)
        return ReprString('\n'.join([shape] + lines))

    # XXX: instead of front_if_spread, we might want to require axes to be contiguous
    #      (ie the caller would have to transpose axes before calling this)
    def combine_axes(self, axes=None, sep='_', wildcard=False, front_if_spread=False):
        """Combine several axes into one.

        Parameters
        ----------
        axes : tuple, list or AxisCollection of axes, optional
            axes to combine. Defaults to all axes.
        sep : str, optional
            delimiter to use for combining. Defaults to '_'.
        wildcard : bool, optional
            whether or not to produce a wildcard axis even if the axes to
            combine are not. This is much faster, but loose axes labels.
        front_if_spread : bool, optional
            whether or not to move the combined axis at the front (it will be
            the first axis) if the combined axes are not next to each other.

        Returns
        -------
        AxisCollection
            New AxisCollection with combined axes.
        """
        axes = self if axes is None else self[axes]
        axes_indices = [self.index(axis) for axis in axes]
        diff = np.diff(axes_indices)
        # combined axes in front
        if front_if_spread and np.any(diff > 1):
            combined_axis_pos = 0
        else:
            combined_axis_pos = min(axes_indices)

        # all anonymous axes => anonymous combined axis
        if all(axis.name is None for axis in axes):
            combined_name = None
        else:
            combined_name = sep.join(str(id_) for id_ in axes.ids)

        if wildcard:
            combined_axis = Axis(combined_name, axes.size)
        else:
            # TODO: the combined keys should be objects which display as:
            # (axis1_label, axis2_label, ...) but which should also store
            # the axes names)
            # Q: Should it be the same object as the NDLGroup?/NDKey?
            # A: yes. On the Pandas backend, we could/should have
            #    separate axes. On the numpy backend we cannot.
            if len(axes) == 1:
                # Q: if axis is a wildcard axis, should the result be a
                #    wildcard axis (and axes_labels discarded?)
                combined_labels = axes[0].labels
            else:
                combined_labels = [sep.join(str(l) for l in p)
                                   for p in product(*axes.labels)]

            combined_axis = Axis(combined_name, combined_labels)
        new_axes = self - axes
        new_axes.insert(combined_axis_pos, combined_axis)
        return new_axes

    def split_axis(self, axis, sep='_', names=None, regex=None):
        """Split one axis and returns a new collection

        Parameters
        ----------
        axis : int, str or Axis
            axis to split. All its labels *must* contain
            the given delimiter string.
        sep : str, optional
            delimiter to use for splitting. Defaults to '_'.
            When `regex` is provided, the delimiter is only used
            on `names` if given as one string or on axis name if
            `names` is None.
        names : str or list of str, optional
            names of resulting axes. Defaults to None.
        regex : str, optional
            use regex instead of delimiter to split labels.
            Defaults to None.

        Returns
        -------
        AxisCollection
        """
        axis = self[axis]
        axis_index = self.index(axis)
        if names is None:
            if sep not in axis.name:
                raise ValueError('{} not found in axis name ({})'
                                 .format(sep, axis.name))
            else:
                names = axis.name.split(sep)
        elif isinstance(names, str):
            if sep not in names:
                raise ValueError('{} not found in names ({})'
                                 .format(sep, names))
            else:
                names = names.split(sep)
        else:
            assert all(isinstance(name, str) for name in names)

        if not regex:
            # gives us an array of lists
            split_labels = np.char.split(axis.labels, sep)
        else:
            rx = re.compile(regex)
            split_labels = [rx.match(l).groups() for l in axis.labels]
        # not using np.unique because we want to keep the original order
        axes_labels = [unique_list(ax_labels) for ax_labels in zip(*split_labels)]
        split_axes = [Axis(name, axis_labels)
                      for name, axis_labels in zip(names, axes_labels)]
        return self[:axis_index] + split_axes + self[axis_index + 1:]


def all(values, axis=None):
    """
    Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    LArray.all
    """
    if isinstance(values, LArray):
        return values.all(axis)
    else:
        return builtins.all(values)


def any(values, axis=None):
    """
    Test whether any array elements along a given axis evaluate to True.

    See Also
    --------
    LArray.any
    """
    if isinstance(values, LArray):
        return values.any(axis)
    else:
        return builtins.any(values)


# commutative modulo float precision errors
def sum(array, *args, **kwargs):
    """
    Sum of array elements.

    See Also
    --------

    LArray.sum
    """
    # XXX: we might want to be more aggressive here (more types to convert),
    #      however, generators should still be computed via the builtin.
    if isinstance(array, (np.ndarray, list)):
        array = LArray(array)
    if isinstance(array, LArray):
        return array.sum(*args, **kwargs)
    else:
        return builtins.sum(array, *args, **kwargs)


def prod(array, *args, **kwargs):
    """
    Product of array elements.

    See Also
    --------
    LArray.prod
    """
    return array.prod(*args, **kwargs)


def cumsum(array, *args, **kwargs):
    """
    Returns the cumulative sum of array elements.

    See Also
    --------
    LArray.cumsum
    """
    return array.cumsum(*args, **kwargs)


def cumprod(array, *args, **kwargs):
    """
    Returns the cumulative product of array elements.

    See Also
    --------
    LArray.cumprod
    """
    return array.cumprod(*args, **kwargs)


def min(array, *args, **kwargs):
    """
    Minimum of array elements.

    See Also
    --------
    LArray.min
    """
    if isinstance(array, LArray):
        return array.min(*args, **kwargs)
    else:
        return builtins.min(array, *args, **kwargs)


def max(array, *args, **kwargs):
    """
    Maximum of array elements.

    See Also
    --------
    LArray.max
    """
    if isinstance(array, LArray):
        return array.max(*args, **kwargs)
    else:
        return builtins.max(array, *args, **kwargs)


def mean(array, *args, **kwargs):
    """
    Computes the arithmetic mean.

    See Also
    --------
    LArray.mean
    """
    return array.mean(*args, **kwargs)


def median(array, *args, **kwargs):
    """
    Computes the median.

    See Also
    --------
    LArray.median
    """
    return array.median(*args, **kwargs)


def percentile(array, *args, **kwargs):
    """
    Computes the qth percentile of the data along the specified axis.

    See Also
    --------
    LArray.percentile
    """
    return array.percentile(*args, **kwargs)


# not commutative
def ptp(array, *args, **kwargs):
    """
    Returns the range of values (maximum - minimum).

    See Also
    --------
    LArray.ptp
    """
    return array.ptp(*args, **kwargs)


def var(array, *args, **kwargs):
    """
    Computes the variance.

    See Also
    --------
    LArray.var
    """
    return array.var(*args, **kwargs)


def std(array, *args, **kwargs):
    """
    Computes the standard deviation.

    See Also
    --------
    LArray.std
    """
    return array.std(*args, **kwargs)


_numeric_kinds = 'buifc'    # Boolean, Unsigned integer, Integer, Float, Complex
_string_kinds = 'SU'        # String, Unicode
_meta_kind = {k: 'str' for k in _string_kinds}
_meta_kind.update({k: 'numeric' for k in _numeric_kinds})


def common_type(arrays):
    """
    Returns a type which is common to the input arrays.
    All input arrays can be safely cast to the returned dtype without loss of information.

    Notes
    -----
    If list of arrays mixes 'numeric' and 'string' types, the function returns 'object'
    as common type.
    """
    arrays = [np.asarray(a) for a in arrays]
    dtypes = [a.dtype for a in arrays]
    meta_kinds = [_meta_kind.get(dt.kind, 'other') for dt in dtypes]
    # mixing string and numeric => object
    if any(mk != meta_kinds[0] for mk in meta_kinds[1:]):
        return object
    elif meta_kinds[0] == 'numeric':
        return np.find_common_type(dtypes, [])
    elif meta_kinds[0] == 'str':
        need_unicode = any(dt.kind == 'U' for dt in dtypes)
        # unicode are coded with 4 bytes
        max_size = max(dt.itemsize // 4 if dt.kind == 'U' else dt.itemsize
                       for dt in dtypes)
        return np.dtype(('U' if need_unicode else 'S', max_size))
    else:
        return object


def concat_empty(axis, arrays_axes, dtype):
    # Get axis by name, so that we do *NOT* check they are "compatible",
    # because it makes sense to append axes of different length
    arrays_axis = [axes[axis] for axes in arrays_axes]
    arrays_labels = [axis.labels for axis in arrays_axis]

    # switch to object dtype if labels are of incompatible types, so that
    # we do not implicitly convert numeric types to strings (numpy should not
    # do this in the first place but that is another story). This can happen for
    # example when we want to add a "total" tick to a numeric axis (eg age).
    labels_type = common_type(arrays_labels)
    if labels_type is object:
        # astype always copies, while asarray only copies if necessary
        arrays_labels = [np.asarray(labels, dtype=object)
                         for labels in arrays_labels]
    new_labels = np.concatenate(arrays_labels)
    combined_axis = Axis(arrays_axis[0].name, new_labels)

    new_axes = [axes.replace(axis, combined_axis)
                for axes, axis in zip(arrays_axes, arrays_axis)]

    # combine all axes (using labels from any side if any)
    result_axes = AxisCollection.union(*new_axes)

    result = empty(result_axes, dtype=dtype)
    lengths = [len(axis) for axis in arrays_axis]
    cumlen = np.cumsum(lengths)
    start_bounds = np.concatenate(([0], cumlen[:-1]))
    stop_bounds = cumlen
    # XXX: wouldn't it be nice to be able to say that? ie translation
    # from position to label on the original axis then translation to
    # position on the actual result axis?
    # result[:axis.i[-1]]
    return result, [result[combined_axis.i[start:stop]]
                    for start, stop in zip(start_bounds, stop_bounds)]


class LArrayIterator(object):
    def __init__(self, array):
        self.array = array
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        array = self.array
        if self.position == len(self.array):
            raise StopIteration
        # result = array.i[array.axes[0].i[self.position]]
        result = array.i[self.position]
        self.position += 1
        return result
    # Python 2
    next = __next__


class LArrayPositionalIndexer(object):
    def __init__(self, array):
        self.array = array

    def _translate_key(self, key):
        """
        Translates key into tuple of PGroup, i.e.
        tuple of collections of labels.
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.array.ndim:
            raise IndexError("key has too many indices (%d) for array with %d "
                             "dimensions" % (len(key), self.array.ndim))
        # no need to create a full nd key as that will be done later anyway
        return tuple(axis.i[axis_key]
                     for axis_key, axis in zip(key, self.array.axes))

    def __getitem__(self, key):
        return self.array[self._translate_key(key)]

    def __setitem__(self, key, value):
        self.array[self._translate_key(key)] = value

    def __len__(self):
        return len(self.array)


class LArrayPointsIndexer(object):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        # TODO: this should generate an "intersection"/points NDGroup and simply
        #       do return self.array[nd_group]
        data = np.asarray(self.array)
        translated_key = self.array._translated_key(key, bool_stuff=True)

        axes = self.array._bool_key_new_axes(translated_key)
        data = data[translated_key]
        # drop length 1 dimensions created by scalar keys
        # data = data.reshape(tuple(len(axis) for axis in axes))
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            return LArray(data, axes)

    # FIXME
    def __setitem__(self, key, value):
        raise NotImplementedError()


class LArrayPositionalPointsIndexer(object):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        data = np.asarray(self.array)

        axes = self.array._bool_key_new_axes(key, wildcard_allowed=False)
        data = data[key]
        # drop length 1 dimensions created by scalar keys
        # data = data.reshape(tuple(len(axis) for axis in axes))
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            return LArray(data, axes)

    def __setitem__(self, key, value):
        data = np.asarray(self.array)
        data[key] = value


def get_axis(obj, i):
    """
    Returns an axis according to its position.

    Parameters
    ----------
    obj : LArray or other array
        Input LArray or any array object which has a shape attribute
        (NumPy or Pandas array).
    i : int
        Position of the axis.

    Returns
    -------
    Axis
        Axis corresponding to the given position if input `obj` is a LArray.
        A new anonymous Axis with the length of the ith dimension of
        the input `obj` otherwise.

    Examples
    --------
    >>> arr = ndtest((2, 2, 2))
    >>> arr
     a | b\c | c0 | c1
    a0 |  b0 |  0 |  1
    a0 |  b1 |  2 |  3
    a1 |  b0 |  4 |  5
    a1 |  b1 |  6 |  7
    >>> get_axis(arr, 1)
    Axis('b', ['b0', 'b1'])
    >>> np_arr = np.zeros((2, 2, 2))
    >>> get_axis(np_arr, 1)
    Axis(None, 2)
    """
    return obj.axes[i] if isinstance(obj, LArray) else Axis(None, obj.shape[i])


def aslarray(a):
    """
    Converts input as LArray if possible.

    Parameters
    ----------
    a : array-like
        Input array to convert into a LArray.

    Returns
    -------
    LArray

    Examples
    --------
    >>> # NumPy array
    >>> np_arr = np.arange(6).reshape((2,3))
    >>> aslarray(np_arr)
    {0}*\{1}* | 0 | 1 | 2
            0 | 0 | 1 | 2
            1 | 3 | 4 | 5
    >>> # Pandas dataframe
    >>> data = {'normal'  : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    ...         'reverse' : pd.Series([3., 2., 1.], index=['a', 'b', 'c'])}
    >>> df = pd.DataFrame(data)
    >>> aslarray(df)
    {0}\{1} | normal | reverse
          a |    1.0 |     3.0
          b |    2.0 |     2.0
          c |    3.0 |     1.0
    """
    if isinstance(a, LArray):
        return a
    elif hasattr(a, '__larray__'):
        return a.__larray__()
    elif isinstance(a, pd.DataFrame):
        return df_aslarray(a)
    else:
        return LArray(a)


_arg_agg = {
    'q':
        """
        q : int in range of [0,100] (or sequence of floats)
            Percentile to compute, which must be between 0 and 100 inclusive.
        """
}

_kwarg_agg = {
    'dtype':
        """
            * dtype : dtype, optional

              The type of the returned array.
              The dtype of the array is used by default.
        """,
    'out':
        """
            * out : LArray, optional

              Alternate output array in which to place the result.
              It must have the same shape as the expected output and
              its type is preserved (e.g., if dtype(out) is float, the
              result will consist of 0.0’s and 1.0’s).
              Axes and labels can be different, only the shape matters.
        """,
    'ddof':
        """
            * ddof : int, optional

              "Delta Degrees of Freedom": the divisor used in the
              calculation is ``N - ddof``, where ``N`` represents
              the number of elements. By default `ddof` is zero.
        """,
    'skipna':
        """
            * skipna : bool, optional

              'skip NaN': Ignore NaN/null values.
              If an entire row/column is NaN, the result will be NaN.
        """,
    'keepaxes':
        """
            * keepaxes : bool, optional

              If this is set to True, the axes which are reduced are
              left in the result as dimensions with size one.
              With this option, the result will broadcast correctly
              against the input array.
        """,
    'interpolation':
        """
            * interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}

              This optional parameter specifies the interpolation method to
              use when the desired quantile lies between two data points ``i < j``:
                * linear: ``i + (j - i) * fraction``, where ``fraction`` is the
                  fractional part of the index surrounded by ``i`` and ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        """
}


def _doc_agg_method(desc, by=False, action="perform",
                    extra_args='', kwargs=''):
    extra_args = extra_args.split(',') if extra_args else []
    kwargs = kwargs.split(',') if kwargs else []

    if by:
        str_doc = "The {0} is {1}ed along all axes except the given one(s). " \
                  "For groups, {0} is {1}ed along groups and non associated axes." \
                  "The default (no axis or group) is to {1} the {0} " \
                  "over all the dimensions of the input array.".format(desc, action)
    else:
        str_doc = "Axis(es) or group(s) along the {0} is {1}ed. " \
                  " The default (no axis or group) is to {1} the {0} " \
                  "over all the dimensions of the input array.".format(desc, action)

    args_doc = \
        """
        Parameters
        ----------
        {0}
        \*args : None or int or str or Axis or Group or any combination of those

            {1}

            An axis can be referred by:

            * its position (integer). Position can be a negative integer,
              in which case it counts from the last to the first axis.
            * its name (str or AxisReference). You can use either a simple
              string ('axis_name') or the special variable x (x.axis_name).
            * a variable (Axis). If the axis has been defined previously
              and assigned to a variable, you can pass it as argument.

            You may not want to {2} the {3} over a whole axis but
            over a selection of specific labels. To do so, you have several
            possibilities:

            * (['a1', 'a3', 'a5'], 'b1, b3, b5') :
              labels separated by commas in a list or a string
            * ('a1:a5:2') : select labels using a slice
              (general syntax is 'start:end:step' where is 'step' is
              optional and 1 by default).
            * (a='a1, a2, a3', x.b['b1, b2, b3']) :
              in case of possible ambiguity, i.e. if labels
              can belong to more than one axis, you must precise the axis.
            * ('a1:a3; a5:a7', b='b0,b2; b1,b3') :
              create several groups with semicolons.
              Names are simply given by the concatenation of labels
              (here: 'a1,a2,a3', 'a5,a6,a7', 'b0,b2' and 'b1,b3')
            * ('a1:a3 >> a123', 'b[b0,b2] >> b12') :
              operator ' >> ' allows to rename groups.

        \**kwargs :
        {4}
        """.format("".join(_arg_agg[arg] for arg in extra_args),
                   str_doc, action, desc,
                   "".join(_kwarg_agg[kw] for kw in kwargs))

    return args_doc


class LArray(object):
    """
    A LArray object represents a multidimensional, homogeneous
    array of fixed-size items with labeled axes.

    The function :func:`aslarray` can be used to convert a
    NumPy array or PandaS DataFrame into a LArray.

    Parameters
    ----------
    data : scalar, tuple, list or NumPy ndarray
        Input data.
    axes : collection (tuple, list or AxisCollection) of axes \
    (int, str or  Axis), optional
        Axes.
    title : str, optional
        Title of array.

    Attributes
    ----------
    data : NumPy ndarray
        Data.
    axes : AxisCollection
        Axes.
    title : str
        Title.

    See Also
    --------
    create_sequential : Create a LArray by sequentially
                        applying modifications to the array along axis.
    ndrange : Create a LArray with increasing elements.
    zeros : Create a LArray, each element of which is zero.
    ones : Create a LArray, each element of which is 1.
    full : Create a LArray filled with a given value.
    empty : Create a LArray, but leave its allocated memory
            unchanged (i.e., it contains “garbage”).

    Examples
    --------
    >>> age = Axis('age', [10, 11, 12])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> time = Axis('time', [2007, 2008, 2009])
    >>> axes = [age, sex, time]
    >>> data = np.zeros((len(axes), len(sex), len(time)))
    >>> LArray(data, axes)
    age | sex\\time | 2007 | 2008 | 2009
     10 |        M |  0.0 |  0.0 |  0.0
     10 |        F |  0.0 |  0.0 |  0.0
     11 |        M |  0.0 |  0.0 |  0.0
     11 |        F |  0.0 |  0.0 |  0.0
     12 |        M |  0.0 |  0.0 |  0.0
     12 |        F |  0.0 |  0.0 |  0.0
    >>> full(axes, 10.0)
    age | sex\\time | 2007 | 2008 | 2009
     10 |        M | 10.0 | 10.0 | 10.0
     10 |        F | 10.0 | 10.0 | 10.0
     11 |        M | 10.0 | 10.0 | 10.0
     11 |        F | 10.0 | 10.0 | 10.0
     12 |        M | 10.0 | 10.0 | 10.0
     12 |        F | 10.0 | 10.0 | 10.0
    >>> arr = empty(axes)
    >>> arr['F'] = 1.0
    >>> arr['M'] = -1.0
    >>> arr
    age | sex\\time | 2007 | 2008 | 2009
     10 |        M | -1.0 | -1.0 | -1.0
     10 |        F |  1.0 |  1.0 |  1.0
     11 |        M | -1.0 | -1.0 | -1.0
     11 |        F |  1.0 |  1.0 |  1.0
     12 |        M | -1.0 | -1.0 | -1.0
     12 |        F |  1.0 |  1.0 |  1.0
    >>> bysex = create_sequential(sex, initial=-1, inc=2)
    >>> bysex
    sex |  M | F
        | -1 | 1
    >>> create_sequential(age, initial=10, inc=bysex)
    sex\\age | 10 | 11 | 12
          M | 10 |  9 |  8
          F | 10 | 11 | 12
    """

    def __init__(self,
                 data,
                 axes=None,
                 title=''   # type: str
                 ):
        data = np.asarray(data)
        ndim = data.ndim
        if axes is None:
            axes = AxisCollection(data.shape)
        else:
            if not isinstance(axes, AxisCollection):
                axes = AxisCollection(axes)
            if axes.ndim != ndim:
                raise ValueError("number of axes (%d) does not match "
                                 "number of dimensions of data (%d)"
                                 % (axes.ndim, ndim))
            if axes.shape != data.shape:
                raise ValueError("length of axes %s does not match "
                                 "data shape %s" % (axes.shape, data.shape))

        # Because __getattr__ and __setattr__ have been overridden
        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'axes', axes)
        object.__setattr__(self, 'title', title)

    # XXX: rename to posnonzero and implement a label version of nonzero
    def nonzero(self):
        """
        Returns the indices of the elements that are non-zero.

        Specifically, it returns a tuple of arrays (one for each dimension)
        containing the indices of the non-zero elements in that dimension.

        Returns
        -------
        tuple of arrays : tuple
            Indices of elements that are non-zero.

        Examples
        --------
        >>> arr = ndtest((2, 3)) % 2
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  0
         a1 |  1 |  0 |  1
        >>> arr.nonzero() # doctest: +SKIP
        [array([0, 1, 1]), array([1, 0, 2])]
        """
        # FIXME: return tuple of PGroup instead (or even NDGroup) so that you
        #  can do a[a.nonzero()]
        return self.data.nonzero()

    def set_axes(self, axes_to_replace=None, new_axis=None, inplace=False, **kwargs):
        """
        Replace one, several or all axes of the array.

        Parameters
        ----------
        axes_to_replace : axis ref or dict {axis ref: axis} or list of tuple (axis ref, axis)
                          or list of Axis or AxisCollection
            Axes to replace. If a single axis reference is given, the `new_axis` argument must be provided.
            If a list of Axis or an AxisCollection is given, all axes will be replaced by the new ones.
            In that case, the number of new axes must match the number of the old ones.
        new_axis : Axis
            New axis if `axes_to_replace` contains a single axis reference.
        inplace : bool
            Whether or not to modify the original object or return a new array and leave the original intact.
        **kwargs : Axis
            New axis for each axis to replace given as a keyword argument.

        Returns
        -------
        LArray
            Array with axes replaced.

        See Also
        --------
        rename : rename one of several axes

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
        >>> row = Axis('row', ['r0', 'r1'])
        >>> column = Axis('column', ['c0', 'c1', 'c2'])

        Replace one axis (second argument `new_axis` must be provided)

        >>> arr.set_axes(x.a, row)
        row\\b | b0 | b1 | b2
           r0 |  0 |  1 |  2
           r1 |  3 |  4 |  5

        Replace several axes (keywords, list of tuple or dictionary)

        >>> arr.set_axes(a=row, b=column) # doctest: +SKIP
        >>> # or
        >>> arr.set_axes([(x.a, row), (x.b, column)]) # doctest: +SKIP
        >>> # or
        >>> arr.set_axes({x.a: row, x.b: column})
        row\\column | c0 | c1 | c2
                r0 |  0 |  1 |  2
                r1 |  3 |  4 |  5

        Replace all axes (list of axes or AxisCollection)

        >>> arr.set_axes([row, column])
        row\\column | c0 | c1 | c2
                r0 |  0 |  1 |  2
                r1 |  3 |  4 |  5
        >>> arr2 = ndrange([row, column])
        >>> arr.set_axes(arr2.axes)
        row\\column | c0 | c1 | c2
                r0 |  0 |  1 |  2
                r1 |  3 |  4 |  5
        """
        if isinstance(axes_to_replace, (list, AxisCollection)) and \
                all([isinstance(axis, Axis) for axis in axes_to_replace]):
            if len(axes_to_replace) != len(self.axes):
                raise ValueError('{} axes given as argument, expected '
                                 '{}'.format(len(axes_to_replace), len(self.axes)))
            axes = axes_to_replace
        else:
            axes = self.axes.copy()
            if isinstance(axes_to_replace, dict):
                items = list(axes_to_replace.items())
            elif isinstance(axes_to_replace, list):
                items = axes_to_replace[:]
            elif isinstance(axes_to_replace, (str, Axis, int)):
                items = [(axes_to_replace, new_axis)]
            else:
                items = []
            items += kwargs.items()
            for old, new in items:
                axes = axes.replace(old, new)
        if inplace:
            object.__setattr__(self, 'axes', axes)
            return self
        else:
            return LArray(self.data, axes, title=self.title)

    def with_axes(self, axes):
        warnings.warn("LArray.with_axes is deprecated, "
                      "use LArray.replace_axes instead",
                      DeprecationWarning)
        return self.set_axes(axes)

    def __getattr__(self, key):
        try:
            return self.axes[key]
        # XXX: maybe I should only catch KeyError here and be more aggressive
        #  in __getitem__ to raise KeyError on any exception
        except Exception:
            return self.__getattribute__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __dir__(self):
        names = set(axis.name for axis in self.axes if axis.name is not None)
        return list(set(dir(self.__class__)) | names)

    @property
    def i(self):
        """
        Allows selection of a subset using positions of labels.

        Examples
        --------
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a | b\\c | c0 | c1 | c2 | c3
        a0 |  b0 |  0 |  1 |  2 |  3
        a0 |  b1 |  4 |  5 |  6 |  7
        a0 |  b2 |  8 |  9 | 10 | 11
        a1 |  b0 | 12 | 13 | 14 | 15
        a1 |  b1 | 16 | 17 | 18 | 19
        a1 |  b2 | 20 | 21 | 22 | 23

        >>> arr.i[:, 0:2, [0,2]]
         a | b\\c | c0 | c2
        a0 |  b0 |  0 |  2
        a0 |  b1 |  4 |  6
        a1 |  b0 | 12 | 14
        a1 |  b1 | 16 | 18
        """
        return LArrayPositionalIndexer(self)

    @property
    def points(self):
        """
        Allows selection of arbitrary items in the array
        based on their N-dimensional label index.

        Examples
        --------
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a | b\\c | c0 | c1 | c2 | c3
        a0 |  b0 |  0 |  1 |  2 |  3
        a0 |  b1 |  4 |  5 |  6 |  7
        a0 |  b2 |  8 |  9 | 10 | 11
        a1 |  b0 | 12 | 13 | 14 | 15
        a1 |  b1 | 16 | 17 | 18 | 19
        a1 |  b2 | 20 | 21 | 22 | 23

        To select the two points with label coordinates
        [a0, b0, c0] and [a1, b2, c2], you must do:

        >>> arr.points['a0,a1', 'b0,b2', 'c0,c2']
        a,b,c | a0,b0,c0 | a1,b2,c2
              |        0 |       22

        The number of label(s) on each dimension must be equal:

        >>> arr.points['a0,a1', 'b0,b2', 'c0,c1,c2'] # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (2,) (3,)
        """
        return LArrayPointsIndexer(self)

    @property
    def ipoints(self):
        """
        Allows selection of arbitrary items in the array based on their
        N-dimensional index.

        Examples
        --------
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a | b\\c | c0 | c1 | c2 | c3
        a0 |  b0 |  0 |  1 |  2 |  3
        a0 |  b1 |  4 |  5 |  6 |  7
        a0 |  b2 |  8 |  9 | 10 | 11
        a1 |  b0 | 12 | 13 | 14 | 15
        a1 |  b1 | 16 | 17 | 18 | 19
        a1 |  b2 | 20 | 21 | 22 | 23

        To select the two points with index coordinates
        [0, 0, 0] and [1, 2, 2], you must do:

        >>> arr.ipoints[[0,1], [0,2], [0,2]]
        a,b,c | a0,b0,c0 | a1,b2,c2
              |        0 |       22

        The number of index(es) on each dimension must be equal:

        >>> arr.ipoints[[0,1], [0,2], [0,1,2]] # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (2,) (3,)
        """
        return LArrayPositionalPointsIndexer(self)

    def to_frame(self, fold_last_axis_name=False, dropna=None):
        """
        Converts LArray into Pandas DataFrame.

        Parameters
        ----------
        fold_last_axis_name : bool, optional.
            False by default.
        dropna : {'any', 'all', None}, optional.
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
            None by default.

        Returns
        -------
        Pandas DataFrame

        Examples
        --------
        >>> arr = ndtest((3, 3, 3))
        >>> arr.to_frame() # doctest: +NORMALIZE_WHITESPACE
        c      c0  c1  c2
        a  b
        a0 b0   0   1   2
           b1   3   4   5
           b2   6   7   8
        a1 b0   9  10  11
           b1  12  13  14
           b2  15  16  17
        a2 b0  18  19  20
           b1  21  22  23
           b2  24  25  26
        >>> arr.to_frame(True) # doctest: +NORMALIZE_WHITESPACE
                c0  c1  c2
        a  b\\c
        a0 b0    0   1   2
           b1    3   4   5
           b2    6   7   8
        a1 b0    9  10  11
           b1   12  13  14
           b2   15  16  17
        a2 b0   18  19  20
           b1   21  22  23
           b2   24  25  26
        """
        columns = pd.Index(self.axes[-1].labels)
        if not fold_last_axis_name:
            columns.name = self.axes[-1].name
        if self.ndim > 1:
            axes_names = self.axes.names[:-1]
            if fold_last_axis_name:
                tmp = axes_names[-1] if axes_names[-1] is not None else ''
                if self.axes[-1].name:
                    axes_names[-1] = "{}\\{}".format(tmp, self.axes[-1].name)

            index = pd.MultiIndex.from_product(self.axes.labels[:-1],
                                               names=axes_names)
        else:
            index = pd.Index([''])
            if fold_last_axis_name:
                index.name = self.axes.names[-1]
        data = np.asarray(self).reshape(len(index), len(columns))
        df = pd.DataFrame(data, index, columns)
        if dropna is not None:
            dropna = dropna if dropna is not True else 'all'
            df.dropna(inplace=True, how=dropna)
        return df
    df = property(to_frame)

    def to_series(self, dropna=False):
        """
        Converts LArray into Pandas Series.

        Parameters
        ----------
        dropna : bool, optional.
            False by default.

        Returns
        -------
        Pandas Series

        Examples
        --------
        >>> arr = ndtest((3, 3), dtype=float)
        >>> arr.to_series() # doctest: +NORMALIZE_WHITESPACE
        a   b
        a0  b0    0.0
            b1    1.0
            b2    2.0
        a1  b0    3.0
            b1    4.0
            b2    5.0
        a2  b0    6.0
            b1    7.0
            b2    8.0
        dtype: float64
        """
        index = pd.MultiIndex.from_product([axis.labels for axis in self.axes],
                                           names=self.axes.names)
        series = pd.Series(np.asarray(self).reshape(self.size), index)
        if dropna:
            series.dropna(inplace=True)
        return series
    series = property(to_series)

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

    def rename(self, renames=None, to=None, inplace=False, **kwargs):
        """Renames axes of the array.

        Parameters
        ----------
        renames : axis ref or dict {axis ref: str} or
                  list of tuple (axis ref, str)
            Renames to apply. If a single axis reference is given, the `to` argument must be used.
        to : string or Axis
            New name if `renames` contains a single axis reference.
        **kwargs :
            New name for each axis given as a keyword argument.

        Returns
        -------
        LArray
            Array with axes renamed.

        See Also
        --------
        set_axes : replace one or several axes

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FO'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = ndrange([nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 0 | 1
             FO | 2 | 3
        >>> arr.rename(x.nat, 'nat2')
        nat2\\sex | M | F
              BE | 0 | 1
              FO | 2 | 3
        >>> arr.rename(nat='nat2', sex='sex2')
        nat2\\sex2 | M | F
               BE | 0 | 1
               FO | 2 | 3
        >>> arr.rename([('nat', 'nat2'), ('sex', 'sex2')])
        nat2\\sex2 | M | F
               BE | 0 | 1
               FO | 2 | 3
        >>> arr.rename({'nat': 'nat2', 'sex': 'sex2'})
        nat2\\sex2 | M | F
               BE | 0 | 1
               FO | 2 | 3
        """
        if isinstance(renames, dict):
            items = list(renames.items())
        elif isinstance(renames, list):
            items = renames[:]
        elif isinstance(renames, (str, Axis, int)):
            items = [(renames, to)]
        else:
            items = []
        items += kwargs.items()
        renames = {self.axes[k]: v for k, v in items}
        axes = [a.rename(renames[a]) if a in renames else a
                for a in self.axes]
        if inplace:
            object.__setattr__(self, 'axes', AxisCollection(axes))
            return self
        else:
            return LArray(self.data, axes)

    def sort_values(self, key):
        """Sorts values of the array.

        Parameters
        ----------
        key : scalar or tuple or Group
            Key along which to sort.
            Must have exactly one dimension less than ndim.

        Returns
        -------
        LArray
            Array with sorted values.

        Examples
        --------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> nat = Axis('nat', ['EU', 'FO', 'BE'])
        >>> xtype = Axis('type', ['type1', 'type2'])
        >>> a = LArray([[10, 2, 4], [3, 7, 1]], [sex, nat])
        >>> a
        sex\\nat | EU | FO | BE
              M | 10 |  2 |  4
              F |  3 |  7 |  1
        >>> a.sort_values('F')
        sex\\nat | BE | EU | FO
              M |  4 | 10 |  2
              F |  1 |  3 |  7
        >>> b = LArray([[[10, 2, 4], [3, 7, 1]], [[5, 1, 6], [2, 8, 9]]],
        ...            [sex, xtype, nat])
        >>> b
        sex | type\\nat | EU | FO | BE
          M |    type1 | 10 |  2 |  4
          M |    type2 |  3 |  7 |  1
          F |    type1 |  5 |  1 |  6
          F |    type2 |  2 |  8 |  9
        >>> b.sort_values(('M', 'type2'))
        sex | type\\nat | BE | EU | FO
          M |    type1 |  4 | 10 |  2
          M |    type2 |  1 |  3 |  7
          F |    type1 |  6 |  5 |  1
          F |    type2 |  9 |  2 |  8
        """
        subset = self[key]
        if subset.ndim > 1:
            raise NotImplementedError("sort_values key must have one "
                                      "dimension less than array.ndim")
        assert subset.ndim == 1
        axis = subset.axes[0]
        posargsort = subset.posargsort()

        # FIXME: .data shouldn't be necessary, but currently, if we do not do
        #  it, we get
        # PGroup(nat | EU | FO | BE
        #            |  1 |  2 |  0, axis='nat')
        # which sorts the *data* correctly, but the labels on the nat axis are
        # not sorted (because the __getitem__ in that case reuse the key
        # axis as-is -- like it should).
        # Both use cases have value, but I think reordering the ticks
        # should be the default. Now, I am unsure where to change this.
        # Probably in PGroupMaker.__getitem__, but then how do I get the
        # "not reordering labels" behavior that I have now?
        # FWIW, using .data, I get PGroup([1, 2, 0], axis='nat'), which works.
        sorter = axis.i[posargsort.data]
        return self[sorter]

    # XXX: rename to sort_axes?
    def sort_axis(self, axes=None, reverse=False):
        """Sorts axes of the array.

        Parameters
        ----------
        axes : axis reference (Axis, str, int) or list of them
            Axis to sort. If None, sorts all axes.
        reverse : bool
            Descending sort (default: False -- ascending)

        Returns
        -------
        LArray
            Array with sorted axes.

        Examples
        --------
        >>> nat = Axis('nat', ['EU', 'FO', 'BE'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> a = ndrange([nat, sex])
        >>> a
        nat\\sex | M | F
             EU | 0 | 1
             FO | 2 | 3
             BE | 4 | 5
        >>> a.sort_axis(x.sex)
        nat\\sex | F | M
             EU | 1 | 0
             FO | 3 | 2
             BE | 5 | 4
        >>> a.sort_axis()
        nat\\sex | F | M
             BE | 5 | 4
             EU | 1 | 0
             FO | 3 | 2
        >>> a.sort_axis((x.sex, x.nat))
        nat\\sex | F | M
             BE | 5 | 4
             EU | 1 | 0
             FO | 3 | 2
        >>> a.sort_axis(reverse=True)
        nat\\sex | M | F
             FO | 2 | 3
             EU | 0 | 1
             BE | 4 | 5
        """
        if axes is None:
            axes = self.axes
        elif not isinstance(axes, (tuple, list, AxisCollection)):
            axes = [axes]

        if not isinstance(axes, AxisCollection):
            axes = self.axes[axes]

        def sort_key(axis):
            key = np.argsort(axis.labels)
            if reverse:
                key = key[::-1]
            return axis.i[key]

        return self[tuple(sort_key(axis) for axis in axes)]

    def _translate_axis_key_chunk(self, axis_key, bool_passthrough=True):
        """
        Translates axis(es) key into axis(es) position(s).

        Parameters
        ----------
        axis_key : any kind of key
            Key to select axis(es).
        bool_passthrough : bool, optional
            True by default.

        Returns
        -------
        PGroup
            Positional group with valid axes (from self.axes)
        """

        if isinstance(axis_key, Group):
            axis = axis_key.axis
            if axis is not None:
                # we have axis information but not necessarily an Axis object
                # from self.axes
                real_axis = self.axes[axis]
                if axis is not real_axis:
                    axis_key = axis_key.with_axis(real_axis)

        # already positional
        if isinstance(axis_key, PGroup):
            if axis is None:
                raise ValueError("positional groups without axis are not "
                                 "supported")
            return axis_key

        # labels but known axis
        if isinstance(axis_key, LGroup) and axis_key.axis is not None:
            axis = axis_key.axis
            try:
                axis_pos_key = axis.translate(axis_key, bool_passthrough)
            except KeyError:
                raise ValueError("%r is not a valid label for any axis"
                                 % axis_key)
            return axis.i[axis_pos_key]

        # otherwise we need to guess the axis
        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        # TODO: use axis_key dtype to only check compatible axes
        for axis in self.axes:
            try:
                axis_pos_key = axis.translate(axis_key, bool_passthrough)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis"
                             % axis_key)
        elif len(valid_axes) > 1:
            # FIXME: .id
            valid_axes = ', '.join(str(a.id) for a in valid_axes)
            raise ValueError('%s is ambiguous (valid in %s)' %
                             (axis_key, valid_axes))
        return valid_axes[0].i[axis_pos_key]

    def _translate_axis_key(self, axis_key, bool_passthrough=True):
        """Same as chunk.

        Returns
        -------
        PGroup
            Positional group with valid axes (from self.axes)
        """
        # TODO: do it for Group without axis too
        # TODO: do it for LArray key too (but using .i[] instead)
        # TODO: we should skip this chunk stuff for keys where the axis is known
        #       otherwise we do translate(key[:1]) without any reason
        #       (in addition to translate(key))
        if isinstance(axis_key, (tuple, list, np.ndarray)):
            axis = None
            # TODO: I should actually do some benchmarks to see if this is
            #       useful, and estimate which numbers to use
            for size in (1, 10, 100, 1000):
                # TODO: do not recheck already checked elements
                key_chunk = axis_key[:size]
                try:
                    tkey = self._translate_axis_key_chunk(key_chunk,
                                                          bool_passthrough)
                    axis = tkey.axis
                    break
                except ValueError:
                    continue
            # the (start of the) key match a single axis
            if axis is not None:
                # make sure we have an Axis object
                # TODO: we should make sure the tkey returned from
                # _translate_axis_key_chunk always contains a real Axis (and
                # thus kill this line)
                axis = self.axes[axis]
                # wrap key in LGroup
                axis_key = axis[axis_key]
                # XXX: reuse tkey chunks and only translate the rest?
            return self._translate_axis_key_chunk(axis_key,
                                                  bool_passthrough)
        else:
            return self._translate_axis_key_chunk(axis_key, bool_passthrough)

    def _guess_axis(self, axis_key):
        if isinstance(axis_key, Group):
            group_axis = axis_key.axis
            if group_axis is not None:
                # we have axis information but not necessarily an Axis object
                # from self.axes
                real_axis = self.axes[group_axis]
                if group_axis is not real_axis:
                    axis_key = axis_key.with_axis(real_axis)
                return axis_key

        # TODO: instead of checking all axes, we should have a big mapping
        # (in AxisCollection or LArray):
        # label -> (axis, index)
        # or possibly (for ambiguous labels)
        # label -> {axis: index}
        # but for Pandas, this wouldn't work, we'd need label -> axis
        valid_axes = []
        for axis in self.axes:
            try:
                axis.translate(axis_key)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            raise ValueError("%s is not a valid label for any axis"
                             % axis_key)
        elif len(valid_axes) > 1:
            # FIXME: .id
            valid_axes = ', '.join(str(a.id) for a in valid_axes)
            raise ValueError('%s is ambiguous (valid in %s)' %
                             (axis_key, valid_axes))
        return valid_axes[0][axis_key]

    # TODO: move this to AxisCollection
    def _translated_key(self, key, bool_stuff=False):
        """Completes and translates key

        Parameters
        ----------
        key : single axis key or tuple of keys or dict {axis_name: axis_key}
           Each axis key can be either a scalar, a list of scalars or
           an LKey.

        Returns
        -------
        Returns a full N dimensional positional key.
        """

        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.bool_) \
                and not bool_stuff:
            return key.nonzero()
        if isinstance(key, LArray) and np.issubdtype(key.dtype, np.bool_) \
                and not bool_stuff:
            # if only the axes order is wrong, transpose
            # FIXME: if the key has both missing and extra axes, it could be
            # the correct size (or even shape, see below)
            if key.size == self.size and key.shape != self.shape:
                return np.asarray(key.transpose(self.axes)).nonzero()
            # otherwise we need to transform the key to integer
            elif key.size != self.size:
                extra_key_axes = key.axes - self.axes
                if extra_key_axes:
                    raise ValueError("subset key %s contains more axes than "
                                     "array %s" % (key.axes, self.axes))

                # do I want to allow key_axis.name to match against
                # axis.num? does not seem like a good idea.
                # but this should work
                # >>> a = ndrange((3, 4))
                # >>> x1, x2 = a.axes
                # >>> a[x2 > 2]

                # the current solution with hash = (name, labels) works
                # but is slow for large axes and broken if axis labels are
                # modified in-place, which I am unsure I want to support
                # anyway
                self.axes.check_compatible(key.axes)
                local_axes = [self.axes[axis] for axis in key.axes]
                map_key = dict(zip(local_axes, np.asarray(key).nonzero()))
                return tuple(map_key.get(axis, slice(None))
                             for axis in self.axes)
            else:
                # correct shape
                # FIXME: if the key has both missing and extra axes (at the
                # position of the missing axes), the shape could be the same
                # while the result should not
                return np.asarray(key).nonzero()

        # convert scalar keys to 1D keys
        if not isinstance(key, (tuple, dict)):
            key = (key,)

        if isinstance(key, tuple):
            # drop slice(None) and Ellipsis since they are meaningless because
            # of guess_axis.
            # XXX: we might want to raise an exception when we find Ellipses
            # or (most) slice(None) because except for a single slice(None)
            # a[:], I don't think there is any point.
            key = [axis_key for axis_key in key
                   if not _isnoneslice(axis_key) and axis_key is not Ellipsis]

            # translate all keys to PGroup
            key = [self._translate_axis_key(axis_key,
                                            bool_passthrough=not bool_stuff)
                   for axis_key in key]

            assert all(isinstance(axis_key, PGroup) for axis_key in key)

            # extract axis from Group keys
            key_items = [(k.axis, k) for k in key]
        else:
            # key axes could be strings or axis references and we want real axes
            key_items = [(self.axes[k], v) for k, v in key.items()]
            # TODO: use _translate_axis_key (to translate to PGroup here too)
            # key_items = [axis.translate(axis_key,
            #                             bool_passthrough=not bool_stuff)
            #              for axis, axis_key in key_items]

        # even keys given as dict can contain duplicates (if the same axis was
        # given under different forms, e.g. name and AxisReference).
        dupe_axes = list(duplicates(axis for axis, axis_key in key_items))
        if dupe_axes:
            dupe_axes = ', '.join(str(axis) for axis in dupe_axes)
            raise ValueError("key has several values for axis: %s"
                             % dupe_axes)

        key = dict(key_items)

        # dict -> tuple (complete and order key)
        assert all(isinstance(k, Axis) for k in key)
        key = [key[axis] if axis in key else slice(None)
               for axis in self.axes]

        # pgroup -> raw positional
        return tuple(axis.translate(axis_key, bool_passthrough=not bool_stuff)
                     for axis, axis_key in zip(self.axes, key))

    # TODO: we only need axes length => move this to AxisCollection
    # (but this backend/numpy-specific so we'll probably need to create a
    #  subclass of it)
    def _cross_key(self, key):
        """
        Returns a key indexing the cross product.

        Parameters
        ----------
        key : complete (contains all dimensions) index-based key.

        Returns
        -------
        key
            A key for indexing the cross product.
        """

        # handle advanced indexing with more than one indexing array:
        # basic indexing (only integer and slices) and advanced indexing
        # with only one indexing array are handled fine by numpy
        if self._needs_advanced_indexing(key):
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
            # result of a[key] can have its dimensions in the wrong order
            # (if the ix_arrays are not next to each other, the corresponding
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

    def _needs_advanced_indexing(self, key):
        sequence = (tuple, list, np.ndarray)
        # count number of indexing arrays (ie non scalar/slices) in tuple
        num_ix_arrays = sum(isinstance(axis_key, sequence) for axis_key in key)
        num_scalars = sum(np.isscalar(axis_key) for axis_key in key)
        num_slices = sum(isinstance(axis_key, slice) for axis_key in key)
        assert len(key) == num_ix_arrays + num_scalars + num_slices
        return num_ix_arrays > 1 or (num_ix_arrays > 0 and num_scalars)

    def _collapse_slices(self, key):
        # isinstance(ndarray, collections.Sequence) is False but it
        # behaves like one
        sequence = (tuple, list, np.ndarray)
        return [_range_to_slice(axis_key, len(axis))
                if isinstance(axis_key, sequence)
                else axis_key
                for axis_key, axis in zip(key, self.axes)]

    def __getitem__(self, key, collapse_slices=False):
        # move this to getattr
        # if isinstance(key, str) and key in ('__array_struct__',
        #                               '__array_interface__'):
        #     raise KeyError("bla")
        if isinstance(key, ExprNode):
            key = key.evaluate(self.axes)

        data = np.asarray(self.data)
        # XXX: I think I should split this into complete_key and translate_key
        # because for LArray keys I need a complete key with axes for subaxis
        #
        translated_key = self._translated_key(key)

        # FIXME: I have a huge problem with boolean labels + non points
        if isinstance(key, (LArray, np.ndarray)) and \
                np.issubdtype(key.dtype, np.bool_):
            return LArray(data[translated_key],
                          self._bool_key_new_axes(translated_key))

        if any(isinstance(axis_key, LArray) for axis_key in translated_key):
            k2 = [k.data if isinstance(k, LArray) else k
                  for k in translated_key]
            res_data = data[k2]
            axes = [axis.subaxis(axis_key)
                    for axis, axis_key in zip(self.axes, translated_key)
                    if not np.isscalar(axis_key)]

            first_col = AxisCollection(axes[0])
            res_axes = first_col.union(*axes[1:])
            return LArray(res_data, res_axes)

        # TODO: if the original key was a list of labels,
        # subaxis(translated_key).labels == orig_key, so we should use
        # orig_axis_key.copy()
        axes = [axis.subaxis(axis_key)
                for axis, axis_key in zip(self.axes, translated_key)
                if not np.isscalar(axis_key)]

        if collapse_slices:
            translated_key = self._collapse_slices(translated_key)
        cross_key = self._cross_key(translated_key)
        data = data[cross_key]
        if not axes:
            # scalars do not need to be wrapped in LArray
            return data
        else:
            # drop length 1 dimensions created by scalar keys
            res_data = data.reshape(tuple(len(axis) for axis in axes))
            assert _equal_modulo_len1(data.shape, res_data.shape)
            return LArray(res_data, axes)

    def __setitem__(self, key, value, collapse_slices=True):
        # TODO: if key or value has more axes than self, we should use
        # total_axes = self.axes + key.axes + value.axes
        # expanded = self.expand(total_axes)
        # data = np.asarray(expanded.data)

        # concerning keys this can make sense in several cases:
        # single bool LArray key with extra axes.
        # tuple of bool LArray keys (eg one for each axis). each could have
        # extra axes. Common axes between keys are not a problem, we can
        # simply "and" them. Though we should avoid explicitly "and"ing them
        # if there is no common axis because that is less efficient than
        # the implicit "and" that is done by numpy __getitem__ (and the fact we
        # need to combine dimensions when any key has more than 1 dim).

        # the bool value represents whether the axis label is taken or not
        # if any bool key (part) has more than one axis, we get combined
        # dimensions out of it.

        # int LArray keys
        # the int value represent a position along ONE particular axis,
        # even if the key has more than one axis.
        if isinstance(key, ExprNode):
            key = key.evaluate(self.axes)

        data = np.asarray(self.data)
        translated_key = self._translated_key(key)

        if isinstance(key, (LArray, np.ndarray)) and \
                np.issubdtype(key.dtype, np.bool_):
            if isinstance(value, LArray):
                new_axes = self._bool_key_new_axes(translated_key,
                                                   wildcard_allowed=True)
                value = value.broadcast_with(new_axes)
            data[translated_key] = value
            return

        if collapse_slices:
            translated_key = self._collapse_slices(translated_key)
        cross_key = self._cross_key(translated_key)

        if isinstance(value, LArray):
            # XXX: we might want to create fakes (or wildcard?) axes in this case,
            # as we only use axes names and axes length, not the ticks, and those
            # could theoretically take a significant time to compute
            if self._needs_advanced_indexing(translated_key):
                # when adv indexing is needed, cross_key converts scalars to lists
                # of 1 element, which does not remove the dimension like scalars
                # normally do
                axes = [axis.subaxis(axis_key) if not np.isscalar(axis_key)
                        else Axis(axis.name, 1)
                        for axis, axis_key in zip(self.axes, translated_key)]
            else:
                axes = [axis.subaxis(axis_key)
                        for axis, axis_key in zip(self.axes, translated_key)
                        if not np.isscalar(axis_key)]
            value = value.broadcast_with(axes)
        else:
            # if value is a "raw" ndarray we rely on numpy broadcasting
            pass

        data[cross_key] = value

    def _bool_key_new_axes(self, key, wildcard_allowed=False):
        """
        Returns an AxisCollection containing combined axes.
        Axes corresponding to scalar key are dropped.

        This method is used in case of boolean key.

        Parameters
        ----------
        key : tuple
            Position-based key
        wildcard_allowed : bool

        Returns
        -------
        AxisCollection

        Notes
        -----
        See examples of properties `points` and `ipoints`.
        """
        # TODO: use AxisCollection.combine_axes. The problem is that combine_axes use product(*axes_labels)
        #       while here we need zip(*axes_labels)
        combined_axes = [axis for axis_key, axis in zip(key, self.axes)
                         if not _isnoneslice(axis_key) and
                            not np.isscalar(axis_key)]
        # scalar axes are not taken, since we want to kill them
        other_axes = [axis for axis_key, axis in zip(key, self.axes)
                      if _isnoneslice(axis_key)]
        assert len(key) > 0
        axes_indices = [self.axes.index(axis) for axis in combined_axes]
        diff = np.diff(axes_indices)
        # this can happen if key has only None slices and scalars
        if not len(combined_axes):
            combined_axis_pos = None
        elif np.any(diff > 1):
            # combined axes in front
            combined_axis_pos = 0
        else:
            combined_axis_pos = axes_indices[0]
        # all anonymous axes => anonymous combined axis
        if all(axis.name is None for axis in combined_axes):
            combined_name = None
        else:
            combined_name = ','.join(str(self.axes.axis_id(axis)) for axis in combined_axes)
        new_axes = other_axes
        if combined_axis_pos is not None:
            if wildcard_allowed:
                lengths = [len(axis_key) for axis_key in key
                           if not _isnoneslice(axis_key) and
                           not np.isscalar(axis_key)]
                combined_axis_len = lengths[0]
                assert all(l == combined_axis_len for l in lengths)
                combined_axis = Axis(combined_name, combined_axis_len)
            else:
                # TODO: the combined keys should be objects which display as:
                # (axis1_label, axis2_label, ...) but which should also store
                # the axis (names?)
                # Q: Should it be the same object as the NDLGroup?/NDKey?
                # A: yes, probably. On the Pandas backend, we could/should have
                #    separate axes. On the numpy backend we cannot.
                axes_labels = [axis.labels[axis_key]
                               for axis_key, axis in zip(key, self.axes)
                               if not _isnoneslice(axis_key) and
                                  not np.isscalar(axis_key)]
                if len(combined_axes) == 1:
                    # Q: if axis is a wildcard axis, should the result be a
                    #    wildcard axis (and axes_labels discarded?)
                    combined_labels = axes_labels[0]
                else:
                    combined_labels = list(zip(*axes_labels))

                # CRAP, this can lead to duplicate labels (especially using
                # .points)
                combined_axis = Axis(combined_name, combined_labels)
            new_axes.insert(combined_axis_pos, combined_axis)
        return AxisCollection(new_axes)

    def set(self, value, **kwargs):
        """
        Sets a subset of array to value.

        * all common axes must be either of length 1 or the same length
        * extra axes in value must be of length 1
        * extra axes in current array can have any length

        Parameters
        ----------
        value : scalar or LArray

        Examples
        --------
        >>> arr = ndtest((3, 3))
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
         a2 |  6 |  7 |  8
        >>> arr['a1:', 'b1:'].set(10)
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 | 10 | 10
         a2 |  6 | 10 | 10
        >>> arr['a1:', 'b1:'].set(ndtest((2, 2)))
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  0 |  1
         a2 |  6 |  2 |  3
        """
        self.__setitem__(kwargs, value)

    def reshape(self, target_axes):
        """
        Given a list of new axes, changes the shape of the array.
        The size of the array (= number of elements) must be equal
        to the product of length of target axes.

        Parameters
        ----------
        target_axes : iterable of Axis
            New axes. The size of the array (= number of stored data)
            must be equal to the product of length of target axes.

        Returns
        -------
        LArray
            New array with new axes but same data.

        Examples
        --------
        >>> arr = ndtest((2, 2, 2))
        >>> arr
         a | b\\c | c0 | c1
        a0 |  b0 |  0 |  1
        a0 |  b1 |  2 |  3
        a1 |  b0 |  4 |  5
        a1 |  b1 |  6 |  7
        >>> new_arr = arr.reshape([Axis('a', ['a0','a1']),
        ... Axis('bc', ['b0c0', 'b0c1', 'b1c0', 'b1c1'])])
        >>> new_arr
        a\\bc | b0c0 | b0c1 | b1c0 | b1c1
          a0 |    0 |    1 |    2 |    3
          a1 |    4 |    5 |    6 |    7
        """
        # this is a dangerous operation, because except for adding
        # length 1 axes (which is safe), it potentially modifies data
        # TODO: add a check/flag? for "unsafe" reshapes (but allow merging
        # several axes & "splitting" axes) etc.
        # eg 4, 3, 2 -> 2, 3, 4 is wrong (even if size is respected)
        #    4, 3, 2 -> 12, 2 is potentially ok (merging adjacent dimensions)
        #            -> 4, 6 is potentially ok (merging adjacent dimensions)
        #            -> 24 is potentially ok (merging adjacent dimensions)
        #            -> 3, 8 WRONG (non adjacent dimensions)
        #            -> 8, 3 WRONG
        #    4, 3, 2 -> 2, 2, 3, 2 is potentially ok (splitting dim)
        data = np.asarray(self).reshape([len(axis) for axis in target_axes])
        return LArray(data, target_axes)

    def reshape_like(self, target):
        """
        Same as reshape but with an array as input.
        Total size (= number of stored data) of the two arrays must be equal.

        See Also
        --------
        reshape : returns a LArray with a new shape given a list of axes.

        Examples
        --------
        >>> arr = zeros((2, 2, 2), dtype=int)
        >>> arr
        {0}* | {1}*\\{2}* | 0 | 1
           0 |         0 | 0 | 0
           0 |         1 | 0 | 0
           1 |         0 | 0 | 0
           1 |         1 | 0 | 0
        >>> new_arr = arr.reshape_like(ndtest((2, 4)))
        >>> new_arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  0 |  0 |  0
         a1 |  0 |  0 |  0 |  0
        """
        return self.reshape(target.axes)

    def broadcast_with(self, other):
        """
        Returns an array that is (NumPy) broadcastable with target.
        Target can be either a LArray or any collection of Axis.

        * all common axes must be either of length 1 or the same length
        * extra axes in source can have any length and will be moved to the
          front
        * extra axes in target can have any length and the result will have axes
          of length 1 for those axes

        This is different from reshape which ensures the result has exactly the
        shape of the target.

        Parameters
        ----------
        other : LArray or collection of Axis
        """
        if isinstance(other, LArray):
            other_axes = other.axes
        else:
            other_axes = other
            if not isinstance(other, AxisCollection):
                other_axes = AxisCollection(other_axes)
        if self.axes == other_axes:
            return self

        target_axes = (self.axes - other_axes) | other_axes

        # XXX: this breaks la['1,5,9'] = la['2,7,3']
        # but that use case should use drop_labels
        # self.axes.check_compatible(target_axes)

        # 1) reorder axes to target order
        array = self.transpose(target_axes & self.axes)

        # 2) add length one axes
        return array.reshape(array.axes.get_all(target_axes))

    # XXX: I wonder if effectively dropping the labels is necessary or not
    # we could perfectly only mark the axis as being a wildcard axis and keep
    # the labels intact. These wildcard axes with labels
    # could be useful in a few situations. For example, Excel sheets could
    # have such behavior: you can slice columns using letters, but that
    # wouldn't prevent doing computation between arrays using different
    # columns. On the other hand, it makes wild axes less obvious and I
    # wonder if there would be a risk of wildcard axes inadvertently leaking.
    # plus it might be confusing if incompatible labels "work".
    def drop_labels(self, axes=None):
        """Drops the labels from axes (replace those axes by "wildcard" axes).

        Useful when you want to apply operations between two arrays
        or subarrays with same shape but incompatible axes
        (different labels).

        Parameters
        ----------
        axes : Axis or list/tuple/AxisCollection of Axis, optional
            Axis(es) on which you want to drop the labels.

        Returns
        -------
        LArray

        Notes
        -----
        Use it at your own risk.

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
        >>> arr1.drop_labels() * arr2
        a\\b | b2 | b3
         a1 |  0 |  1
         a2 |  4 |  9
        >>> arr1.drop_labels(x.a) * arr2.drop_labels(x.b)
        a\\b | b1 | b2
         a1 |  0 |  1
         a2 |  4 |  9
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
            return table2str(list(self.as_table(maxlines=200, edgeitems=5)),
                             'nan', fullinfo=True, maxwidth=200,
                             keepcols=self.ndim - 1)
    __repr__ = __str__

    def __iter__(self):
        return LArrayIterator(self)

    def as_table(self, maxlines=None, edgeitems=5):
        """
        Generator. Returns next line of the table representing an array.

        Parameters
        ----------
        maxlines : int, optional
            Maximum number of lines to show.
        edgeitems : int, optional
            If number of lines to display is greater than `maxlines`,
            only the first and last `edgeitems` lines are displayed.
            Only active if `maxlines` is not None.
            Equals to 5 by default.

        Returns
        -------
        list
            Next line of the table as a list.
        """
        if not self.ndim:
            return

        # ert    | unit | geo\time | 2012   | 2011   | 2010
        # NEER27 | I05  | AT       | 101.41 | 101.63 | 101.63
        # NEER27 | I05  | AU       | 134.86 | 125.29 | 117.08
        width = self.shape[-1]
        height = int(np.prod(self.shape[:-1]))
        data = np.asarray(self).reshape(height, width)

        # get list of names of axes
        axes_names = self.axes.display_names[:]
        # transforms ['a', 'b', 'c', 'd'] into ['a', 'b', 'c\\d']
        if len(axes_names) > 1:
            axes_names[-2] = '\\'.join(axes_names[-2:])
            axes_names.pop()
        # get list of labels for each axis except the last one.
        labels = [axis.labels.tolist() for axis in self.axes[:-1]]
        # creates vertical lines (ticks is a list of list)
        if self.ndim == 1:
            # There is no vertical axis, so the axis name should not have
            # any "tick" below it and we add an empty "tick".
            ticks = [['']]
        else:
            ticks = product(*labels)
        # returns the first line (axes names + labels of last axis)
        yield axes_names + self.axes[-1].labels.tolist()
        # summary if needed
        if maxlines is not None and height > maxlines:
            # replace middle lines of the table by '...'.
            # We show only the first and last edgeitems lines.
            startticks = islice(ticks, edgeitems)
            midticks = [["..."] * (self.ndim - 1)]
            endticks = list(islice(rproduct(*labels), edgeitems))[::-1]
            ticks = chain(startticks, midticks, endticks)
            data = chain(data[:edgeitems].tolist(),
                         [["..."] * width],
                         data[-edgeitems:].tolist())
            for tick, dataline in izip(ticks, data):
                # returns next line (labels of N-1 first axes + data)
                yield list(tick) + dataline
        else:
            for tick, dataline in izip(ticks, data):
                # returns next line (labels of N-1 first axes + data)
                yield list(tick) + dataline.tolist()

    def dump(self, header=True):
        """Dump array as a 2D nested list

        Parameters
        ----------
        header : bool
            Whether or not to output axes names and labels.

        Returns
        -------
        2D nested list
        """
        if not header:
            # flatten all dimensions except the last one
            return self.data.reshape(-1, self.shape[-1]).tolist()
        else:
            return list(self.as_table())

    # XXX: should filter(geo=['W']) return a view by default? (collapse=True)
    # I think it would be dangerous to make it the default
    # behavior, because that would introduce a subtle difference between
    # filter(dim=[a, b]) and filter(dim=[a]) even though it would be faster
    # and uses less memory. Maybe I should have a "view" argument which
    # defaults to 'auto' (ie collapse by default), can be set to False to
    # force a copy and to True to raise an exception if a view is not possible.
    def filter(self, collapse=False, **kwargs):
        """Filters the array along the axes given as keyword arguments.

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
            An aggregate function with this signature:
            func(a, axis=None, dtype=None, out=None, keepdims=False)
        axes : tuple of axes, optional
            Each axis can be an Axis object, str or int
        out : LArray, optional
            Alternative output array in which to place the result.
            It must have the same shape as the expected output
        keepaxes : bool or scalar, optional
            If this is set to True, the axes which are reduced are
            left in the result as dimensions with size one.

        Returns
        -------
        LArray or scalar
        """
        src_data = np.asarray(self)
        axes = self.axes[list(axes)] if axes else self.axes
        axes_indices = tuple(self.axes.index(a) for a in axes) if axes != self.axes else None
        if op.__name__ == 'ptp':
            if axes_indices is not None and len(axes) > 1:
                raise ValueError('ptp can only be applied along a single axis or all axes, not multiple arbitrary axes')
            elif axes_indices is not None:
                axes_indices = axes_indices[0]
        else:
            kwargs['keepdims'] = bool(keepaxes)
        if out is not None:
            assert isinstance(out, LArray)
            kwargs['out'] = out.data
        res_data = op(src_data, axis=axes_indices, **kwargs)
        if keepaxes:
            label = op.__name__.replace('nan', '') if keepaxes is True else keepaxes
            new_axes = [Axis(axis.name, [label]) for axis in axes]
            res_axes = self.axes.replace(axes, new_axes)
        else:
            res_axes = self.axes - axes
        if not res_axes:
            # scalars don't need to be wrapped in LArray
            return res_data
        else:
            return LArray(res_data, res_axes)

    def _cum_aggregate(self, op, axis):
        """
        op is a numpy cumulative aggregate function: func(arr, axis=0).
        axis is an Axis object, a str or an int. Contrary to other aggregate
        functions this only supports one axis at a time.
        """
        # TODO: accept a single group in axis, to filter & aggregate in one shot
        return LArray(op(np.asarray(self), axis=self.axes.index(axis)),
                      self.axes)

    # TODO: now that items is never a (k, v), it should be renamed to
    # something else: args? (groups would be misleading because each "item"
    # can contain several groups)
    # TODO: experiment implementing this using ufunc.reduceat
    # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ufunc.reduceat.html
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
                # they should all have the same axis (this is already checked
                # in _prepare_aggregate though)
                assert all(g.axis.equals(axis) for g in groups[1:])
                killaxis = False
            else:
                # item is in fact a single group
                assert isinstance(item, Group), type(item)
                groups = (item,)
                axis = item.axis
                # it is easier to kill the axis after the fact
                killaxis = True

            axis, axis_idx = res.axes[axis], res.axes.index(axis)
            # potentially translate axis reference to real axes
            groups = tuple(g.with_axis(axis) for g in groups)
            res_shape[axis_idx] = len(groups)
            res_dtype = res.dtype if op not in (np.mean, np.nanmean) else float
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
                    key = _to_key(group.key)
                    assert not isinstance(key, Group)
                    if np.isscalar(key):
                        key = [key]
                    # we do not care about the name at this point
                    group = LGroup(key, axis=group.axis)

                arr = res.__getitem__(group, collapse_slices=True)
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

    def _prepare_aggregate(self, op, args, kwargs=None, commutative=False,
                           stack_depth=1):
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
        def standardise_kw_arg(axis_name, key, stack_depth=1):
            if isinstance(key, str):
                key = to_keys(key, stack_depth + 1)
            if isinstance(key, tuple):
                # XXX +2?
                return tuple(standardise_kw_arg(axis_name, k, stack_depth + 1)
                             for k in key)
            if isinstance(key, LGroup):
                return key
            return self.axes[axis_name][key]

        def to_labelgroup(key, stack_depth=1):
            if isinstance(key, str):
                key = to_keys(key, stack_depth + 1)
            if isinstance(key, tuple):
                # a tuple is supposed to be several groups on the same axis
                # TODO: it would be better to use
                # self._translate_axis_key directly
                # (so that we do not need to do the label -> position
                #  translation twice) but this fails because the groups are
                # also used as ticks on the new axis, and pgroups are not the
                # same that LGroups in this regard (I wonder if
                # ideally it shouldn't be the same???)
                # groups = tuple(self._translate_axis_key(k)
                #                for k in key)
                groups = tuple(self._guess_axis(_to_key(k, stack_depth + 1)) for k in key)
                axis = groups[0].axis
                if not all(g.axis.equals(axis) for g in groups[1:]):
                    raise ValueError("group with different axes: %s"
                                     % str(key))
                return groups
            if isinstance(key, (Group, int, basestring, list, slice)):
                return self._guess_axis(key)
            else:
                raise NotImplementedError("%s has invalid type (%s) for a "
                                          "group aggregate key"
                                          % (key, type(key).__name__))

        def standardise_arg(arg, stack_depth=1):
            if self.axes.isaxis(arg):
                return self.axes[arg]
            else:
                return to_labelgroup(arg, stack_depth + 1)

        operations = [standardise_arg(a, stack_depth=stack_depth + 2)
                      for a in args if a is not None] + \
                     [standardise_kw_arg(k, v, stack_depth=stack_depth + 2)
                      for k, v in sorted_kwargs]
        if not operations:
            # op() without args is equal to op(all_axes)
            operations = self.axes
        return operations

    def _aggregate(self, op, args, kwargs=None, keepaxes=False, by_agg=False,
                   commutative=False, out=None, extra_kwargs={}):
        operations = self._prepare_aggregate(op, args, kwargs, commutative,
                                             stack_depth=3)
        if by_agg and operations != self.axes:
            # get axes to aggregate
            flat_op = chain.from_iterable([(o,) if isinstance(o, (Group, Axis))
                                           else o for o in operations])
            axes = [o.axis if isinstance(o, Group) else o for o in flat_op]
            to_agg = self.axes - axes

            # add groups to axes to aggregate
            def is_or_contains_group(o):
                return isinstance(o, Group) or \
                       (isinstance(o, tuple) and isinstance(o[0], Group))

            operations = list(to_agg) + \
                         [o for o in operations if is_or_contains_group(o)]

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
        Add aggregated values (sum by default) along each axis.
        A user defined label can be given to specified the computed values.

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

        Examples
        --------
        >>> arr = ndtest((3, 3))
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
         a2 |  6 |  7 |  8
        >>> arr.with_total()
          a\\b | b0 | b1 | b2 | total
           a0 |  0 |  1 |  2 |     3
           a1 |  3 |  4 |  5 |    12
           a2 |  6 |  7 |  8 |    21
        total |  9 | 12 | 15 |    36
        >>> arr.with_total(op=prod, label='product')
            a\\b | b0 | b1 | b2 | product
             a0 |  0 |  1 |  2 |       0
             a1 |  3 |  4 |  5 |      60
             a2 |  6 |  7 |  8 |     336
        product |  0 | 28 | 80 |       0
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
        operations = self._prepare_aggregate(op, args, kwargs, False,
                                             stack_depth=2)
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
    # arr[x.sex.i[arr.posargmin(x.sex)]] <- fails
    # and
    # arr[arr.argmin(x.sex)] <- fails
    # should both be equal to arr.min(x.sex)
    # the versions where axis is None already work as expected in the simple
    # case (no ambiguous labels):
    # arr.i[arr.posargmin()]
    # arr[arr.argmin()]
    # for the case where axis is None, we should return an NDGroup
    # so that arr[arr.argmin()] works even if the minimum is on ambiguous labels
    def argmin(self, axis=None):
        """Returns labels of the minimum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices
        corresponding to the first occurrence are returned.

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.argmin(x.sex)
        nat | BE | FR | IT
            |  M |  F |  M
        >>> arr.argmin()
        ('BE', 'M')
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            data = axis.labels[self.data.argmin(axis_idx)]
            return LArray(data, self.axes - axis)
        else:
            indices = np.unravel_index(self.data.argmin(), self.shape)
            return tuple(axis.labels[i] for i, axis in zip(indices, self.axes))

    def posargmin(self, axis=None):
        """Returns indices of the minimum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices
        corresponding to the first occurrence are returned.

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.posargmin(x.sex)
        nat | BE | FR | IT
            |  0 |  1 |  0
        >>> arr.posargmin()
        (0, 0)
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            return LArray(self.data.argmin(axis_idx), self.axes - axis)
        else:
            return np.unravel_index(self.data.argmin(), self.shape)

    def argmax(self, axis=None):
        """Returns labels of the maximum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the maximum values, the labels
        corresponding to the first occurrence are returned.

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.argmax(x.sex)
        nat | BE | FR | IT
            |  F |  M |  F
        >>> arr.argmax()
        ('IT', 'F')
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            data = axis.labels[self.data.argmax(axis_idx)]
            return LArray(data, self.axes - axis)
        else:
            indices = np.unravel_index(self.data.argmax(), self.shape)
            return tuple(axis.labels[i] for i, axis in zip(indices, self.axes))

    def posargmax(self, axis=None):
        """Returns indices of the maximum values along a given axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to work. If not specified, works on the full array.

        Returns
        -------
        LArray

        Notes
        -----
        In case of multiple occurrences of the maximum values, the labels
        corresponding to the first occurrence are returned.

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.posargmax(x.sex)
        nat | BE | FR | IT
            |  1 |  0 |  1
        >>> arr.posargmax()
        (2, 1)
        """
        if axis is not None:
            axis, axis_idx = self.axes[axis], self.axes.index(axis)
            return LArray(self.data.argmax(axis_idx), self.axes - axis)
        else:
            return np.unravel_index(self.data.argmax(), self.shape)

    def argsort(self, axis=None, kind='quicksort'):
        """Returns the labels that would sort this array.

        Perform an indirect sort along the given axis using the algorithm
        specified by the `kind` keyword. It returns an array of labels of the
        same shape as `a` that index data along the given axis in sorted order.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to sort. This can be omitted if array has only
            one axis.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Defaults to 'quicksort'.

        Returns
        -------
        LArray

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = LArray([[0, 1], [3, 2], [2, 5]], [nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 0 | 1
             FR | 3 | 2
             IT | 2 | 5
        >>> arr.argsort(x.sex)
        nat\\sex | 0 | 1
             BE | M | F
             FR | F | M
             IT | M | F
        """
        if axis is None:
            if self.ndim > 1:
                raise ValueError("array has ndim > 1 and no axis specified for argsort")
            axis = self.axes[0]
        axis = self.axes[axis]
        pos = self.posargsort(axis, kind=kind)
        return LArray(axis.labels[pos.data], pos.axes)

    def posargsort(self, axis=None, kind='quicksort'):
        """Returns the indices that would sort this array.

        Performs an indirect sort along the given axis using the algorithm
        specified by the `kind` keyword. It returns an array of indices
        with the same axes as `a` that index data along the given axis in
        sorted order.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to sort. This can be omitted if array has only
            one axis.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Defaults to 'quicksort'.

        Returns
        -------
        LArray

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FR', 'IT'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> arr = LArray([[1, 5], [3, 2], [0, 4]], [nat, sex])
        >>> arr
        nat\\sex | M | F
             BE | 1 | 5
             FR | 3 | 2
             IT | 0 | 4
        >>> arr.posargsort(x.nat)
        nat\\sex | M | F
              0 | 2 | 1
              1 | 0 | 2
              2 | 1 | 0
        """
        if axis is None:
            if self.ndim > 1:
                raise ValueError("array has ndim > 1 and no axis specified for posargsort")
            axis = self.axes[0]
        axis, axis_idx = self.axes[axis], self.axes.index(axis)
        data = self.data.argsort(axis_idx, kind=kind)
        new_axis = Axis(axis.name, np.arange(len(axis)))
        return LArray(data, self.axes.replace(axis, new_axis))

    def copy(self):
        """Returns a copy of the array.
        """
        return LArray(self.data.copy(), axes=self.axes[:], title=self.title)

    @property
    def info(self):
        """Describes a LArray (title + shape and labels for each axis).

        Returns
        -------
        str
            Description of the array (title + shape and labels for each axis).

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FO'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> mat0 = ones([nat, sex])
        >>> mat0.info
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'M' 'F'
        >>> mat1 = LArray(np.ones((2, 2)), [nat, sex], 'test matrix')
        >>> mat1.info
        test matrix
        2 x 2
         nat [2]: 'BE' 'FO'
         sex [2]: 'M' 'F'
        """
        if self.title:
            return ReprString(self.title + '\n' + self.axes.info)
        else:
            return self.axes.info

    def ratio(self, *axes):
        """Returns an array with all values divided by the
         sum of values along given axes.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            array / array.sum(axes)

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FO'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> a = LArray([[4, 6], [2, 8]], [nat, sex])
        >>> a
        nat\\sex | M | F
             BE | 4 | 6
             FO | 2 | 8
        >>> a.sum()
        20
        >>> a.ratio()
        nat\\sex |   M |   F
             BE | 0.2 | 0.3
             FO | 0.1 | 0.4
        >>> a.ratio(x.sex)
        nat\\sex |   M |   F
             BE | 0.4 | 0.6
             FO | 0.2 | 0.8
        >>> a.ratio('M')
        nat\\sex |   M |   F
             BE | 1.0 | 1.5
             FO | 1.0 | 4.0
        """
        # # this does not work, but I am unsure it should
        # # >>> a.sum(age[[0, 1]], age[2]) / a.sum(age)
        # >>> a.sum(([0, 1], 2)) / a.sum(age)
        # # >>> a / a.sum(([0, 1], 2))
        # >>> a.sum(x.sex)
        # >>> a.sum(x.age)
        # >>> a.sum(x.sex) / a.sum(x.age)
        # >>> a.ratio('F')
        # could mean
        # >>> a.sum('F') / a.sum(a.get_axis('F'))
        # >>> a.sum('F') / a.sum(x.sex)
        # age |   0 |   1 |              2
        #     | 1.0 | 0.6 | 0.555555555556
        # OR (current meaning)
        # >>> a / a.sum('F')
        # age\\sex |              M |   F
        #       0 |            0.0 | 1.0
        #       1 | 0.666666666667 | 1.0
        #       2 |            0.8 | 1.0
        # One solution is to add an argument
        # >>> a.ratio(what='F', by=x.sex)
        # age |   0 |   1 |              2
        #     | 1.0 | 0.6 | 0.555555555556
        # >>> a.sum('F') / a.sum(x.sex)

        # >>> a.sum((age[[0, 1]], age[[1, 2]])) / a.sum(age)
        # >>> a.ratio((age[[0, 1]], age[[1, 2]]), by=age)

        # >>> a.sum((x.age[[0, 1]], x.age[[1, 2]])) / a.sum(x.age)
        # >>> a.ratio((x.age[[0, 1]], x.age[[1, 2]], by=x.age)

        # >>> lalala.sum(([0, 1], [1, 2])) / lalala.sum(x.age)
        # >>> lalala.ratio(([0, 1], [1, 2]), by=x.age)

        # >>> b = a.sum((age[[0, 1]], age[[1, 2]]))
        # >>> b
        # age\sex | M | F
        #   [0 1] | 2 | 4
        #   [1 2] | 6 | 8
        # >>> b / b.sum(x.age)
        # age\\sex |    M |              F
        #   [0 1] | 0.25 | 0.333333333333
        #   [1 2] | 0.75 | 0.666666666667
        # >>> b / a.sum(x.age)
        # age\\sex |              M |              F
        #   [0 1] | 0.333333333333 | 0.444444444444
        #   [1 2] |            1.0 | 0.888888888889
        # # >>> a.ratio([0, 1], [2])
        # # >>> a.ratio(x.age[[0, 1]], x.age[2])
        # >>> a.ratio((x.age[[0, 1]], x.age[2]))
        # nat\\sex |            M |   F
        #      BE |          0.0 | 1.0
        #      FO | 0.6666666666 | 1.0
        return self / self.sum(*axes)

    def rationot0(self, *axes):
        """Returns a LArray with values array / array.sum(axes) where the sum
        is not 0, 0 otherwise.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            array / array.sum(axes)

        Examples
        --------
        >>> a = Axis('a', 'a0,a1')
        >>> b = Axis('b', 'b0,b1,b2')
        >>> arr = LArray([[6, 0, 2],
        ...               [4, 0, 8]], [a, b])
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  6 |  0 |  2
         a1 |  4 |  0 |  8
        >>> arr.sum()
        20
        >>> arr.rationot0()
        a\\b |  b0 |  b1 |  b2
         a0 | 0.3 | 0.0 | 0.1
         a1 | 0.2 | 0.0 | 0.4
        >>> arr.rationot0(x.a)
        a\\b |  b0 |  b1 |  b2
         a0 | 0.6 | 0.0 | 0.2
         a1 | 0.4 | 0.0 | 0.8

        for reference, the normal ratio method would return:

        >>> arr.ratio(x.a)
        a\\b |  b0 |  b1 |  b2
         a0 | 0.6 | nan | 0.2
         a1 | 0.4 | nan | 0.8
        """
        return self.divnot0(self.sum(*axes))

    def percent(self, *axes):
        """Returns an array with values given as
         percent of the total of all values along given axes.

        Parameters
        ----------
        *axes

        Returns
        -------
        LArray
            array / array.sum(axes) * 100

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FO'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> a = LArray([[4, 6], [2, 8]], [nat, sex])
        >>> a
        nat\\sex | M | F
             BE | 4 | 6
             FO | 2 | 8
        >>> a.percent()
        nat\\sex |    M |    F
             BE | 20.0 | 30.0
             FO | 10.0 | 40.0
        >>> a.percent(x.sex)
        nat\\sex |    M |    F
             BE | 40.0 | 60.0
             FO | 20.0 | 80.0
        """
        # dividing by self.sum(*axes) * 0.01 would be faster in many cases but
        # I suspect it loose more precision.
        return self * 100 / self.sum(*axes)

    # aggregate method factory
    def _agg_method(npfunc, nanfunc=None, name=None, commutative=False, by_agg=False):
        def method(self, *args, **kwargs):
            keepaxes = kwargs.pop('keepaxes', False)
            skipna = kwargs.pop('skipna', None)
            if skipna is None:
                skipna = nanfunc is not None
            if skipna and nanfunc is None:
                raise ValueError("skipna is not available for %s" % name)
            # func = npfunc
            func = nanfunc if skipna else npfunc
            return self._aggregate(func, args, kwargs, by_agg=by_agg,
                                   keepaxes=keepaxes,
                                   commutative=commutative)
        if name is None:
            name = npfunc.__name__
        if by_agg:
            name += "_by"
        method.__name__ = name
        return method

    all = _agg_method(np.all, commutative=True)
    all.__doc__ = """
        Test whether all selected elements evaluate to True.

        {}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.all_by, LArray.any, LArray.any_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> barr = arr < 6
        >>> barr
        a\\b |    b0 |    b1 |    b2 |    b3
         a0 |  True |  True |  True |  True
         a1 |  True |  True | False | False
         a2 | False | False | False | False
         a3 | False | False | False | False
        >>> barr.all()
        False
        >>> # along axis 'a'
        >>> barr.all(x.a)
        b |    b0 |    b1 |    b2 |    b3
          | False | False | False | False
        >>> # along axis 'b'
        >>> barr.all(x.b)
        a |   a0 |    a1 |    a2 |    a3
          | True | False | False | False

        Select some rows only

        >>> barr.all(['a0', 'a1'])
        b |   b0 |   b1 |    b2 |    b3
          | True | True | False | False
        >>> # or equivalently
        >>> # barr.all('a0,a1')

        Split an axis in several parts

        >>> barr.all((['a0', 'a1'], ['a2', 'a3']))
          a\\b |    b0 |    b1 |    b2 |    b3
        a0,a1 |  True |  True | False | False
        a2,a3 | False | False | False | False
        >>> # or equivalently
        >>> # barr.all('a0,a1;a2,a3')

        Same with renaming

        >>> barr.all((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b |    b0 |    b1 |    b2 |    b3
        a01 |  True |  True | False | False
        a23 | False | False | False | False
        >>> # or equivalently
        >>> # barr.all('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("AND reduction", kwargs="out,skipna,keepaxes"))

    all_by = _agg_method(np.all, commutative=True, by_agg=True)
    all_by.__doc__ = """
        Test whether all selected elements evaluate to True.

        {}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.all, LArray.any, LArray.any_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> barr = arr < 6
        >>> barr
        a\\b |    b0 |    b1 |    b2 |    b3
         a0 |  True |  True |  True |  True
         a1 |  True |  True | False | False
         a2 | False | False | False | False
         a3 | False | False | False | False
        >>> barr.all_by()
        False
        >>> # by axis 'a'
        >>> barr.all_by(x.a)
        a |   a0 |    a1 |    a2 |    a3
          | True | False | False | False
        >>> # by axis 'b'
        >>> barr.all_by(x.b)
        b |    b0 |    b1 |    b2 |    b3
          | False | False | False | False

        Select some rows only

        >>> barr.all_by(['a0', 'a1'])
        False
        >>> # or equivalently
        >>> # barr.all_by('a0,a1')

        Split an axis in several parts

        >>> barr.all_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          | False | False
        >>> # or equivalently
        >>> # barr.all_by('a0,a1;a2,a3')

        Same with renaming

        >>> barr.all_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a |   a01 |   a23
          | False | False
        >>> # or equivalently
        >>> # barr.all_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("AND reduction", by=True, kwargs="out,skipna,keepaxes"))

    any = _agg_method(np.any, commutative=True)
    any.__doc__ = """
        Test whether any selected elements evaluate to True.

        {}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.any_by, LArray.all, LArray.all_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> barr = arr < 6
        >>> barr
        a\\b |    b0 |    b1 |    b2 |    b3
         a0 |  True |  True |  True |  True
         a1 |  True |  True | False | False
         a2 | False | False | False | False
         a3 | False | False | False | False
        >>> barr.any()
        True
        >>> # along axis 'a'
        >>> barr.any(x.a)
        b |   b0 |   b1 |   b2 |   b3
          | True | True | True | True
        >>> # along axis 'b'
        >>> barr.any(x.b)
        a |   a0 |   a1 |    a2 |    a3
          | True | True | False | False

        Select some rows only

        >>> barr.any(['a0', 'a1'])
        b |   b0 |   b1 |   b2 |   b3
          | True | True | True | True
        >>> # or equivalently
        >>> # barr.any('a0,a1')

        Split an axis in several parts

        >>> barr.any((['a0', 'a1'], ['a2', 'a3']))
          a\\b |    b0 |    b1 |    b2 |    b3
        a0,a1 |  True |  True |  True |  True
        a2,a3 | False | False | False | False
        >>> # or equivalently
        >>> # barr.any('a0,a1;a2,a3')

        Same with renaming

        >>> barr.any((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b |    b0 |    b1 |    b2 |    b3
        a01 |  True |  True |  True |  True
        a23 | False | False | False | False
        >>> # or equivalently
        >>> # barr.any('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("OR reduction", kwargs="out,skipna,keepaxes"))

    any_by = _agg_method(np.any, commutative=True, by_agg=True)
    any_by.__doc__ = """
        Test whether any selected elements evaluate to True.

        {}

        Returns
        -------
        LArray of bool or bool

        See Also
        --------
        LArray.any, LArray.all, LArray.all_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> barr = arr < 6
        >>> barr
        a\\b |    b0 |    b1 |    b2 |    b3
         a0 |  True |  True |  True |  True
         a1 |  True |  True | False | False
         a2 | False | False | False | False
         a3 | False | False | False | False
        >>> barr.any_by()
        True
        >>> # by axis 'a'
        >>> barr.any_by(x.a)
        a |   a0 |   a1 |    a2 |    a3
          | True | True | False | False
        >>> # by axis 'b'
        >>> barr.any_by(x.b)
        b |   b0 |   b1 |   b2 |   b3
          | True | True | True | True

        Select some rows only

        >>> barr.any_by(['a0', 'a1'])
        True
        >>> # or equivalently
        >>> # barr.any_by('a0,a1')

        Split an axis in several parts

        >>> barr.any_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |  True | False
        >>> # or equivalently
        >>> # barr.any_by('a0,a1;a2,a3')

        Same with renaming

        >>> barr.any_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a |  a01 |   a23
          | True | False
        >>> # or equivalently
        >>> # barr.any_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("OR reduction", by=True, kwargs="out,skipna,keepaxes"))

    # commutative modulo float precision errors

    sum = _agg_method(np.sum, np.nansum, commutative=True)
    sum.__doc__ = """
        Computes the sum of array elements along given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.sum_by, LArray.prod, LArray.prod_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.sum()
        120
        >>> # along axis 'a'
        >>> arr.sum(x.a)
        b | b0 | b1 | b2 | b3
          | 24 | 28 | 32 | 36
        >>> # along axis 'b'
        >>> arr.sum(x.b)
        a | a0 | a1 | a2 | a3
          |  6 | 22 | 38 | 54

        Select some rows only

        >>> arr.sum(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  4 |  6 |  8 | 10
        >>> # or equivalently
        >>> # arr.sum('a0,a1')

        Split an axis in several parts

        >>> arr.sum((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  4 |  6 |  8 | 10
        a2,a3 | 20 | 22 | 24 | 26
        >>> # or equivalently
        >>> # arr.sum('a0,a1;a2,a3')

        Same with renaming

        >>> arr.sum((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  4 |  6 |  8 | 10
        a23 | 20 | 22 | 24 | 26
        >>> # or equivalently
        >>> # arr.sum('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("sum", kwargs="dtype,out,skipna,keepaxes"))

    sum_by = _agg_method(np.sum, np.nansum, commutative=True, by_agg=True)
    sum_by.__doc__ = """
        Computes the sum of array elements for the given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.sum, LArray.prod, LArray.prod_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.sum_by()
        120
        >>> # along axis 'a'
        >>> arr.sum_by(x.a)
        a | a0 | a1 | a2 | a3
          |  6 | 22 | 38 | 54
        >>> # along axis 'b'
        >>> arr.sum_by(x.b)
        b | b0 | b1 | b2 | b3
          | 24 | 28 | 32 | 36

        Select some rows only

        >>> arr.sum_by(['a0', 'a1'])
        28
        >>> # or equivalently
        >>> # arr.sum_by('a0,a1')

        Split an axis in several parts

        >>> arr.sum_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |    28 |    92
        >>> # or equivalently
        >>> # arr.sum_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.sum_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 | a23
          |  28 |  92
        >>> # or equivalently
        >>> # arr.sum_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("sum", by=True, kwargs="dtype,out,skipna,keepaxes"))

    # nanprod needs numpy 1.10
    prod = _agg_method(np.prod, np_nanprod, commutative=True)
    prod.__doc__ = """
        Computes the product of array elements along given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.prod_by, LArray.sum, LArray.sum_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.prod()
        0
        >>> # along axis 'a'
        >>> arr.prod(x.a)
        b | b0 |  b1 |   b2 |   b3
          |  0 | 585 | 1680 | 3465
        >>> # along axis 'b'
        >>> arr.prod(x.b)
        a | a0 |  a1 |   a2 |    a3
          |  0 | 840 | 7920 | 32760

        Select some rows only

        >>> arr.prod(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  0 |  5 | 12 | 21
        >>> # or equivalently
        >>> # arr.prod('a0,a1')

        Split an axis in several parts

        >>> arr.prod((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 |  b1 |  b2 |  b3
        a0,a1 |  0 |   5 |  12 |  21
        a2,a3 | 96 | 117 | 140 | 165
        >>> # or equivalently
        >>> # arr.prod('a0,a1;a2,a3')

        Same with renaming

        >>> arr.prod((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 |  b1 |  b2 |  b3
        a01 |  0 |   5 |  12 |  21
        a23 | 96 | 117 | 140 | 165
        >>> # or equivalently
        >>> # arr.prod('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("product", kwargs="dtype,out,skipna,keepaxes"))

    prod_by = _agg_method(np.prod, np_nanprod, commutative=True, by_agg=True)
    prod_by.__doc__ = """
        Computes the product of array elements for the given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.prod, LArray.sum, LArray.sum_by,
        LArray.cumsum, LArray.cumprod

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.prod_by()
        0
        >>> # along axis 'a'
        >>> arr.prod_by(x.a)
        a | a0 |  a1 |   a2 |    a3
          |  0 | 840 | 7920 | 32760
        >>> # along axis 'b'
        >>> arr.prod_by(x.b)
        b | b0 |  b1 |   b2 |   b3
          |  0 | 585 | 1680 | 3465

        Select some rows only

        >>> arr.prod_by(['a0', 'a1'])
        0
        >>> # or equivalently
        >>> # arr.prod_by('a0,a1')

        Split an axis in several parts

        >>> arr.prod_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 |     a2,a3
          |     0 | 259459200
        >>> # or equivalently
        >>> # arr.prod_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.prod_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 |       a23
          |   0 | 259459200
        >>> # or equivalently
        >>> # arr.prod_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("product", by=True, kwargs="dtype,out,skipna,keepaxes"))

    min = _agg_method(np.min, np.nanmin, commutative=True)
    min.__doc__ = """
        Get minimum of array elements along given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.min_by, LArray.max, LArray.max_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.min()
        0
        >>> # along axis 'a'
        >>> arr.min(x.a)
        b | b0 | b1 | b2 | b3
          |  0 |  1 |  2 |  3
        >>> # along axis 'b'
        >>> arr.min(x.b)
        a | a0 | a1 | a2 | a3
          |  0 |  4 |  8 | 12

        Select some rows only

        >>> arr.min(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  0 |  1 |  2 |  3
        >>> # or equivalently
        >>> # arr.min('a0,a1')

        Split an axis in several parts

        >>> arr.min((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  0 |  1 |  2 |  3
        a2,a3 |  8 |  9 | 10 | 11
        >>> # or equivalently
        >>> # arr.min('a0,a1;a2,a3')

        Same with renaming

        >>> arr.min((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  0 |  1 |  2 |  3
        a23 |  8 |  9 | 10 | 11
        >>> # or equivalently
        >>> # arr.min('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("minimum", action="search",
                                   kwargs="out,skipna,keepaxes"))

    min_by = _agg_method(np.min, np.nanmin, commutative=True, by_agg=True)
    min_by.__doc__ = """
        Get minimum of array elements for the given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.min, LArray.max, LArray.max_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.min_by()
        0
        >>> # along axis 'a'
        >>> arr.min_by(x.a)
        a | a0 | a1 | a2 | a3
          |  0 |  4 |  8 | 12
        >>> # along axis 'b'
        >>> arr.min_by(x.b)
        b | b0 | b1 | b2 | b3
          |  0 |  1 |  2 |  3

        Select some rows only

        >>> arr.min_by(['a0', 'a1'])
        0
        >>> # or equivalently
        >>> # arr.min_by('a0,a1')

        Split an axis in several parts

        >>> arr.min_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |     0 |     8
        >>> # or equivalently
        >>> # arr.min_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.min_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 | a23
          |   0 |   8
        >>> # or equivalently
        >>> # arr.min_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("minimum", by=True, action="search",
                                   kwargs="out,skipna,keepaxes"))

    max = _agg_method(np.max, np.nanmax, commutative=True)
    max.__doc__ = """
        Get maximum of array elements along given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.max_by, LArray.min, LArray.min_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.max()
        15
        >>> # along axis 'a'
        >>> arr.max(x.a)
        b | b0 | b1 | b2 | b3
          | 12 | 13 | 14 | 15
        >>> # along axis 'b'
        >>> arr.max(x.b)
        a | a0 | a1 | a2 | a3
          |  3 |  7 | 11 | 15

        Select some rows only

        >>> arr.max(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  4 |  5 |  6 |  7
        >>> # or equivalently
        >>> # arr.max('a0,a1')

        Split an axis in several parts

        >>> arr.max((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  4 |  5 |  6 |  7
        a2,a3 | 12 | 13 | 14 | 15
        >>> # or equivalently
        >>> # arr.max('a0,a1;a2,a3')

        Same with renaming

        >>> arr.max((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  4 |  5 |  6 |  7
        a23 | 12 | 13 | 14 | 15
        >>> # or equivalently
        >>> # arr.max('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("maximum", action="search",
                                   kwargs="out,skipna,keepaxes"))

    max_by = _agg_method(np.max, np.nanmax, commutative=True, by_agg=True)
    max_by.__doc__ = """
        Get maximum of array elements for the given axes/groups.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.max, LArray.min, LArray.min_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.max_by()
        15
        >>> # along axis 'a'
        >>> arr.max_by(x.a)
        a | a0 | a1 | a2 | a3
          |  3 |  7 | 11 | 15
        >>> # along axis 'b'
        >>> arr.max_by(x.b)
        b | b0 | b1 | b2 | b3
          | 12 | 13 | 14 | 15

        Select some rows only

        >>> arr.max_by(['a0', 'a1'])
        7
        >>> # or equivalently
        >>> # arr.max_by('a0,a1')

        Split an axis in several parts

        >>> arr.max_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |     7 |    15
        >>> # or equivalently
        >>> # arr.max_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.max_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 | a23
          |   7 |  15
        >>> # or equivalently
        >>> # arr.max_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("maximum", by=True, action="search",
                                   kwargs="out,skipna,keepaxes"))

    mean = _agg_method(np.mean, np.nanmean, commutative=True)
    mean.__doc__ = """
        Computes the arithmetic mean.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.mean_by, LArray.median, LArray.median_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.mean()
        7.5
        >>> # along axis 'a'
        >>> arr.mean(x.a)
        b |  b0 |  b1 |  b2 |  b3
          | 6.0 | 7.0 | 8.0 | 9.0
        >>> # along axis 'b'
        >>> arr.mean(x.b)
        a |  a0 |  a1 |  a2 |   a3
          | 1.5 | 5.5 | 9.5 | 13.5

        Select some rows only

        >>> arr.mean(['a0', 'a1'])
        b |  b0 |  b1 |  b2 |  b3
          | 2.0 | 3.0 | 4.0 | 5.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1')

        Split an axis in several parts

        >>> arr.mean((['a0', 'a1'], ['a2', 'a3']))
          a\\b |   b0 |   b1 |   b2 |   b3
        a0,a1 |  2.0 |  3.0 |  4.0 |  5.0
        a2,a3 | 10.0 | 11.0 | 12.0 | 13.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1;a2,a3')

        Same with renaming

        >>> arr.mean((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b |   b0 |   b1 |   b2 |   b3
        a01 |  2.0 |  3.0 |  4.0 |  5.0
        a23 | 10.0 | 11.0 | 12.0 | 13.0
        >>> # or equivalently
        >>> # arr.mean('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("mean", kwargs="dtype,out,skipna,keepaxes"))

    mean_by = _agg_method(np.mean, np.nanmean, commutative=True, by_agg=True)
    mean_by.__doc__ = """
        Computes the arithmetic mean.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.mean, LArray.median, LArray.median_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.mean()
        7.5
        >>> # along axis 'a'
        >>> arr.mean_by(x.a)
        a |  a0 |  a1 |  a2 |   a3
          | 1.5 | 5.5 | 9.5 | 13.5
        >>> # along axis 'b'
        >>> arr.mean_by(x.b)
        b |  b0 |  b1 |  b2 |  b3
          | 6.0 | 7.0 | 8.0 | 9.0

        Select some rows only

        >>> arr.mean_by(['a0', 'a1'])
        3.5
        >>> # or equivalently
        >>> # arr.mean_by('a0,a1')

        Split an axis in several parts

        >>> arr.mean_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |   3.5 |  11.5
        >>> # or equivalently
        >>> # arr.mean_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.mean_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 |  a23
          | 3.5 | 11.5
        >>> # or equivalently
        >>> # arr.mean_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("mean", by=True, kwargs="dtype,out,skipna,keepaxes"))

    median = _agg_method(np.median, np.nanmedian, commutative=True)
    median.__doc__ = """
        Computes the arithmetic median.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.median_by, LArray.mean, LArray.mean_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9], \
                        [5, 8, 3, 7], \
                        [6, 2, 0, 9], \
                        [9, 10, 5, 6]]
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 | 10 |  7 |  5 |  9
         a1 |  5 |  8 |  3 |  7
         a2 |  6 |  2 |  0 |  9
         a3 |  9 | 10 |  5 |  6
        >>> arr.median()
        6.5
        >>> # along axis 'a'
        >>> arr.median(x.a)
        b |  b0 |  b1 |  b2 |  b3
          | 7.5 | 7.5 | 4.0 | 8.0
        >>> # along axis 'b'
        >>> arr.median(x.b)
        a |  a0 |  a1 |  a2 |  a3
          | 8.0 | 6.0 | 4.0 | 7.5

        Select some rows only

        >>> arr.median(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  7 |  7 |  4 |  8
        >>> # or equivalently
        >>> # arr.median('a0,a1')

        Split an axis in several parts

        >>> arr.median((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  7 |  7 |  4 |  8
        a2,a3 |  7 |  6 |  2 |  7
        >>> # or equivalently
        >>> # arr.median('a0,a1;a2,a3')

        Same with renaming

        >>> arr.median((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  7 |  7 |  4 |  8
        a23 |  7 |  6 |  2 |  7
        >>> # or equivalently
        >>> # arr.median('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("median", kwargs="out,skipna,keepaxes"))

    median_by = _agg_method(np.median, np.nanmedian, commutative=True, by_agg=True)
    median_by.__doc__ = """
        Computes the arithmetic median.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.median, LArray.mean, LArray.mean_by,
        LArray.var, LArray.var_by, LArray.std, LArray.std_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9], \
                        [5, 8, 3, 7], \
                        [6, 2, 0, 9], \
                        [9, 10, 5, 6]]
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 | 10 |  7 |  5 |  9
         a1 |  5 |  8 |  3 |  7
         a2 |  6 |  2 |  0 |  9
         a3 |  9 | 10 |  5 |  6
        >>> arr.median_by()
        6.5
        >>> # along axis 'a'
        >>> arr.median_by(x.a)
        a |  a0 |  a1 |  a2 |  a3
          | 8.0 | 6.0 | 4.0 | 7.5
        >>> # along axis 'b'
        >>> arr.median_by(x.b)
        b |  b0 |  b1 |  b2 |  b3
          | 7.5 | 7.5 | 4.0 | 8.0

        Select some rows only

        >>> arr.median_by(['a0', 'a1'])
        7.0
        >>> # or equivalently
        >>> # arr.median_by('a0,a1')

        Split an axis in several parts

        >>> arr.median_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |   7.0 |  5.75
        >>> # or equivalently
        >>> # arr.median_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.median_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 |  a23
          | 7.0 | 5.75
        >>> # or equivalently
        >>> # arr.median_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("median", by=True, kwargs="out,skipna,keepaxes"))

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

    percentile.__doc__ = """
        Computes the qth percentile of the data along the specified axis.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.percentile_by, LArray.mean, LArray.mean_by,
        LArray.median, LArray.median_by, LArray.var, LArray.var_by,
        LArray.std, LArray.std_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.percentile(25)
        3.75
        >>> # along axis 'a'
        >>> arr.percentile(25, x.a)
        b |  b0 |  b1 |  b2 |  b3
          | 3.0 | 4.0 | 5.0 | 6.0
        >>> # along axis 'b'
        >>> arr.percentile(25, x.b)
        a |   a0 |   a1 |   a2 |    a3
          | 0.75 | 4.75 | 8.75 | 12.75

        Select some rows only

        >>> arr.percentile(25, ['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  1 |  2 |  3 |  4
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1')

        Split an axis in several parts

        >>> arr.percentile(25, (['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  1 |  2 |  3 |  4
        a2,a3 |  9 | 10 | 11 | 12
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1;a2,a3')

        Same with renaming

        >>> arr.percentile(25, (x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  1 |  2 |  3 |  4
        a23 |  9 | 10 | 11 | 12
        >>> # or equivalently
        >>> # arr.percentile(25, 'a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("qth percentile", extra_args="q",
                                   kwargs="out,interpolation,skipna,keepaxes"))

    def percentile_by(self, q, *args, **kwargs):
        keepaxes = kwargs.pop('keepaxes', False)
        skipna = kwargs.pop('skipna', None)
        if skipna is None:
            skipna = True
        func = np.nanpercentile if skipna else np.percentile
        return self._aggregate(func, args, kwargs, keepaxes=keepaxes, by_agg=True,
                               commutative=True, extra_kwargs={'q': q})

    percentile_by.__doc__ = """
        Computes the qth percentile of the data for the specified axis.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.percentile, LArray.mean, LArray.mean_by,
        LArray.median, LArray.median_by, LArray.var, LArray.var_by,
        LArray.std, LArray.std_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.percentile_by(25)
        3.75
        >>> # along axis 'a'
        >>> arr.percentile_by(25, x.a)
        a |   a0 |   a1 |   a2 |    a3
          | 0.75 | 4.75 | 8.75 | 12.75
        >>> # along axis 'b'
        >>> arr.percentile_by(25, x.b)
        b |  b0 |  b1 |  b2 |  b3
          | 3.0 | 4.0 | 5.0 | 6.0

        Select some rows only

        >>> arr.percentile_by(25, ['a0', 'a1'])
        1.75
        >>> # or equivalently
        >>> # arr.percentile_by('a0,a1')

        Split an axis in several parts

        >>> arr.percentile_by(25, (['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 | a2,a3
          |  1.75 |  9.75
        >>> # or equivalently
        >>> # arr.percentile_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.percentile_by(25, (x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a |  a01 |  a23
          | 1.75 | 9.75
        >>> # or equivalently
        >>> # arr.percentile_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("qth percentile", by=True, extra_args="q",
                                   kwargs="out,interpolation,skipna,keepaxes"))

    # not commutative
    ptp = _agg_method(np.ptp)
    ptp.__doc__ = """
        Returns the range of values (maximum - minimum).

        The name of the function comes from the acronym for ‘peak to peak’.

        {}

        Returns
        -------
        LArray or scalar

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.ptp()
        15
        >>> # along axis 'a'
        >>> arr.ptp(x.a)
        b | b0 | b1 | b2 | b3
          | 12 | 12 | 12 | 12
        >>> # along axis 'b'
        >>> arr.ptp(x.b)
        a | a0 | a1 | a2 | a3
          |  3 |  3 |  3 |  3

        Select some rows only

        >>> arr.ptp(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  4 |  4 |  4 |  4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1')

        Split an axis in several parts

        >>> arr.ptp((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  4 |  4 |  4 |  4
        a2,a3 |  4 |  4 |  4 |  4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1;a2,a3')

        Same with renaming

        >>> arr.ptp((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  4 |  4 |  4 |  4
        a23 |  4 |  4 |  4 |  4
        >>> # or equivalently
        >>> # arr.ptp('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("ptp", kwargs="out"))

    var = _agg_method(np.var, np.nanvar)
    var.__doc__ = """
        Computes the variance.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.var_by, LArray.std, LArray.std_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9], \
                        [5, 8, 3, 7], \
                        [6, 2, 0, 9], \
                        [9, 10, 5, 6]]
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 | 10 |  7 |  5 |  9
         a1 |  5 |  8 |  3 |  7
         a2 |  6 |  2 |  0 |  9
         a3 |  9 | 10 |  5 |  6
        >>> arr.var()
        7.96484375
        >>> # along axis 'a'
        >>> arr.var(x.a)
        b |   b0 |     b1 |     b2 |     b3
          | 4.25 | 8.6875 | 4.1875 | 1.6875
        >>> # along axis 'b'
        >>> arr.var(x.b)
        a |     a0 |     a1 |      a2 |   a3
          | 3.6875 | 3.6875 | 12.1875 | 4.25

        Select some rows only

        >>> arr.var(['a0', 'a1'])
        b | b0 | b1 | b2 | b3
          |  6 |  0 |  1 |  1
        >>> # or equivalently
        >>> # arr.var('a0,a1')

        Split an axis in several parts

        >>> arr.var((['a0', 'a1'], ['a2', 'a3']))
          a\\b | b0 | b1 | b2 | b3
        a0,a1 |  6 |  0 |  1 |  1
        a2,a3 |  2 | 16 |  6 |  2
        >>> # or equivalently
        >>> # arr.var('a0,a1;a2,a3')

        Same with renaming

        >>> arr.var((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b | b0 | b1 | b2 | b3
        a01 |  6 |  0 |  1 |  1
        a23 |  2 | 16 |  6 |  2
        >>> # or equivalently
        >>> # arr.var('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("variance", kwargs="dtype,out,ddof,skipna,keepaxes"))

    var_by = _agg_method(np.var, np.nanvar, by_agg=True)
    var_by.__doc__ = """
        Computes the variance.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.var, LArray.std, LArray.std_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr[:,:] = [[10, 7, 5, 9], \
                        [5, 8, 3, 7], \
                        [6, 2, 0, 9], \
                        [9, 10, 5, 6]]
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 | 10 |  7 |  5 |  9
         a1 |  5 |  8 |  3 |  7
         a2 |  6 |  2 |  0 |  9
         a3 |  9 | 10 |  5 |  6
        >>> arr.var_by()
        7.96484375
        >>> # along axis 'a'
        >>> arr.var_by(x.a)
        a |     a0 |     a1 |      a2 |   a3
          | 3.6875 | 3.6875 | 12.1875 | 4.25
        >>> # along axis 'b'
        >>> arr.var_by(x.b)
        b |   b0 |     b1 |     b2 |     b3
          | 4.25 | 8.6875 | 4.1875 | 1.6875

        Select some rows only

        >>> arr.var_by(['a0', 'a1'])
        0.0
        >>> # or equivalently
        >>> # arr.var_by('a0,a1')

        Split an axis in several parts

        >>> arr.var_by((['a0', 'a1'], ['a2', 'a3']))
        a | a0,a1 |         a2,a3
          |   0.0 | 15.7509765625
        >>> # or equivalently
        >>> # arr.var_by('a0,a1;a2,a3')

        Same with renaming

        >>> arr.var_by((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a | a01 |           a23
          | 0.0 | 15.7509765625
        >>> # or equivalently
        >>> # arr.var_by('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("variance", by=True,
                                   kwargs="dtype,out,ddof,skipna,keepaxes"))

    std = _agg_method(np.std, np.nanstd)
    std.__doc__ = """
        Computes the standard deviation.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.std_by, LArray.var, LArray.var_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4), dtype=float)
        >>> arr[:,:] = [[10, 5, 7, 12], \
                        [5, 8, 7, 9], \
                        [5, 5, 3, 9], \
                        [10, 8, 3, 12]]
        >>> arr
        a\\b |   b0 |  b1 |  b2 |   b3
         a0 | 10.0 | 5.0 | 7.0 | 12.0
         a1 |  5.0 | 8.0 | 7.0 |  9.0
         a2 |  5.0 | 5.0 | 3.0 |  9.0
         a3 | 10.0 | 8.0 | 3.0 | 12.0
        >>> arr.std()
        2.7810744326608736
        >>> # along axis 'a'
        >>> arr.std(x.a)
        b |  b0 |  b1 |  b2 |  b3
          | 2.5 | 1.5 | 2.0 | 1.5

        Select some rows only

        >>> arr.std(['a0', 'a1'])
        b |  b0 |  b1 |  b2 |  b3
          | 2.5 | 1.5 | 0.0 | 1.5
        >>> # or equivalently
        >>> # arr.std('a0,a1')

        Split an axis in several parts

        >>> arr.std((['a0', 'a1'], ['a2', 'a3']))
          a\\b |  b0 |  b1 |  b2 |  b3
        a0,a1 | 2.5 | 1.5 | 0.0 | 1.5
        a2,a3 | 2.5 | 1.5 | 0.0 | 1.5
        >>> # or equivalently
        >>> # arr.std('a0,a1;a2,a3')

        Same with renaming

        >>> arr.std((x.a['a0', 'a1'] >> 'a01', x.a['a2', 'a3'] >> 'a23'))
        a\\b |  b0 |  b1 |  b2 |  b3
        a01 | 2.5 | 1.5 | 0.0 | 1.5
        a23 | 2.5 | 1.5 | 0.0 | 1.5
        >>> # or equivalently
        >>> # arr.std('a0,a1>>a01;a2,a3>>a23')
        """.format(_doc_agg_method("standard deviation",
                                   kwargs="dtype,out,ddof,skipna,keepaxes"))

    std_by = _agg_method(np.std, np.nanstd, by_agg=True)
    std_by.__doc__ = """
        Computes the standard deviation.

        {}

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.std_by, LArray.var, LArray.var_by,
        LArray.mean, LArray.mean_by, LArray.median, LArray.median_by,
        LArray.percentile, LArray.percentile_by

        Examples
        --------
        >>> arr = ndtest((4, 4), dtype=float)
        >>> arr[:,:] = [[10, 5, 7, 12], \
                        [5, 8, 7, 9], \
                        [5, 5, 3, 9], \
                        [10, 8, 3, 12]]
        >>> arr
        a\\b |   b0 |  b1 |  b2 |   b3
         a0 | 10.0 | 5.0 | 7.0 | 12.0
         a1 |  5.0 | 8.0 | 7.0 |  9.0
         a2 |  5.0 | 5.0 | 3.0 |  9.0
         a3 | 10.0 | 8.0 | 3.0 | 12.0
        >>> arr.std_by()
        2.7810744326608736
        >>> # along axis 'b'
        >>> arr.std_by(x.b)
        b |  b0 |  b1 |  b2 |  b3
          | 2.5 | 1.5 | 2.0 | 1.5

        Select some rows only

        >>> arr.std_by(['b0', 'b1'])
        0.5
        >>> # or equivalently
        >>> # arr.std_by('b0,b1')

        Split an axis in several parts

        >>> arr.std_by((['b0', 'b1'], ['b2', 'b3']))
        b | b0,b1 | b2,b3
          |   0.5 |  0.25
        >>> # or equivalently
        >>> # arr.std_by('b0,b1;b2,b3')

        Same with renaming

        >>> arr.std_by((x.b['b0', 'b1'] >> 'b01', x.b['b2', 'b3'] >> 'b23'))
        b | b01 |  b23
          | 0.5 | 0.25
        >>> # or equivalently
        >>> # arr.std_by('b0,b1>>b01;b2,b3>>b23')
        """.format(_doc_agg_method("standard deviation", by=True,
                                   kwargs="dtype,out,ddof,skipna,keepaxes"))

    # cumulative aggregates
    def cumsum(self, axis=-1):
        """
        Returns the cumulative sum of array elements along an axis.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to perform the cumulative sum.
            If given as position, it can be a negative integer,
            in which case it counts from the last to the first axis.
            By default, the cumulative sum is performed
            along the last axis.

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.cumprod, LArray.sum, LArray.sum_by,
        LArray.prod, LArray.prod_by

        Notes
        -----
        Cumulative aggregation functions accept only one axis

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.cumsum()
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  3 |  6
         a1 |  4 |  9 | 15 | 22
         a2 |  8 | 17 | 27 | 38
         a3 | 12 | 25 | 39 | 54
        >>> arr.cumsum(x.a)
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  6 |  8 | 10
         a2 | 12 | 15 | 18 | 21
         a3 | 24 | 28 | 32 | 36
        """
        return self._cum_aggregate(np.cumsum, axis)

    def cumprod(self, axis=-1):
        """
        Returns the cumulative product of array elements.

        Parameters
        ----------
        axis : int or str or Axis, optional
            Axis along which to perform the cumulative product.
            If given as position, it can be a negative integer,
            in which case it counts from the last to the first axis.
            By default, the cumulative product is performed
            along the last axis.

        Returns
        -------
        LArray or scalar

        See Also
        --------
        LArray.cumsum, LArray.sum, LArray.sum_by,
        LArray.prod, LArray.prod_by

        Notes
        -----
        Cumulative aggregation functions accept only one axis.

        Examples
        --------
        >>> arr = ndtest((4, 4))
        >>> arr
        a\\b | b0 | b1 | b2 | b3
         a0 |  0 |  1 |  2 |  3
         a1 |  4 |  5 |  6 |  7
         a2 |  8 |  9 | 10 | 11
         a3 | 12 | 13 | 14 | 15
        >>> arr.cumprod()
        a\\b | b0 |  b1 |   b2 |    b3
         a0 |  0 |   0 |    0 |     0
         a1 |  4 |  20 |  120 |   840
         a2 |  8 |  72 |  720 |  7920
         a3 | 12 | 156 | 2184 | 32760
        >>> arr.cumprod(x.a)
        a\\b | b0 |  b1 |   b2 |   b3
         a0 |  0 |   1 |    2 |    3
         a1 |  0 |   5 |   12 |   21
         a2 |  0 |  45 |  120 |  231
         a3 |  0 | 585 | 1680 | 3465
        """
        return self._cum_aggregate(np.cumprod, axis)

    # element-wise method factory
    def _binop(opname):
        fullname = '__%s__' % opname
        super_method = getattr(np.ndarray, fullname)

        def opmethod(self, other):
            res_axes = self.axes

            if isinstance(other, ExprNode):
                other = other.evaluate(self.axes)

            # we could pass scalars through aslarray too but it is too costly
            # performance-wise for only suppressing one isscalar test and an
            # if statement.
            # TODO: ndarray should probably be converted to larrays because
            # that would harmonize broadcasting rules, but it makes some
            # tests fail for some reason.
            if not isinstance(other, (LArray, np.ndarray)) and \
                    not np.isscalar(other):
                other = aslarray(other)

            if isinstance(other, LArray):
                # TODO: first test if it is not already broadcastable
                (self, other), res_axes = \
                    make_numpy_broadcastable([self, other])
                other = other.data
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

    def __matmul__(self, other):
        """
        Overrides operator @ for matrix multiplication.

        Notes
        -----
        Only available with Python >= 3.5

        Examples
        --------
        >>> arr1d = ndtest(3)
        >>> arr1d
        a | a0 | a1 | a2
          |  0 |  1 |  2
        >>> arr2d = ndtest((3, 3))
        >>> arr2d
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
         a2 |  6 |  7 |  8
        >>> arr1d @ arr1d # doctest: +SKIP
        5
        >>> arr1d @ arr2d # doctest: +SKIP
        b | b0 | b1 | b2
          | 15 | 18 | 21
        >>> arr2d @ arr1d # doctest: +SKIP
        a | a0 | a1 | a2
          |  5 | 14 | 23
        >>> arr3d = ndrange('c=c0..c2;d=d0..d2;e=e0..e2')
        >>> arr1d @ arr3d # doctest: +SKIP
        c\\e | e0 | e1 | e2
         c0 | 15 | 18 | 21
         c1 | 42 | 45 | 48
         c2 | 69 | 72 | 75
        >>> arr3d @ arr1d # doctest: +SKIP
        c\\d | d0 | d1 | d2
         c0 |  5 | 14 | 23
         c1 | 32 | 41 | 50
         c2 | 59 | 68 | 77
        >>> arr3d @ arr3d # doctest: +SKIP
         c | d\\e |   e0 |   e1 |   e2
        c0 |  d0 |   15 |   18 |   21
        c0 |  d1 |   42 |   54 |   66
        c0 |  d2 |   69 |   90 |  111
        c1 |  d0 |  366 |  396 |  426
        c1 |  d1 |  474 |  513 |  552
        c1 |  d2 |  582 |  630 |  678
        c2 |  d0 | 1203 | 1260 | 1317
        c2 |  d1 | 1392 | 1458 | 1524
        c2 |  d2 | 1581 | 1656 | 1731
        """
        current = self[:]
        axes = self.axes
        if not isinstance(other, (LArray, np.ndarray)):
            raise NotImplementedError("matrix multiplication not "
                                      "implemented for %s" % type(other))
        if isinstance(other, np.ndarray):
            other = LArray(other)
        other_axes = other.axes

        combined_axes = axes[:-2] + other_axes[:-2]
        if self.ndim > 2 and other.ndim > 2:
            current = current.expand(combined_axes).transpose(combined_axes)
            other = other.expand(combined_axes).transpose(combined_axes)

        # XXX : What doc of Numpy matmul says:
        # The behavior depends on the arguments in the following way:
        # * If both arguments are 2-D they are multiplied like conventional matrices.
        # * If either argument is N-D, N > 2, it is treated as a stack of matrices
        #   residing in the last two indexes and broadcast accordingly.
        # * If the first argument is 1-D, it is promoted to a matrix by
        #   prepending a 1 to its dimensions. After matrix multiplication
        #   the prepended 1 is removed.
        # * If the second argument is 1-D, it is promoted to a matrix by
        #   appending a 1 to its dimensions. After matrix multiplication
        #   the appended 1 is removed.
        res_data = current.data.__matmul__(other.data)

        res_axes = list(combined_axes)
        if self.ndim > 1:
            res_axes += [axes[-2]]
        if other.ndim > 1:
            res_axes += [other_axes[-1].copy()]
        return LArray(res_data, res_axes)

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            other = LArray(other)
        if not isinstance(other, LArray):
            raise NotImplementedError("matrix multiplication not "
                                      "implemented for %s" % type(other))
        return other.__matmul__(self)

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

    def __round__(self, n=0):
        # XXX: use the ufuncs.round instead?
        return np.round(self, decimals=n)

    def divnot0(self, other):
        """Divides array by other, but returns 0.0 where other is 0.

        Parameters
        ----------
        other : scalar or LArray
            What to divide by.

        Returns
        -------
        LArray
            Array divided by other, 0.0 where other is 0

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FO'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> a = ndrange((nat, sex))
        >>> a
        nat\\sex | M | F
             BE | 0 | 1
             FO | 2 | 3
        >>> b = ndrange(sex)
        >>> b
        sex | M | F
            | 0 | 1
        >>> a / b
        nat\\sex |   M |   F
             BE | nan | 1.0
             FO | inf | 3.0
        >>> a.divnot0(b)
        nat\\sex |   M |   F
             BE | 0.0 | 1.0
             FO | 0.0 | 3.0
        """
        if np.isscalar(other):
            if other == 0:
                return zeros_like(self, dtype=float)
            else:
                return self / other
        else:
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            res = self / other
            np.seterr(**old_settings)
            res[other == 0] = 0
            return res

    # XXX: rename/change to "add_axes" ?
    # TODO: add a flag copy=True to force a new array.
    def expand(self, target_axes=None, out=None, readonly=False):
        """Expands array to target_axes.

        Target axes will be added to array if not present.
        In most cases this function is not needed because
        LArray can do operations with arrays having different
        (compatible) axes.

        Parameters
        ----------
        target_axes : list of Axis or AxisCollection, optional
            Self can contain axes not present in `target_axes`.
            The result axes will be: [self.axes not in target_axes] + target_axes
        out : LArray, optional
            Output array, must have the correct shape
        readonly : bool, optional
            Whether returning a readonly view is acceptable or not (this is
            much faster)

        Returns
        -------
        LArray
            Original array if possible (and out is None).

        Examples
        --------
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

        # TODO: this is not necessary if out is not None ([:] already broadcast)
        broadcasted = self.broadcast_with(target_axes)
        # this can only happen if only the order of axes differed
        if out is None and broadcasted.axes == target_axes:
            return broadcasted

        if out is None:
            if readonly:
                # requires numpy 1.10
                return LArray(np.broadcast_to(broadcasted, target_axes.shape),
                              target_axes)
            else:
                out = LArray(np.empty(target_axes.shape, dtype=self.dtype),
                             target_axes)
        out[:] = broadcasted
        return out

    def append(self, axis, value, label=None):
        """Adds an array to self along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis reference
            Axis along with to append input array (`value`).
        value : LArray
            Array with compatible axes.
        label : str, optional
            Label for the new item in axis

        Returns
        -------
        LArray
            Array expanded with `value` along `axis`.

        Examples
        --------
        >>> a = ones('nat=BE,FO;sex=M,F')
        >>> a
        nat\\sex |   M |   F
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
        >>> a.append(x.sex, a.sum(x.sex), 'M+F')
        nat\\sex |   M |   F | M+F
             BE | 1.0 | 1.0 | 2.0
             FO | 1.0 | 1.0 | 2.0
        >>> a.append(x.nat, 2, 'Other')
        nat\\sex |   M |   F
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
          Other | 2.0 | 2.0
        >>> b = zeros('type=type1,type2')
        >>> b
        type | type1 | type2
             |   0.0 |   0.0
        >>> a.append(x.nat, b, 'Other')
          nat | sex\\type | type1 | type2
           BE |        M |   1.0 |   1.0
           BE |        F |   1.0 |   1.0
           FO |        M |   1.0 |   1.0
           FO |        F |   1.0 |   1.0
        Other |        M |   0.0 |   0.0
        Other |        F |   0.0 |   0.0
        """
        axis = self.axes[axis]
        if np.isscalar(value):
            value = LArray(np.asarray(value, self.dtype))
        # This does not prevent value to have more axes than self.
        # We do not use .expand(..., readonly=True) so that the code is more
        # similar to .prepend().
        target_axes = self.axes.replace(axis, Axis(axis.name, [label]))
        value = value.broadcast_with(target_axes)
        return self.extend(axis, value)

    def prepend(self, axis, value, label=None):
        """Adds an array before self along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis reference
            Axis along which to prepend input array (`value`)
        value : LArray
            Array with compatible axes.
        label : str, optional
            Label for the new item in axis

        Returns
        -------
        LArray
            Array expanded with 'value' at the start of 'axis'.

        Examples
        --------
        >>> a = ones('nat=BE,FO;sex=M,F')
        >>> a
        nat\sex |   M |   F
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
        >>> a.prepend(x.sex, a.sum(x.sex), 'M+F')
        nat\\sex | M+F |   M |   F
             BE | 2.0 | 1.0 | 1.0
             FO | 2.0 | 1.0 | 1.0
        >>> a.prepend(x.nat, 2, 'Other')
        nat\\sex |   M |   F
          Other | 2.0 | 2.0
             BE | 1.0 | 1.0
             FO | 1.0 | 1.0
        >>> b = zeros('type=type1,type2')
        >>> b
        type | type1 | type2
             |   0.0 |   0.0
        >>> a.prepend(x.nat, b, 'Other')
         type | nat\sex |   M |   F
        type1 |   Other | 0.0 | 0.0
        type1 |      BE | 1.0 | 1.0
        type1 |      FO | 1.0 | 1.0
        type2 |   Other | 0.0 | 0.0
        type2 |      BE | 1.0 | 1.0
        type2 |      FO | 1.0 | 1.0
        """
        axis = self.axes[axis]
        if np.isscalar(value):
            value = LArray(np.asarray(value, self.dtype))
        # This does not prevent value to have more axes than self
        target_axes = self.axes.replace(axis, Axis(axis.name, [label]))
        # We cannot simply add the "new" axis to value (e.g. using expand)
        # because the resulting axes would not be in the correct order.
        value = value.broadcast_with(target_axes)
        return value.extend(axis, self)

    def extend(self, axis, other):
        """Adds an to self along an axis.

        The two arrays must have compatible axes.

        Parameters
        ----------
        axis : axis
            Axis along which to extend with input array (`other`)
        other : LArray
            Array with compatible axes

        Returns
        -------
        LArray
            Array expanded with 'other' along 'axis'.

        Examples
        --------
        >>> nat = Axis('nat', ['BE', 'FO'])
        >>> sex = Axis('sex', ['M', 'F'])
        >>> sex2 = Axis('sex', ['U'])
        >>> xtype = Axis('type', ['type1', 'type2'])
        >>> arr1 = ones([sex, xtype])
        >>> arr1
        sex\\type | type1 | type2
               M |   1.0 |   1.0
               F |   1.0 |   1.0
        >>> arr2 = zeros([sex2, xtype])
        >>> arr2
        sex\\type | type1 | type2
               U |   0.0 |   0.0
        >>> arr1.extend(x.sex, arr2)
        sex\\type | type1 | type2
               M |   1.0 |   1.0
               F |   1.0 |   1.0
               U |   0.0 |   0.0
        >>> arr3 = zeros([sex2, nat])
        >>> arr3
        sex\\nat |  BE |  FO
              U | 0.0 | 0.0
        >>> arr1.extend(x.sex, arr3)
        sex | type\\nat |  BE |  FO
          M |    type1 | 1.0 | 1.0
          M |    type2 | 1.0 | 1.0
          F |    type1 | 1.0 | 1.0
          F |    type2 | 1.0 | 1.0
          U |    type1 | 0.0 | 0.0
          U |    type2 | 0.0 | 0.0
        """
        result, (self_target, other_target) = \
            concat_empty(axis, (self.axes, other.axes), self.dtype)
        self_target[:] = self
        other_target[:] = other
        return result

    def transpose(self, *args):
        """Reorder axes.

        Parameters
        ----------
        args
            Accepts either a tuple of axes specs or
            axes specs as `*args`.

        Returns
        -------
        LArray
            LArray with reordered axes.

        Examples
        --------
        >>> a = ndrange([('nat', 'BE,FO'),
        ...              ('sex', 'M,F'),
        ...              ('alive', [False, True])])
        >>> a
        nat | sex\\alive | False | True
         BE |         M |     0 |    1
         BE |         F |     2 |    3
         FO |         M |     4 |    5
         FO |         F |     6 |    7
        >>> a.transpose(x.alive, x.sex, x.nat)
        alive | sex\\nat | BE | FO
        False |       M |  0 |  4
        False |       F |  2 |  6
         True |       M |  1 |  5
         True |       F |  3 |  7
        >>> a.transpose(x.alive)
        alive | nat\\sex | M | F
        False |      BE | 0 | 2
        False |      FO | 4 | 6
         True |      BE | 1 | 3
         True |      FO | 5 | 7
        """
        if len(args) == 1 and isinstance(args[0],
                                         (tuple, list, AxisCollection)):
            axes = args[0]
        elif len(args) == 0:
            axes = self.axes[::-1]
        else:
            axes = args

        axes = self.axes[axes]
        axes_indices = [self.axes.index(axis) for axis in axes]
        indices_present = set(axes_indices)
        missing_indices = [i for i in range(len(self.axes))
                           if i not in indices_present]
        axes_indices = axes_indices + missing_indices
        src_data = np.asarray(self)
        res_data = src_data.transpose(axes_indices)
        return LArray(res_data, self.axes[axes_indices])

        # if len(args) == 1 and isinstance(args[0],
        #                                  (tuple, list, AxisCollection)):
        #     axes = args[0]
        # else:
        #     axes = args
        # # this SHOULD work but does not currently for positional axes
        # # on anonymous axes. e.g. axes = (1, 2) because that ends up
        # # trying to do:
        # # self.axes[(1, 2)] | self.axes
        # # self.axes[(1, 2)] | self.axes[0]
        # # since self.axes[0] does not exist in self.axes[1, 2], BUT has
        # # a position < len(self.axes[1, 2]), it tries to match
        # # against self.axes[1, 2][0], (ie self.axes[1]) which breaks
        # # the problem is that AxisCollection.union should not try to match
        # # by position in this case.
        # res_axes = (self.axes[axes] | self.axes) if axes else self.axes[::-1]
        # axes_indices = [self.axes.index(axis) for axis in res_axes]
        # res_data = np.asarray(self).transpose(axes_indices)
        # return LArray(res_data, res_axes)
    T = property(transpose)

    def clip(self, a_min, a_max, out=None):
        """Clip (limit) the values in an array.

        Given an interval, values outside the interval are clipped to the interval edges.
        For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
        and values larger than 1 become 1.

        Parameters
        ----------
        a_min : scalar or array-like
            Minimum value.
        a_max : scalar or array-like
            Maximum value.
        out : LArray, optional
            The results will be placed in this array.

        Returns
        -------
        LArray
            An array with the elements of the current array, but where
            values < `a_min` are replaced with `a_min`, and those > `a_max`
            with `a_max`.

        Notes
        -----
        If `a_min` and/or `a_max` are array_like, broadcast will occur between
        self, `a_min` and `a_max`.
        """
        from larray.ufuncs import clip
        return clip(self, a_min, a_max, out)

    def to_csv(self, filepath, sep=',', na_rep='', transpose=True,
               dropna=None, dialect='default', **kwargs):
        """
        Writes array to a csv file.

        Parameters
        ----------
        filepath : str
            path where the csv file has to be written.
        sep : str
            seperator for the csv file.
        na_rep : str
            replace NA values with na_rep.
        transpose : boolean
            transpose = True  => transpose over last axis.
            transpose = False => no transpose.
        dialect : 'default' | 'classic'
            Whether or not to write the last axis name (using '\' )
        dropna : None, 'all', 'any' or True, optional
            Drop lines if 'all' its values are NA, if 'any' value is NA or do
            not drop any line (default). True is equivalent to 'all'.

        Examples
        --------
        >>> from .tests.test_la import abspath
        >>> fpath = abspath('test.csv')
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> a
        nat\\sex | M | F
             BE | 0 | 1
             FO | 2 | 3
        >>> a.to_csv(fpath)
        >>> with open(fpath) as f:
        ...     print(f.read().strip())
        nat\\sex,M,F
        BE,0,1
        FO,2,3
        >>> a.to_csv(fpath, sep=';', transpose=False)
        >>> with open(fpath) as f:
        ...     print(f.read().strip())
        nat;sex;0
        BE;M;0
        BE;F;1
        FO;M;2
        FO;F;3
        >>> a.to_csv(fpath, dialect='classic')
        >>> with open(fpath) as f:
        ...     print(f.read().strip())
        nat,M,F
        BE,0,1
        FO,2,3
        """
        fold = dialect == 'default'
        if transpose:
            frame = self.to_frame(fold, dropna)
            frame.to_csv(filepath, sep=sep, na_rep=na_rep, **kwargs)
        else:
            series = self.to_series(dropna is not None)
            series.to_csv(filepath, sep=sep, na_rep=na_rep, header=True,
                          **kwargs)

    def to_hdf(self, filepath, key, *args, **kwargs):
        """
        Writes array to a HDF file.

        A HDF file can contain multiple arrays.
        The 'key' parameter is a unique identifier for the array.

        Parameters
        ----------
        filepath : str
            Path where the hdf file has to be written.
        key : str
            Name of the array within the HDF file.
        *args
        **kargs

        Examples
        --------
        >>> from .tests.test_la import abspath
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> a.to_hdf(abspath('test.h5'), 'a')
        """
        self.to_frame().to_hdf(filepath, key, *args, **kwargs)

    def to_excel(self, filepath=None, sheet_name=None, position='A1',
                 overwrite_file=False, clear_sheet=False, header=True,
                 transpose=False, engine=None, *args, **kwargs):
        """
        Writes array in the specified sheet of specified excel workbook.

        Parameters
        ----------
        filepath : str or int or None, optional
            Path where the excel file has to be written. If None (default),
            creates a new Excel Workbook in a live Excel instance
            (Windows only). Use -1 to use the currently active Excel
            Workbook. Use a name without extension (.xlsx) to use any
            *unsaved* workbook.
        sheet_name : str or int or None, optional
            Sheet where the data has to be written. Defaults to None,
            Excel standard name if adding a sheet to an existing file,
            "Sheet1" otherwise. sheet_name can also refer to the position of
            the sheet (e.g. 0 for the first sheet, -1 for the last one).
        position : str or tuple of integers, optional
            Integer position (row, column) must be 1-based. Defaults to 'A1'.
        overwrite_file : bool, optional
            Whether or not to overwrite the existing file (or just modify the
            specified sheet). Defaults to False.
        clear_sheet : bool, optional
            Whether or not to clear the existing sheet (if any) before writing.
            Defaults to False.
        header : bool, optional
            Whether or not to write a header (axes names and labels).
            Defaults to True.
        transpose : bool, optional
            Whether or not to transpose the resulting array. This can be used,
            for example, for writing one dimensional arrays vertically.
            Defaults to False.
        engine : 'xlwings' | 'openpyxl' | 'xlsxwriter' | 'xlwt' | None, optional
            Engine to use to make the output. If None (default), it will use
            'xlwings' by default if the module is installed and relies on
            Pandas default writer otherwise.
        *args
        **kargs

        Examples
        --------
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> # write to a new (unnamed) sheet
        >>> a.to_excel('test.xlsx')  # doctest: +SKIP
        >>> # write to top-left corner of an existing sheet
        >>> a.to_excel('test.xlsx', 'Sheet1')  # doctest: +SKIP
        >>> # add to existing sheet starting at position A15
        >>> a.to_excel('test.xlsx', 'Sheet1', 'A15')  # doctest: +SKIP
        """
        df = self.to_frame(fold_last_axis_name=True)
        if engine is None:
            engine = 'xlwings' if xw is not None else None

        if engine == 'xlwings':
            from .excel import open_excel

            wb = open_excel(filepath, overwrite_file=overwrite_file)

            close = False
            new_workbook = False
            if filepath is None:
                new_workbook = True
            elif isinstance(filepath, str):
                basename, ext = os.path.splitext(filepath)
                if ext:
                    if not os.path.isfile(filepath):
                        new_workbook = True
                    close = True

            if new_workbook:
                sheet = wb.sheets[0]
                if sheet_name is not None:
                    sheet.name = sheet_name
            elif sheet_name is not None and sheet_name in wb:
                sheet = wb.sheets[sheet_name]
                if clear_sheet:
                    sheet.clear()
            else:
                sheet = wb.sheets.add(sheet_name, after=wb.sheets[-1])

            options = dict(header=header, index=header, transpose=transpose)
            sheet[position].options(**options).value = df
            # TODO: implement transpose via/in dump
            # sheet[position] = self.dump(header=header, transpose=transpose)
            if close:
                wb.save()
                wb.close()
        else:
            if sheet_name is None:
                sheet_name = 'Sheet1'
            # TODO: implement position in this case
            # startrow, startcol
            df.to_excel(filepath, sheet_name, *args, engine=engine, **kwargs)

    def to_clipboard(self, *args, **kwargs):
        """Sends the content of the array to clipboard.

        Using to_clipboard() makes it possible to paste the content
        of the array into a file (Excel, ascii file,...).

        Examples
        --------
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> a.to_clipboard()  # doctest: +SKIP
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

    @property
    def plot(self):
        """Plots the data of the array into a graph (window pop-up).

        The graph can be tweaked to achieve the desired formatting and can be
        saved to a .png file.

        Examples
        --------
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> a.plot()  # doctest: +SKIP
        """
        combined = self.combine_axes(self.axes[:-1], sep=' ') if self.ndim > 2 else self
        if combined.ndim == 1:
            return combined.to_series().plot
        else:
            return combined.transpose().to_frame().plot

    @property
    def shape(self):
        """Returns the shape of the array as a tuple.

        Returns
        -------
        tuple
            Tuple representing the current shape.

        Examples
        --------
        >>> a = ndrange('nat=BE,FO;sex=M,F;type=type1,type2,type3')
        >>> a.shape  # doctest: +SKIP
        (2, 2, 3)
        """
        return self.data.shape

    @property
    def ndim(self):
        """Returns the number of dimensions of the array.

        Returns
        -------
        int
            Number of dimensions of a LArray.

        Examples
        --------
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> a.ndim
        2
        """
        return self.data.ndim

    @property
    def size(self):
        """Returns the number of elements in array.

        Returns
        -------
        int
            Number of elements in array.

        Examples
        --------
        >>> a = ndrange('sex=M,F;type=type1,type2,type3')
        >>> a.size
        6
        """
        return self.data.size

    @property
    def nbytes(self):
        """Returns the number of bytes used to store the array in memory.

        Returns
        -------
        int
            Number of bytes in array.

        Examples
        --------
        >>> a = ndrange('sex=M,F;type=type1,type2,type3', dtype=float)
        >>> a.nbytes
        48
        """
        return self.data.nbytes

    @property
    def memory_used(self):
        """Returns the memory consumed by the array in human readable form.

        Returns
        -------
        str
            Memory used by the array.

        Examples
        --------
        >>> a = ndrange('sex=M,F;type=type1,type2,type3', dtype=float)
        >>> a.memory_used
        '48 bytes'
        """
        return size2str(self.data.nbytes)

    @property
    def dtype(self):
        """Returns the type of the data of the array.

        Returns
        -------
        dtype
            Type of the data of the array.

        Examples
        --------
        >>> a = zeros('sex=M,F;type=type1,type2,type3')
        >>> a.dtype
        dtype('float64')
        """
        return self.data.dtype

    @property
    def item(self):
        return self.data.item

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    __array_priority__ = 100

    def set_labels(self, axis, labels, inplace=False):
        """Replaces the labels of an axis of array.

        Parameters
        ----------
        axis : string or Axis
            Axis for which we want to replace the labels.
        labels : int or iterable
            Integer or list of values usable as the collection of labels for an Axis.
        inplace : bool
            Whether or not to modify the original object or return a new array and leave the original intact.

        Returns
        -------
        LArray
            Array with modified labels.

        Examples
        --------
        >>> a = ndrange('nat=BE,FO;sex=M,F')
        >>> a
        nat\\sex | M | F
             BE | 0 | 1
             FO | 2 | 3
        >>> a.set_labels(x.sex, 'Men,Women')
        nat\\sex | Men | Women
             BE |   0 |     1
             FO |   2 |     3
        >>> a.set_labels(x.sex, ['Men', 'Women'])
        nat\\sex | Men | Women
             BE |   0 |     1
             FO |   2 |     3
        """
        axis = self.axes[axis]
        if inplace:
            axis.labels = labels
            return self
        else:
            return LArray(self.data, self.axes.replace(axis, Axis(axis.name, labels)))

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return LArray(self.data.astype(dtype, order, casting, subok, copy),
                      self.axes)
    astype.__doc__ = np.ndarray.astype.__doc__

    def shift(self, axis, n=1):
        """Shifts the cells of the array n-times to the left along axis.

        Parameters
        ----------
        axis : int, str or Axis
            Axis for which we want to perform the shift.
        n : int
            Number of cells to shift.

        Returns
        -------
        LArray

        Examples
        --------
        >>> a = ndrange('sex=M,F;type=type1,type2,type3')
        >>> a
        sex\\type | type1 | type2 | type3
               M |     0 |     1 |     2
               F |     3 |     4 |     5
        >>> a.shift(x.type)
        sex\\type | type2 | type3
               M |     0 |     1
               F |     3 |     4
        >>> a.shift(x.type, n=-1)
        sex\\type | type1 | type2
               M |     1 |     2
               F |     4 |     5
        """
        axis = self.axes[axis]
        if n > 0:
            return self[axis.i[:-n]].set_labels(axis, axis.labels[n:])
        elif n < 0:
            return self[axis.i[-n:]].set_labels(axis, axis.labels[:n])
        else:
            return self[:]

    # TODO: add support for groups as axis (like aggregates)
    # eg a.diff(x.year[2018:]) instead of
    #    a[2018:].diff(x.year)
    def diff(self, axis=-1, d=1, n=1, label='upper'):
        """Calculates the n-th order discrete difference along a given axis.

        The first order difference is given by out[n] = a[n + 1] - a[n] along the
        given axis, higher order differences are calculated by using diff
        recursively.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis along which the difference is taken.
            Defaults to the last axis.

        d : int, optional
            Periods to shift for forming difference. Defaults to 1.

        n : int, optional
            The number of times values are differenced. Defaults to 1.

        label : {'lower', 'upper'}, optional
            The new labels in `axis` will have the labels of either
            the array being subtracted ('lower') or the array it is
            subtracted from ('upper'). Defaults to 'upper'.

        Returns
        -------
        LArray :
            The n-th order differences. The shape of the output is the same
            as `a` except for `axis` which is smaller by `n` * `d`.

        Examples
        --------
        >>> a = ndrange('sex=M,F;type=type1,type2,type3').cumsum(x.type)
        >>> a
        sex\\type | type1 | type2 | type3
               M |     0 |     1 |     3
               F |     3 |     7 |    12
        >>> a.diff()
        sex\\type | type2 | type3
               M |     1 |     2
               F |     4 |     5
        >>> a.diff(n=2)
        sex\\type | type3
               M |     1
               F |     1
        >>> a.diff(x.sex)
        sex\\type | type1 | type2 | type3
               F |     3 |     6 |     9
        """
        array = self
        for _ in range(n):
            axis_obj = array.axes[axis]
            left = array[axis_obj.i[d:]]
            right = array[axis_obj.i[:-d]]
            if label == 'upper':
                right = right.drop_labels(axis)
            else:
                left = left.drop_labels(axis)
            array = left - right
        return array

    # XXX: this is called pct_change in Pandas (but returns the same results,
    # not results * 100, which I find silly). Maybe change_rate would be
    # better (because growth is not always positive)?
    # TODO: add support for groups as axis (like aggregates)
    # eg a.growth_rate(x.year[2018:]) instead of
    #    a[2018:].growth_rate(x.year)
    def growth_rate(self, axis=-1, d=1, label='upper'):
        """Calculates the growth along a given axis.

        Roughly equivalent to a.diff(axis, d, label) / a[axis.i[:-d]]

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis along which the difference is taken.
            Defaults to the last axis.

        d : int, optional
            Periods to shift for forming difference. Defaults to 1.

        label : {'lower', 'upper'}, optional
            The new labels in `axis` will have the labels of either
            the array being subtracted ('lower') or the array it is
            subtracted from ('upper'). Defaults to 'upper'.

        Returns
        -------
        LArray

        Examples
        --------
        >>> sex = Axis('sex', ['M', 'F'])
        >>> year = Axis('year', range(2016, 2020))
        >>> a = LArray([[1.0, 2.0, 3.0, 3.0], [2.0, 3.0, 1.5, 3.0]],
        ...            [sex, year])
        >>> a
        sex\\year | 2016 | 2017 | 2018 | 2019
               M |  1.0 |  2.0 |  3.0 |  3.0
               F |  2.0 |  3.0 |  1.5 |  3.0
        >>> a.growth_rate()
        sex\\year | 2017 | 2018 | 2019
               M |  1.0 |  0.5 |  0.0
               F |  0.5 | -0.5 |  1.0
        >>> a.growth_rate(d=2)
        sex\\year |  2018 | 2019
               M |   2.0 |  0.5
               F | -0.25 |  0.0
        """
        diff = self.diff(axis=axis, d=d, label=label)
        axis_obj = self.axes[axis]
        return diff / self[axis_obj.i[:-d]].drop_labels(axis)

    def compact(self):
        """Detects and removes "useless" axes
        (ie axes for which values are constant over the whole axis)

        Returns
        -------
        LArray or scalar
            Array with constant axes removed.

        Examples
        --------
        >>> a = LArray([[1, 2],
        ...             [1, 2]], [Axis('sex', 'M,F'), Axis('nat', 'BE,FO')])
        >>> a
        sex\\nat | BE | FO
              M |  1 |  2
              F |  1 |  2
        >>> a.compact()
        nat | BE | FO
            |  1 |  2
        """
        res = self
        for axis in res.axes:
            if (res == res[axis.i[0]]).all():
                res = res[axis.i[0]]
        return res

    def combine_axes(self, axes=None, sep='_', wildcard=False):
        """Combine several axes into one.

        Parameters
        ----------
        axes : tuple, list or AxisCollection of axes, optional
            axes to combine. Defaults to all axes.
        sep : str, optional
            delimiter to use for combining. Defaults to '_'.
        wildcard : bool, optional
            whether or not to produce a wildcard axis even if the axes to
            combine are not. This is much faster, but loose axes labels.

        Returns
        -------
        LArray
            Array with combined axes.

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
        >>> arr.combine_axes()
        a_b | a0_b0 | a0_b1 | a0_b2 | a1_b0 | a1_b1 | a1_b2
            |     0 |     1 |     2 |     3 |     4 |     5
        >>> arr.combine_axes(sep='/')
        a/b | a0/b0 | a0/b1 | a0/b2 | a1/b0 | a1/b1 | a1/b2
            |     0 |     1 |     2 |     3 |     4 |     5
        >>> arr = ndtest((2, 3, 4))
        >>> arr
         a | b\\c | c0 | c1 | c2 | c3
        a0 |  b0 |  0 |  1 |  2 |  3
        a0 |  b1 |  4 |  5 |  6 |  7
        a0 |  b2 |  8 |  9 | 10 | 11
        a1 |  b0 | 12 | 13 | 14 | 15
        a1 |  b1 | 16 | 17 | 18 | 19
        a1 |  b2 | 20 | 21 | 22 | 23
        >>> arr.combine_axes((x.a, x.c))
        a_c\\b | b0 | b1 | b2
        a0_c0 |  0 |  4 |  8
        a0_c1 |  1 |  5 |  9
        a0_c2 |  2 |  6 | 10
        a0_c3 |  3 |  7 | 11
        a1_c0 | 12 | 16 | 20
        a1_c1 | 13 | 17 | 21
        a1_c2 | 14 | 18 | 22
        a1_c3 | 15 | 19 | 23
        """
        axes = self.axes if axes is None else self.axes[axes]
        # transpose all axes next to each other, using position of first axis
        axes_indices = [self.axes.index(axis) for axis in axes]
        min_axis_index = min(axes_indices)
        transposed_axes = self.axes[:min_axis_index] + axes + self.axes
        transposed = self.transpose(transposed_axes)

        new_axes = transposed.axes.combine_axes(axes, sep=sep, wildcard=wildcard)
        return transposed.reshape(new_axes)

    def split_axis(self, axis, sep='_', names=None, regex=None):
        """Split one axis and returns a new array

        Parameters
        ----------
        axis : int, str or Axis
            axis to split. All its labels *must* contain
            the given delimiter string.
        sep : str, optional
            delimiter to use for splitting. Defaults to '_'.
            When `regex` is provided, the delimiter is only used
            on `names` if given as one string or on axis name if
            `names` is None.
        names : str or list of str, optional
            names of resulting axes. Defaults to None.
        regex : str, optional
            use regex instead of delimiter to split labels.
            Defaults to None.

        Returns
        -------
        LArray

        Examples
        --------
        >>> arr = ndtest((2, 3))
        >>> arr
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
        >>> combined = arr.combine_axes()
        >>> combined
        a_b | a0_b0 | a0_b1 | a0_b2 | a1_b0 | a1_b1 | a1_b2
            |     0 |     1 |     2 |     3 |     4 |     5
        >>> combined.split_axis(x.a_b)
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5

        Split labels using regex

        >>> combined = ndrange('a_b = a0b0..a1b2')
        >>> combined
        a_b | a0b0 | a0b1 | a0b2 | a1b0 | a1b1 | a1b2
            |    0 |    1 |    2 |    3 |    4 |    5
        >>> combined.split_axis(x.a_b, regex='(\w{2})(\w{2})')
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |  5
        """
        return self.reshape(self.axes.split_axis(axis, sep, names, regex))


def parse(s):
    """
    Used to parse the "folded" axis ticks (usually periods).
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
    Returns unique labels for each dimension.
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


def df_aslarray(df, sort_rows=False, sort_columns=False, raw=False, parse_header=True, **kwargs):
    # the dataframe was read without index at all (ie 2D dataframe), irrespective of the actual data dimensionality
    if raw:
        columns = df.columns.values.tolist()
        try:
            # take the first column which contains '\'
            # pos_last = next(i for i, v in enumerate(columns) if '\\' in str(v))
            pos_last = next(i for i, v in enumerate(columns) if isinstance(v, basestring) and '\\' in v)
            onedim = False
        except StopIteration:
            # we assume first column will not contain data
            pos_last = 0
            onedim = True

        axes_names = columns[:pos_last + 1]
        if onedim:
            df = df.iloc[:, 1:]
        else:
            # This is required to handle int column names (otherwise we can simply use column positions in set_index).
            # This is NOT the same as df.columns[list(range(...))] !
            index_columns = [df.columns[i] for i in range(pos_last + 1)]
            # TODO: we should pass a flag to df_aslarray so that we can use inplace=True here
            # df.set_index(index_columns, inplace=True)
            df = df.set_index(index_columns)
    else:
        axes_names = [decode(name, 'utf8') for name in df.index.names]

    # handle 2 or more dimensions with the last axis name given using \
    if isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]:
        last_axes = [name.strip() for name in axes_names[-1].split('\\')]
        axes_names = axes_names[:-1] + last_axes
    # handle 1D
    elif len(df) == 1 and axes_names == [None]:
        axes_names = [df.columns.name]
    # handle 2 or more dimensions with the last axis name given as the columns index name
    elif len(df) > 1:
        axes_names += [df.columns.name]

    if len(axes_names) > 1:
        df, axes_labels = cartesian_product_df(df, sort_rows=sort_rows, sort_columns=sort_columns, **kwargs)
    else:
        axes_labels = []

    # we could inline df_aslarray into the functions that use it, so that the
    # original (non-cartesian) df is freed from memory at this point, but it
    # would be much uglier and would not lower the peak memory usage which
    # happens during cartesian_product_df.reindex

    # Pandas treats column labels as column names (strings) so we need to convert them to values
    last_axis_labels = [parse(cell) for cell in df.columns.values] if parse_header else list(df.columns.values)
    axes_labels.append(last_axis_labels)
    axes_names = [str(name) if name is not None else name
                  for name in axes_names]

    axes = [Axis(name, labels) for name, labels in zip(axes_names, axes_labels)]
    data = df.values.reshape([len(axis) for axis in axes])
    return LArray(data, axes)


def from_lists(data, nb_index=None, index_col=None):
    """
    initialize array from a list of lists (lines)

    Parameters
    ----------
    data : sequence (tuple, list, ...)
        Input data. All data is supposed to already have the correct type (e.g. strings are not parsed).
    nb_index : int, optional
        Number of leading index columns (ex. 4). Defaults to None, in which case it guesses the number of index columns
        by using the position of the first '\' in the first line.
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]). Defaults to None (see nb_index above).

    Returns
    -------
    LArray

    Examples
    --------
    >>> from_lists([['sex', 'M', 'F'],
    ...             ['',      0,   1]])
    sex | M | F
        | 0 | 1
    >>> from_lists([['sex\\year', 1991, 1992, 1993],
    ...             [ 'M',           0,    1,    2],
    ...             [ 'F',           3,    4,    5]])
    sex\\year | 1991 | 1992 | 1993
           M |    0 |    1 |    2
           F |    3 |    4 |    5
    >>> from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
    ...             [  'M', 'BE',           1,    0,    0],
    ...             [  'M', 'FO',           2,    0,    0],
    ...             [  'F', 'BE',           0,    0,    1]])
    sex | nat\\year | 1991 | 1992 | 1993
      M |       BE |  1.0 |  0.0 |  0.0
      M |       FO |  2.0 |  0.0 |  0.0
      F |       BE |  0.0 |  0.0 |  1.0
      F |       FO |  nan |  nan |  nan
    >>> from_lists([['sex', 'nat', 1991, 1992, 1993],
    ...             [  'M', 'BE',     1,    0,    0],
    ...             [  'M', 'FO',     2,    0,    0],
    ...             [  'F', 'BE',     0,    0,    1]], nb_index=2)
    sex | nat\\{2} | 1991 | 1992 | 1993
      M |      BE |  1.0 |  0.0 |  0.0
      M |      FO |  2.0 |  0.0 |  0.0
      F |      BE |  0.0 |  0.0 |  1.0
      F |      FO |  nan |  nan |  nan
    """
    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    df = pd.DataFrame(data[1:], columns=data[0])
    if index_col is not None:
        df.set_index([df.columns[c] for c in index_col], inplace=True)

    return df_aslarray(df, raw=index_col is None, parse_header=False)


def from_string(s, nb_index=None, index_col=None, sep=',', **kwargs):
    """Create an array from a multi-line string.

    Parameters
    ----------
    s : str
        input string.
    nb_index : int, optional
        Number of leading index columns (ex. 4). Defaults to None, in which case it guesses the number of index columns
        by using the position of the first '\' in the first line.
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]). Defaults to None (see nb_index above).
    sep : str
        delimiter used to split each line into cells.
    \**kwargs
        See arguments of Pandas read_csv function.

    Returns
    -------
    LArray

    Examples
    --------
    >>> from_string("sex,M,F\\n,0,1")
    sex | M | F
        | 0 | 1
    >>> from_string("nat\\sex,M,F\\nBE,0,1\\nFO,2,3")
    nat\sex | M | F
         BE | 0 | 1
         FO | 2 | 3

    Each label is stripped of leading and trailing whitespace, so this is valid too:

    >>> from_string('''nat\\sex, M, F
    ...                BE,       0, 1
    ...                FO,       2, 3''')
    nat\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> from_string('''age,nat\\sex, M, F
    ...                0,  BE,       0, 1
    ...                0,  FO,       2, 3
    ...                1,  BE,       4, 5
    ...                1,  FO,       6, 7''')
    age | nat\sex | M | F
      0 |      BE | 0 | 1
      0 |      FO | 2 | 3
      1 |      BE | 4 | 5
      1 |      FO | 6 | 7

    Empty lines at the beginning or end are ignored, so one can also format the string like this:

    >>> from_string('''
    ... nat\\sex, M, F
    ... BE,       0, 1
    ... FO,       2, 3
    ... ''')
    nat\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    """

    return read_csv(StringIO(s), nb_index=nb_index, index_col=index_col, sep=sep, skipinitialspace=True, **kwargs)


def read_csv(filepath_or_buffer, nb_index=None, index_col=None, sep=',', headersep=None, na=np.nan,
             sort_rows=False, sort_columns=False, dialect='larray', **kwargs):
    """
    Reads csv file and returns an array with the contents.

    Notes
    -----
    csv file format:
    arr,ages,sex,nat\time,1991,1992,1993
    A1,BI,H,BE,1,0,0
    A1,BI,H,FO,2,0,0
    A1,BI,F,BE,0,0,1
    A1,BI,F,FO,0,0,0
    A1,A0,H,BE,0,0,0

    Parameters
    ----------
    filepath_or_buffer : str or any file-like object
        Path where the csv file has to be read or a file handle.
    nb_index : int, optional
        Number of leading index columns (ex. 4).
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]).
    sep : str, optional
        Separator.
    headersep : str or None, optional
        Separator for headers.
    na : scalar, optional
        Value for NaN (Not A Number). Defaults to NumPy NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    dialect : 'classic' | 'larray' | 'liam2', optional
        Name of dialect. Defaults to 'larray'.
    **kwargs

    Returns
    -------
    LArray

    Examples
    --------
    >>> from .tests.test_la import abspath
    >>> fpath = abspath('test.csv')
    >>> a = ndrange('nat=BE,FO;sex=M,F')

    >>> a.to_csv(fpath)
    >>> read_csv(fpath)
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> read_csv(fpath, sort_columns=True)
    nat\\sex | F | M
         BE | 1 | 0
         FO | 3 | 2
    >>> fpath = abspath('no_axis_name.csv')
    >>> a.to_csv(fpath, dialect='classic')
    >>> read_csv(fpath, nb_index=1)
    nat\\{1} | M | F
         BE | 0 | 1
         FO | 2 | 3
    """
    if dialect == 'liam2':
        # read axes names. This needs to be done separately instead of reading the whole file with Pandas then
        # manipulating the dataframe because the header line must be ignored for the column types to be inferred
        # correctly. Note that to read one line, this is faster than using Pandas reader.
        with csv_open(filepath_or_buffer) as f:
            reader = csv.reader(f, delimiter=sep)
            line_stream = skip_comment_cells(strip_rows(reader))
            axes_names = next(line_stream)

        if nb_index is not None or index_col is not None:
            raise ValueError("nb_index and index_col are not compatible with dialect='liam2'")
        if len(axes_names) > 1:
            nb_index = len(axes_names) - 1
        # use the second data line for column headers (excludes comments and blank lines before counting)
        kwargs['header'] = 1
        kwargs['comment'] = '#'

    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    if headersep is not None:
        if index_col is None:
            index_col = [0]

    df = pd.read_csv(filepath_or_buffer, index_col=index_col, sep=sep, **kwargs)
    if dialect == 'liam2':
        if len(axes_names) > 1:
            df.index.names = axes_names[:-1]
        df.columns.name = axes_names[-1]
        raw = False
    else:
        raw = index_col is None

    if headersep is not None:
        combined_axes_names = df.index.name
        df.index = df.index.str.split(headersep, expand=True)
        df.index.names = combined_axes_names.split(headersep)
        raw = False

    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=na, raw=raw)


def read_tsv(filepath_or_buffer, **kwargs):
    return read_csv(filepath_or_buffer, sep='\t', **kwargs)


def read_eurostat(filepath_or_buffer, **kwargs):
    """Reads EUROSTAT TSV (tab-separated) file into an array.

    EUROSTAT TSV files are special because they use tabs as data separators but comas to separate headers.

    Parameters
    ----------
    filepath_or_buffer : str or any file-like object
        Path where the tsv file has to be read or a file handle.
    kwargs
        Arbitrary keyword arguments are passed through to read_csv.

    Returns
    -------
    LArray
    """
    return read_csv(filepath_or_buffer, sep='\t', headersep=',', **kwargs)


def read_hdf(filepath_or_buffer, key, na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """Reads an array named key from a HDF5 file in filepath (path+name)

    Parameters
    ----------
    filepath_or_buffer : str or pandas.HDFStore
        Path and name where the HDF5 file is stored or a HDFStore object.
    key : str
        Name of the array.

    Returns
    -------
    LArray
    """
    df = pd.read_hdf(filepath_or_buffer, key, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=na, parse_header=False)


def read_excel(filepath, sheetname=0, nb_index=None, index_col=None, na=np.nan, sort_rows=False, sort_columns=False,
               engine=None, **kwargs):
    """
    Reads excel file from sheet name and returns an LArray with the contents

    Parameters
    ----------
    filepath : str
        Path where the Excel file has to be read.
    sheetname : str or int, optional
        Name or index of the Excel sheet containing the array to be read.
        By default the array is read from the first sheet.
    nb_index : int, optional
        Number of leading index columns (ex. 4). Defaults to 1.
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]).
        Default to [0].
    na : scalar, optional
        Value for NaN (Not A Number). Defaults to NumPy NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    engine : {'xlrd', 'xlwings'}, optional
        Engine to use to read the Excel file. If None (default), it will use 'xlwings' by default if the module is
        installed and relies on Pandas default reader otherwise.
    **kwargs
    """
    if engine is None:
        engine = 'xlwings' if xw is not None else None

    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    if engine == 'xlwings':
        if kwargs:
            raise TypeError("'{}' is an invalid keyword argument for this function when using the xlwings backend"
                            .format(list(kwargs.keys())[0]))
        from .excel import open_excel
        with open_excel(filepath) as wb:
            return wb[sheetname].load(index_col=index_col)
    else:
        df = pd.read_excel(filepath, sheetname, index_col=index_col, engine=engine, **kwargs)
        return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, raw=index_col is None, fill_value=na)


def read_sas(filepath, nb_index=None, index_col=None, na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """
    Reads sas file and returns an LArray with the contents
        nb_index: number of leading index columns (e.g. 4)
    or
        index_col: list of columns for the index (e.g. [0, 1, 3])
    """
    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    df = pd.read_sas(filepath, index=index_col, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=na)


def zeros(axes, title='', dtype=float, order='C'):
    """Returns an array with the specified axes and filled with zeros.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.
        Default is `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> zeros([('nat', ['BE', 'FO']),
    ...        ('sex', ['M', 'F'])])
    nat\sex |   M |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0
    >>> zeros('nat=BE,FO;sex=M,F')
    nat\sex |   M |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> zeros([nat, sex])
    nat\sex |   M |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0
    """
    axes = AxisCollection(axes)
    return LArray(np.zeros(axes.shape, dtype, order), axes, title)


def zeros_like(array, title='', dtype=None, order='K'):
    """Returns an array with the same axes as array and filled with zeros.

    Parameters
    ----------
    array : LArray
         Input array.
    title : str, optional
        Title.
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

    Examples
    --------
    >>> a = ndrange((2, 3))
    >>> zeros_like(a)
    {0}*\\{1}* | 0 | 1 | 2
            0 | 0 | 0 | 0
            1 | 0 | 0 | 0
    """
    if not title:
        title = array.title
    return LArray(np.zeros_like(array, dtype, order), array.axes, title)


def ones(axes, title='', dtype=float, order='C'):
    """Returns an array with the specified axes and filled with ones.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> ones([nat, sex])
    nat\\sex |   M |   F
         BE | 1.0 | 1.0
         FO | 1.0 | 1.0
    """
    axes = AxisCollection(axes)
    return LArray(np.ones(axes.shape, dtype, order), axes, title)


def ones_like(array, title='', dtype=None, order='K'):
    """Returns an array with the same axes as array and filled with ones.

    Parameters
    ----------
    array : LArray
        Input array.
    title : str, optional
        Title.
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

    Examples
    --------
    >>> a = ndrange((2, 3))
    >>> ones_like(a)
    {0}*\\{1}* | 0 | 1 | 2
            0 | 1 | 1 | 1
            1 | 1 | 1 | 1
    """
    axes = array.axes
    if not title:
        title = array.title
    return LArray(np.ones_like(array, dtype, order), axes, title)


def empty(axes, title='', dtype=float, order='C'):
    """Returns an array with the specified axes and uninitialized (arbitrary) data.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> empty([nat, sex])  # doctest: +SKIP
    nat\\sex |                  M |                  F
         BE | 2.47311483356e-315 | 2.47498446195e-315
         FO |                0.0 | 6.07684618082e-31
    """
    axes = AxisCollection(axes)
    return LArray(np.empty(axes.shape, dtype, order), axes, title)


def empty_like(array, title='', dtype=None, order='K'):
    """Returns an array with the same axes as array and uninitialized (arbitrary) data.

    Parameters
    ----------
    array : LArray
        Input array.
    title : str, optional
        Title.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the data type of array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' (default) means match the layout of `a` as closely
        as possible.

    Returns
    -------
    LArray

    Examples
    --------
    >>> a = ndrange((3, 2))
    >>> empty_like(a)   # doctest: +SKIP
    -\- |                  0 |                  1
      0 | 2.12199579097e-314 | 6.36598737388e-314
      1 | 1.06099789568e-313 | 1.48539705397e-313
      2 | 1.90979621226e-313 | 2.33419537056e-313
    """
    if not title:
        title = array.title
    # cannot use empty() because order == 'K' is not understood
    return LArray(np.empty_like(array.data, dtype, order), array.axes, title)


def full(axes, fill_value, title='', dtype=None, order='C'):
    """Returns an array with the specified axes and filled with fill_value.

    Parameters
    ----------
    axes : int, tuple of int or tuple/list/AxisCollection of Axis
        Collection of axes or a shape.
    fill_value : scalar or LArray
        Value to fill the array
    title : str, optional
        Title.
    dtype : data-type, optional
        Desired data-type for the array. Default is the data type of fill_value.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- (default) or
        Fortran-contiguous (row- or column-wise) order in memory.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> full([nat, sex], 42.0)
    nat\\sex |    M |    F
         BE | 42.0 | 42.0
         FO | 42.0 | 42.0
    >>> initial_value = ndrange([sex])
    >>> initial_value
    sex | M | F
        | 0 | 1
    >>> full([nat, sex], initial_value)
    nat\\sex | M | F
         BE | 0 | 1
         FO | 0 | 1
    """
    if dtype is None:
        dtype = np.asarray(fill_value).dtype
    res = empty(axes, title, dtype, order)
    res[:] = fill_value
    return res


def full_like(array, fill_value, title='', dtype=None, order='K'):
    """Returns an array with the same axes and type as input array and filled with fill_value.

    Parameters
    ----------
    array : LArray
        Input array.
    fill_value : scalar or LArray
        Value to fill the array
    title : str, optional
        Title.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the data type of array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' (default) means match the layout of `a` as closely
        as possible.

    Returns
    -------
    LArray

    Examples
    --------
    >>> a = ndrange((2, 3))
    >>> full_like(a, 5)
    {0}*\\{1}* | 0 | 1 | 2
            0 | 5 | 5 | 5
            1 | 5 | 5 | 5
    """
    if not title:
        title = array.title
    # cannot use full() because order == 'K' is not understood
    # cannot use np.full_like() because it would not handle LArray fill_value
    res = empty_like(array, title, dtype, order)
    res[:] = fill_value
    return res


# XXX: would it be possible to generalize to multiple axes and deprecate
#      ndrange (or rename this one to ndrange)? ndrange is only ever used to
#      create test data (except for 1d)
# see https://github.com/pydata/pandas/issues/4567
def create_sequential(axis, initial=0, inc=None, mult=1, func=None, axes=None, title=''):
    """
    Creates an array by sequentially applying modifications to the array along
    axis.

    The value for each label in axis will be given by sequentially transforming
    the value for the previous label. This transformation on the previous label
    value consists of applying the function "func" on that value if provided, or
    to multiply it by mult and increment it by inc otherwise.

    Parameters
    ----------
    axis : axis reference (Axis, str, int)
        Axis along which to apply mod.
    initial : scalar or LArray, optional
        Value for the first label of axis. Defaults to 0.
    inc : scalar, LArray, optional
        Value to increment the previous value by. Defaults to 0 if mult is
        provided, 1 otherwise.
    mult : scalar, LArray, optional
        Value to multiply the previous value by. Defaults to 1.
    func : function/callable, optional
        Function to apply to the previous value. Defaults to None. Note that
        this is much slower than using inc and/or mult.
    axes : int, tuple of int or tuple/list/AxisCollection of Axis, optional
        Axes of the result. Defaults to the union of axes present in other
        arguments.
    title : str, optional
        Title.

    Examples
    --------
    >>> year = Axis('year', range(2016, 2020))
    >>> sex = Axis('sex', ['M', 'F'])
    >>> create_sequential(year)
    year | 2016 | 2017 | 2018 | 2019
         |    0 |    1 |    2 |    3
    >>> create_sequential(year, 1.0, 0.5)
    year | 2016 | 2017 | 2018 | 2019
         |  1.0 |  1.5 |  2.0 |  2.5
    >>> create_sequential(year, 1.0, mult=1.5)
    year | 2016 | 2017 | 2018 |  2019
         |  1.0 |  1.5 | 2.25 | 3.375
    >>> inc = LArray([1, 2], [sex])
    >>> inc
    sex | M | F
        | 1 | 2
    >>> create_sequential(year, 1.0, inc)
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  3.0 |  4.0
           F |  1.0 |  3.0 |  5.0 |  7.0
    >>> mult = LArray([2, 3], [sex])
    >>> mult
    sex | M | F
        | 2 | 3
    >>> create_sequential(year, 1.0, mult=mult)
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  4.0 |  8.0
           F |  1.0 |  3.0 |  9.0 | 27.0
    >>> initial = LArray([3, 4], [sex])
    >>> initial
    sex | M | F
        | 3 | 4
    >>> create_sequential(year, initial, inc, mult)
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |    3 |    7 |   15 |   31
           F |    4 |   14 |   44 |  134
    >>> def modify(prev_value):
    ...     return prev_value / 2
    >>> create_sequential(year, 8, func=modify)
    year | 2016 | 2017 | 2018 | 2019
         |    8 |    4 |    2 |    1
    >>> create_sequential(3)
    {0}* | 0 | 1 | 2
         | 0 | 1 | 2
    >>> create_sequential(x.year, axes=(sex, year))
    sex\\year | 2016 | 2017 | 2018 | 2019
           M |    0 |    1 |    2 |    3
           F |    0 |    1 |    2 |    3

    create_sequential can be used as the inverse of growth_rate:

    >>> a = LArray([1.0, 2.0, 3.0, 3.0], year)
    >>> a
    year | 2016 | 2017 | 2018 | 2019
         |  1.0 |  2.0 |  3.0 |  3.0
    >>> g = a.growth_rate() + 1
    >>> g
    year | 2017 | 2018 | 2019
         |  2.0 |  1.5 |  1.0
    >>> create_sequential(a.axes.year, a[2016], mult=g)
    year | 2016 | 2017 | 2018 | 2019
         |  1.0 |  2.0 |  3.0 |  3.0
    """
    if inc is None:
        inc = 1 if mult is 1 else 0
    if isinstance(axis, int):
        axis = Axis(None, axis)
    elif isinstance(axis, Group):
        axis = Axis(axis.axis.name, list(axis))
    if axes is None:
        def strip_axes(col):
            return get_axes(col) - axis
        # we need to remove axis if present, because it might be incompatible
        axes = strip_axes(initial) | strip_axes(inc) | strip_axes(mult) | axis
    else:
        axes = AxisCollection(axes)
    axis = axes[axis]
    res_dtype = np.dtype(common_type((initial, inc, mult)))
    res = empty(axes, title=title, dtype=res_dtype)
    res[axis.i[0]] = initial
    def has_axis(a, axis):
        return isinstance(a, LArray) and axis in a.axes
    if func is not None:
        for i in range(1, len(axis)):
            res[axis.i[i]] = func(res[axis.i[i - 1]])
    elif has_axis(inc, axis) and has_axis(mult, axis):
        # This case is more complicated to vectorize. It seems
        # doable (probably by adding a fictive axis), but let us wait until
        # someone requests it. The trick is to be able to write this:
        # a[i] = initial * prod(mult[j]) + inc[1] * prod(mult[j]) + ...
        #                 j=1..i                    j=2..i
        #      + inc[i-2] * prod(mult[j]) + inc[i-1] * mult[i] + inc[i]
        #                 j=i-1..i

        # a[0] = initial
        # a[1] = initial * mult[1]
        #      + inc[1]
        # a[2] = initial * mult[1] * mult[2]
        #      + inc[1] * mult[2]
        #      + inc[2]
        # a[3] = initial * mult[1] * mult[2] * mult[3]
        #      + inc[1] * mult[2] * mult[3]
        #      + inc[2]           * mult[3]
        #      + inc[3]
        # a[4] = initial * mult[1] * mult[2] * mult[3] * mult[4]
        #      + inc[1] * mult[2] * mult[3] * mult[4]
        #      + inc[2]           * mult[3] * mult[4]
        #      + inc[3]                     * mult[4]
        #      + inc[4]

        # a[1:] = initial * cumprod(mult[1:]) + ...
        def index_if_exists(a, axis, i):
            if isinstance(a, LArray) and axis in a.axes:
                a_axis = a.axes[axis]
                return a[a_axis[axis.labels[i]]]
            else:
                return a
        for i in range(1, len(axis)):
            i_mult = index_if_exists(mult, axis, i)
            i_inc = index_if_exists(inc, axis, i)
            res[axis.i[i]] = res[axis.i[i - 1]] * i_mult + i_inc
    else:
        # TODO: use cumprod and cumsum to avoid the explicit loop
        # it is easy for constant inc OR constant mult.
        # it is easy for array inc OR array mult.
        # it is a bit more complicated for constant inc AND constant mult
        #
        # it gets hairy for array inc AND array mult. It seems doable but let us
        #    wait until someone requests it.
        def array_or_full(a, axis, initial):
            dt = common_type((a, initial))
            r = empty((get_axes(a) - axis) | axis, title=title, dtype=dt)
            r[axis.i[0]] = initial
            if isinstance(a, LArray) and axis in a.axes:
                # not using axis.i[1:] because a could have less ticks
                # on axis than axis
                r[axis.i[1:]] = a[axis[axis.labels[1]:]]
            else:
                r[axis.i[1:]] = a
            return r

        # inc only (integer scalar)
        if np.isscalar(mult) and mult == 1 and np.isscalar(inc) and \
                res_dtype.kind == 'i':
            # stop is not included
            stop = initial + inc * len(axis)
            data = np.arange(initial, stop, inc)
            res[:] = LArray(data, axis)
        # inc only (other scalar)
        elif np.isscalar(mult) and mult == 1 and np.isscalar(inc):
            # stop is included
            stop = initial + inc * (len(axis) - 1)
            data = np.linspace(initial, stop=stop, num=len(axis))
            res[:] = LArray(data, axis)
        # inc only (array)
        elif np.isscalar(mult) and mult == 1:
            inc_array = array_or_full(inc, axis, initial)
            res[axis.i[1:]] = inc_array.cumsum(axis)[axis.i[1:]]
        # mult only (scalar or array)
        elif np.isscalar(inc) and inc == 0:
            mult_array = array_or_full(mult, axis, initial)
            res[axis.i[1:]] = mult_array.cumprod(axis)[axis.i[1:]]
        # both inc and mult defined but scalars or axis not present
        else:
            mult_array = array_or_full(mult, axis, 1.0)
            cum_mult = mult_array.cumprod(axis)[axis.i[1:]]
            res[axis.i[1:]] = \
                ((1 - cum_mult) / (1 - mult)) * inc + initial * cum_mult
    return res


def ndrange(axes, start=0, title='', dtype=int):
    """Returns an array with the specified axes and filled with increasing int.

    Parameters
    ----------
    axes : single axis or tuple/list/AxisCollection of axes
        Axes of the array to create.
        Each axis can be given as either:

        * Axis object: actual axis object to use.
        * single int: length of axis. will create a wildcard axis of that
                      length.
        * str: coma separated list of labels, with optional leading '=' to
               set the name of the axis. eg. "a,b,c" or "sex=F,M"
        * (name, labels) pair: name and labels of axis
    start : number, optional
    title : str, optional
        Title.
    dtype : dtype, optional
        The type of the output array.  Defaults to int.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> ndrange([nat, sex])
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> ndrange([('nat', ['BE', 'FO']),
    ...          ('sex', ['M', 'F'])])
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> ndrange([('nat', 'BE,FO'),
    ...          ('sex', 'M,F')])
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> ndrange(['nat=BE,FO', 'sex=M,F'])
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> ndrange('nat=BE,FO;sex=M,F')
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> ndrange([2, 3], dtype=float)
    {0}*\\{1}* |   0 |   1 |   2
            0 | 0.0 | 1.0 | 2.0
            1 | 3.0 | 4.0 | 5.0
    >>> ndrange(3, start=2)
    {0}* | 0 | 1 | 2
         | 2 | 3 | 4
    >>> ndrange('a,b,c')
    {0} | a | b | c
        | 0 | 1 | 2
    """
    # XXX: implement something like:
    # >>> mat = ndrange([['BE', 'FO'], ['M', 'F']], axes=['nat', 'sex'])
    # >>> mat = ndrange(['BE,FO', 'M,F'], axes=['nat', 'sex'])
    # XXX: try to come up with a syntax where start is before "end". For ndim
    #  > 1, I cannot think of anything nice.
    axes = AxisCollection(axes)
    data = np.arange(start, start + axes.size, dtype=dtype)
    return LArray(data.reshape(axes.shape), axes, title)


def ndtest(shape, start=0, label_start=0, title='', dtype=int):
    """Returns test array with given shape.

    Axes are named by single letters starting from 'a'. Axes labels are
    constructed using a '{axis_name}{label_pos}' pattern (e.g. 'a0'). Values
    start from `start` increase by steps of 1.

    Parameters
    ----------
    shape : int, tuple or list
        Shape of the array to create. An int can be used directly for one
        dimensional arrays.
    start : int or float, optional
        Start value
    label_start : int, optional
        Label position for each axis is `label_start + position`.
        `label_start` defaults to 0.
    title : str, optional
        Title.
    dtype : type or np.dtype, optional
        Type of resulting array.

    Returns
    -------
    LArray

    Examples
    --------
    >>> ndtest(6)
    a | a0 | a1 | a2 | a3 | a4 | a5
      |  0 |  1 |  2 |  3 |  4 |  5
    >>> ndtest((2, 3))
    a\\b | b0 | b1 | b2
     a0 |  0 |  1 |  2
     a1 |  3 |  4 |  5
    >>> ndtest((2, 3), label_start=1)
    a\\b | b1 | b2 | b3
     a1 |  0 |  1 |  2
     a2 |  3 |  4 |  5
    """
    a = ndrange(shape, start=start, dtype=dtype, title=title)
    # TODO: move this to a class method on AxisCollection
    assert a.ndim <= 26
    axes_names = [chr(ord('a') + i) for i in range(a.ndim)]
    label_ranges = [range(label_start, label_start + length)
                    for length in a.shape]
    new_axes = [Axis(name, [name + str(i) for i in label_range])
                for name, label_range in zip(axes_names, label_ranges)]
    return LArray(a.data, new_axes)


def kth_diag_indices(shape, k):
    indices = np.diag_indices(min(shape), ndim=len(shape))
    if len(shape) == 2 and k != 0:
        rows, cols = indices
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
    elif k != 0:
        raise NotImplementedError("k != 0 and len(axes) != 2")
    else:
        return indices


def diag(a, k=0, axes=(0, 1), ndim=2, split=True):
    """
    Extracts a diagonal or construct a diagonal array.

    Parameters
    ----------
    a : LArray
        If `a` has 2 dimensions or more, return a copy of its `k`-th diagonal.
        If `a` has 1 dimension, return an array with `ndim` dimensions on the
        `k`-th diagonal.
    k : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).
    axes : tuple or list or AxisCollection of axes references, optional
        Axes along which the diagonals should be taken.  Use None for all axes.
        Defaults to the first two axes (0, 1).
    ndim : int, optional
        Target number of dimensions when constructing a diagonal array from
        an array without axes names/labels. Defaults to 2.
    split : bool, optional
        Whether or not to try to split the axis name and labels

    Returns
    -------
    LArray
        The extracted diagonal or constructed diagonal array.

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> a = ndrange([nat, sex], start=1)
    >>> a
    nat\\sex | M | F
         BE | 1 | 2
         FO | 3 | 4
    >>> d = diag(a)
    >>> d
    nat,sex | BE,M | FO,F
            |    1 |    4
    >>> diag(d)
    nat\\sex | M | F
         BE | 1 | 0
         FO | 0 | 4
    >>> a = ndrange(sex, start=1)
    >>> a
    sex | M | F
        | 1 | 2
    >>> diag(a)
    sex\\sex | M | F
          M | 1 | 0
          F | 0 | 2
    """
    if a.ndim == 1:
        axis = a.axes[0]
        axis_name = axis.name
        if k != 0:
            raise NotImplementedError("k != 0 not supported for 1D arrays")
        if split and isinstance(axis_name, str) and ',' in axis_name:
            axes_names = axis_name.split(',')
            axes_labels = list(zip(*np.char.split(axis.labels, ',')))
            axes = [Axis(name, labels)
                    for name, labels in zip(axes_names, axes_labels)]
        else:
            axes = [axis] + [axis.copy() for _ in range(ndim - 1)]
        res = zeros(axes, dtype=a.dtype)
        diag_indices = kth_diag_indices(res.shape, k)
        res.ipoints[diag_indices] = a
        return res
    else:
        if k != 0 and len(axes) > 2:
            raise NotImplementedError("k != 0 and len(axes) > 2")
        if axes is None:
            axes = a.axes
        else:
            axes = a.axes[axes]
        axes_indices = kth_diag_indices(axes.shape, k)
        indexer = tuple(axis.i[indices]
                        for axis, indices in zip(axes, axes_indices))
        return a.points[indexer]


def labels_array(axes, title=''):
    """Returns an array with specified axes and the combination of
    corresponding labels as values.

    Parameters
    ----------
    axes : Axis or collection of Axis
    title : str, optional
        Title.

    Returns
    -------
    LArray

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> labels_array(sex)
    sex | M | F
        | M | F
    >>> labels_array((nat, sex))
    nat | sex\\axis | nat | sex
     BE |        M |  BE |   M
     BE |        F |  BE |   F
     FO |        M |  FO |   M
     FO |        F |  FO |   F
    """
    # >>> labels_array((nat, sex))
    # nat\\sex |    M |    F
    #      BE | BE,M | BE,F
    #      FO | FO,M | FO,F
    axes = AxisCollection(axes)
    if len(axes) > 1:
        res_axes = axes + Axis('axis', axes.names)
        res_data = np.empty(res_axes.shape, dtype=object)
        res_data.flat[:] = list(product(*axes.labels))
        # XXX: I wonder if it wouldn't be better to return LGroups or a
        # similar object which would display as "a,b" but where each label is
        # stored separately.
        # flat_data = np.array([p for p in product(*axes.labels)])
        # res_data = flat_data.reshape(axes.shape)
    else:
        res_axes = axes
        res_data = axes[0].labels
    return LArray(res_data, res_axes, title)


def identity(axis):
    raise NotImplementedError("identity(axis) is deprecated. In most cases, "
                              "you can now use the axis directly. For example, "
                              "'identity(age) < 10' can be replaced by "
                              "'age < 10'. In other cases, you should use "
                              "labels_array(axis) instead.")


def eye(rows, columns=None, k=0, title='', dtype=None):
    """Returns a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    rows : int or Axis
        Rows of the output.
    columns : int or Axis, optional
        Columns in the output. If None, defaults to rows.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal, a
        positive value refers to an upper diagonal, and a negative value to a
        lower diagonal.
    title : str, optional
        Title.
    dtype : data-type, optional
        Data-type of the returned array. Defaults to float.

    Returns
    -------
    LArray of shape (rows, columns)
        An array where all elements are equal to zero, except for the k-th
        diagonal, whose values are equal to one.

    Examples
    --------
    >>> eye(2, dtype=int)
    {0}*\\{1}* | 0 | 1
            0 | 1 | 0
            1 | 0 | 1
    >>> sex = Axis('sex', ['M', 'F'])
    >>> eye(sex)
    sex\\sex |   M |   F
          M | 1.0 | 0.0
          F | 0.0 | 1.0
    >>> eye(3, k=1)
    {0}*\\{1}* |   0 |   1 |   2
            0 | 0.0 | 1.0 | 0.0
            1 | 0.0 | 0.0 | 1.0
            2 | 0.0 | 0.0 | 0.0
    """
    if columns is None:
        columns = rows.copy() if isinstance(rows, Axis) else rows
    axes = AxisCollection([rows, columns])
    shape = axes.shape
    data = np.eye(shape[0], shape[1], k, dtype)
    return LArray(data, axes, title)


# XXX: we could change the syntax to use *args
#      => less punctuation but forces kwarg
#      => potentially longer
#      => unsure for now. The most important point is that it should be
#         consistent with other functions.
# stack(a1, a2, axis=Axis('sex', 'M,F'))
# stack(('M', a1), ('F', a2), axis='sex')
# stack(a1, a2, axis='sex')
def stack(arrays, axis=None, title=''):
    """
    Combines several arrays along an axis.

    Parameters
    ----------
    arrays : tuple or list of values.
        Arrays to stack. values can be scalars, arrays or (label, value) pairs.
    axis : str or Axis, optional
        Axis to create. If None, defaults to a range() axis.
    title : str, optional
        Title.

    Returns
    -------
    LArray
        A single array combining arrays.

    Examples
    --------
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> arr1 = ones(nat)
    >>> arr1
    nat |  BE |  FO
        | 1.0 | 1.0
    >>> arr2 = zeros(nat)
    >>> arr2
    nat |  BE |  FO
        | 0.0 | 0.0
    >>> stack([('M', arr1), ('F', arr2)], 'sex')
    nat\\sex |   M |   F
         BE | 1.0 | 0.0
         FO | 1.0 | 0.0

    It also works when reusing an existing axis (though the first syntax
    should be preferred because it is more obvious which label correspond to
    what array):

    >>> sex = Axis('sex', ['M', 'F'])
    >>> stack((arr1, arr2), sex)
    nat\\sex |   M |   F
         BE | 1.0 | 0.0
         FO | 1.0 | 0.0

    or for arrays with different axes:

    >>> stack((arr1, 0), sex)
    nat\\sex |   M |   F
         BE | 1.0 | 0.0
         FO | 1.0 | 0.0

    or even with no axis at all:

    >>> stack((arr1, arr2))
    nat\\{1}* |   0 |   1
          BE | 1.0 | 0.0
          FO | 1.0 | 0.0

    >>> # TODO: move this to unit test
    >>> # not using the same length as nat, otherwise numpy gets confused :(
    >>> nd = LArray([arr1, zeros(Axis('type', [1, 2, 3]))], sex)
    >>> stack(nd, sex)
    nat | type\\sex |   M |   F
     BE |        1 | 1.0 | 0.0
     BE |        2 | 1.0 | 0.0
     BE |        3 | 1.0 | 0.0
     FO |        1 | 1.0 | 0.0
     FO |        2 | 1.0 | 0.0
     FO |        3 | 1.0 | 0.0
    """
    # LArray arrays could be interesting
    if isinstance(arrays, LArray):
        if axis is None:
            axis = -1
        axis = arrays.axes[axis]
        values = [arrays[k] for k in axis]
    else:
        assert isinstance(arrays, (tuple, list))
        if all(isinstance(a, tuple) for a in arrays):
            assert all(len(a) == 2 for a in arrays)
            keys = [k for k, v in arrays]
            assert all(np.isscalar(k) for k in keys)
            if isinstance(axis, Axis):
                assert np.array_equal(axis.labels, keys)
            else:
                # None or str
                axis = Axis(axis, keys)
            values = [v for k, v in arrays]
        else:
            values = arrays
            if axis is None or isinstance(axis, basestring):
                axis = Axis(axis, len(arrays))
            else:
                assert len(axis) == len(arrays)
    result_axes = AxisCollection.union(*[get_axes(v) for v in values])
    result_axes.append(axis)
    result = empty(result_axes, title=title, dtype=common_type(values))
    for k, v in zip(axis, values):
        result[k] = v
    return result


class ExprNode(object):
    # method factory
    def _binop(opname):
        def opmethod(self, other):
            return BinaryOp(opname, self, other)

        opmethod.__name__ = '__%s__' % opname
        return opmethod

    __matmul__ = _binop('matmul')
    __ror__ = _binop('ror')
    __or__ = _binop('or')
    __rxor__ = _binop('rxor')
    __xor__ = _binop('xor')
    __rand__ = _binop('rand')
    __and__ = _binop('and')
    __rrshift__ = _binop('rrshift')
    __rshift__ = _binop('rshift')
    __rlshift__ = _binop('rlshift')
    __lshift__ = _binop('lshift')
    __rpow__ = _binop('rpow')
    __pow__ = _binop('pow')
    __rdivmod__ = _binop('rdivmod')
    __divmod__ = _binop('divmod')
    __rmod__ = _binop('rmod')
    __mod__ = _binop('mod')
    __rfloordiv__ = _binop('rfloordiv')
    __floordiv__ = _binop('floordiv')
    __rtruediv__ = _binop('rtruediv')
    __truediv__ = _binop('truediv')
    if sys.version < '3':
        __div__ = _binop('div')
        __rdiv__ = _binop('rdiv')
    __rmul__ = _binop('rmul')
    __mul__ = _binop('mul')
    __rsub__ = _binop('rsub')
    __sub__ = _binop('sub')
    __radd__ = _binop('radd')
    __add__ = _binop('add')
    __ge__ = _binop('ge')
    __gt__ = _binop('gt')
    __ne__ = _binop('ne')
    __eq__ = _binop('eq')
    __le__ = _binop('le')
    __lt__ = _binop('lt')

    def _unaryop(opname):
        def opmethod(self):
            return UnaryOp(opname, self)

        opmethod.__name__ = '__%s__' % opname
        return opmethod

    # unary ops do not need broadcasting so do not need to be overridden
    __neg__ = _unaryop('neg')
    __pos__ = _unaryop('pos')
    __abs__ = _unaryop('abs')
    __invert__ = _unaryop('invert')

    def evaluate(self, context):
        raise NotImplementedError()


def expr_eval(expr, context):
    return expr.evaluate(context) if isinstance(expr, ExprNode) else expr


class AxisReference(ExprNode, Axis):
    def __init__(self, name):
        self.name = name
        self._labels = None
        self._iswildcard = False

    def translate(self, key):
        raise NotImplementedError("an AxisReference (x.) cannot translate "
                                  "labels")

    def __repr__(self):
        return 'AxisReference(%r)' % self.name

    def evaluate(self, context):
        """
        Parameters
        ----------
        context : AxisCollection
            Use axes from this collection
        """
        return context[self]

    # needed because ExprNode.__hash__ (which is object.__hash__)
    # takes precedence over Axis.__hash__
    def __hash__(self):
        return id(self)


class BinaryOp(ExprNode):
    def __init__(self, op, expr1, expr2):
        self.op = op
        self.expr1 = expr1
        self.expr2 = expr2

    def evaluate(self, context):
        # TODO: implement eval via numexpr
        expr1 = expr_eval(self.expr1, context)
        expr2 = expr_eval(self.expr2, context)
        return getattr(expr1, '__%s__' % self.op)(expr2)


class UnaryOp(ExprNode):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

    def evaluate(self, context):
        # TODO: implement eval via numexpr
        expr = expr_eval(self.expr, context)
        return getattr(expr, '__%s__' % self.op)()


class AxisReferenceFactory(object):
    def __getattr__(self, key):
        return AxisReference(key)

    def __getitem__(self, key):
        return AxisReference(key)

x = AxisReferenceFactory()


def get_axes(value):
    return value.axes if isinstance(value, LArray) else AxisCollection([])


def _strip_shape(shape):
    return tuple(s for s in shape if s != 1)


def _equal_modulo_len1(shape1, shape2):
    return _strip_shape(shape1) == _strip_shape(shape2)


# assigning a temporary name to anonymous axes before broadcasting and
# removing it afterwards is not a good idea after all because it copies the
# axes/change the object, and thus "flatten" wouldn't work with position axes:
# a[ones(a.axes[axes], dtype=bool)]
# but if we had assigned axes names from the start (without dropping them)
# this wouldn't be a problem.
def make_numpy_broadcastable(values):
    """
    Returns values where LArrays are (NumPy) broadcastable between them.
    For that to be possible, all common axes must be compatible
    (see Axis class documentation).
    Extra axes (in any array) can have any length.

    * the resulting arrays will have the combination of all axes found in the
      input arrays, the earlier arrays defining the order of axes. Axes with
      labels take priority over wildcard axes.
    * length 1 wildcard axes will be added for axes not present in input

    Parameters
    ----------
    values : iterable of arrays
        Arrays that requires to be (NumPy) broadcastable between them.

    Returns
    -------
    list of arrays
        List of arrays broadcastable between them.
        Arrays will have the combination of all axes found in the
        input arrays, the earlier arrays defining the order of axes.
    AxisCollection
        Collection of axes of all input arrays.

    See Also
    --------
    Axis.iscompatible : tests if axes are compatible between them.
    """
    all_axes = AxisCollection.union(*[get_axes(v) for v in values])
    return [v.broadcast_with(all_axes) if isinstance(v, LArray) else v
            for v in values], all_axes


# excel IO tools in Python
# - openpyxl, the slowest but most-complete package but still lags behind
#   PHPExcel from which it was ported. despite the drawbacks the API is very
#   complete.
#   biggest drawbacks:
#   * you can get either the "cached" value of cells OR their formulas but NOT
#     BOTH and this is a file-wide setting (data_only=True).
#     if you have an excel file and want to add a sheet to it, you either loose
#     all cached values (which is problematic in many cases since you do not
#     necessarily have linked files) or loose all formulas.
#   * it loose "charts" on read. => cannot append/update a sheet to a file with
#     charts, which is precisely what many users asked. => users need to
#     create their charts using code.
# - xlsxwriter: faster and slightly more feature-complete than openpyxl
#   regarding writing but does not read anything => cannot update an existing
#   file. API seems extremely complete.
# - pyexcelerate: yet faster but also write only. Didn't check whether API
#   is more featured than xlsxwriter or not.
# - xlwings: wraps win32com & equivalent on mac, so can potentially do
#   everything (I guess) but this is SLOW and needs a running excel instance,
#   etc.
