import fnmatch
import re
import warnings
from itertools import product, chain

import numpy as np
import pandas as pd

from typing import Union, List, Tuple, Any, Sequence, Iterable

from larray.core.abstractbases import ABCAxis, ABCAxisReference, ABCArray
from larray.util.oset import OrderedSet
from larray.util.misc import (
    unique_list, find_closing_chr, _parse_bound, _seq_summary, _isintstring,
    renamed_to, LHDFStore, PY312_OR_LATER
)
from larray.util.types import Scalar, Key


def _slice_to_str(key: slice, repr_func=str) -> str:
    r"""
    Convert a slice to a string.

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
    start = repr_func(key.start) if key.start is not None else ''
    stop = repr_func(key.stop) if key.stop is not None else ''
    step = (":" + repr_func(key.step)) if key.step is not None else ''
    return f'{start}:{stop}{step}'


def irange(start, stop, step=None) -> range:
    r"""Create a range, with inclusive stop bound and automatic sign for step.

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
    elif step <= 0:
        raise ValueError("step must be a positive integer or None")
    step = step if start <= stop else -step
    stop = stop + 1 if start <= stop else stop - 1
    return range(start, stop, step)


_range_bound_pattern = re.compile(r'([0-9]+|[a-zA-Z]+)')


def generalized_range(start, stop, step=1) -> Union[range, List[str], List[int]]:
    r"""Create a range, with inclusive stop bound and automatic sign for step. Bounds can be strings.

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

    >>> generalized_range('A8', 'A10')
    ['A8', 'A9', 'A10']

    one may use zero padding on numbers

    >>> generalized_range('A08', 'A10')
    ['A08', 'A09', 'A10']

    consecutive letters create all combinations

    >>> generalized_range('AA', 'CC')
    ['AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC']

    one cannot go from a integer to a letter and vice versa

    >>> generalized_range('1', 'F')
    Traceback (most recent call last):
    ...
    ValueError: expected an integer for the stop bound (because the start bound is an integer) but got 'F' instead

    when using special characters, they must be the same on both sides

    >>> generalized_range('a|1', 'a/2')
    Traceback (most recent call last):
    ...
    ValueError: Special characters must be the same for start and stop
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
                if not stop_part.isdigit():
                    raise ValueError(f"expected an integer for the stop bound (because the start bound is an integer) "
                                     f"but got '{stop_part}' instead")
                rng = irange(int(start_part), int(stop_part))
                start_pad = len(start_part) if start_part.startswith('0') else None
                stop_pad = len(stop_part) if stop_part.startswith('0') else None
                if start_pad is not None and stop_pad is not None and start_pad != stop_pad:
                    raise ValueError(f"Inconsistent zero padding for start and stop ({start_pad} vs {stop_pad}) of the "
                                     f"numerical part. Must be either the same on both sides or no padding on either "
                                     f"side")
                elif start_pad is None and stop_pad is None:
                    r = [str(num) for num in rng]
                else:
                    pad = start_pad if stop_pad is None else stop_pad
                    r = ['%0*d' % (pad, num) for num in rng]
            elif start_part.isalpha():
                assert stop_part.isalpha()
                int_start = [ord(c) for c in start_part]
                int_stop = [ord(c) for c in stop_part]
                sranges = [[chr(c) for c in irange(r_start, r_stop) if chr(c).isalnum()]
                           for r_start, r_stop in zip(int_start, int_stop)]
                r = [''.join(p) for p in product(*sranges)]
            else:
                # special characters
                if start_part != stop_part:
                    raise ValueError("Special characters must be the same for start and stop")
                r = [start_part]
            ranges.append(r)
        res = [''.join(p) for p in product(*ranges)]
        return res if step == 1 else res[::step]
    else:
        return irange(start, stop, step)


_range_str_pattern = re.compile(r'(?P<start>[^\s.]+)?\s*\.\.\s*(?P<stop>[^\s.]+)?(\s+step\s+(?P<step>\d+))?')


def _range_str_to_range(s, stack_depth=1) -> Union[range, List[str], List[int]]:
    r"""
    Convert a range string to a range (of values).
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

    0 padding
    >>> list(_range_str_to_range('01..12'))
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    any special character except . and spaces should work
    >>> _range_str_to_range('a|+*@-b .. a|+*@-d')
    ['a|+*@-b', 'a|+*@-c', 'a|+*@-d']
    """
    s = s.strip()
    m = _range_str_pattern.match(s)

    groups = m.groupdict()
    start, stop, step = groups['start'], groups['stop'], groups['step']
    start = _parse_bound(start, stack_depth + 1) if start is not None else 0
    if stop is None:
        raise ValueError(f"no stop bound provided in range: {s!r}")
    stop = _parse_bound(stop, stack_depth + 1)
    if isinstance(start, str) or isinstance(stop, str):
        start, stop = str(start), str(stop)
    # TODO: use parse_bound
    step = int(step) if step is not None else 1
    return generalized_range(start, stop, step)


def _normalize_idx(idx, length) -> int:
    if idx < - length:
        return - length - 1
    elif -length <= idx < 0:
        return idx + length
    elif 0 <= idx < length:
        return idx
    elif idx > length:
        return length


def _idx_seq_to_slice(seq, length) -> slice:
    r"""
    Transform a sequence of indices into a slice if possible and normalize it.

    If a constant step between indices of the sequence can be inferred (the sequence is something like
    [start, start+step, start+2*step, ...]), a normalized (with has few negative indices as possible) slice is
    returned, otherwise the sequence is returned unmodified.

    Parameters
    ----------
    seq : sequence-like of integers
        Sequence (list, tuple or ndarray) of indices.
    length : int
        Length of the sequence the indices refer to.

    Returns
    -------
    slice or sequence-like
        return the input sequence if a slice cannot be defined.

    Examples
    --------
    >>> _idx_seq_to_slice([3, 4, 5], 10)
    slice(3, 6, None)
    >>> _idx_seq_to_slice([3, 4, 6], 10)
    [3, 4, 6]
    >>> _idx_seq_to_slice([3, 5, 7], 10)
    slice(3, 8, 2)
    >>> _idx_seq_to_slice([-3, -2], 10)
    slice(7, 9, None)
    >>> _idx_seq_to_slice([-1, -2], 10)
    slice(9, 7, -1)
    >>> _idx_seq_to_slice([2, 1], 10)
    slice(2, 0, -1)
    >>> _idx_seq_to_slice([1, 0], 10)
    slice(1, -11, -1)
    >>> _idx_seq_to_slice([1], 10)
    slice(1, 2, None)
    >>> _idx_seq_to_slice([], 10)
    []
    >>> _idx_seq_to_slice([3, 1], 10)
    slice(3, 0, -2)
    """
    if len(seq) < 1:
        return seq
    first_idx = _normalize_idx(seq[0], length)
    if len(seq) == 1:
        return slice(first_idx, first_idx + 1)
    second_idx = _normalize_idx(seq[1], length)
    step = second_idx - first_idx
    prev_idx = second_idx
    for raw_idx in seq[2:]:
        idx = _normalize_idx(raw_idx, length)
        if idx != prev_idx + step:
            return seq
        prev_idx = idx
    last_idx = prev_idx
    step_sign = 1 if step > 0 else -1
    # we stop just after last_idx instead of using a full "step" because for negative steps, it could get us a negative
    # stop which isn't what we want
    stop = last_idx + step_sign
    if last_idx == 0 and step < 0:
        stop -= length
    if step == 1:
        step = None
    return slice(first_idx, stop, step)


def _is_object_array(array) -> bool:
    return isinstance(array, np.ndarray) and array.dtype.type == np.object_


def _can_have_groups(seq) -> bool:
    return _is_object_array(seq) or isinstance(seq, (tuple, list))


def _contain_group_ticks(ticks) -> bool:
    return _can_have_groups(ticks) and any(isinstance(tick, Group) for tick in ticks)


def _seq_group_to_name(seq) -> Sequence[Any]:
    if _can_have_groups(seq):
        return [v.name if isinstance(v, Group) else v for v in seq]
    else:
        return seq


def _to_tick(v) -> Scalar:
    r"""
    Convert any value to a tick (ie makes it hashable, and acceptable as an ndarray element).

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
    # the fact that an "aggregated tick" is passed as a LGroup or as a string should be as irrelevant as possible.
    # The thing is that we cannot (currently) use the more elegant _to_tick(e.key) that means the LGroup is not
    # available in Axis.__init__ after to_ticks, and we need it to update the mapping if it was named. Effectively,
    # this creates two entries in the mapping for a single tick. Besides, I like having the LGroup as the tick, as it
    # provides extra info as to where it comes from.
    if np.isscalar(v):
        return v
    elif isinstance(v, Group):
        return v.name if v.name is not None else _to_tick(v.to_label())
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


def _to_ticks(s, parse_single_int=False) -> Iterable[Scalar]:
    r"""
    Make a (list of) value(s) usable as the collection of labels for an Axis (ie hashable).

    Strip strings, split them on ',' and translate "range strings" to list of values **including the end point** !

    Parameters
    ----------
    s : iterable
        List of values usable as the collection of labels for an Axis.

    Returns
    -------
    Numpy 1D array containing labels

    Examples
    --------
    >>> list(_to_ticks('M , F'))                # doctest: +NORMALIZE_WHITESPACE
    ['M', 'F']
    >>> list(_to_ticks('A,C..E,F..G,Z'))        # doctest: +NORMALIZE_WHITESPACE
    ['A', 'C', 'D', 'E', 'F', 'G', 'Z']
    >>> list(_to_ticks('U'))                    # doctest: +NORMALIZE_WHITESPACE
    ['U']
    >>> list(_to_ticks('..3'))                  # doctest: +NORMALIZE_WHITESPACE
    [0, 1, 2, 3]
    >>> list(_to_ticks('01..12'))               # doctest: +NORMALIZE_WHITESPACE
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    >>> list(_to_ticks('01,02,03,10,11,12'))    # doctest: +NORMALIZE_WHITESPACE
    ['01', '02', '03', '10', '11', '12']
    """
    if isinstance(s, ABCAxis):
        return s.labels
    if isinstance(s, Group):
        # a single LGroup used for all ticks of an Axis
        return _to_ticks(s.eval())
    elif isinstance(s, np.ndarray):
        # we assume it has already been translated
        # XXX: Is it a safe assumption?
        return s

    if isinstance(s, pd.Index):
        ticks = s.values
    elif isinstance(s, (list, tuple)):
        ticks = [_to_tick(e) for e in s]
    elif isinstance(s, range):
        ticks = s
    elif isinstance(s, str):
        seq = _seq_str_to_seq(s, parse_single_int=parse_single_int)
        if isinstance(seq, slice):
            raise ValueError("using : to define axes is deprecated, please use .. instead")
        ticks = [seq] if isinstance(seq, (str, int)) else seq
    elif hasattr(s, '__array__'):
        ticks = s.__array__()
    else:
        try:
            ticks = list(s)
        except TypeError:
            raise TypeError(f"ticks must be iterable ({type(s)} is not)")
    return np.asarray(ticks)


_axis_name_pattern = re.compile(r'\s*(([A-Za-z0-9]\w*)(\.i)?\s*\[)?(.*)')


def _seq_str_to_seq(s, stack_depth=1, parse_single_int=False) -> Union[slice, List[Scalar], range, Scalar]:
    r"""
    Convert a sequence string to its sequence (or scalar).

    Parameters
    ----------
    s : str
        string to parse

    Returns
    -------
    scalar, slice, range or list
    """
    numcolons = s.count(':')
    # stack_depth + 2 before Python 3.12 because the list comp had its own frame
    extra_depth = 1 if PY312_OR_LATER else 2
    if numcolons:
        assert numcolons <= 2
        # bounds can be of len 2 or 3 (if step is provided)
        bounds = [_parse_bound(b, stack_depth + extra_depth) for b in s.split(':')]
        return slice(*bounds)
    elif ',' in s and '..' in s:
        # strip extremity commas to avoid empty string sequence elements
        s = s.strip(',')

        def to_seq(b, stack_depth=1):
            if '..' in b:
                return _range_str_to_range(b, stack_depth + 1)
            else:
                parsed = _parse_bound(b, stack_depth + 1)
                return (parsed,)

        return list(chain(*[to_seq(b, stack_depth + extra_depth) for b in s.split(',')]))
    elif ',' in s:
        # strip extremity commas to avoid empty string sequence elements
        s = s.strip(',')
        return [_parse_bound(b, stack_depth + extra_depth) for b in s.split(',')]
    elif '..' in s:
        return _range_str_to_range(s, stack_depth + 1)
    else:
        return _parse_bound(s, stack_depth + 1, parse_int=parse_single_int)


def _to_key(v, stack_depth=1, parse_single_int=False) -> Key:
    r"""
    Convert a value to a key usable for indexing (slice object, list of values,...).
    Strings are split on ',' and stripped. Colons (:) are interpreted as slices.

    Parameters
    ----------
    v : int or str or tuple or list or slice or Array or Group
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
    >>> _to_key('a..c')
    ['a', 'b', 'c']
    >>> _to_key('a,c..e,g..h,z')
    ['a', 'c', 'd', 'e', 'g', 'h', 'z']
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
    >>> _to_key('0[a0]')
    0['a0']
    >>> _to_key('0.i[a0]')
    0.i['a0']
    >>> ext = [1, 2, 3]
    >>> _to_key('{ext} >> ext')
    LGroup([1, 2, 3]) >> 'ext'
    >>> answer = 42
    >>> _to_key('{answer}')
    42
    >>> _to_key('{answer} >> answer')
    LGroup(42) >> 'answer'
    >>> _to_key('10:{answer} >> answer')
    LGroup(slice(10, 42, None)) >> 'answer'
    >>> _to_key('4,{answer},2 >> answer')
    LGroup([4, 42, 2]) >> 'answer'
    >>> list(_to_key('40..{answer}'))
    [40, 41, 42]
    >>> _to_key('4,40..{answer},2')
    [4, 40, 41, 42, 2]
    >>> _to_key('4,40..{answer},2 >> answer')
    LGroup([4, 40, 41, 42, 2]) >> 'answer'
    """
    if isinstance(v, tuple):
        return list(v)
    elif isinstance(v, str):
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
            if _isintstring(axis):
                axis = int(axis)
            axis_bracket_open = m.end(1) - 1
            # check that the string parentheses are correctly balanced
            _ = find_closing_chr(v, axis_bracket_open)
            # strip closing bracket (it should be at the end because we took care of the name earlier)
            assert key[-1] == ']'
            key = key[:-1]
        if name is not None or axis is not None:
            cls = IGroup if positional else LGroup
            key = _to_key(key, stack_depth + 1, parse_single_int=positional)
            return cls(key, name=name, axis=axis)
        else:
            return _seq_str_to_seq(v, stack_depth + 1, parse_single_int=parse_single_int)
    elif (v is Ellipsis or np.isscalar(v) or
          isinstance(v, (Group, slice, list, np.ndarray, ABCArray, OrderedSet,
                         pd.arrays.ArrowStringArray))):
        return v
    else:
        raise TypeError(f"{v} has an invalid type ({type(v).__name__}) for a key")


def _to_keys(value, stack_depth=1) -> Union[Key, Tuple[Key]]:
    r"""
    Convert a (collection of) group(s) to a structure usable for indexing.

    'label' or ['l1', 'l2'] or [['l1', 'l2'], ['l3']]

    Parameters
    ----------
    value : int or str or tuple or list or slice or Array or Group
        (collection of) value(s) to convert into key(s) usable for indexing

    Returns
    -------
    list of keys

    Examples
    --------
    It is only used for .sum(axis=xxx)
    >>> _to_keys('P01,P02')  # <-- one group => collapse dimension
    ['P01', 'P02']
    >>> _to_keys(('P01,P02',))  # <-- do not collapse dimension
    (['P01', 'P02'],)
    >>> _to_keys('P01;P02,P03;:')
    ('P01', ['P02', 'P03'], slice(None, None, None))

    >>> ext = 'P03'
    >>> _to_keys('P01,P02,{ext}')
    ['P01', 'P02', 'P03']
    >>> _to_keys('P01;P02;{ext}')
    ('P01', 'P02', 'P03')

    >>> _to_keys('age[10:19] >> teens ; year.i[-1]')
    (age[10:19] >> 'teens', year.i[-1])

    # >>> to_keys('P01,P02,:') # <-- INVALID !
    # it should have an explicit failure

    # we allow this, even though it is a dubious syntax
    >>> _to_keys(('P01', 'P02', ':'))
    ('P01', 'P02', slice(None, None, None))

    # it is better to use explicit groups
    >>> _to_keys(('P01,', 'P02,', ':'))
    (['P01'], ['P02'], slice(None, None, None))

    # or even the ugly duck...
    >>> _to_keys((('P01',), ('P02',), ':'))
    (['P01'], ['P02'], slice(None, None, None))
    """
    if isinstance(value, str) and ';' in value:
        value = tuple(value.split(';'))

    if isinstance(value, tuple):
        # stack_depth + 2 before Python 3.12 because the list comp had its own frame
        extra_depth = 1 if PY312_OR_LATER else 2
        return tuple([_to_key(group, stack_depth + extra_depth) for group in value])
    else:
        return _to_key(value, stack_depth + 1)


# forbidden characters in sheet names
_sheet_name_pattern = re.compile(r'[\\/?*\[\]:]')


def _translate_sheet_name(sheet_name) -> str:
    if isinstance(sheet_name, Group):
        sheet_name = str(_to_tick(sheet_name))
    if isinstance(sheet_name, str):
        sheet_name = _sheet_name_pattern.sub('_', sheet_name)
        if len(sheet_name) > 31:
            raise ValueError("Sheet names cannot exceed 31 characters")
    return sheet_name


# forbidden characters for dataset names in HDF files
_key_hdf_pattern = re.compile(r'[\\/]')


def _translate_group_key_hdf(key) -> str:
    if isinstance(key, Group):
        key = _key_hdf_pattern.sub('_', str(_to_tick(key)))
    return key


# TODO: kill this function
def union(*args) -> List[Any]:
    # TODO: add support for LGroup and lists
    r"""
    Return the union of several "value strings" as a list.

    Parameters
    ----------
    *args
        (collection of) value(s) to be converted into label(s). Repeated values are taken only once.

    Returns
    -------
    list of labels

    Examples
    --------
    >>> union('a', 'a, b, c, d', ['d', 'e', 'f'], '..2')
    ['a', 'b', 'c', 'd', 'e', 'f', 0, 1, 2]
    """
    if args:
        return unique_list(chain(*(_to_ticks(arg) for arg in args)))
    else:
        return []


class IGroupMaker:
    r"""
    Generates a new instance of IGroup for a given axis and key.

    Attributes
    ----------
    axis : Axis
        an axis.

    Notes
    -----
    This class is used by the method `Axis.i`
    """

    __slots__ = ('axis',)

    def __init__(self, axis):
        assert isinstance(axis, ABCAxis)
        self.axis = axis

    def __getitem__(self, key) -> 'IGroup':
        if isinstance(key, (int, np.integer)) and not isinstance(self.axis, ABCAxisReference) and key >= len(self.axis):
            raise IndexError(f"{key} is out of range for axis of length {len(self.axis)}")
        return IGroup(key, None, self.axis)

    def __len__(self) -> int:
        return len(self.axis)


# We need a separate class for LGroup and cannot simply create a new Axis with a subset of values/ticks/labels:
# the subset of ticks/labels of the LGroup need to correspond to its *Axis* indices
class Group:
    """Abstract Group.
    """

    __slots__ = ('key', 'name', 'axis')

    format_string = None

    def __init__(self, key, name=None, axis=None):
        if isinstance(key, tuple):
            key = list(key)
        if isinstance(key, Group):
            # FIXME: this is wrong for slices. Triggering a problem with
            #        this needs axes with labels in different order.
            #        We should call .eval() instead, but we might want
            #        to keep the slice as is (for performance reasons) when
            #        possible (when slice evaluates to the same labels on
            #        both axes). See #1155.
            # key = key.eval()
            key = key.to_label()
        self.key = remove_nested_groups(key)

        # we do NOT assign a name automatically when missing because that makes it impossible to know whether a name
        # was explicitly given or not
        self.name = _to_tick(name) if name is not None else name
        assert axis is None or isinstance(axis, (str, int, ABCAxis)), f"invalid axis '{axis}' ({type(axis).__name__})"

        # we could check the key is valid but this can be slow and could be useless
        # TODO: for performance reasons, we should cache the result. This will need to be invalidated correctly
        # axis.translate(key)

        # we store the Axis object and not its name like we did previously so that groups on anonymous axes are more
        # meaningful and that we can iterate on a slice of an axis (an LGroup). The reason to store the name instead of
        # the object was to make sure that a Group from an axis (or without axis) could be used on another axis with
        # the same name. See test_array.py:test_...
        self.axis = axis

    def __repr__(self) -> str:
        key = self.key

        # eval only returns a slice for groups without an Axis object
        if isinstance(key, slice):
            key_repr = _slice_to_str(key, repr_func=repr)
        elif isinstance(key, (tuple, list, np.ndarray, OrderedSet)):
            key_repr = _seq_summary(key, n=1000, repr_func=repr, sep=', ')
        else:
            key_repr = repr(key)

        axis_name = self.axis.name if isinstance(self.axis, ABCAxis) else self.axis
        if axis_name is not None:
            axis_name = f'X.{axis_name}' if isinstance(self.axis, ABCAxisReference) else axis_name
            s = self.format_string.format(axis=axis_name, key=key_repr)
        else:
            if self.axis is not None:
                # anonymous axis
                axis_ref = f', axis={self.axis!r}'
            else:
                axis_ref = ''

            if isinstance(key, slice):
                key_repr = repr(key)
            elif isinstance(key, list):
                key_repr = f'[{key_repr}]'

            s = f'{self.__class__.__name__}({key_repr}{axis_ref})'
        return f"{s} >> {self.name!r}" if self.name is not None else s

    def __str__(self) -> str:
        return str(self.eval())

    # TODO: rename to "to_positional"
    def translate(self, bound=None, stop=False) -> Union[int, slice, Sequence[int]]:
        r"""
        Translate key to a position if it is not already.

        Parameters
        ----------
        bound : any, optional
        stop : bool, optional

        Returns
        -------
        int-based key (single int, slice of int or tuple/list/array of them)
        """
        raise NotImplementedError()

    def eval(self) -> Union[Scalar, Sequence[Scalar]]:
        r"""
        Translate key to labels, if it is not already, expanding slices in the process.

        Returns
        -------
        label-based key (single scalar or tuple/list/array of them)
        """
        raise NotImplementedError()

    def to_label(self) -> Union[Scalar, slice, Sequence[Scalar]]:
        r"""
        Translate key to labels, if it is not already.

        Returns
        -------
        label-based key (single scalar, slice of scalars or tuple/list/array of them)
        """
        raise NotImplementedError()

    def retarget_to(self, target_axis) -> 'Group':
        r"""Retarget group to another axis.

        It will be translated to an LGroup using its former axis, if necessary.

        Parameters
        ----------
        target_axis : Axis
            axis to conform to

        Returns
        -------
        Group with axis, raise ValueError if retargeting is not possible
        """
        assert isinstance(target_axis, ABCAxis)
        if self.axis is target_axis:
            return self
        elif isinstance(self.axis, str) or isinstance(self.axis, ABCAxisReference):
            axis_name = self.axis.name if isinstance(self.axis, ABCAxisReference) else self.axis
            if axis_name != target_axis.name:
                raise ValueError('cannot retarget a Group defined without a real axis object (e.g. using '
                                 'an AxisReference (X.)) to an axis with a different name')
            return self.__class__(self.key, self.name, target_axis)
        elif isinstance(self.axis, int) or self.axis.equals(target_axis):
            # in the case of isinstance(self.axis, int), we can only hope the axis corresponds. This is the
            # case if we come from _translate_axis_key_chunk, but if the users calls this manually, we cannot know.
            # XXX: maybe changing this to retarget_to_axes would be a good idea after all?

            # just change the axis object
            return self.__class__(self.key, self.name, target_axis)
        else:
            # to retarget to another (non-equal) Axis, we need to translate to labels and expand slices
            return LGroup(self.eval(), self.name, target_axis)

    def __len__(self) -> int:
        # XXX: we probably want to_label instead of .eval (so that we do not expand slices)
        value = self.eval()
        # for some reason this breaks having LGroup ticks/labels on an axis
        # if isinstance(value, (tuple, list, Array, np.ndarray, str)):
        if hasattr(value, '__len__'):
            return len(value)
        elif isinstance(value, slice):
            start, stop = value.start, value.stop
            # not using stop - start because that does not work for string bounds
            # (and it is different for LGroup & IGroup)
            start_pos = self.translate(start)
            stop_pos = self.translate(stop)
            return stop_pos - start_pos
        else:
            raise TypeError(f'len() of unsized object ({value})')

    def __iter__(self) -> Iterable['LGroup']:
        # XXX: use translate/IGroup instead, so that it works even in the presence of duplicate labels
        #      possibly, only if axis is set?
        axis = self.axis
        return iter([LGroup(v, axis=axis) for v in self.eval()])

    def named(self, name) -> 'Group':
        r"""Return group with a different name.

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

    def with_axis(self, axis) -> 'Group':
        r"""Return group with a different axis.

        Parameters
        ----------
        axis : int, str, Axis
            new axis for group

        Returns
        -------
        Group
        """
        return self.__class__(self.key, self.name, axis)

    def by(self, length, step=None, template=None) -> Tuple['Group']:
        r"""Split group into several groups of specified length.

        Parameters
        ----------
        length : int
            length of new groups
        step : int, optional
            step between groups. Defaults to length.
        template : str, optional
            template describing how group names are generated. It is a string containing specific arguments
            written inside brackets {}. Available arguments are {start} and {end} representing the first and last label
            of each group. By default, template is defined as '{start}:{end}'.

        Notes
        -----
        step can be smaller than length, in which case, this will produce overlapping groups.

        Returns
        -------
        list of Group

        Examples
        --------
        >>> from larray import Axis, X
        >>> age = Axis('age=0..100')
        >>> young_children = age[0:6]
        >>> young_children.by(3)
        (age.i[0:3] >> '0:2', age.i[3:6] >> '3:5', age.i[6:7] >> '6')
        >>> young_children.by(3, step=2)
        (age.i[0:3] >> '0:2', age.i[2:5] >> '2:4', age.i[4:7] >> '4:6', age.i[6:7] >> '6')
        >>> young_children.by(3, template='{start}-{end}')
        (age.i[0:3] >> '0-2', age.i[3:6] >> '3-5', age.i[6:7] >> '6')
        """
        def make_group(start, length, name_template) -> 'Group':
            g = self[start:start + length]
            labels = g.eval()
            g.name = name_template.format(start=labels[0], end=labels[-1]) if len(labels) > 1 else str(labels[0])
            return g

        if step is None:
            step = length
        if template is None:
            template = '{start}:{end}'
        return tuple(make_group(start, length, template) for start in range(0, len(self), step))

    # TODO: __getitem__ should work by label and .i[] should work by position. I guess it would be more consistent this
    # way even if the usefulness of subsetting a group with labels is dubious (but it is sometimes practical to treat
    # the group as if it was an axis).
    # >>> vla = geo['...']
    # >>> # first 10 regions of flanders (this could have some use)
    # >>> vla.i[:10]  # => IGroup on geo
    # >>> vla["antwerp", "gent"]  # => LGroup on geo

    # LGroup[] => LGroup
    # IGroup[] => LGroup
    # IGroup.i[] => IGroup
    # LGroup.i[] => IGroup
    def __getitem__(self, key) -> 'Group':
        r"""

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
            orig_start, orig_stop, orig_step = orig_key.start, orig_key.stop, orig_key.step
            if orig_step is None:
                orig_step = 1

            orig_start_pos = self.translate(orig_start) if orig_start is not None else 0
            if isinstance(key, slice):
                key_start, key_stop, key_step = key.start, key.stop, key.step
                if key_step is None:
                    key_step = 1

                orig_stop_pos = self.translate(orig_stop, stop=True) if orig_stop is not None else len(self)
                new_start = orig_start_pos + key_start * orig_step
                new_stop = min(orig_start_pos + key_stop * orig_step, orig_stop_pos)
                new_step = orig_step * key_step
                if new_step == 1:
                    new_step = None
                return IGroup(slice(new_start, new_stop, new_step), None, self.axis)
            elif isinstance(key, int):
                return IGroup(orig_start_pos + key * orig_step, None, self.axis)
            elif isinstance(key, (tuple, list)):
                return IGroup([orig_start_pos + k * orig_step for k in key], None, self.axis)
        elif isinstance(orig_key, ABCArray):
            # XXX: why .i ?
            return cls(orig_key.i[key], None, self.axis)
        elif np.isscalar(orig_key):
            # give the opportunity to subset the label/key itself (for example for string keys)
            value = self.eval()
            return value[key]
        else:
            raise TypeError(f"cannot take a subset of {self.key} because it has a '{type(self.key)}' key")

    def _ipython_key_completions_(self) -> List[Scalar]:
        return list(self.eval())

    # method factory
    def _binop(opname):
        op_fullname = f'__{opname}__'

        # TODO: implement this in a delayed fashion for axes references
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
    # div and rdiv are not longer used on Python3+
    __rfloordiv__ = _binop('rfloordiv')
    __floordiv__ = _binop('floordiv')
    __rtruediv__ = _binop('rtruediv')
    __truediv__ = _binop('truediv')
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

    # having ne and eq use .eval on a slice group creates an ndarray, for which __eq__ does not return a single value,
    # which means, it cannot be in a mapping/Axis, but this is no longer a problem, since we do not create axes with
    # LGroup labels anymore anyway
    __ne__ = _binop('ne')
    __eq__ = _binop('eq')

    def equals(self, other) -> bool:
        r"""
        Check if this group is equal to another group.
        Two groups are equal if they have the same group and axis names and correspond to the same labels.

        Parameters
        ----------
        other : Group
            Group to compare with.

        Returns
        -------
        bool
            True if the other group is equal to this group, False otherwise.

        Examples
        --------
        >>> from larray import Axis
        >>> a = Axis('a=a0..a3')
        >>> a02 = a['a0:a2'] >> 'group_a'

        Same group names, axis names and labels

        >>> a02.equals(a02)
        True

        Different group names (one is None)

        >>> a02.equals(a['a0:a2'])
        False

        Different axis name

        >>> other_axis = a.rename('other_name')
        >>> a02.equals(other_axis['a0:a2'] >> 'group_a')
        False

        Different labels

        >>> a02.equals(a['a1:a3'] >> 'group_a')
        False

        Mixing slice and list groups

        >>> a['a0:a2'].equals(a['a0,a1,a2'])
        True

        Mixing LGroup and IGroup

        >>> a['a0:a2'].equals(a.i[0:3])
        True
        """
        if self is other:
            return True
        if self.name != other.name:
            return False
        self_axis_name = self.axis.name if isinstance(self.axis, ABCAxis) else self.axis
        other_axis_name = other.axis.name if isinstance(other.axis, ABCAxis) else other.axis
        if self_axis_name != other_axis_name:
            return False
        res = self == other
        return res if isinstance(res, bool) else all(res)

    def set(self) -> 'LSet':
        r"""Create LSet from this group.

        Returns
        -------
        LSet
        """
        return LSet(self.eval(), self.name, self.axis)

    def union(self, other) -> 'LSet':
        r"""Return (set) union of this label group and other.

        Labels relative order will be kept intact, but only unique labels will be returned. Labels from this group will
        be before labels from other.

        Parameters
        ----------
        other : Group or any sequence of labels
            other labels

        Returns
        -------
        LSet

        Examples
        --------
        >>> from larray import Axis
        >>> a = Axis('a=a0..a2')
        >>> a['a0', 'a1'].union(a['a1', 'a2'])
        a['a0', 'a1', 'a2'].set()
        >>> a['a0', 'a1'].union('a1,a2')
        a['a0', 'a1', 'a2'].set()
        >>> a['a0', 'a1'].union(['a1', 'a2'])
        a['a0', 'a1', 'a2'].set()
        """
        return self.set().union(other)

    def intersection(self, other) -> 'LSet':
        r"""Return (set) intersection of this label group and other.

        In other words, this will return labels from this group which are also in other. Labels relative order will be
        kept intact, but only unique labels will be returned.

        Parameters
        ----------
        other : Group or any sequence of labels
            other labels

        Returns
        -------
        LSet

        Examples
        --------
        >>> from larray import Axis
        >>> a = Axis('a=a0..a2')
        >>> a['a0', 'a1'].intersection(a['a1', 'a2'])
        a['a1'].set()
        >>> a['a0', 'a1'].intersection('a1,a2')
        a['a1'].set()
        >>> a['a0', 'a1'].intersection(['a1', 'a2'])
        a['a1'].set()
        """
        return self.set().intersection(other)

    def difference(self, other) -> 'LSet':
        r"""Return (set) difference of this label group and other.

        In other words, this will return labels from this group without those in other. Labels relative order will be
        kept intact, but only unique labels will be returned.

        Parameters
        ----------
        other : Group or any sequence of labels
            other labels

        Returns
        -------
        LSet

        Examples
        --------
        >>> from larray import Axis
        >>> a = Axis('a=a0..a2')
        >>> a['a0', 'a1'].difference(a['a1', 'a2'])
        a['a0'].set()
        >>> a['a0', 'a1'].difference('a1,a2')
        a['a0'].set()
        >>> a['a0', 'a1'].difference(['a1', 'a2'])
        a['a0'].set()
        """
        return self.set().difference(other)

    def __contains__(self, item) -> bool:
        if isinstance(item, Group):
            item = item.eval()
        return item in self.eval()

    def startingwith(self, prefix) -> 'LGroup':
        r"""
        Return a group with the labels starting with the specified string.

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
        >>> from larray import Axis
        >>> people = Axis(['Bruce Wayne', 'Arthur Dent', 'Harvey Dent'], 'people')
        >>> group = people.endingwith('Dent')
        >>> group
        people['Arthur Dent', 'Harvey Dent']
        >>> group.startingwith('Art')
        people['Arthur Dent']
        """
        if isinstance(prefix, Group):
            prefix = prefix.eval()
        return LGroup([v for v in self.eval() if v.startswith(prefix)], axis=self.axis)

    def endingwith(self, suffix) -> 'LGroup':
        r"""
        Return a group with the labels ending with the specified string.

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
        >>> from larray import Axis
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Arthur Dent'], 'people')
        >>> group = people.startingwith('Bru')
        >>> group
        people['Bruce Wayne', 'Bruce Willis']
        >>> people.endingwith('yne')
        people['Bruce Wayne']
        """
        if isinstance(suffix, Group):
            suffix = suffix.eval()
        return LGroup([v for v in self.eval() if v.endswith(suffix)], axis=self.axis)

    def matching(self, deprecated=None, pattern=None, regex=None) -> 'LGroup':
        r"""
        Return a group with all the labels matching the specified pattern or regular expression.

        Parameters
        ----------
        pattern : str or Group
            Pattern to match.

            - `?`     matches any single character
            - `*`     matches any number of characters
            - [seq]   matches any character in seq
            - [!seq]  matches any character not in seq

            To match any of the special characters above, wrap the character in brackets. For example, `[?]` matches
            the character `?`.
        regex : str or Group
            Regular expression pattern to match. Regular expressions are more powerful than what the simple patterns
            supported by the `pattern` argument but are also more complex to write.
            See `Regular Expression <https://docs.python.org/3/library/re.html>`_ for more details about how to build
            a regular expression pattern.

        Returns
        -------
        LGroup
            Group containing all the labels matching the pattern.

        Examples
        --------
        >>> from larray import Axis
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Arthur Dent'], 'people')

        Let us create a group with all names starting with B

        >>> group = people.startingwith('B')
        >>> group
        people['Bruce Wayne', 'Bruce Willis']

        Within that group, all labels containing any characters then W then any characters then s are given by

        >>> group.matching(pattern='*W*s')
        people['Bruce Willis']

        Regular expressions are more powerful but usually harder to write and less readable. For example,
        here are the labels not containing the letter "i".

        >>> group.matching(regex='^[^i]*$')
        people['Bruce Wayne']
        """
        if deprecated is not None:
            assert pattern is None and regex is None
            regex = deprecated
            warnings.warn("Group.matching() first argument will change to `pattern` in a later release. "
                          "If your pattern is a regular expression, use Group.matching(regex='yourpattern')."
                          "If your pattern is a 'simple pattern', use Group.matching(pattern='yourpattern').",
                          FutureWarning, stacklevel=2)
        if pattern is not None and regex is not None:
            raise ValueError("Cannot use both `pattern` and `regex` arguments at the same time in Group.matching()")
        if pattern is None and regex is None:
            raise ValueError("Must provide either `pattern` or `regex` argument in Group.matching()")
        if isinstance(regex, Group):
            regex = regex.eval()
        if pattern is not None:
            if isinstance(pattern, Group):
                pattern = pattern.eval()
            regex = fnmatch.translate(pattern)
        match = re.compile(regex).match
        return LGroup([v for v in self.eval() if match(v)], axis=self.axis)

    def containing(self, substring) -> 'LGroup':
        r"""
        Return a group with all the labels containing the specified substring.

        Parameters
        ----------
        substring : str or Group
            The substring to search for.

        Returns
        -------
        LGroup
            Group containing all the labels containing the substring.

        Examples
        --------
        >>> from larray import Axis
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Arthur Dent'], 'people')
        >>> group = people.startingwith('Bru')
        >>> group
        people['Bruce Wayne', 'Bruce Willis']
        >>> group.containing('Will')
        people['Bruce Willis']
        """
        if isinstance(substring, Group):
            substring = substring.eval()
        return LGroup([v for v in self.eval() if substring in v], axis=self.axis)

    def to_hdf(self, filepath, key=None, axis_key=None) -> None:
        r"""
        Write group to a HDF file.

        A HDF file can contain multiple groups.
        The 'key' parameter is a unique identifier for the group.
        The 'axis_key' parameter is the unique identifier for the associated axis.
        The associated axis will be saved if not already present in the HDF file.

        Parameters
        ----------
        filepath : str
            Path where the hdf file has to be written.
        key : str or Group, optional
            Key (path) of the group within the HDF file (see Notes below).
            If None, the name of the group is used.
            Defaults to None.
        axis_key : str, optional
            Key (path) of the associated axis in the HDF file (see Notes below).
            If None, the name of the axis associated with the group is used.
            Defaults to None.

        Notes
        -----
        Objects stored in a HDF file can be grouped together in `HDF groups`.
        If an object 'my_obj' is stored in a HDF group 'my_group',
        the key associated with this object is then 'my_group/my_obj'.
        Be aware that a HDF group can have subgroups.

        Examples
        --------
        >>> from larray import Axis
        >>> a = Axis("a=a0..a2")
        >>> a.to_hdf('test.h5')                     # doctest: +SKIP
        >>> a01 = a['a0,a1'] >> 'a01'

        Save group

        >>> # by default, the key is the name of the group
        >>> # and axis_key the name of the associated axis
        >>> a01.to_hdf('test.h5')                   # doctest: +SKIP

        Save group with a specific key

        >>> a01.to_hdf('test.h5', 'a_01')           # doctest: +SKIP

        Save group in a specific HDF group

        >>> a.to_hdf('test.h5', 'groups/a01')       # doctest: +SKIP

        The associated axis is saved with the group if not already present in the HDF file

        >>> b = Axis("b=b0..b2")
        >>> b01 = b['b0,b1'] >> 'b01'
        >>> # save both the group 'b01' and the associated axis 'b'
        >>> b01.to_hdf('test.h5')                   # doctest: +SKIP
        """
        if key is None:
            if self.name is None:
                raise ValueError("Argument key must be provided explicitly in case of anonymous group")
            key = self.name
        key = _translate_group_key_hdf(key)
        if axis_key is None:
            if self.axis.name is None:
                raise ValueError("Argument axis_key must be provided explicitly if the associated axis is anonymous")
            axis_key = self.axis.name
        data = self.eval()
        dtype_kind = data.dtype.kind if isinstance(data, np.ndarray) else ''
        if dtype_kind == 'U':
            data = np.char.encode(data, 'utf-8')
        s = pd.Series(data=data, name=self.name)
        with LHDFStore(filepath) as store:
            store.put(key, s)
            store.get_storer(key).attrs.type = 'Group'
            store.get_storer(key).attrs.dtype_kind = dtype_kind
            if axis_key not in store:
                self.axis.to_hdf(store, key=axis_key)
            store.get_storer(key).attrs.axis_key = axis_key

    # this makes range(LGroup(int)) possible
    def __index__(self):
        return self.eval().__index__()

    def __int__(self) -> int:
        # 'str' objects have no '__int__' attribute, so this is better than calling __int__ explicitly
        return int(self.eval())

    def __float__(self) -> float:
        # 'str' objects have no '__float__' attribute, so this is better than calling __float__ explicitly
        return float(self.eval())

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.eval(), dtype=dtype)

    def __dir__(self) -> List[str]:
        # called by dir() and tab-completion at the interactive prompt, must return a list of any valid getattr key.
        # dir() takes care of sorting but not uniqueness, so we must ensure that.
        attributes = self.__slots__
        return list(set(dir(self.eval())) | set(attributes) | set(dir(self.__class__)))

    def __getattr__(self, key) -> Scalar:
        if key == '__array_struct__':
            raise AttributeError("'Group' object has no attribute '__array_struct__'")
        else:
            return getattr(self.eval(), key)

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before .key exists, which leads to
    # an infinite recursion)
    def __getstate__(self):
        return self.key, self.name, self.axis

    def __setstate__(self, d):
        self.key, self.name, self.axis = d

    def __hash__(self) -> int:
        # to_tick & to_key are partially opposite operations but this standardize on a single notation so that they can
        # all target each other. eg, this removes spaces in "list strings", instead of hashing them directly
        # XXX: but we might want to include that normalization feature in to_tick directly, instead of using to_key
        #      explicitly here
        # XXX: we might want to make hash use the position along the axis instead of the labels so that if an axis has
        #      ambiguous labels, they do not hash to the same thing.
        # XXX: for performance reasons, I think hash should not evaluate slices. It should only translate pos bounds to
        #      labels or vice versa. We would loose equality between list Groups and equivalent slice groups but that
        #      is a small price to pay if the performance impact is large.
        # the problem with using self.translate() is that we cannot compare groups without axis
        # return hash(_to_tick(self.translate()))
        return hash(_to_tick(self.key))


def remove_nested_groups(key) -> Any:
    # "struct" key with Group elements -> key without Group
    # TODO: ideally if all key elements are groups on the same Axis, we should make a group on that axis
    #       for slice bounds, watch out for None
    if isinstance(key, slice):
        key_start, key_stop = key.start, key.stop
        start = key_start.to_label() if isinstance(key_start, Group) else key_start
        stop = key_stop.to_label() if isinstance(key_stop, Group) else key_stop
        return slice(start, stop, key.step)
    elif isinstance(key, (tuple, list)):
        res = [k.to_label() if isinstance(k, Group) else k for k in key]
        return tuple(res) if isinstance(key, tuple) else res
    else:
        return key


class LGroup(Group):
    r"""Label group.

    Represents a subset of labels of an axis.

    Parameters
    ----------
    key : key
        Anything usable for indexing. A key should be either sequence of labels, a slice with label bounds or a string.
    name : str, optional
        Name of the group.
    axis : int, str, Axis, optional
        Axis for group.

    Examples
    --------
    >>> from larray import Axis, X
    >>> age = Axis('0..100', 'age')
    >>> teens = X.age[10:19].named('teens')
    >>> teens
    X.age[10:19] >> 'teens'
    >>> teens = X.age[10:19] >> 'teens'
    >>> teens
    X.age[10:19] >> 'teens'
    """

    __slots__ = ()
    format_string = "{axis}[{key}]"

    def __init__(self, key, name=None, axis=None):
        key = _to_key(key)
        # we only check for LGroup and not Group because for scalar and list
        # IGroup, it already worked thanks to .to_label() in Group.__init__
        # and for slice IGroup, the case causing problem is subtle and we need
        # a different error message because it currently evaluates to a weird
        # thing.
        if (isinstance(key, LGroup) and isinstance(key.axis, ABCAxis) and
               not isinstance(key.axis, ABCAxisReference)):
            # unsure whether we should copy the name and axis if they are not
            # provided
            # this is what we SHOULD do but do not for backward compatibility
            # (see #1154)
            # key = key.eval()
            if (isinstance(axis, ABCAxis) and
                    not isinstance(axis, ABCAxisReference)):
                from larray.core.axis import check_warn_retarget
                check_warn_retarget(key, axis, stacklevel=4)
        Group.__init__(self, key, name, axis)

    # XXX: return IGroup instead?
    def translate(self, bound=None, stop=False) -> int:
        r"""
        compute position(s) of group.
        """
        if bound is None:
            bound = self.key
        if isinstance(self.axis, ABCAxis):
            pos = self.axis.index(bound)
            return (pos + int(stop)) if np.isscalar(pos) else pos
        else:
            raise ValueError("Cannot translate an LGroup without axis")

    def to_label(self) -> Union[Scalar, slice, Sequence[Scalar]]:
        return self.key

    def eval(self) -> Union[Scalar, Sequence[Scalar]]:
        if isinstance(self.key, slice):
            if isinstance(self.axis, ABCAxis):
                # expand slices
                return self.axis.labels[self.translate()]
            else:
                return self.key
                # raise ValueError("Cannot evaluate a slice group without axis")
        else:
            # we do not check the group labels are actually valid on Axis
            return self.key


class LSet(LGroup):
    r"""Label set.

    Represents a set of (unique) labels of an axis.

    Parameters
    ----------
    key : key
        Anything usable for indexing. A key should be either sequence of labels, a slice with label bounds or a string.
    name : str, optional
        Name of the set.
    axis : int, str, Axis, optional
        Axis for set.

    Examples
    --------
    >>> from larray import Axis
    >>> letters = Axis('letters=a..z')
    >>> abc = letters[':c'].set() >> 'abc'
    >>> abc
    letters['a', 'b', 'c'].set() >> 'abc'
    >>> abc & letters['b:d']
    letters['b', 'c'].set()
    """

    __slots__ = ()
    format_string = "{axis}[{key}].set()"

    def __init__(self, key, name=None, axis=None):
        key = _to_key(key)
        if isinstance(key, Group):
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
        op_fullname = f'__{opname}__'

        # TODO: implement this in a delayed fashion for reference axes
        def opmethod(self, other):
            if not isinstance(other, LSet):
                other = LSet(other)
            axis = self.axis if self.axis is not None else other.axis

            # setting a meaningful name is hard when either one has no name
            if self.name is not None and other.name is not None:
                name = f'{self.name} {c} {other.name}'
            else:
                name = None
            # TODO: implement this in a more efficient way for ndarray keys which can be large
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


class IGroup(Group):
    r"""Index Group.

    Represents a subset of indices of an axis.

    Parameters
    ----------
    key : key
        Anything usable for indexing. A key should be either a single position, a sequence of positions, or a slice
        with integer bounds.
    name : str, optional
        Name of the group.
    axis : int, str, Axis, optional
        Axis for group.
    """

    __slots__ = ()
    format_string = "{axis}.i[{key}]"

    def translate(self, bound=None, stop=False) -> Union[int, slice, Sequence[int]]:
        r"""
        compute position(s) of group.
        """
        if bound is not None:
            return bound
        else:
            return self.key

    def to_label(self) -> Union[Scalar, slice, Sequence[Scalar]]:
        if isinstance(self.axis, ABCAxis):
            labels = self.axis.labels
            key = self.key
            if isinstance(key, slice):
                start = labels[key.start] if key.start is not None else None
                # FIXME: this probably breaks for reverse slices
                # - 1 because IGroup slice stop is excluded while LGroup slice stop is included
                stop = labels[key.stop - 1] if key.stop is not None else None
                return slice(start, stop, key.step)
            else:
                # key is a single int or tuple/list/array of them
                return labels[key]
        else:
            raise ValueError("Cannot evaluate a positional group without axis")

    def eval(self) -> Union[Scalar, Sequence[Scalar]]:
        if isinstance(self.axis, ABCAxis):
            return self.axis.labels[self.key]
        else:
            raise ValueError("Cannot evaluate a positional group without axis")

    def __hash__(self):
        return hash(('IGroup', _to_tick(self.key)))


PGroup = renamed_to(IGroup, 'PGroup', raise_error=True)
