import fnmatch
import re
import warnings
from itertools import product
from collections import defaultdict

from typing import Union, Iterable, Tuple, List, Dict, Sequence, Optional, Any

import numpy as np
import pandas as pd

from larray.core.abstractbases import ABCAxis, ABCAxisReference, ABCArray
from larray.core.expr import ExprNode
from larray.core.group import (Group, LGroup, IGroup, IGroupMaker, _to_tick, _to_ticks, _to_key, _seq_summary,
                               _idx_seq_to_slice, _seq_group_to_name, _translate_group_key_hdf, remove_nested_groups)
from larray.util.oset import OrderedSet
from larray.util.misc import (duplicates, array_lookup2, ReprString, index_by_id, renamed_to, LHDFStore,
                              lazy_attribute, _isnoneslice, unique_list, unique_multi, Product, argsort, has_duplicates,
                              exactly_one, concatenate_ndarrays)
from larray.util.misc import first
from larray.util.types import Scalar

np_frompyfunc = np.frompyfunc


def array_equal(a1: np.ndarray, a2: np.ndarray):
    if a1.shape != a2.shape:
        return False
    # fast path when we have the same simple dtype on both sides
    # for object dtype the tobytes representation can be different
    if a1.dtype == a2.dtype and a1.dtype != object:
        return a1.data.tobytes() == a2.data.tobytes()
    else:
        # equivalent to np.array_equal, except we assume we already have numpy arrays
        return bool((a1 == a2).all())


class Axis(ABCAxis):
    r"""
    Represents an axis. It consists of a name and a list of labels.

    Parameters
    ----------
    labels : array-like or int
        collection of values usable as labels, i.e. numbers or strings or the size of the axis.
        In the last case, a wildcard axis is created.
    name : str or Axis, optional
        name of the axis or another instance of Axis. In the second case, the name of the other axis is simply copied.
        By default None.

    Attributes
    ----------
    labels : array-like or int
        collection of values usable as labels, i.e. numbers or strings
    name : str
        name of the axis. None in the case of an anonymous axis.

    Examples
    --------
    >>> gender = Axis(['M', 'F'], 'gender')
    >>> gender
    Axis(['M', 'F'], 'gender')
    >>> gender.name
    'gender'
    >>> list(gender.labels)
    ['M', 'F']

    using a string definition

    >>> gender = Axis('gender=M,F')
    >>> gender
    Axis(['M', 'F'], 'gender')
    >>> age = Axis('age=0..9')
    >>> age
    Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'age')
    >>> code = Axis('code=A,C..E,F..G,Z')
    >>> code
    Axis(['A', 'C', 'D', 'E', 'F', 'G', 'Z'], 'code')

    a wildcard axis only needs a length

    >>> row = Axis(10, 'row')
    >>> row
    Axis(10, 'row')
    >>> row.labels
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    axes can also be defined without name

    >>> anonymous = Axis('0..4')
    >>> anonymous
    Axis([0, 1, 2, 3, 4], None)
    """

    __slots__ = ('name', '__mapping', '__sorted_keys', '__sorted_values', '_labels', '_length', '_iswildcard')

    # ticks instead of labels?
    def __init__(self, labels, name=None):
        if isinstance(labels, Group) and name is None:
            name = labels.axis
        if isinstance(name, Axis):
            name = name.name
        if isinstance(labels, str):
            if '=' in labels:
                if name is not None:
                    raise ValueError("Axis(labels, name=None) cannot have "
                                     "both a string labels with an '=' sign "
                                     "and a value for the name argument")
                name, labels = [o.strip() for o in labels.split('=')]
            elif '..' not in labels and ',' not in labels:
                warnings.warn("Arguments 'name' and 'labels' of Axis constructor have been inverted in "
                              "version 0.22 of larray. Please check you are passing labels first and name "
                              "as second argument.", FutureWarning, stacklevel=2)
                name, labels = labels, name

        # make sure we do not have np.str_ as it causes problems down the
        # line with xlwings. Cannot use isinstance to check that though.
        name_is_python_str = type(name) is str or type(name) is bytes  # noqa: E721
        if isinstance(name, str) and not name_is_python_str:
            name = str(name)
        if name is not None and not isinstance(name, (int, str)):
            nametype = type(name).__name__
            raise TypeError(f"Axis name should be None, int or str but is: {name} ({nametype})")
        self.name: str = name
        self._labels = None
        self.__mapping = None
        self.__sorted_keys = None
        self.__sorted_values = None
        self._length = None
        self._iswildcard = False
        # set _labels, _length and _iswildcard via the property
        self.labels = labels

    @property
    def _mapping(self) -> Dict[Scalar, int]:
        # To map labels with their positions
        mapping = self.__mapping
        if mapping is None:
            labels = self._labels
            # TODO: this would be more efficient for wildcard axes but does not work in all cases
            # mapping = labels
            mapping = {label: i for i, label in enumerate(labels)}
            if not self._iswildcard:
                # we have no choice but to do that, otherwise we could not make geo['Brussels'] work efficiently
                # (we could have to traverse the whole mapping checking for each name, which is not an option)
                # TODO: only do this if labels.dtype is object, or add "contains_lgroup" flag in above code
                # (if any(...))
                mapping.update({label.name: i for i, label in enumerate(labels) if isinstance(label, Group)})
            self.__mapping = mapping
        return mapping

    def _update_key_values(self) -> Tuple[np.ndarray, np.ndarray]:
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
    def _sorted_keys(self) -> np.ndarray:
        if self.__sorted_keys is None:
            keys, _ = self._update_key_values()
        return self.__sorted_keys

    @property
    def _sorted_values(self) -> np.ndarray:
        values = self.__sorted_values
        if values is None:
            _, values = self._update_key_values()
        return values

    @lazy_attribute
    def i(self) -> IGroupMaker:
        r"""
        Take a subset using positions along the axis instead of labels.

        Examples
        --------
        >>> from larray import ndtest
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> arr = ndtest([sex, time])
        >>> arr
        sex\\time  2007  2008  2009  2010
               M     0     1     2     3
               F     4     5     6     7
        >>> arr[time.i[0, -1]]
        sex\\time  2007  2010
               M     0     3
               F     4     7
        """
        return IGroupMaker(self)

    @property
    def labels(self) -> Iterable[Scalar]:
        r"""
        Labels of the axis.
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            raise TypeError("labels should be a sequence, a string or a single int but not None")
        if isinstance(labels, (int, np.integer)):
            length = labels
            labels = np.arange(length)
            iswildcard = True
        else:
            labels = _to_ticks(labels, parse_single_int=True)
            length = len(labels)
            iswildcard = False

        self._length = length
        self._labels = labels
        self._iswildcard = iswildcard

    def by(self, length, step=None, template=None) -> Tuple[Group, ...]:
        r"""Split axis into several groups of specified length.

        Parameters
        ----------
        length : int
            length of groups
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
        >>> age = Axis('age=0..6')
        >>> age
        Axis([0, 1, 2, 3, 4, 5, 6], 'age')
        >>> age.by(3)
        (age.i[0:3] >> '0:2', age.i[3:6] >> '3:5', age.i[6:7] >> '6')
        >>> age.by(3, step=2)
        (age.i[0:3] >> '0:2', age.i[2:5] >> '2:4', age.i[4:7] >> '4:6', age.i[6:7] >> '6')
        >>> age.by(3, template='{start}-{end}')
        (age.i[0:3] >> '0-2', age.i[3:6] >> '3-5', age.i[6:7] >> '6')
        """
        return self[:].by(length, step, template)

    def astype(self, dtype: Union[str, np.dtype], casting: str = 'unsafe') -> 'Axis':
        """
        Cast labels to a specified type.

        Parameters
        ----------
        dtype: str or dtype
            Typecode or data-type to which the labels are cast.

        casting: str, optional
            Controls what kind of data casting may occur. Defaults to `unsafe`.

                * `no` means the data types should not be cast at all.
                * `equiv` means only byte-order changes are allowed.
                * `safe` means only casts which can preserve values are allowed.
                * `same_kind` means only safe casts or casts within a kind, like float64 to float32, are allowed.
                * `unsafe` means any data conversions may be done.

        Returns
        -------
        Axis
            Axis with labels converted to the new type.

        Examples
        --------
        >>> from larray import ndtest
        >>> arr = ndtest('time=2015..2020')
        >>> arr = arr.with_total()
        >>> arr
        time  2015  2016  2017  2018  2019  2020  total
                 0     1     2     3     4     5     15
        >>> arr = arr.drop('total')
        >>> time = arr.time
        >>> time
        Axis([2015, 2016, 2017, 2018, 2019, 2020], 'time')
        >>> time.dtype
        dtype('O')
        >>> time = time.astype(int)
        >>> time.dtype                      # doctest: +SKIP
        dtype('int64')
        """
        return Axis(labels=self.labels.astype(dtype=dtype, casting=casting), name=self.name)

    def extend(self, labels) -> 'Axis':
        r"""
        Append new labels to an axis or increase its length in case of wildcard axis.
        Note that `extend` does not occur in-place: a new axis object is allocated, filled and returned.

        Parameters
        ----------
        labels : int, iterable or Axis
            New labels to append to the axis. Passing directly another Axis is also possible.
            If the current axis is a wildcard axis, passing a length is enough.

        Returns
        -------
        Axis
            A copy of the axis with new labels appended to it or with increased length (if wildcard).

        Examples
        --------
        >>> time = Axis([2007, 2008], 'time')
        >>> time
        Axis([2007, 2008], 'time')
        >>> time.extend([2009, 2010])
        Axis([2007, 2008, 2009, 2010], 'time')
        >>> waxis = Axis(10, 'wildcard_axis')
        >>> waxis
        Axis(10, 'wildcard_axis')
        >>> waxis.extend(5)
        Axis(15, 'wildcard_axis')
        >>> waxis.extend([11, 12, 13, 14])
        Traceback (most recent call last):
        ...
        ValueError: Axis to append must (not) be wildcard if this axis is (not) wildcard
        """
        other = labels if isinstance(labels, Axis) else Axis(labels)
        if self.iswildcard != other.iswildcard:
            raise ValueError("Axis to append must (not) be wildcard if this axis is (not) wildcard")
        labels = self._length + other._length if self.iswildcard else np.append(self.labels, other.labels)
        return Axis(labels, self.name)

    def split(self, sep='_', names=None, regex=None, return_labels=False) \
            -> Union[List['Axis'], Tuple[List['Axis'], Tuple[Group]]]:
        r"""Split axis and returns a list of Axis.

        Parameters
        ----------
        sep : str, optional
            Delimiter to use for splitting. Defaults to '_'.
            When `regex` is provided, the delimiter is only used on `names` if given as one string or on axis name if
            `names` is None.
        names : str or list of str, optional
            Names of resulting axes. Defaults to None.
        regex : str, optional
            Use regex instead of delimiter to split labels. Defaults to None.
        return_labels : bool, optional
            Whether split labels must be returned (as a tuple of tuples). These labels are suitable for indexing
            via array.points[labels]. Defaults to False.

        Returns
        -------
        list of Axis or (list of Axis, array-like)

        Examples
        --------
        >>> a_b = Axis('a_b=a0_b0,a0_b1,a0_b2,a1_b0,a1_b1,a1_b2')
        >>> a_b.split()
        [Axis(['a0', 'a1'], 'a'), Axis(['b0', 'b1', 'b2'], 'b')]
        >>> ab = Axis('ab=a0_b0,a0_b1')
        >>> ab.split(names=["a", "b"])
        [Axis(['a0'], 'a'), Axis(['b0', 'b1'], 'b')]
        """
        if names is None:
            if self.name is None:
                names = None
            elif sep not in self.name:
                raise ValueError(f'{sep} not found in axis name ({self.name})')
            else:
                names = self.name.split(sep)
        # TODO: this is so ugly, I wonder if it should be allowed
        elif isinstance(names, str):
            if sep not in names:
                raise ValueError(f'{sep} not found in names ({names})')
            else:
                names = names.split(sep)
        else:
            assert all(isinstance(name, str) for name in names)
        if not regex:
            # np.char.split does not work on arrays with object dtype
            labels = self.labels if self.labels.dtype.kind != 'O' else self.labels.astype(str)
            # split_labels is an array of lists
            split_labels = np.char.split(labels, sep)
        else:
            match = re.compile(regex).match
            # split_labels is a list of tuples
            split_labels = [match(label).groups() for label in self.labels]
        first_split_label_length = len(split_labels[0])
        # TODO: when our lowest supported version will be Python 3.10, we should use
        #       strict=True instead of checking lengths explicitly
        if any(len(split_label) != first_split_label_length
               for split_label in split_labels):
            raise ValueError("not all labels have the same number of separators")
        indexing_labels = tuple(zip(*split_labels))
        if names is None:
            names = [None] * first_split_label_length
        num_axes = len(indexing_labels)
        if num_axes != len(names):
            raise ValueError(f"number of resulting axes ({num_axes}) differs "
                             f"from number of resulting axes names "
                             f"({len(names)})")
        # not using np.unique because we want to keep the original order
        split_axes = [Axis(unique_list(ax_labels), name) for ax_labels, name in zip(indexing_labels, names)]
        if return_labels:
            assert len(split_axes) == num_axes
            indexing_labels = tuple(axis[labels] for axis, labels in zip(split_axes, indexing_labels))
            return split_axes, indexing_labels
        else:
            return split_axes

    def insert(self, new_labels, before=None, after=None) -> 'Axis':
        r"""
        Return a new axis with `new_labels` inserted before `before` or after `after`.

        Parameters
        ----------
        new_labels : scalar, tuple/list/array of scalars, Group or Axis
            New label(s) to append to the axis.
        before : scalar or Group, optional
            Label or group before which to insert `new_labels`.
        after : scalar or Group, optional
            Label or group after which to insert `new_labels`.

        Returns
        -------
        Axis
            A copy of the axis with the new labels inserted.

        Examples
        --------
        >>> time = Axis([2007, 2009], 'time')
        >>> time.insert(2008, before=2009)
        Axis([2007, 2008, 2009], 'time')
        >>> time.insert(2008, after=2007)
        Axis([2007, 2008, 2009], 'time')
        >>> time.insert(2008, before=time.i[1])
        Axis([2007, 2008, 2009], 'time')
        >>> time.insert(2008, after=time.i[0])
        Axis([2007, 2008, 2009], 'time')
        >>> b = Axis(['b1', 'b2'], 'b')
        >>> b.insert('b1.5', before='b2')
        Axis(['b1', 'b1.5', 'b2'], 'b')
        >>> b.insert(['b1.1', 'b1.2'], before='b2')
        Axis(['b1', 'b1.1', 'b1.2', 'b2'], 'b')
        >>> c = Axis(['c1', 'c2'], 'c')
        >>> b.insert(c, before='b2')
        Axis(['b1', 'c1', 'c2', 'b2'], 'b')
        """
        if not exactly_one(before is not None, after is not None):
            raise ValueError("must specify exactly one of before or after")
        if before is not None:
            before = self.index(before)
        else:
            assert after is not None
            before = self.index(after) + 1

        if isinstance(new_labels, Axis):
            new_labels = new_labels.labels
        elif isinstance(new_labels, Group):
            new_labels = new_labels.eval()
            assert isinstance(new_labels, np.ndarray)
        else:
            if np.isscalar(new_labels):
                new_labels = [new_labels]
            new_labels = np.asarray(new_labels)

        # not using np.insert to avoid truncating inserted string labels (because of current_labels.dtype)
        return Axis(concatenate_ndarrays((self.labels[:before], new_labels, self.labels[before:])), self.name)

    @property
    def iswildcard(self) -> bool:
        return self._iswildcard

    def _group(self, *args, **kwargs):
        raise NotImplementedError('Axis._group is deprecated. Use "axis[key] >> group_name" instead.')

    def group(self, *args, name=None):
        key = args[0] if len(args) == 1 else args
        axis_name = self.name if self.name else 'axis'
        syntax = f'{axis_name}[{key}]'
        if name is not None:
            syntax += f' >> {name!r}'
        raise NotImplementedError(f'Axis.group is deprecated. Use {syntax} instead.')

    def all(self, name=None):
        r"""
        (Deprecated) Return a group containing all labels.

        Parameters
        ----------
        name : str, optional
            Name of the group. If not provided, name is set to 'all'.
        """
        axis_name = self.name if self.name else 'axis'
        group_name = name if name else 'all'
        raise NotImplementedError(f'Axis.all is deprecated. Use {axis_name}[:] >> {group_name!r} instead.')

    # TODO: make this method private
    def subaxis(self, key) -> 'Axis':
        r"""
        Return an axis for a sub-array.

        Parameters
        ----------
        key : int, or collection (list, slice, array, Array) of them
            Indices-based key to use for the new axis.

        Returns
        -------
        Axis
            Subaxis. If key is a None slice, the original Axis is returned.
            If key is an Array, the list of axes is returned.

        Examples
        --------
        >>> age = Axis(range(100), 'age')
        >>> age.subaxis(range(10, 19))
        Axis([10, 11, 12, 13, 14, 15, 16, 17, 18], 'age')
        """
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            return self
        if isinstance(key, ABCArray):
            return key.axes

        # TODO: compute length for wildcard axes more efficiently
        labels = len(self.labels[key]) if self.iswildcard else self.labels[key]

        # we must NOT modify the axis name, even though this creates a new axis that is independent from the original
        # one because the original name is probably what users will want to use to filter
        return Axis(labels, self.name)

    def iscompatible(self, other) -> bool:
        r"""
        Check if this axis is compatible with another axis.

        * two axes are compatible if they have compatible names and labels
        * names are compatible if they are the same or missing
        * non-wildcard labels are compatible if they are the same
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
        >>> a10  = Axis(range(10), 'a')
        >>> wa10 = Axis(10, 'a')
        >>> wa1  = Axis(1, 'a')
        >>> b10  = Axis(range(10), 'b')
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

    def equals(self, other) -> bool:
        r"""
        Check if this axis is equal to another axis.
        Two axes are equal if they have the same name and label(s).

        Parameters
        ----------
        other : Axis
            Axis to compare with.

        Returns
        -------
        bool
            True if other axis is equal to this axis, False otherwise.

        Examples
        --------
        >>> age = Axis(range(5), 'age')
        >>> age_2 = Axis(5, 'age')
        >>> age_3 = Axis(range(5), 'young children')
        >>> age_4 = Axis([0, 1, 2, 3, 4], 'age')
        >>> age.equals(age_2)
        False
        >>> age.equals(age_3)
        False
        >>> age.equals(age_4)
        True
        """
        if self is other:
            return True

        # this might need to change if we ever support wildcard axes with real labels
        return (isinstance(other, Axis)
                and self.name == other.name
                and self.iswildcard == other.iswildcard
                and (len(self) == len(other) if self.iswildcard else array_equal(self.labels, other.labels)))

    def min(self) -> Scalar:
        """
        Get minimum of labels.

        Returns
        -------
        label
            Label with minimum value.

        Warnings
        --------
        Fails on non-numeric labels.

        Examples
        --------
        >>> time = Axis('time=1991..2020')
        >>> time.min()
        1991
        """
        return np.nanmin(self.labels)

    def max(self) -> Scalar:
        """
        Get maximum of labels.

        Returns
        -------
        label
            Label with maximum value.

        Warnings
        --------
        Fails on non-numeric labels.

        Examples
        --------
        >>> time = Axis('time=1991..2020')
        >>> time.max()
        2020
        """
        return np.nanmax(self.labels)

    def matching(self, deprecated=None, pattern=None, regex=None) -> LGroup:
        r"""
        Return a group with all the labels matching the specified pattern or regular expression.

        Parameters
        ----------
        pattern : str or Group, optional
            Pattern to match.
            * `?`     matches any single character
            * `*`     matches any number of characters
            * [seq]   matches any character in seq
            * [!seq]  matches any character not in seq

            To match any of the special characters above, wrap the character in brackets. For example, `[?]` matches
            the character `?`.
        regex : str or Group, optional
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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')

        >>> # All labels starting with "A" and ending with "t"
        >>> people.matching(pattern='A*t')
        people['Arthur Dent']
        >>> # All labels containing "W" and ending with "s"
        >>> people.matching(pattern='*W*s')
        people['Bruce Willis']
        >>> # All labels with exactly 5 characters
        >>> people.matching(pattern='?????')
        people['Waldo']
        >>> # All labels starting with either "A" or "B"
        >>> people.matching(pattern='[AB]*')
        people['Bruce Wayne', 'Bruce Willis', 'Arthur Dent']

        Regular expressions are more powerful but usually harder to write and less readable

        >>> # All labels starting with "W" and ending with "o"
        >>> people.matching(regex='A.*t')
        people['Arthur Dent']
        >>> # All labels not containing character "a"
        >>> people.matching(regex='^[^a]*$')
        people['Bruce Willis', 'Arthur Dent']
        """
        if deprecated is not None:
            assert pattern is None and regex is None
            regex = deprecated
            warnings.warn("Axis.matching() first argument will change to `pattern` in a later release. "
                          "If your pattern is a regular expression, use Axis.matching(regex='yourpattern')."
                          "If your pattern is a 'simple pattern', use Axis.matching(pattern='yourpattern').",
                          FutureWarning, stacklevel=2)
        if pattern is not None and regex is not None:
            raise ValueError("Cannot use both `pattern` and `regex` arguments at the same time in Axis.matching()")
        if pattern is None and regex is None:
            raise ValueError("Must provide either `pattern` or `regex` argument in Axis.matching()")
        if isinstance(regex, Group):
            regex = regex.eval()
        if pattern is not None:
            if isinstance(pattern, Group):
                pattern = pattern.eval()
            regex = fnmatch.translate(pattern)
        match = re.compile(regex).match
        return LGroup([v for v in self.labels if match(v)], axis=self)

    matches = renamed_to(matching, 'matches', raise_error=True)

    def startingwith(self, prefix) -> LGroup:
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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')
        >>> people.startingwith('Bru')
        people['Bruce Wayne', 'Bruce Willis']
        """
        if isinstance(prefix, Group):
            prefix = prefix.eval()
        return LGroup([v for v in self.labels if v.startswith(prefix)], axis=self)

    startswith = renamed_to(startingwith, 'startswith', raise_error=True)

    def endingwith(self, suffix) -> LGroup:
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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')
        >>> people.endingwith('Dent')
        people['Arthur Dent', 'Harvey Dent']
        """
        if isinstance(suffix, Group):
            suffix = suffix.eval()
        return LGroup([v for v in self.labels if v.endswith(suffix)], axis=self)

    endswith = renamed_to(endingwith, 'endswith', raise_error=True)

    def containing(self, substring) -> LGroup:
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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Arthur Dent'], 'people')
        >>> people.containing('Will')
        people['Bruce Willis']
        """
        if isinstance(substring, Group):
            substring = substring.eval()
        return LGroup([v for v in self.labels if substring in v], axis=self)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterable[IGroup]:
        return iter([IGroup(i, None, self) for i in range(self._length)])

    def __getitem__(self, key) -> Union[LGroup, List[Group], Tuple[Group, ...]]:
        r"""
        Return a group (list or unique element) of label(s) usable in .sum or .filter.

        key is a label-based key (other axis, slice and fancy indexing are supported)

        Returns
        -------
        Group
            group containing selected label(s)/position(s).

        Notes
        -----
        key is label-based (slice and fancy indexing are supported)
        """
        # if isinstance(key, str):
        #     key = to_keys(key)

        def isscalar(k):
            return np.isscalar(k) or (isinstance(k, Group) and np.isscalar(k.key))

        if key is self:
            key = slice(None)
        elif isinstance(key, Axis):
            key = key.labels
        # handle axis[several groups or sequences]
        # the "not all(np.isscalar)" part is necessary to also support axis[a, b, c] and axis[[a, b, c]]
        elif isinstance(key, (tuple, list)) and not all(isscalar(k) for k in key):
            # this creates a group for each key if it wasn't and retargets IGroup
            list_res = [self[k] for k in key]
            return list_res if isinstance(key, list) else tuple(list_res)
        # allow targeting a label from an aggregated axis with the group which created it
        elif (
            not isinstance(self, AxisReference)
            and isinstance(key, Group)
            and isinstance(key.axis, Axis)
            and key.axis.name == self.name
            and key.name in self
        ):
            msg = "Using a Group object which was used to create an aggregate to " \
                  "target its aggregated label is deprecated. " \
                  "Please use the aggregated label directly instead. " \
                  f"In this case, you should use {key.name!r} instead of " \
                  f"using {key!r}."
            # let us hope the stacklevel does not vary by codepath
            warnings.warn(msg, FutureWarning, stacklevel=7)
            return LGroup(key.name, None, self)
        # elif isinstance(key, str) and key in self:
            # TODO: this is an awful workaround to avoid the "processing" of string keys which exist as is in the axis
            #       (probably because the string was used in an aggregate function to create the label)
            # res = LGroup(slice(None), None, self)
            # res.key = key
            # return res

        name = key.name if isinstance(key, Group) else None
        return LGroup(key, name, self)

    def _ipython_key_completions_(self) -> List[Scalar]:
        return list(self.labels)

    def __contains__(self, key) -> bool:
        # TODO: ideally, _to_tick shouldn't be necessary, the __hash__ and __eq__ of Group should include this
        return _to_tick(key) in self._mapping

    # use the default hash. We have to specify it explicitly because we define __eq__
    __hash__ = object.__hash__

    def _is_key_type_compatible(self, key) -> bool:
        key_kind = np.dtype(type(key)).kind
        label_kind = self.labels.dtype.kind
        # object kind can match anything
        return key_kind == label_kind or key_kind == 'O' or label_kind == 'O'

    def index(self, key) -> Union[int, np.ndarray, slice]:
        r"""
        Translate a label key to its numerical index counterpart.

        Parameters
        ----------
        key : key
            Everything usable as a key.

        Returns
        -------
        int, slice, np.ndarray or Arrray
            Numerical index(ices) of (all) label(s) represented by the key

        Notes
        -----
        Fancy index with boolean vectors are passed through unmodified

        Examples
        --------
        >>> people = Axis(['John Doe', 'Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')
        >>> people.index('Waldo')
        3
        >>> people.index(people.containing('Bruce'))
        array([1, 2])
        """
        mapping = self._mapping

        if isinstance(key, Group) and key.axis is not self and key.axis is not None:
            try:
                # XXX: this is potentially very expensive if key.key is an array or list and should be tried as a last
                # resort
                potential_tick = _to_tick(key)

                # avoid matching 0 against False or 0.0, note that None has object dtype and so always pass this test
                if self._is_key_type_compatible(potential_tick):
                    try:
                        res_idx = mapping[potential_tick]
                        if potential_tick != key.key:
                            # only warn if no KeyError was raised (potential_tick is in mapping)
                            msg = "Using a Group object which was used to create an aggregate to " \
                                  "target its aggregated label is deprecated. " \
                                  "Please use the aggregated label directly instead. " \
                                  f"In this case, you should use {potential_tick!r} instead of " \
                                  f"using {key!r}."
                            # let us hope the stacklevel does not vary by codepath
                            warnings.warn(msg, FutureWarning, stacklevel=8)
                        return res_idx
                    except KeyError:
                        pass
            # we must catch TypeError because key might not be hashable (eg slice)
            # IndexError is for when mapping is an ndarray
            except (KeyError, TypeError, IndexError):
                pass

        if isinstance(key, str):
            # try the key as-is to allow getting at ticks with special characters (",", ":", ...)
            try:
                # avoid matching 0 against False or 0.0, note that Group keys have object dtype and so always pass this
                # test
                if self._is_key_type_compatible(key):
                    return mapping[key]
            # we must catch TypeError because key might not be hashable (eg slice)
            # IndexError is for when mapping is an ndarray
            except (KeyError, TypeError, IndexError):
                pass

            # transform "specially formatted strings" for slices, lists, LGroup and IGroup to actual objects
            key = _to_key(key)

        if isinstance(key, range):
            key = list(key)

        # this can happen when key was passed as a string and converted to a Group via _to_key
        if isinstance(key, Group) and isinstance(key.axis, str) and key.axis != self.name:
            raise KeyError(key)

        if isinstance(key, IGroup):
            if isinstance(key.axis, Axis):
                assert key.axis is self
            return key.key

        if isinstance(key, LGroup):
            # at this point we do not care about the axis nor the name
            key = key.key

        if isinstance(key, slice):
            start = mapping[key.start] if key.start is not None else None
            # stop is inclusive in the input key and exclusive in the output !
            stop = mapping[key.stop] + 1 if key.stop is not None else None
            return slice(start, stop, key.step)
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
            # TODO: benchmark this against the tuple/list version above when mapping is large
            # array_lookup is O(len(key) * log(len(mapping)))
            # vs
            # tuple/list version is O(len(key)) (dict.getitem is O(1))
            # XXX: we might want to special case dtype bool, because in that case the mapping will in most case be
            # {False: 0, True: 1} or {False: 1, True: 0} and in those case key.astype(int) and (~key).astype(int)
            # are MUCH faster
            # see C:\Users\gdm\devel\lookup_methods.py and C:\Users\gdm\Desktop\lookup_methods.html
            try:
                return array_lookup2(_seq_group_to_name(key), self._sorted_keys, self._sorted_values)
            except KeyError:
                return array_lookup2(key, self._sorted_keys, self._sorted_values)
        elif isinstance(key, ABCArray):
            from .array import Array
            return Array(self.index(key.data), key.axes)
        else:
            # the first mapping[key] above will cover most cases.
            # This code path is only used if the key was given in "non normalized form"
            assert np.isscalar(key), f"{key} ({type(key)}) is not scalar"
            # key is scalar (integer, float, string, ...)
            if self._is_key_type_compatible(key):
                return mapping[key]
            else:
                # print("diff dtype", )
                raise KeyError(key)

    translate = renamed_to(index, 'translate', raise_error=True)

    @property
    def id(self) -> str:
        raise NotImplementedError('Axis.id is deprecated. Please use Axis.name instead.')

    def __str__(self) -> str:
        name = str(self.name) if self.name is not None else '{?}'
        return (name + '*') if self.iswildcard else name

    def __repr__(self) -> str:
        labels = len(self) if self.iswildcard else list(self.labels)
        return f'Axis({labels!r}, {self.name!r})'

    def labels_summary(self) -> str:
        r"""
        Return a short representation of the labels.

        Examples
        --------
        >>> Axis(100, 'age').labels_summary()
        '0 1 2 ... 97 98 99'
        """
        def repr_on_strings(v):
            return repr(v) if isinstance(v, str) else str(v)
        return _seq_summary(self.labels, repr_func=repr_on_strings)

    # method factory
    def _binop(opname):
        r"""
        Method factory to create binary operators special methods.
        """
        fullname = f'__{opname}__'

        def opmethod(self, other):
            if isinstance(other, (Axis, ABCArray)) or np.isscalar(other):
                from .array import labels_array
                self_array = labels_array(self)
                if isinstance(other, Axis):
                    other = labels_array(other)
                return getattr(self_array, fullname)(other)
            # give a chance to AxisCollection.__rXXX__ ops to trigger
            elif isinstance(other, AxisCollection):
                # in this case it is indeed return NotImplemented, not raise NotImplementedError!
                return NotImplemented
            else:
                raise TypeError(f"{opname} not supported between instances of 'Axis' and '{type(other).__name__}'")
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
    # div and rdiv are not longer used on Python3+
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
        r"""
        Return axis as Array.
        """
        from .array import labels_array
        return labels_array(self)

    def copy(self) -> 'Axis':
        r"""
        Return a copy of the axis.
        """
        new_axis = Axis([], self.name)
        # XXX: I wonder if we should make a copy of the labels + mapping. There should at least be an option.
        new_axis._labels = self._labels
        new_axis.__mapping = self.__mapping
        new_axis._length = self._length
        new_axis._iswildcard = self._iswildcard
        new_axis.__sorted_keys = self.__sorted_keys
        new_axis.__sorted_values = self.__sorted_values
        return new_axis

    def replace(self, old, new=None) -> 'Axis':
        r"""
        Return a new axis with some labels replaced.

        Parameters
        ----------
        old : any scalar (bool, int, str, ...), tuple/list/array of scalars, or a mapping.
            the label(s) to be replaced. Old can be a mapping {old1: new1, old2: new2, ...}
        new : any scalar (bool, int, str, ...) or tuple/list/array of scalars, optional
            the new label(s). This is argument must not be used if old is a mapping.

        Returns
        -------
        Axis
            a new Axis with the old labels replaced by new labels.

        Examples
        --------
        >>> sex = Axis('sex=M,F')
        >>> sex
        Axis(['M', 'F'], 'sex')
        >>> sex.replace('M', 'Male')
        Axis(['Male', 'F'], 'sex')
        >>> sex.replace({'M': 'Male', 'F': 'Female'})
        Axis(['Male', 'Female'], 'sex')
        >>> sex.replace(['M', 'F'], ['Male', 'Female'])
        Axis(['Male', 'Female'], 'sex')
        """
        if isinstance(old, dict):
            new = list(old.values())
            old = list(old.keys())
        elif np.isscalar(old):
            assert new is not None and np.isscalar(new), f"{new} is not a scalar but a {type(new).__name__}"
            old = [old]
            new = [new]
        else:
            seq = (tuple, list, np.ndarray)
            assert isinstance(old, seq), f"{old} is not a sequence but a {type(old).__name__}"
            assert isinstance(new, seq), f"{new} is not a sequence but a {type(new).__name__}"
            assert len(old) == len(new)
        # using object dtype because new labels length can be larger than the fixed str length in the self.labels array
        labels = self.labels.astype(object)
        indices = self.index(old)
        labels[indices] = new
        return Axis(labels, self.name)

    def apply(self, func) -> 'Axis':
        r"""
        Return a new axis with the labels transformed by func.

        Parameters
        ----------
        func : callable
            A callable which takes a single argument and returns a single value.

        Returns
        -------
        Axis
            a new Axis with the transformed labels.

        Examples
        --------
        >>> sex = Axis('sex=MALE,FEMALE')
        >>> sex.apply(str.capitalize)
        Axis(['Male', 'Female'], 'sex')
        """
        return Axis(np_frompyfunc(func, 1, 1)(self.labels), self.name)

    # XXX: rename to named like Group?
    def rename(self, name) -> 'Axis':
        r"""
        Rename the axis.

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
        >>> sex = Axis('sex=M,F')
        >>> sex
        Axis(['M', 'F'], 'sex')
        >>> sex.rename('gender')
        Axis(['M', 'F'], 'gender')
        """
        res = self.copy()
        if isinstance(name, Axis):
            name = name.name
        res.name = name
        return res
    _rename = renamed_to(rename, '_rename', raise_error=True)

    def union(self, other) -> 'Axis':
        r"""Return axis with the union of this axis labels and other labels.

        Labels relative order will be kept intact, but only unique labels will be returned. Labels from this axis will
        be before labels from other.

        Parameters
        ----------
        other : Axis, Group or any sequence of labels
            other labels

        Returns
        -------
        Axis

        Examples
        --------
        >>> a = Axis('a=a0..a2')
        >>> a.union('a1')
        Axis(['a0', 'a1', 'a2'], 'a')
        >>> a.union('a3')
        Axis(['a0', 'a1', 'a2', 'a3'], 'a')
        >>> a.union(Axis('a=a1..a3'))
        Axis(['a0', 'a1', 'a2', 'a3'], 'a')
        >>> a.union('a1..a3')
        Axis(['a0', 'a1', 'a2', 'a3'], 'a')
        >>> a.union(['a1', 'a2', 'a3'])
        Axis(['a0', 'a1', 'a2', 'a3'], 'a')
        """
        non_string_scalar = np.isscalar(other) and not isinstance(other, str)
        other = [other] if non_string_scalar else _to_ticks(other)
        return Axis(unique_multi((self.labels, other)), self.name)

    def intersection(self, other) -> 'Axis':
        r"""Return axis with the (set) intersection of this axis labels and other labels.

        In other words, this will use labels from this axis if they are also in
        other. Duplicate labels and labels relative order will be kept intact.

        Parameters
        ----------
        other : Axis, Group or any sequence of labels
            other labels

        Returns
        -------
        Axis

        Examples
        --------
        >>> a = Axis('a=a0..a2')
        >>> a.intersection('a1')
        Axis(['a1'], 'a')
        >>> a.intersection('a3')
        Axis([], 'a')
        >>> a.intersection(Axis('a=a1..a3'))
        Axis(['a1', 'a2'], 'a')
        >>> a.intersection('a1..a3')
        Axis(['a1', 'a2'], 'a')
        >>> a.intersection(['a1', 'a2', 'a3'])
        Axis(['a1', 'a2'], 'a')
        >>> Axis('a=a0,a1,a0').intersection('a1,a0')
        Axis(['a0', 'a1', 'a0'], 'a')
        """
        non_string_scalar = np.isscalar(other) and not isinstance(other, str)
        other = [other] if non_string_scalar else _to_ticks(other)
        to_keep = set(other)
        return Axis([label for label in self.labels if label in to_keep], self.name)

    def difference(self, other) -> 'Axis':
        r"""Return axis with the (set) difference of this axis labels and other labels.

        In other words, this will use labels from this axis if they are not in other. Labels relative order will be
        kept intact.

        Parameters
        ----------
        other : Axis, Group or any sequence of labels
            other labels

        Returns
        -------
        Axis

        Examples
        --------
        >>> a = Axis('a=a0..a2')
        >>> a.difference('a1')
        Axis(['a0', 'a2'], 'a')
        >>> a.difference('a3')
        Axis(['a0', 'a1', 'a2'], 'a')
        >>> a.difference(Axis('a=a1..a3'))
        Axis(['a0'], 'a')
        >>> a.difference('a1..a3')
        Axis(['a0'], 'a')
        >>> a.difference(['a1', 'a2', 'a3'])
        Axis(['a0'], 'a')
        """
        non_string_scalar = np.isscalar(other) and not isinstance(other, str)
        other = [other] if non_string_scalar else _to_ticks(other)
        to_drop = set(other)
        return Axis([label for label in self.labels if label not in to_drop], self.name)

    def align(self, *other, join='outer') -> 'Axis':
        r"""Align axis with other object using specified join method.

        Parameters
        ----------
        *other : Axis or label sequence
        join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
            Defaults to 'outer'.

        Returns
        -------
        Axis
            Aligned axis

        See Also
        --------
        Array.align

        Examples
        --------
        >>> axis1 = Axis('a=a0..a2')
        >>> axis2 = Axis('a=a1..a3')
        >>> axis1.align(axis2)
        Axis(['a0', 'a1', 'a2', 'a3'], 'a')
        >>> axis1.align(axis2, join='inner')
        Axis(['a1', 'a2'], 'a')
        >>> axis1.align(axis2, join='left')
        Axis(['a0', 'a1', 'a2'], 'a')
        >>> axis1.align(axis2, join='right')
        Axis(['a1', 'a2', 'a3'], 'a')
        >>> axis1.align(axis2, join='exact')   # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: align method with join='exact' expected
        Axis(['a0', 'a1', 'a2'], 'a') to be equal to Axis(['a1', 'a2', 'a3'], 'a')
        """
        bad_objs = [obj for obj in other if not isinstance(obj, Axis)]
        if bad_objs:
            for obj in bad_objs:
                obj_type = type(obj).__name__
                warnings.warn(f"aligning an Axis to a non-Axis object "
                              f"({obj_type}) is deprecated. Please convert to "
                              f"an Axis first.", FutureWarning, stacklevel=2)
            other = [obj if isinstance(obj, Axis) else Axis(obj)
                     for obj in other]
        return align_axes((self, *other), join=join)

    def to_hdf(self, filepath, key=None) -> None:
        r"""
        Write axis to a HDF file.

        A HDF file can contain multiple axes.
        The 'key' parameter is a unique identifier for the axis.

        Parameters
        ----------
        filepath : str
            Path where the hdf file has to be written.
        key : str or Group, optional
            Key (path) of the axis within the HDF file (see Notes below).
            If None, the name of the axis is used.
            Defaults to None.

        Notes
        -----
        Objects stored in a HDF file can be grouped together in `HDF groups`.
        If an object 'my_obj' is stored in a HDF group 'my_group',
        the key associated with this object is then 'my_group/my_obj'.
        Be aware that a HDF group can have subgroups.

        Examples
        --------
        >>> a = Axis("a=a0..a2")

        Save axis

        >>> # by default, the key is the name of the axis
        >>> a.to_hdf('test.h5')            # doctest: +SKIP

        Save axis with a specific key

        >>> a.to_hdf('test.h5', 'a')       # doctest: +SKIP

        Save axis in a specific HDF group

        >>> a.to_hdf('test.h5', 'axes/a')  # doctest: +SKIP
        """
        if key is None:
            if self.name is None:
                raise ValueError("Argument key must be provided explicitly in case of anonymous axis")
            key = self.name
        key = _translate_group_key_hdf(key)
        dtype_kind = self.labels.dtype.kind
        data = np.char.encode(self.labels, 'utf-8') if dtype_kind == 'U' else self.labels
        s = pd.Series(data=data, name=self.name)
        with LHDFStore(filepath) as store:
            store.put(key, s)
            store.get_storer(key).attrs.type = 'Axis'
            store.get_storer(key).attrs.dtype_kind = dtype_kind
            store.get_storer(key).attrs.wildcard = self.iswildcard

    @property
    def dtype(self) -> np.dtype:
        return self._labels.dtype

    def ignore_labels(self) -> 'Axis':
        r"""Return a wildcard axis with the same name and length than this axis.

        Useful when you want to apply operations between two arrays with the same shape but incompatible axes
        (different labels).

        Returns
        -------
        Axis

        Examples
        --------
        >>> a = Axis('a=a1,a2')
        >>> a
        Axis(['a1', 'a2'], 'a')
        >>> a.ignore_labels()
        Axis(2, 'a')
        """
        return Axis(len(self), self.name)


def align_axes(axes: Sequence[Axis], join: str = 'outer') -> Axis:
    if not all(isinstance(axis, Axis) for axis in axes):
        raise TypeError("all objects to align must be Axis objects")

    if join not in {'outer', 'inner', 'left', 'right', 'exact'}:
        raise ValueError(f"join must be one of 'outer', 'inner', 'left', "
                         f"'right' or 'exact', got {join!r}")

    names = [axis.name for axis in axes]
    first_name = first((name for name in names if name is not None),
                       default=None)
    if first_name is not None:
        if not all(name is None or name == first_name for name in names):
            raise ValueError("In align, all axes must be anonymous or "
                             "have the same name: "
                             f"{', '.join(repr(name) for name in names)}")

    def join_left(axis1, axis2):
        return axis1
    def join_right(axis1, axis2):
        return axis2
    def join_exact(axis1, axis2):
        if not axis1.equals(axis2):
            raise ValueError(f"align method with join='exact' expected "
                             f"{axis1!r} to be equal to {axis2!r}")
        else:
            return axis1
    if join == 'outer':
        join_labels_func = Axis.union
    elif join == 'inner':
        join_labels_func = Axis.intersection
    elif join == 'left':
        join_labels_func = join_left
    elif join == 'right':
        join_labels_func = join_right
    else:
        assert join == 'exact'
        join_labels_func = join_exact
    aligned_axis = axes[0]
    for axis in axes[1:]:
        aligned_axis = join_labels_func(aligned_axis, axis)
    return aligned_axis


def _make_axis(obj) -> Axis:
    if isinstance(obj, Axis):
        return obj
    elif isinstance(obj, tuple):
        assert len(obj) == 2
        labels, name = obj
        return Axis(labels, name)
    elif isinstance(obj, Group):
        return Axis(obj.eval(), obj.axis)
    else:
        # int, str, list, ndarray
        return Axis(obj)


# not using a dict because it does not support indices-based getitem
# not using namedtuple because we have to know the fields in advance (it is a one-off class) and we need more
# functionality than just a named tuple
class AxisCollection:
    __slots__ = ('_list', '_map')
    r"""
    Represents a collection of axes.

    Parameters
    ----------
    axes : sequence of Axis or int or tuple or str, optional
        An axis can be given as an Axis object, an int or a
        tuple (labels, name) or a string of the kind
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
    >>> age = Axis(range(10), 'age')
    >>> AxisCollection([3, age, (['M', 'F'], 'sex'), 'time = 2007, 2008, 2009, 2010'])
    AxisCollection([
        Axis(3, None),
        Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'age'),
        Axis(['M', 'F'], 'sex'),
        Axis([2007, 2008, 2009, 2010], 'time')
    ])
    >>> AxisCollection('age=0..9; sex=M,F; time=2007..2010')
    AxisCollection([
        Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'age'),
        Axis(['M', 'F'], 'sex'),
        Axis([2007, 2008, 2009, 2010], 'time')
    ])
    """
    def __init__(self, axes=None):
        if axes is None:
            axes = []
        elif isinstance(axes, (int, Group, Axis)):
            axes = [axes]
        elif isinstance(axes, str):
            axes = [axis.strip() for axis in axes.split(';')]

        axes = [_make_axis(axis) for axis in axes]
        assert all(isinstance(a, Axis) for a in axes)
        # check for duplicate axes
        dupe_axes = list(duplicates(axes))
        if dupe_axes:
            axis = dupe_axes[0]
            raise ValueError(f"Cannot have multiple occurrences of the same axis object in a collection ('{axis.name}'"
                             f" -- {axis.labels_summary()} with id {id(axis):d}). Several axes with the same name are "
                             f"allowed though (but not recommended).")
        self._list = axes
        self._map = {axis.name: axis for axis in axes if axis.name is not None}

        # # check dupes on each axis
        # for axis in axes:
        #     axis_dupes = list(duplicates(axis.labels))
        #     if axis_dupes:
        #         dupe_labels = ', '.join(str(l) for l in axis_dupes)
        #         warnings.warn(f"duplicate labels found for axis {axis.name}: {dupe_labels}",
        #                       category=UserWarning, stacklevel=2)
        #
        # # check dupes between axes. Using unique to not spot the dupes within the same axis that we just displayed.
        # all_labels = chain(*[np.unique(axis.labels) for axis in axes])
        # dupe_labels = list(duplicates(all_labels))
        # if dupe_labels:
        #     label_axes = [(label, ', '.join(display_name for axis, display_name in zip(axes, self.display_names)
        #                                     if label in axis))
        #                   for label in dupe_labels]
        #     dupes = '\n'.join(f"{label} is valid in {{{axes}}}" for label, axes in label_axes)
        #     warnings.warn(f"ambiguous labels found:\n{dupes}", category=UserWarning, stacklevel=5)

    def __dir__(self):
        # called by dir() and tab-completion at the interactive prompt, must return a list of any valid getattr key.
        # dir() takes care of sorting but not uniqueness, so we must ensure that.
        names = set(axis.name for axis in self._list if axis.name is not None)
        return list(set(dir(self.__class__)) | names)

    def __iter__(self) -> Iterable[Axis]:
        return iter(self._list)

    # TODO: move a few doctests to unit tests
    def iter_labels(self, axes=None, ascending=True) -> Sequence[Tuple[IGroup, ...]]:
        r"""Return a view of the axes labels.

        Parameters
        ----------
        axes : int, str or Axis or tuple of them, optional
            Axis or axes along which to iterate and in which order. Defaults to None (all axes in the order they are
            in the collection).
        ascending : bool, optional
            Whether to iterate the axes in ascending order (from start to end). Defaults to True.

        Returns
        -------
        Sequence
            An object you can iterate (loop) on and index by position.

        Examples
        --------
        >>> from larray import ndtest
        >>> axes = ndtest((2, 2)).axes
        >>> axes
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1'], 'b')
        ])
        >>> axes.iter_labels()[0]
        (a.i[0], b.i[0])
        >>> for index in axes.iter_labels():
        ...     print(index)
        (a.i[0], b.i[0])
        (a.i[0], b.i[1])
        (a.i[1], b.i[0])
        (a.i[1], b.i[1])
        >>> axes.iter_labels(ascending=False)[0]
        (a.i[1], b.i[1])
        >>> for index in axes.iter_labels(ascending=False):
        ...     print(index)
        (a.i[1], b.i[1])
        (a.i[1], b.i[0])
        (a.i[0], b.i[1])
        (a.i[0], b.i[0])
        >>> axes.iter_labels(('b', 'a'))[0]
        (b.i[0], a.i[0])
        >>> for index in axes.iter_labels(('b', 'a')):
        ...     print(index)
        (b.i[0], a.i[0])
        (b.i[0], a.i[1])
        (b.i[1], a.i[0])
        (b.i[1], a.i[1])
        >>> axes.iter_labels('b')[0]
        (b.i[0],)
        >>> for index in axes.iter_labels('b'):
        ...     print(index)
        (b.i[0],)
        (b.i[1],)
        """
        axes = self if axes is None else self[axes]
        if not isinstance(axes, AxisCollection):
            axes = (axes,)
        # we need .i because Product uses len and [] on axes and not iter; and [] creates LGroup and not IGroup
        p = Product([axis.i for axis in axes])
        if not ascending:
            p = p[::-1]
        return p

    def _combined_iflat(self, flat_key, sep='_'):
        """Return the combined axes AxisCollection indexed by flat_key.

        Parameters
        ----------
        flat_key : int or array of int
            Indices-based key for indexing the combined axis.
        sep : str, optional
            Separator to use for creating combined axis name and labels. Defaults to '_'.

        Returns
        -------
        AxisCollection

        Examples
        --------
        >>> axes = AxisCollection("a=a0..a1;b=b0..b2")
        >>> axes._combined_iflat([0, 2, 3])
        AxisCollection([
            Axis(['a0_b0', 'a0_b2', 'a1_b0'], 'a_b')
        ])
        """
        nd_key = np.unravel_index(flat_key, self.shape)
        return self._adv_keys_to_combined_axes(nd_key, sep=sep)

    def _iflat(self, key):
        """Return labels for each axis corresponding to indexing an array with the AxisCollection axes.

        Parameters
        ----------
        key : int or array of int
            Indices-based key for indexing the combined axis.

        Returns
        -------
        tuple

        Examples
        --------
        >>> axes = AxisCollection("a=a0..a1;b=b0..b2")
        >>> axes._iflat([0, 2, 3])
        (array(['a0', 'a0', 'a1'],
              dtype='<U2'), array(['b0', 'b2', 'b0'],
              dtype='<U2'))
        """
        axes_indices = np.unravel_index(key, self.shape)
        return tuple(axis.labels[axis_indices] for axis_indices, axis in zip(axes_indices, self))

    def __getattr__(self, key) -> Axis:
        try:
            return self._map[key]
        except KeyError:
            return self.__getattribute__(key)

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before _map exists, which leads to
    # an infinite recursion)
    def __getstate__(self) -> List[Axis]:
        return self._list

    def __setstate__(self, state):
        self._list = state
        self._map = {axis.name: axis for axis in state if axis.name is not None}

    def __getitem__(self, key) -> Union[Axis, 'AxisCollection']:
        if isinstance(key, Axis):
            try:
                key = self.index(key)
            # transform ValueError to KeyError
            except ValueError:
                if key.name is None:
                    raise KeyError(f"axis '{key}' not found in {self}")
                else:
                    # we should NOT check that the object is the same, so that we can use AxisReference objects to
                    # target real axes
                    key = key.name

        if isinstance(key, (int, np.integer)):
            return self._list[key]
        elif isinstance(key, (tuple, list)):
            if any(axis is Ellipsis for axis in key):
                ellipsis_idx = index_by_id(key, Ellipsis)
                # going through lists (instead of doing self[key[:ellipsis_idx]] to avoid problems with anonymous axes
                before_ellipsis = [self[k] for k in key[:ellipsis_idx]]
                after_ellipsis = [self[k] for k in key[ellipsis_idx + 1:]]
                selected_axes = set(before_ellipsis + after_ellipsis)
                ellipsis_axes = [axis for axis in self._list if axis not in selected_axes]
                return AxisCollection(before_ellipsis + ellipsis_axes + after_ellipsis)
            # XXX: also use get_by_pos if tuple/list of Axis?
            return AxisCollection([self[k] for k in key])
        elif isinstance(key, AxisCollection):
            return AxisCollection([self.get_by_pos(k, i)
                                   for i, k in enumerate(key)])
        elif isinstance(key, slice):
            return AxisCollection(self._list[key])
        elif key is None:
            raise KeyError(f"axis '{key}' not found in {self}")
        else:
            assert isinstance(key, str), type(key)
            if key in self._map:
                return self._map[key]
            else:
                raise KeyError(f"axis '{key}' not found in {self}")

    def _ipython_key_completions_(self) -> List[str]:
        return list(self._map.keys())

    # XXX: I wonder if this whole positional crap should really be part of AxisCollection or the default behavior.
    # It could either be moved to make_numpy_broadcastable or made non default
    def get_by_pos(self, key, i) -> Axis:
        r"""
        Return axis corresponding to a key, or to position i if the key has no name and key object not found.

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
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection([age, sex, time])
        >>> col.get_by_pos('sex', 1)
        Axis(['M', 'F'], 'sex')
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
                        raise ValueError(f"axis {res} is not compatible with {key}")
                # XXX: KeyError instead?
                raise ValueError(f"axis {key} not found in {self}")
        else:
            return self[key]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            assert isinstance(value, (tuple, list, AxisCollection))

            def slice_bound(bound):
                if bound is None or isinstance(bound, int):
                    # out of bounds integer bounds are allowed in slice setitem so we cannot use .index
                    return bound
                else:
                    return self.index(bound)
            start_idx = slice_bound(key.start)
            # XXX: we might want to make the stop bound inclusive, which makes more sense for label bounds (but
            #      prevents inserts via setitem)
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
            if not isinstance(value, Axis):
                value = Axis(value)
            idx = self.index(key)
            step = 1 if idx >= 0 else -1
            self[idx:idx + step:step] = [value]

    def __delitem__(self, key) -> None:
        if isinstance(key, slice):
            self[key] = []
        else:
            idx = self.index(key)
            axis = self._list.pop(idx)
            if axis.name is not None:
                del self._map[axis.name]

    def union(self, *args, validate=True, replace_wildcards=True) -> 'AxisCollection':
        result = self[:]
        for a in args:
            if not isinstance(a, AxisCollection):
                a = AxisCollection(a)
            result.extend(a, validate=validate, replace_wildcards=replace_wildcards)
        return result
    __or__ = union
    # TODO: deprecate method (should use | or union instead)
    __add__ = union

    # TODO: deprecate method (should use | or union instead) but implement __ror__ !)
    def __radd__(self, other) -> 'AxisCollection':
        result = AxisCollection(other)
        result.extend(self)
        return result

    def __and__(self, other) -> 'AxisCollection':
        r"""
        Return the intersection of this collection and other.
        """
        if not isinstance(other, AxisCollection):
            other = AxisCollection(other)

        # XXX: add iscompatible when matching by position?
        # TODO: move this to a class method (possibly private) so that we are sure we use same heuristic than in .extend
        def contains(col, i, axis):
            return axis in col or (axis.name is None and i in col)

        return AxisCollection([axis for i, axis in enumerate(self) if contains(other, i, axis)])

    def __eq__(self, other: Any) -> bool:
        r"""
        Other collection compares equal if all axes compare equal and in the same order. Works with a list.
        """
        if self is other:
            return True
        elif isinstance(other, AxisCollection):
            other = other._list
        elif not isinstance(other, (list, tuple)):
            return False
        return len(self._list) == len(other) and all(a.equals(b) for a, b in zip(self._list, other))

    # for python2, we need to define it explicitly
    def __ne__(self, other) -> bool:
        return not self == other

    def __contains__(self, key) -> bool:
        if isinstance(key, int):
            return -len(self) <= key < len(self)
        elif isinstance(key, Axis):
            # the special case is just a performance optimization to avoid scanning through the whole list
            if key.name is not None:
                return key.name in self._map
            else:
                try:
                    self.index(key)
                    return True
                except ValueError:
                    return False
        # key can be anything, it should just return False in case of weird types
        return key in self._map

    def isaxis(self, value) -> bool:
        r"""
        Test if input is an Axis object or the name of an axis contained in self.

        Parameters
        ----------
        value : Axis or str
            Input axis or string

        Returns
        -------
        bool
            True if input is an Axis object or the name of an axis contained in the current AxisCollection instance,
            False otherwise.

        Examples
        --------
        >>> a = Axis('a=a0,a1')
        >>> b = Axis('b=b0,b1')
        >>> col = AxisCollection([a, b])
        >>> col.isaxis(a)
        True
        >>> col.isaxis('b')
        True
        >>> col.isaxis('c')
        False
        """
        # this is tricky. 0 and 1 can be both axes indices and axes ticks.
        # not sure what's worse:
        # 1) disallow aggregates(axis_num): users could still use arr.sum(arr.axes[0])
        #    we could also provide an explicit kwarg (ie this would effectively forbid having an axis named "axis").
        #    arr.sum(axis=0). I think this is the sanest option. The error message in case we use it without the
        #    keyword needs to be clearer though.
        return isinstance(value, Axis) or (isinstance(value, str) and value in self)
        # 2) slightly inconsistent API: allow aggregate over single labels if they are string, but not int
        #    arr.sum(0) would sum on the first axis, but arr.sum('M') would
        #    sum a single tick. I don't like this option.
        # 3) disallow single tick aggregates. Single labels make little sense in the context of an aggregate,
        #    but you don't always know/want to differenciate the code in that case anyway.
        #    It would be annoying for e.g. Brussels
        # 4) give priority to axes,
        #    arr.sum(0) would sum on the first axis but arr.sum(5) would
        #    sum a single tick (assuming there is a int axis and less than six axes).
        # return value in self

    def __len__(self) -> int:
        return len(self._list)
    ndim = property(__len__)

    def __str__(self) -> str:
        names = ', '.join(self.display_names)
        return f"{{{names}}}"

    def __repr__(self) -> str:
        if len(self):
            repr_per_axis = [repr(axis) for axis in self._list]
            axes_repr = ',\n    '.join(repr_per_axis)
            axes_repr = f"\n    {axes_repr}\n"
        else:
            axes_repr = ""
        return f"AxisCollection([{axes_repr}])"

    # TODO: kill name argument (does not seem to be used anywhere
    def get(self, key, default=None, name=None) -> Axis:
        r"""
        Return axis corresponding to key. If not found, the argument `name` is used to create a new Axis.
        If `name` is None, the `default` axis is then returned.

        Parameters
        ----------
        key : key
            Key corresponding to an axis of the current AxisCollection.
        default : axis, optional
            Default axis to return if key doesn't correspond to any axis of the collection and argument `name` is None.
        name : str, optional
            If key doesn't correspond to any axis of the collection, a new Axis with this name is created and returned.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection([age, time])
        >>> col.get('time')
        Axis([2007, 2008, 2009, 2010], 'time')
        >>> col.get('sex', sex)
        Axis(['M', 'F'], 'sex')
        >>> col.get('nb_children', None, 'nb_children')
        Axis(1, 'nb_children')
        """
        # XXX: use if key in self?
        try:
            return self[key]
        except KeyError:
            if name is None:
                return default
            else:
                return Axis(1, name)

    def get_all(self, key) -> 'AxisCollection':
        r"""
        Return all axes from key if present and length 1 wildcard axes otherwise.

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
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> city = Axis(['London', 'Paris', 'Rome'], 'city')
        >>> col = AxisCollection([age, sex, time])
        >>> col2 = AxisCollection([age, city, time])
        >>> col.get_all(col2)
        AxisCollection([
            Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'age'),
            Axis(1, 'city'),
            Axis([2007, 2008, 2009, 2010], 'time')
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
                    return Axis(1, k.name if k.name is not None else i)

        return AxisCollection([get_pos_default(k, i) for i, k in enumerate(key)])

    def keys(self) -> List[str]:
        r"""
        Return list of all axis names.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age, sex, time]).keys()
        ['age', 'sex', 'time']
        """
        # XXX: include id/num for anonymous axes? I think I should
        return [a.name for a in self._list]

    def pop(self, axis=-1) -> Axis:
        r"""
        Remove and return an axis.

        Parameters
        ----------
        axis : key, optional
            Axis to remove and return. Default value is -1 (last axis).

        Returns
        -------
        Axis
            If no argument is provided, the last axis is removed and returned.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection([age, sex, time])
        >>> col.pop('age')
        Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'age')
        >>> col
        AxisCollection([
            Axis(['M', 'F'], 'sex'),
            Axis([2007, 2008, 2009, 2010], 'time')
        ])
        >>> col.pop()
        Axis([2007, 2008, 2009, 2010], 'time')
        """
        axis = self[axis]
        del self[axis]
        return axis

    def append(self, axis) -> None:
        r"""
        Append axis at the end of the collection.

        Parameters
        ----------
        axis : Axis
            Axis to append.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection([age, sex])
        >>> col.append(time)
        >>> col
        AxisCollection([
            Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'age'),
            Axis(['M', 'F'], 'sex'),
            Axis([2007, 2008, 2009, 2010], 'time')
        ])
        """
        self[len(self):len(self)] = [axis]

    def check_compatible(self, axes) -> None:
        r"""
        Check if axes passed as argument are compatible with those contained in the collection.
        Raises ValueError if not.

        See Also
        --------
        Axis.iscompatible
        """
        for i, axis in enumerate(axes):
            local_axis = self.get_by_pos(axis, i)
            if not local_axis.iscompatible(axis):
                raise ValueError(f"incompatible axes:\n{axis!r}\nvs\n{local_axis!r}")

    # XXX: deprecate method (functionality is duplicated in union)?
    #      I am not so sure anymore we need to actually deprecate the method: having both methods with the same
    #      semantic like we currently have is useless indeed but I think we should have both a set-like method (union)
    #      and the possibility to add an axis unconditionally (append or extend). That is, add an axis, even if that
    #      name already exists. This is especially important for anonymous axes (see my comments in stack for example)
    # TODO: deprecate validate argument (unused)
    # TODO: deprecate replace_wildcards argument (unused)
    def extend(self, axes, validate=True, replace_wildcards=False) -> None:
        r"""
        Extend the collection by appending the axes from `axes`.

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
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection(age)
        >>> col.extend([sex, time])
        >>> col
        AxisCollection([
            Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'age'),
            Axis(['M', 'F'], 'sex'),
            Axis([2007, 2008, 2009, 2010], 'time')
        ])
        """
        # axes should be a sequence
        if not isinstance(axes, (tuple, list, AxisCollection)):
            raise TypeError(f"AxisCollection can only be extended by a sequence of Axis, not {type(axes).__name__}")
        # check that common axes are the same
        # if validate:
        #     self.check_compatible(axes)

        # TODO: factorize with get_by_pos
        def get_axis(col, i, axis) -> Optional[Axis]:
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
                    raise ValueError(f"incompatible axes:\n{axis!r}\nvs\n{old_axis!r}")
                if replace_wildcards and old_axis.iswildcard:
                    self[old_axis] = axis

    def index(self, axis, compatible=False) -> int:
        r"""
        Return the index of axis.

        `axis` can be a name or an Axis object (or an index). If the Axis object itself exists in the list, index()
        will return it. Otherwise, it will return the index of the local axis with the same name than the key (whether
        it is compatible or not).

        Parameters
        ----------
        axis : Axis or int or str
            Can be the axis itself or its position (returned if represents a valid index) or its name.
        compatible : bool, optional
            If axis is an Axis, whether to find an exact match (using Axis.equals) or any
            compatible axis (using Axis.iscompatible)

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
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection([age, sex, time])
        >>> col.index(time)
        2
        >>> col.index('sex')
        1
        """
        if isinstance(axis, AxisReference):
            axis = axis.name
        # not using an elif because name can be an int
        if isinstance(axis, int):
            if -len(self) <= axis < len(self):
                return axis
            else:
                raise ValueError(f"axis {axis} is not in collection")
        elif isinstance(axis, Axis):
            try:
                # 1) first look for that particular axis object

                # This avoids testing labels of each axis and makes sure the result is correct even if there are
                # several axes with the same name and labels.
                return index_by_id(self._list, axis)
            except ValueError:
                # 2) look for an axis with the same name and labels using axis.equals

                # This makes sure that if there are several axes with the same name but different labels, it returns
                # the index of the one with the correct labels. This is especially important for anonymous axes but is
                # also useful for other axes. Note that this shouldn't be too slow as labels will only be actually
                # checked if the name match.

                if compatible:
                    for i, item in enumerate(self._list):
                        if item.iscompatible(axis):
                            return i
                else:
                    # We cannot use self._list.index because it use Axis.__eq__ which produces an Array
                    for i, item in enumerate(self._list):
                        if item.equals(axis):
                            return i

                # 3) otherwise look for any axis with the same name
                name = axis.name
        else:
            name = axis
        if name is None:
            raise ValueError(f"{axis!r} is not in collection")
        return self.names.index(name)

    # XXX: we might want to return a new AxisCollection (same question for other inplace operations:
    # append, extend, pop, __delitem__, __setitem__)
    def insert(self, index, axis) -> None:
        r"""
        Insert axis before index.

        Parameters
        ----------
        index : int
            position of the inserted axis.
        axis : Axis
            axis to insert.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> col = AxisCollection([age, time])
        >>> col.insert(1, sex)
        >>> col
        AxisCollection([
            Axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'age'),
            Axis(['M', 'F'], 'sex'),
            Axis([2007, 2008, 2009, 2010], 'time')
        ])
        """
        self[index:index] = [axis]

    def copy(self) -> 'AxisCollection':
        r"""
        Return a copy.
        """
        return self[:]

    def rename(self, renames=None, to=None, **kwargs) -> 'AxisCollection':
        r"""Rename axes of the collection.

        Parameters
        ----------
        renames : axis ref or dict {axis ref: str} or list of tuple (axis ref, str), optional
            Rename to apply. If a single axis reference is given, the `to` argument must be used.
        to : str or Axis, optional
            New name if `renames` contains a single axis reference.
        **kwargs : str or Axis
            New name for each axis given as a keyword argument.

        Returns
        -------
        AxisCollection
            collection with axes renamed.

        Examples
        --------
        >>> nat = Axis('nat=BE,FO')
        >>> sex = Axis('sex=M,F')
        >>> axes = AxisCollection([nat, sex])
        >>> axes
        AxisCollection([
            Axis(['BE', 'FO'], 'nat'),
            Axis(['M', 'F'], 'sex')
        ])
        >>> axes.rename(nat, 'nat2')
        AxisCollection([
            Axis(['BE', 'FO'], 'nat2'),
            Axis(['M', 'F'], 'sex')
        ])
        >>> axes.rename(nat='nat2', sex='sex2')
        AxisCollection([
            Axis(['BE', 'FO'], 'nat2'),
            Axis(['M', 'F'], 'sex2')
        ])
        >>> axes.rename([('nat', 'nat2'), ('sex', 'sex2')])
        AxisCollection([
            Axis(['BE', 'FO'], 'nat2'),
            Axis(['M', 'F'], 'sex2')
        ])
        >>> axes.rename({'nat': 'nat2', 'sex': 'sex2'})
        AxisCollection([
            Axis(['BE', 'FO'], 'nat2'),
            Axis(['M', 'F'], 'sex2')
        ])
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
        renames = {self[k]: v for k, v in items}
        return AxisCollection([a.rename(renames[a]) if a in renames else a
                               for a in self])

    # XXX: what's the point in supporting a list of Axis or AxisCollection in axes_to_replace?
    #      it is used in Array.set_axes but if it is only there, shouldn't the support for that be
    #      moved there?
    def replace(self, axes_to_replace=None, new_axis=None, inplace=False, **kwargs) -> 'AxisCollection':
        r"""Replace one, several or all axes of the collection.

        Parameters
        ----------
        axes_to_replace : axis ref or dict {axis ref: axis} or list of (tuple or Axis) or AxisCollection, optional
            Axes to replace. If a single axis reference is given, the `new_axis` argument must be provided.
            If a list of Axis or an AxisCollection is given, all axes will be replaced by the new ones.
            In that case, the number of new axes must match the number of the old ones. Defaults to None.
            If a list of tuple is given, it must be pairs of (reference to old axis, new axis).
        new_axis : axis ref, optional
            New axis if `axes_to_replace` contains a single axis reference. Defaults to None.
        inplace : bool, optional
            Whether to modify the original object or return a new AxisCollection and leave the original intact.
            Defaults to False.
        **kwargs : Axis
            New axis for each axis to replace given as a keyword argument.

        Returns
        -------
        AxisCollection
            AxisCollection with axes replaced.

        Examples
        --------
        >>> from larray import ndtest
        >>> axes = ndtest((2, 3)).axes
        >>> axes
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])
        >>> row = Axis(['r0', 'r1'], 'row')
        >>> column = Axis(['c0', 'c1', 'c2'], 'column')

        Replace one axis (second argument `new_axis` must be provided)

        >>> axes.replace(X.a, row)  # doctest: +SKIP
        >>> # or
        >>> axes.replace(X.a, "row=r0,r1")
        AxisCollection([
            Axis(['r0', 'r1'], 'row'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])

        Replace several axes (keywords, list of tuple or dictionary)

        >>> axes.replace(a=row, b=column)  # doctest: +SKIP
        >>> # or
        >>> axes.replace(a="row=r0,r1", b="column=c0,c1,c2")  # doctest: +SKIP
        >>> # or
        >>> axes.replace([(X.a, row), (X.b, column)])  # doctest: +SKIP
        >>> # or
        >>> axes.replace({X.a: row, X.b: column})
        AxisCollection([
            Axis(['r0', 'r1'], 'row'),
            Axis(['c0', 'c1', 'c2'], 'column')
        ])

        Replace all axes (list of axes or AxisCollection)

        >>> axes.replace([row, column])
        AxisCollection([
            Axis(['r0', 'r1'], 'row'),
            Axis(['c0', 'c1', 'c2'], 'column')
        ])
        >>> arr = ndtest([row, column])
        >>> axes.replace(arr.axes)
        AxisCollection([
            Axis(['r0', 'r1'], 'row'),
            Axis(['c0', 'c1', 'c2'], 'column')
        ])
        """
        if isinstance(axes_to_replace, zip):
            axes_to_replace = list(axes_to_replace)

        if isinstance(axes_to_replace, (list, AxisCollection)) and \
                all([isinstance(axis, Axis) for axis in axes_to_replace]):
            if len(axes_to_replace) != len(self):
                raise ValueError(f'{len(axes_to_replace)} axes given as argument, expected {len(self)}')
            axes = axes_to_replace
        else:
            axes = self if inplace else self[:]
            if isinstance(axes_to_replace, dict):
                items = list(axes_to_replace.items())
            elif isinstance(axes_to_replace, list):
                assert all([isinstance(item, tuple) and len(item) == 2 for item in axes_to_replace])
                items = axes_to_replace[:]
            elif isinstance(axes_to_replace, (str, Axis, int)):
                items = [(axes_to_replace, new_axis)]
            else:
                items = []
            items += kwargs.items()
            for old, new in items:
                axes[old] = new
        if inplace:
            return self
        else:
            return AxisCollection(axes)

    def set_labels(self, axis=None, labels=None, inplace=False, **kwargs) -> 'AxisCollection':
        r"""Replace the labels of one or several axes.

        Parameters
        ----------
        axis : Axis, str, int or dict
            Axis for which we want to replace labels, or mapping {axis: changes} where changes can either be the
            complete list of labels, a mapping {old_label: new_label} or a function to transform labels.
            If there is no ambiguity (two or more axes have the same labels), `axis` can be a direct mapping
            {old_label: new_label}.
        labels : int, str, iterable or mapping or function, optional
            Integer or list of values usable as the collection of labels for an Axis. If this is mapping, it must be
            {old_label: new_label}. If it is a function, it must be a function accepting a single argument (a
            label) and returning a single value. This argument must not be used if axis is a mapping.
        inplace : bool, optional
            Whether to modify the original object or return a new AxisCollection and leave the original intact.
            Defaults to False.
        **kwargs :
            `axis`=`labels` for each axis you want to set labels.

        Returns
        -------
        AxisCollection
            AxisCollection with modified labels.

        Warnings
        --------
        Not passing a mapping but the complete list of new labels as the 'labels' argument must be done with caution.
        Make sure that the order of new labels corresponds to the exact same order of previous labels.

        Examples
        --------
        >>> from larray import ndtest
        >>> axes = AxisCollection('nat=BE,FO;sex=M,F')
        >>> axes
        AxisCollection([
            Axis(['BE', 'FO'], 'nat'),
            Axis(['M', 'F'], 'sex')
        ])
        >>> axes.set_labels('sex', ['Men', 'Women'])
        AxisCollection([
            Axis(['BE', 'FO'], 'nat'),
            Axis(['Men', 'Women'], 'sex')
        ])

        when passing a single string as labels, it will be interpreted to create the list of labels, so that one can
        use the same syntax than during axis creation.

        >>> axes.set_labels('sex', 'Men,Women')
        AxisCollection([
            Axis(['BE', 'FO'], 'nat'),
            Axis(['Men', 'Women'], 'sex')
        ])

        to replace only some labels, one must give a mapping giving the new label for each label to replace

        >>> axes.set_labels('sex', {'M': 'Men'})
        AxisCollection([
            Axis(['BE', 'FO'], 'nat'),
            Axis(['Men', 'F'], 'sex')
        ])

        to transform labels by a function, use any function accepting and returning a single argument:

        >>> axes.set_labels('nat', str.lower)
        AxisCollection([
            Axis(['be', 'fo'], 'nat'),
            Axis(['M', 'F'], 'sex')
        ])

        to replace labels for several axes at the same time, one should give a mapping giving the new labels for each
        changed axis

        >>> axes.set_labels({'sex': 'Men,Women', 'nat': 'Belgian,Foreigner'})
        AxisCollection([
            Axis(['Belgian', 'Foreigner'], 'nat'),
            Axis(['Men', 'Women'], 'sex')
        ])

        or use keyword arguments

        >>> axes.set_labels(sex='Men,Women', nat='Belgian,Foreigner')
        AxisCollection([
            Axis(['Belgian', 'Foreigner'], 'nat'),
            Axis(['Men', 'Women'], 'sex')
        ])

        one can also replace some labels in several axes by giving a mapping of mappings

        >>> axes.set_labels({'sex': {'M': 'Men'}, 'nat': {'BE': 'Belgian'}})
        AxisCollection([
            Axis(['Belgian', 'FO'], 'nat'),
            Axis(['Men', 'F'], 'sex')
        ])

        when there is no ambiguity (two or more axes have the same labels), it is possible to give a mapping
        between old and new labels

        >>> axes.set_labels({'M': 'Men', 'BE': 'Belgian'})
        AxisCollection([
            Axis(['Belgian', 'FO'], 'nat'),
            Axis(['Men', 'F'], 'sex')
        ])
        >>> axes.set_labels({'M': 'Men', 'F': 'Women'})
        AxisCollection([
            Axis(['BE', 'FO'], 'nat'),
            Axis(['Men', 'Women'], 'sex')
        ])
        """
        if axis is None:
            changes = {}
        elif isinstance(axis, dict):
            changes = axis
        elif isinstance(axis, (str, Axis, int)):
            changes = {axis: labels}
        else:
            raise ValueError("Expected None or a string/int/Axis/dict instance for axis argument")
        changes.update(kwargs)

        # handle {label1: new_label1, label2: new_label2}
        if any(axis_ref not in self for axis_ref in changes.keys()):
            changes_per_axis = defaultdict(list)
            for selection, new_labels in changes.items():
                group = self._guess_axis(selection)
                axis = group.axis
                changes_per_axis[axis].append((selection, new_labels))
            changes = {axis: dict(axis_changes) for axis, axis_changes in changes_per_axis.items()}

        new_axes = []
        for old_axis, axis_changes in changes.items():
            real_axis = self[old_axis]
            if isinstance(axis_changes, dict):
                new_axis = real_axis.replace(axis_changes)
            # TODO: we should implement the non-dict behavior in Axis.replace, so that we can simplify this code to:
            # new_axes = [self[old_axis].replace(axis_changes) for old_axis, axis_changes in changes.items()]
            elif callable(axis_changes):
                new_axis = real_axis.apply(axis_changes)
            else:
                new_axis = Axis(axis_changes, real_axis.name)
            new_axes.append((real_axis, new_axis))
        return self.replace(new_axes, inplace=inplace)

    # TODO: deprecate method (should use __sub__ instead)
    def without(self, axes) -> 'AxisCollection':
        r"""
        Return a new collection without some axes.

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
        >>> age = Axis('age=0..5')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis('time=2015..2017')
        >>> col = AxisCollection([age, sex, time])
        >>> col.without([age, sex])
        AxisCollection([
            Axis([2015, 2016, 2017], 'time')
        ])
        >>> col.without(0)
        AxisCollection([
            Axis(['M', 'F'], 'sex'),
            Axis([2015, 2016, 2017], 'time')
        ])
        >>> col.without('sex,time')
        AxisCollection([
            Axis([0, 1, 2, 3, 4, 5], 'age')
        ])
        """
        return self - axes

    def __sub__(self, axes) -> 'AxisCollection':
        r"""
        See Also
        --------
        without
        """
        if isinstance(axes, str):
            axes = axes.split(',')
        elif isinstance(axes, (int, Axis)):
            axes = [axes]

        def index_first_compatible(axis):
            try:
                return self.index(axis, compatible=True)
            except ValueError:
                return -1
        # only keep indices (as this works for unnamed axes too)
        to_remove = set(index_first_compatible(axis) for axis in axes)
        # -1 in to_remove are not a problem since enumerate starts at 0
        return AxisCollection([axis for i, axis in enumerate(self) if i not in to_remove])

    def _translate_nice_key(self, axis_key):
        # TODO: instead of checking all axes, we should have a big mapping (in AxisCollection):
        #       label -> (axis, index) but for sparse/multi-index, this would not work, we'd need label -> axis
        valid_axes = []
        # TODO: use axis_key dtype to only check compatible axes
        for axis in self:
            try:
                axis_pos_key = axis.index(axis_key)
                valid_axes.append(axis)
            except KeyError:
                continue
        if not valid_axes:
            # transform string key to object (Group, list, slice, range, scalar)
            nicer_key = _to_key(axis_key)
            sequence_types = (tuple, list, np.ndarray, ABCArray)
            if (isinstance(nicer_key, sequence_types) or
                (isinstance(nicer_key, Group) and isinstance(nicer_key.key, sequence_types))):

                # ... and check for partial matches
                if isinstance(nicer_key, Group):
                    nicer_key = nicer_key.eval()

                # we transform arrays with ndim > 1 to flat arrays because
                # otherwise the elements are arrays themselves which are not
                # hashable and thus we cannot compute a set of them
                if isinstance(nicer_key, (ABCArray, np.ndarray)):
                    key_flat_values = nicer_key.data.flat if isinstance(nicer_key, ABCArray) else nicer_key.flat
                    array_key = True
                    msg = (f"The values of the array key:\n\n{axis_key!r}\n\n"
                           f"do not all correspond to labels of a single axis "
                           f"of the subsetted array which has the following "
                           f"axes:\n\n{self._axes_summary()}\n")
                else:
                    key_flat_values = nicer_key
                    array_key = False
                    msg = (f"{axis_key!r} is not a valid subset for any axis:\n"
                           f"{self._axes_summary()}")

                key_label_set = set(key_flat_values)
                partial_matches = {}
                for axis in self:
                    missing_labels = key_label_set - set(axis.labels)
                    if missing_labels < key_label_set:
                        partial_matches[axis] = missing_labels

                if partial_matches:
                    partial_matches_str = '\n'.join(
                        f" * axis '{self.axis_id(axis)}' contains {len(key_label_set) - len(missing_labels)}"
                        f' out of {len(key_label_set)}'
                        f' labels (labels not found: {", ".join(repr(label) for label in missing_labels)})'
                        for axis, missing_labels in partial_matches.items()
                    )
                    what = 'key values' if array_key else 'labels'
                    msg += f"\nSome of those {what} correspond though:\n{partial_matches_str}"

                # if we have a single partial match and an la.Array key (we
                # don't do it for np.ndarray keys), we compute the problematic
                # parts of the key and mention them if they are small enough
                if len(partial_matches) == 1 and isinstance(nicer_key, ABCArray):
                    axis = next(iter(partial_matches.keys()))
                    is_bad_key_value = (~nicer_key.isin(axis.labels)).compact()
                    bad_indices_per_axis = is_bad_key_value.data.nonzero()
                    SMALL_BAD_PART_THRESHOLD = 5
                    small_bad_parts_locations = [
                        f" {axis.name}: " +
                            ' '.join(repr(label)
                                     for label in axis.labels[axis_indices])
                        for axis, axis_indices in zip(nicer_key.axes, bad_indices_per_axis)
                        if len(axis_indices) <= SMALL_BAD_PART_THRESHOLD
                    ]
                    if small_bad_parts_locations:
                        msg += ("\n\nNote that all the bad key values are "
                                "located within the following labels:\n")
                        msg += '\n'.join(small_bad_parts_locations)
            else:
                # we have a single label
                msg = f"{axis_key!r} is not a valid label for any axis:\n{self._axes_summary()}"

            raise ValueError(msg)
        elif len(valid_axes) > 1:
            raise ValueError(f'{axis_key!r} is ambiguous, it is valid in the following axes:\n'
                             f'{self._axes_summary(valid_axes)}')
        real_axis = valid_axes[0]
        return real_axis, axis_pos_key

    def _guess_axis(self, axis_key):
        """
        Translate any *single axis* key to an LGroup on the real axis.
        """
        if isinstance(axis_key, Group):
            group_axis = axis_key.axis
            if group_axis is not None:
                # we have axis information but not necessarily an Axis object from self
                real_axis = self[group_axis]
                if group_axis is not real_axis:
                    axis_key = axis_key.with_axis(real_axis)
                return axis_key

        real_axis, axis_pos_key = self._translate_nice_key(axis_key)
        return real_axis[axis_key]

    def _translate_axis_key_chunk(self, axis_key):
        """
        Translate any *single axis* key to an (axis, indices) pair.

        Parameters
        ----------
        axis_key : any kind of key
            Key to select axis. Supports all kinds of label-based keys and IGroups.

        Returns
        -------
        (axis, indices)
            Indices group with a valid axis (from self)
        """
        axis_key = remove_nested_groups(axis_key)

        if isinstance(axis_key, IGroup):
            if axis_key.axis is None:
                raise ValueError("positional groups without axis are not supported")

            # retarget to real axis, if needed
            # only retarget IGroup and not LGroup to give the opportunity for axis.translate to try the "ticks"
            # version of the group ONLY if key.axis is not real_axis (for performance reasons)
            if axis_key.axis in self:
                real_axis = self[axis_key.axis]
                axis_key = axis_key.retarget_to(real_axis)
                # still positional after retarget
                if isinstance(axis_key, IGroup):
                    return axis_key.axis, axis_key.key
            else:
                # axis associated with axis_key may not belong to self.
                # In that case, we translate IGroup to labels and search for a compatible axis
                # (see end of this method)
                axis_key = axis_key.to_label()

        # labels but known axis
        if isinstance(axis_key, LGroup) and axis_key.axis is not None:
            try:
                real_axis = self[axis_key.axis]
                try:
                    axis_pos_key = real_axis.index(axis_key)
                except KeyError:
                    raise ValueError(f"{axis_key!r} is not a valid label for the {real_axis.name!r} axis "
                                     f"with labels: {', '.join(repr(label) for label in real_axis.labels)}")
                return real_axis, axis_pos_key
            except KeyError:
                # axis associated with axis_key may not belong to self.
                # In that case, we translate LGroup to labels and search for a compatible axis
                # (see end of this method)
                axis_key = axis_key.to_label()

        # otherwise we need to guess the axis
        return self._translate_nice_key(axis_key)

    def _translate_axis_key(self, axis_key):
        """
        Translate single axis label-based key to (axis, indices).

        Parameters
        ----------
        axis_key : any valid key
            Key to select axis.

        Returns
        -------
        (axis, indices)
            Indices group with a valid axis (from self)
        """
        # called from _key_to_igroups

        # Need to convert string keys to groups otherwise command like
        # >>> ndtest((5, 5)).drop('1[a0]')
        # will work although it shouldn't
        if isinstance(axis_key, str):
            key = _to_key(axis_key)
            if isinstance(key, Group):
                axis_key = key

        # translate Axis keys to LGroup keys
        # FIXME: this should be simply:
        # if isinstance(axis_key, Axis):
        #     axis_key = axis_key[:]
        # but it does not work for some reason (the retarget does not seem to happen)
        if isinstance(axis_key, Axis):
            real_axis = self[axis_key]
            if isinstance(axis_key, AxisReference) or axis_key.equals(real_axis):
                axis_key = real_axis[:]
            else:
                axis_key = axis_key.labels

        return self._translate_axis_key_chunk(axis_key)

    def _key_to_axis_indices_dict(self, key):
        """
        Translate any label-based key to an {axis: indices} dict.

        Parameters
        ----------
        key : scalar, list/array of scalars, Group or tuple or dict of them
            any key supported by Array.__get|setitem__

        Returns
        -------
        dict
            dict {axis: indices}, with axis from this array.
            The order of the dict axes/keys is *not* guaranteed to be the same as the order of axes.

        See Also
        --------
        Axis.index
        """
        from .array import Array

        if isinstance(key, dict):
            # key axes could be strings or axis references and we want real axes
            key = tuple(self[axis][axis_key] for axis, axis_key in key.items())
        elif not isinstance(key, tuple):
            # convert scalar keys to 1D keys
            key = (key,)

        filtered_key = []
        for axis_key in key:
            # handle ExprNode
            if isinstance(axis_key, ExprNode):
                axis_key = axis_key.evaluate(self)

            # handle boolean filter keys
            if isinstance(axis_key, np.ndarray) and np.issubdtype(axis_key.dtype, np.bool_):
                if axis_key.shape != self.shape:
                    raise ValueError(f"boolean key with a different shape ({axis_key.shape}) than array ({self.shape})")
                filtered_key.extend(Array(axis_key, self).nonzero())
            elif isinstance(axis_key, Array) and np.issubdtype(axis_key.dtype, np.bool_):
                extra_key_axes = axis_key.axes - self
                if extra_key_axes:
                    raise ValueError(f"boolean subset key contains more axes ({axis_key.axes}) than array ({self})")

                # TODO: factorize with check_compatible
                for i, subset_axis in enumerate(axis_key.axes):
                    array_axis = self.get_by_pos(subset_axis, i)
                    if not array_axis.iscompatible(subset_axis):
                        msg = f"""boolean subset array has incompatible axes with array:
array axes: {self}
subset array axes: {axis_key.axes}
incompatible axes:
    array axis:
        {array_axis!r}
    subset array axis:
        {subset_axis!r}
"""
                        raise ValueError(msg)

                # nonzero (currently) returns a tuple of IGroups containing 1D Arrays (one IGroup per axis)
                filtered_key.extend(axis_key.nonzero())
            # drop slice(None) and Ellipsis since they are meaningless because of guess_axis.
            # XXX: we might want to raise an exception when we find Ellipses or (most) slice(None) because except for
            #      a single slice(None) a[:], I don't think there is any point.
            elif not _isnoneslice(axis_key) and axis_key is not Ellipsis:
                filtered_key.append(axis_key)

        key = tuple(filtered_key)

        # translate all keys to (axis, indices) pairs
        key_items = tuple(self._translate_axis_key(axis_key) for axis_key in filtered_key)

        assert all(isinstance(axis, Axis) for axis, axis_key in key_items)

        # even keys given as dict can contain duplicates (if the same axis was
        # given under different forms, e.g. name and AxisReference).
        if has_duplicates(axis for axis, axis_key in key_items):
            dupe_axes = duplicates(axis for axis, axis_key in key_items)
            dupe_axes_str = ', '.join(str(axis) for axis in dupe_axes)
            raise ValueError(f"key has several values for axis: {dupe_axes_str}\nkey: {key}")

        # ((axis, indices), (axis, indices), ...) -> dict
        return dict(key_items)

    def _key_to_raw_and_axes(self, key, collapse_slices=False, translate_key=True, points=False, wildcard=False):
        r"""
        Transform any key (from Array.__getitem__) to a raw numpy key, the resulting axes, and potentially a tuple
        of indices to transpose axes back to where they were.

        Parameters
        ----------
        key : scalar, list/array of scalars, Group or tuple or dict of them
            any key supported by Array.__getitem__
        collapse_slices : bool, optional
            Whether to convert ranges to slices. Defaults to False.
        translate_key : bool, optional
            Whether to consider key as label-based. Defaults to True.

        Returns
        -------
        raw_key, res_axes, transposed_indices
        """
        from .array import raw_broadcastable, Array, sequence

        ndim = self.ndim
        if translate_key:
            # complete key & translate (those two cannot be dissociated because to complete
            # the key we need to know which axis each key belongs to and to do that, we need to
            # translate the key to indices)

            # any key -> {axis: indices, axis: indices, ...}
            dict_key = self._key_to_axis_indices_dict(key)

            # dict -> tuple (complete and order key)
            key = tuple(dict_key[axis] if axis in dict_key else slice(None)
                        for axis in self)
        else:
            # if we do not translate the key, we still need to make sure it is complete
            if not isinstance(key, tuple):
                key = (key,)
            key_len = len(key)
            if key_len < ndim:
                # complete with slice(None)
                key = key + (slice(None),) * (ndim - key_len)
            elif key_len > ndim:
                raise IndexError(f"key is too long ({key_len}) for array with {ndim} dimensions")

        # TODO: check if we can move those to globals without a performance hit
        INTEGER_TYPES = (np.integer, int)
        SEQ_TYPES = (tuple, list, np.ndarray)

        # fastpath for indexing a single element or a scalar array (ideally this should not be needed)
        num_ints = sum(isinstance(axis_key, INTEGER_TYPES) for axis_key in key)
        if num_ints == ndim or not ndim:
            return key, None, None

        # explicitly raise if we get an ndarray with ndim > 1 since we do not support them yet
        if any(isinstance(axis_key, np.ndarray) and axis_key.ndim > 1 for axis_key in key):
            raise NotImplementedError("ndarray keys with ndim > 1 are not supported yet")

        if points:
            # transform non-Array advanced keys to Array with a combined axis
            key = self._adv_keys_to_combined_axis_la_keys(key, wildcard=wildcard)

        # transform ranges to slices if needed
        if collapse_slices:
            # TODO: we should only do this if there are no Array key (with axes corresponding to the range)
            #       otherwise we will be translating them back to a range afterwards
            key = tuple(_idx_seq_to_slice(axis_key, len(axis)) if isinstance(axis_key, SEQ_TYPES) else axis_key
                        for axis_key, axis in zip(key, self))

        # fastpath when no broadcasting is needed (ideally the "broadcasting path" should be fast enough to make this
        # special case useless but we are not there yet)
        num_arrays = sum(isinstance(axis_key, Array) for axis_key in key)
        # TODO: ideally I think we should do this (but then the fastpath code needs to change)
        # num_sequences = sum(isinstance(axis_key, (list, np.ndarray, Array)) for axis_key in key)
        # need_broadcasting = (num_sequences == 1 and num_ints > 0) or (num_sequences > 1)
        # 1D np.ndarray do not need broadcasting (can be considered sequences)
        list_or_array = (list, np.ndarray)
        num_sequences = sum(isinstance(axis_key, list_or_array) for axis_key in key)
        need_broadcasting = (num_arrays > 0) or (num_sequences == 1 and num_ints > 0) or (num_sequences > 1)
        if not need_broadcasting:
            res_axes = AxisCollection([axis.subaxis(axis_key)
                                       for axis, axis_key in zip(self, key)
                                       if not isinstance(axis_key, INTEGER_TYPES)])
            return key, res_axes, None

        # transform non-Array advanced keys (list and ndarray) to Array
        def to_la_ikey(axis, axis_key):
            if isinstance(axis_key, (int, np.integer, slice, Array)):
                return axis_key
            else:
                assert isinstance(axis_key, (list, np.ndarray)), f"unsupported key type: {axis_key} ({type(axis_key)})"
                res_axis = axis.subaxis(axis_key)
                # TODO: for perf reasons, we should bypass creating an actual Array by returning axes and key_data
                # but then we will need to implement a function similar to make_numpy_broadcastable which works on axes
                # and rawdata instead of arrays
                return Array(axis_key, res_axis)

        key = tuple(to_la_ikey(axis, axis_key) for axis, axis_key in zip(self, key))

        # transform slice keys to Array too IF they refer to axes present in advanced key (so that those axes
        # broadcast together instead of being duplicated, which is not what we want)
        def get_axes(value):
            return value.axes if isinstance(value, Array) else AxisCollection([])

        def slice_to_sequence(axis, axis_key):
            if isinstance(axis_key, slice) and axis in la_key_axes:
                # TODO: sequence assumes the axis in the la_key is in the same order. It will be easier to solve when
                # make_numpy_broadcastable automatically aligns all arrays
                start, stop, step = axis_key.indices(len(axis))
                return sequence(axis.subaxis(axis_key), initial=start, inc=step)
            else:
                return axis_key

        # XXX: can we avoid computing this twice? (here and in make_numpy_broadcastable)
        la_key_axes = AxisCollection.union(*[get_axes(k) for k in key])
        key = tuple(slice_to_sequence(axis, axis_key) for axis, axis_key in zip(self, key))

        # start with the simple (slice) keys
        # scalar keys are ignored since they do not produce any resulting axis
        res_axes = AxisCollection([axis.subaxis(axis_key)
                                   for axis, axis_key in zip(self, key)
                                   if isinstance(axis_key, slice)])
        transpose_indices = None

        # if there are only simple keys, do not bother going via the "advanced indexing" code path
        if all(isinstance(axis_key, (int, np.integer, slice)) for axis_key in key):
            raw_broadcasted_key = key
        else:
            # Now that we know advanced indexing comes into play, we need to compute were the subspace created by the
            # advanced indexes will be inserted. Note that there is only ever a SINGLE combined subspace (even if it
            # has multiple axes) because all the non slice indexers MUST broadcast together to a single
            # "advanced indexer"

            # to determine where the "subspace" axes will be inserted, a scalar key counts as "advanced" indexing
            adv_axes_indices = [i for i, axis_key in enumerate(key)
                                if not isinstance(axis_key, slice)]
            diff = np.diff(adv_axes_indices)
            if np.any(diff > 1):
                # insert advanced indexing subspace in front
                adv_key_subspace_pos = 0

                # If all (non scalar) adv_keys are 1D and have a different axis name, we will index the cross product.
                # In that case, store their original order so that we can transpose them back to where they were.
                adv_keys = [axis_key for axis_key in key if not isinstance(axis_key, (int, np.integer, slice))]
                if all(axis_key.ndim == 1 for axis_key in adv_keys):
                    # we can only handle the non-anonymous axes case since anonymous axes will not broadcast to the
                    # cross product anyway
                    if len(set(axis_key.axes[0].name for axis_key in adv_keys)) == len(adv_keys):
                        # 0, 1, 2, 3, 4, 5 <- original axes indices
                        # A  X  A  S  S  A <- key (A = adv, X = scalar/remove, S = slice)
                        # 0, 2, 5, 3, 4    <- result
                        # 0, 2, 3, 4, 5    <- desired result
                        # 0, 1, 3, 4, 2    <- what I need to feed to transpose to get the correct result
                        adv_axes_indices = [i for i, axis_key in enumerate(key)
                                            if not isinstance(axis_key, (int, np.integer, slice))]
                        # not taking scalar axes since they will disappear
                        slice_axes_indices = [i for i, axis_key in enumerate(key)
                                              if isinstance(axis_key, slice)]
                        result_axes_indices = adv_axes_indices + slice_axes_indices
                        transpose_indices = argsort(result_axes_indices)
            else:
                # the advanced indexing subspace keep its position (insert at position of first concerned axis)
                adv_key_subspace_pos = adv_axes_indices[0]

            # scalar/slice keys are ignored by make_numpy_broadcastable, which is exactly what we need
            raw_broadcasted_key, adv_key_dest_axes = raw_broadcastable(key)

            # insert advanced indexing subspace
            res_axes[adv_key_subspace_pos:adv_key_subspace_pos] = adv_key_dest_axes

        return raw_broadcasted_key, res_axes, transpose_indices

    @property
    def labels(self) -> List[Iterable[Scalar]]:
        r"""
        Return the list of labels of the axes.

        Returns
        -------
        list
            List of labels of the axes.

        Examples
        --------
        >>> age = Axis(range(10), 'age')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age, time]).labels  # doctest: +NORMALIZE_WHITESPACE
        [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
         array([2007, 2008, 2009, 2010])]
        """
        return [axis.labels for axis in self._list]

    @property
    def names(self) -> List[str]:
        r"""
        Return the list of (raw) names of the axes.

        Returns
        -------
        list
            List of names of the axes.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age, sex, time]).names
        ['age', 'sex', 'time']
        """
        return [axis.name for axis in self._list]

    # providing idx is just an optimization to avoid the relatively expensive self.index(axis)
    def _display_name(self, axis, idx=None):
        if axis.name is None:
            if idx is None:
                idx = self.index(axis)
            name = f'{{{idx}}}'
        else:
            # using str() because name can be an integer
            name = str(axis.name)
        return (name + '*') if axis.iswildcard else name

    def _axes_summary(self, axes=None):
        def axis_summary(axis, idx=None):
            return f" {self._display_name(axis, idx)} [{len(axis)}]: {axis.labels_summary()}"

        if axes is None:
            parts = [axis_summary(axis, i) for i, axis in enumerate(self._list)]
        else:
            parts = [axis_summary(axis) for axis in axes]
        return '\n'.join(parts)

    @property
    def display_names(self) -> List[str]:
        r"""
        Return the list of (display) names of the axes.

        Returns
        -------
        list
            List of names of the axes. Wildcard axes are displayed with an attached \*.
            Anonymous axes (name = None) are replaced by their position wrapped in braces.

        Examples
        --------
        >>> a = Axis(['a1', 'a2'], 'a')
        >>> b = Axis(2, 'b')
        >>> c = Axis(['c1', 'c2'])
        >>> d = Axis(3)
        >>> AxisCollection([a, b, c, d]).display_names
        ['a', 'b*', '{2}', '{3}*']
        """
        return [self._display_name(axis, i) for i, axis in enumerate(self._list)]

    @property
    def ids(self) -> List[Union[str, int]]:
        r"""
        Return the list of ids of the axes.

        Returns
        -------
        list
            List of ids of the axes.

        See Also
        --------
        axis_id

        Examples
        --------
        >>> a = Axis(2, 'a')
        >>> b = Axis(2)
        >>> c = Axis(2, 'c')
        >>> AxisCollection([a, b, c]).ids
        ['a', 1, 'c']
        """
        return [axis.name if axis.name is not None else i
                for i, axis in enumerate(self._list)]

    def axis_id(self, axis) -> Union[str, int]:
        r"""
        Return the id of an axis.

        Returns
        -------
        str or int
            Id of axis, which is its name if defined and its position otherwise.

        Examples
        --------
        >>> a = Axis(2, 'a')
        >>> b = Axis(2)
        >>> c = Axis(2, 'c')
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
    def shape(self) -> Tuple[int, ...]:
        r"""
        Return the shape of the collection.

        Returns
        -------
        tuple
            Tuple of lengths of axes.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age, sex, time]).shape
        (20, 2, 4)
        """
        # axis._length is almost twice as fast as the nicer len(axis)
        return tuple(axis._length for axis in self._list)

    @property
    def size(self) -> int:
        r"""
        Return the size of the collection, i.e.
        the number of elements of the array.

        Returns
        -------
        int
            Number of elements of the array.

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age, sex, time]).size
        160
        """
        return np.prod(self.shape)

    @property
    def info(self) -> str:
        r"""
        Describe the collection (shape and labels for each axis).

        Returns
        -------
        str
            Description of the AxisCollection (shape and labels for each axis).

        Examples
        --------
        >>> age = Axis(20, 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age, sex, time]).info
        20 x 2 x 4
         age* [20]: 0 1 2 ... 17 18 19
         sex [2]: 'M' 'F'
         time [4]: 2007 2008 2009 2010
        """
        shape_str = " x ".join(str(s) for s in self.shape)
        return ReprString(f"{shape_str}\n{self._axes_summary()}")

    # XXX: instead of front_if_spread, we might want to require axes to be contiguous
    #      (ie the caller would have to transpose axes before calling this)
    def combine_axes(self, axes=None, sep='_', wildcard=False, front_if_spread=False) -> 'AxisCollection':
        r"""Combine several axes into one.

        Parameters
        ----------
        axes : tuple, list, AxisCollection of axes or list of combination of those or dict, optional
            axes to combine. Tuple, list or AxisCollection will combine several axes into one. To chain several axes
            combinations, pass a list of tuple/list/AxisCollection of axes. To set the name(s) of resulting axis(es),
            use a {(axes, to, combine): 'new_axis_name'} dictionary. Defaults to all axes.
        sep : str, optional
            delimiter to use for combining. Defaults to '_'.
        wildcard : bool, optional
            whether to produce a wildcard axis even if the axes to combine are not.
            This is much faster, but loose axes labels.
        front_if_spread : bool, optional
            whether to move the combined axis at the front (it will be the first axis) if the combined axes are
            not next to each other.

        Returns
        -------
        AxisCollection
            New AxisCollection with combined axes.

        Examples
        --------
        >>> axes = AxisCollection('a=a0,a1;b=b0..b2')
        >>> axes
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])
        >>> axes.combine_axes()
        AxisCollection([
            Axis(['a0_b0', 'a0_b1', 'a0_b2', 'a1_b0', 'a1_b1', 'a1_b2'], 'a_b')
        ])
        >>> axes.combine_axes(sep='/')
        AxisCollection([
            Axis(['a0/b0', 'a0/b1', 'a0/b2', 'a1/b0', 'a1/b1', 'a1/b2'], 'a/b')
        ])
        >>> axes += AxisCollection('c=c0..c2;d=d0,d1')
        >>> axes.combine_axes(('a', 'c'))
        AxisCollection([
            Axis(['a0_c0', 'a0_c1', 'a0_c2', 'a1_c0', 'a1_c1', 'a1_c2'], 'a_c'),
            Axis(['b0', 'b1', 'b2'], 'b'),
            Axis(['d0', 'd1'], 'd')
        ])
        >>> axes.combine_axes({('a', 'c'): 'ac'})
        AxisCollection([
            Axis(['a0_c0', 'a0_c1', 'a0_c2', 'a1_c0', 'a1_c1', 'a1_c2'], 'ac'),
            Axis(['b0', 'b1', 'b2'], 'b'),
            Axis(['d0', 'd1'], 'd')
        ])

        # make several combinations at once

        >>> axes.combine_axes([('a', 'c'), ('b', 'd')])
        AxisCollection([
            Axis(['a0_c0', 'a0_c1', 'a0_c2', 'a1_c0', 'a1_c1', 'a1_c2'], 'a_c'),
            Axis(['b0_d0', 'b0_d1', 'b1_d0', 'b1_d1', 'b2_d0', 'b2_d1'], 'b_d')
        ])
        >>> axes.combine_axes({('a', 'c'): 'ac', ('b', 'd'): 'bd'})
        AxisCollection([
            Axis(['a0_c0', 'a0_c1', 'a0_c2', 'a1_c0', 'a1_c1', 'a1_c2'], 'ac'),
            Axis(['b0_d0', 'b0_d1', 'b1_d0', 'b1_d1', 'b2_d0', 'b2_d1'], 'bd')
        ])
        """
        if axes is None:
            axes = {tuple(self): None}
        elif isinstance(axes, AxisCollection):
            axes = {tuple(self[axes]): None}
        elif isinstance(axes, (list, tuple)):
            # checks for nested tuple/list
            if all(isinstance(axis, (list, tuple, AxisCollection)) for axis in axes):
                axes = {tuple(self[axes_to_combine]): None for axes_to_combine in axes}
            else:
                axes = {tuple(self[axes]): None}
        # axes should be a dict at this time
        assert isinstance(axes, dict)

        new_axes = self[:]
        for _axes, name in axes.items():
            _axes = new_axes[_axes]
            axes_indices = [new_axes.index(axis) for axis in _axes]
            diff = np.diff(axes_indices)
            # combined axes in front
            if front_if_spread and np.any(diff > 1):
                combined_axis_pos = 0
            else:
                combined_axis_pos = min(axes_indices)

            if name is not None:
                combined_name = name
            # all anonymous axes => anonymous combined axis
            elif all(axis.name is None for axis in _axes):
                combined_name = None
            else:
                combined_name = sep.join(str(id_) for id_ in _axes.ids)

            if wildcard:
                combined_axis = Axis(_axes.size, combined_name)
            else:
                # TODO: the combined keys should be objects which display as: (axis1_label, axis2_label, ...) but
                # which should also store the axes names)
                # Q: Should it be the same object as the NDLGroup?/NDKey?
                # A: yes. On the Pandas backend, we could/should have separate axes. On the numpy backend we cannot.
                if len(_axes) == 1:
                    # Q: if axis is a wildcard axis, should the result be a wildcard axis (and axes_labels discarded?)
                    combined_labels = _axes[0].labels
                else:
                    sepjoin = sep.join
                    axes_labels = [np.asarray(label, str) for label in _axes.labels]
                    combined_labels = [sepjoin(p) for p in product(*axes_labels)]
                combined_axis = Axis(combined_labels, combined_name)
            new_axes = new_axes - _axes
            new_axes.insert(combined_axis_pos, combined_axis)
        return new_axes

    def split_axes(self, axes=None, sep='_', names=None, regex=None) -> 'AxisCollection':
        r"""Split axes and returns a new collection.

        The split axes are inserted where the combined axis was.

        Parameters
        ----------
        axes : int, str, Axis or any combination of those, optional
            axes to split. All labels *must* contain the given delimiter string. To split several axes at once, pass
            a list or tuple of axes to split. To set the names of resulting axes, use a {'axis_to_split': (new, axes)}
            dictionary. Defaults to all axes whose name contains the `sep` delimiter.
        sep : str, optional
            delimiter to use for splitting. Defaults to '_'. When `regex` is provided, the delimiter is only used on
            `names` if given as one string or on axis name if `names` is None.
        names : str or list of str, optional
            names of resulting axes. Defaults to None.
        regex : str, optional
            use the `regex` regular expression to split labels instead of the `sep` delimiter. Defaults to None.

        See Also
        --------
        Axis.split
        Array.split_axes

        Returns
        -------
        AxisCollection

        Examples
        --------
        >>> col = AxisCollection('a=a0,a1;b=b0..b2')
        >>> col
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])
        >>> combined = col.combine_axes()
        >>> combined
        AxisCollection([
            Axis(['a0_b0', 'a0_b1', 'a0_b2', 'a1_b0', 'a1_b1', 'a1_b2'], 'a_b')
        ])
        >>> combined.split_axes()
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])

        Split labels using a regular expression

        >>> combined = AxisCollection('ab = a0b0..a1b2')
        >>> combined
        AxisCollection([
            Axis(['a0b0', 'a0b1', 'a0b2', 'a1b0', 'a1b1', 'a1b2'], 'ab')
        ])
        >>> # The pattern for each resulting axis should be enclosed in parentheses
        >>> combined.split_axes('ab', names=['a', 'b'], regex=r'(..)(..)')
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])

        Split several axes at once

        >>> combined = AxisCollection('a_b = a0_b0..a1_b1; c_d = c0_d0..c1_d1')
        >>> combined
        AxisCollection([
            Axis(['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1'], 'a_b'),
            Axis(['c0_d0', 'c0_d1', 'c1_d0', 'c1_d1'], 'c_d')
        ])
        >>> # equivalent to combined.split_axes() which split all axes
        >>> # containing the delimiter defined by the argument `sep`
        >>> combined.split_axes(['a_b', 'c_d'])
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1'], 'b'),
            Axis(['c0', 'c1'], 'c'),
            Axis(['d0', 'd1'], 'd')
        ])
        >>> combined.split_axes({'a_b': ('A', 'B'), 'c_d': ('C', 'D')})
        AxisCollection([
            Axis(['a0', 'a1'], 'A'),
            Axis(['b0', 'b1'], 'B'),
            Axis(['c0', 'c1'], 'C'),
            Axis(['d0', 'd1'], 'D')
        ])
        """
        axes = self._prepare_split_axes(axes, names, sep)
        new_axes = self[:]
        for axis, names in axes.items():
            axis = new_axes[axis]
            axis_index = new_axes.index(axis)
            split_axes = axis.split(sep, names, regex)
            new_axes = new_axes[:axis_index] + split_axes + new_axes[axis_index + 1:]
        return new_axes

    def _prepare_split_axes(self, axes, names, sep) -> dict:
        if axes is None:
            # use all axes where sep is present
            axes = [axis for axis in self if axis.name is not None and sep in axis.name]
        elif isinstance(axes, (int, str, Axis)):
            axes = {axes: names}

        if isinstance(axes, (list, tuple)):
            if any(not isinstance(axis, (int, str, Axis)) for axis in axes):
                raise TypeError("Expected tuple or list of int, string or Axis instances")

            if names is None:
                axes = {axis: None for axis in axes}
            else:
                axes = dict(zip(axes, names))

        # axes should be a dict at this time
        assert isinstance(axes, dict)
        return axes

    split_axis = renamed_to(split_axes, 'split_axis', raise_error=True)

    def align(self, *other, join='outer', axes=None) -> Tuple['AxisCollection']:
        r"""Align this AxisCollection with (an)other AxisCollection(s).

        This ensures all common axes are compatible.

        Parameters
        ----------
        *other : AxisCollection
            AxisCollection(s) to align with this one.
        join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
            Defaults to 'outer'.
        axes : AxisReference or sequence of them, optional
            Axes to align. Need to be valid in all axis collections.
            Defaults to None (all common axes). This must be specified
            when mixing anonymous and non-anonymous axes.

        Returns
        -------
        tuple of AxisCollection
            Aligned collections

        See Also
        --------
        Array.align

        Examples
        --------
        >>> col1 = AxisCollection("a=a0..a1;b=b0..b2")
        >>> col1
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])
        >>> col2 = AxisCollection("a=a0..a2;c=c0..c0;b=b0..b1")
        >>> col2
        AxisCollection([
            Axis(['a0', 'a1', 'a2'], 'a'),
            Axis(['c0'], 'c'),
            Axis(['b0', 'b1'], 'b')
        ])
        >>> aligned1, aligned2 = col1.align(col2)
        >>> aligned1
        AxisCollection([
            Axis(['a0', 'a1', 'a2'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])
        >>> aligned2
        AxisCollection([
            Axis(['a0', 'a1', 'a2'], 'a'),
            Axis(['c0'], 'c'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])

        Using anonymous axes

        >>> col1 = AxisCollection("a0..a1;b0..b2")
        >>> col1
        AxisCollection([
            Axis(['a0', 'a1'], None),
            Axis(['b0', 'b1', 'b2'], None)
        ])
        >>> col2 = AxisCollection("a0..a2;b0..b1;c0..c0")
        >>> col2
        AxisCollection([
            Axis(['a0', 'a1', 'a2'], None),
            Axis(['b0', 'b1'], None),
            Axis(['c0'], None)
        ])
        >>> aligned1, aligned2 = col1.align(col2)
        >>> aligned1
        AxisCollection([
            Axis(['a0', 'a1', 'a2'], None),
            Axis(['b0', 'b1', 'b2'], None)
        ])
        >>> aligned2
        AxisCollection([
            Axis(['a0', 'a1', 'a2'], None),
            Axis(['b0', 'b1', 'b2'], None),
            Axis(['c0'], None)
        ])
        """
        # For backward compatibility with older code using align with a
        # non-AxisCollection second argument, we only support aligning more
        # than two collection when other contains actual AxisCollection objects
        bad_objs = [obj for obj in other if not isinstance(obj, AxisCollection)]
        if bad_objs:
            for obj in bad_objs:
                obj_type = type(obj).__name__
                warnings.warn(f"aligning an AxisCollection to a "
                              f"non-AxisCollection object ({obj_type}) is "
                              f"deprecated. Please convert to an AxisCollection "
                              f"first.", FutureWarning, stacklevel=2)
            other = [AxisCollection(obj) for obj in other]

        return align_axis_collections((self, *other), join=join, axes=axes)

    # XXX: make this into a public method/property? AxisCollection.flat_labels[flat_indices]?
    def _flat_lookup(self, flat_indices):
        r"""Return labels corresponding to indices into the flattened axes.

        Parameters
        ----------
        flat_indices : array-like
            indices to get

        Examples
        --------
        >>> from larray import ndtest, Array
        >>> arr = ndtest((2, 3))
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
        >>> indices = Array([2, 5, 0], 'draw=d0..d2')
        >>> indices
        draw  d0  d1  d2
               2   5   0
        >>> arr.axes._flat_lookup(indices)
        draw\axis   a   b
               d0  a0  b2
               d1  a1  b2
               d2  a0  b0
        """
        from larray.core.array import asarray, Array, stack

        flat_indices = asarray(flat_indices)
        # This could return an Array with object dtype because axes labels can have different types (but not length)
        # TODO: this should be:
        # return stack({axis.name: axis.i[inds] for axis, inds in zip(axes, axes_indices)}, axis='axis')
        flat_axes = flat_indices.axes
        axes_labels = self._iflat(flat_indices.data)
        return stack({axis.name: Array(axis_labels, flat_axes) for axis, axis_labels in zip(self, axes_labels)}, 'axis')

    def _adv_keys_to_combined_axis_la_keys(self, key, wildcard=False, sep='_'):
        r"""
        Return key with the non-Array "advanced indexing" key parts transformed to Arrays with a combined axis.
        Scalar, slice and Array key parts are just left as is.

        Parameters
        ----------
        key : tuple
            Complete (len(key) == self.ndim) indices-based key.
        wildcard : bool, optional
            Whether to produce a wildcard axis. Defaults to False.
        sep : str, optional
            Separator to use for creating combined axis name and labels (when wildcard is False). Defaults to '_'.

        Returns
        -------
        tuple
        """
        from larray.core.array import Array
        combined_axes = self._adv_keys_to_combined_axes(key, wildcard=wildcard, sep=sep)
        if combined_axes is None:
            return key

        # transform all advanced non-Array keys to Array with the combined axis
        ignored_types = (int, np.integer, slice, Array)
        return tuple(axis_key if isinstance(axis_key, ignored_types) else Array(axis_key, combined_axes)
                     for axis_key in key)

    def _adv_keys_to_combined_axes(self, key, wildcard=False, sep='_'):
        r"""
        Return an AxisCollection corresponding to the combined axis of the non-Array "advanced indexing" key parts.
        Scalar, slice and Array key parts are ignored.

        Parameters
        ----------
        key : tuple
            Complete (len(key) == self.ndim) indices-based key.
        wildcard : bool, optional
            Whether to produce a wildcard axis. Defaults to False.
        sep : str, optional
            Separator to use for creating combined axis name and labels (when wildcard is False). Defaults to '_'.

        Returns
        -------
        AxisCollection or None
        """
        from larray.core.array import Array

        assert isinstance(key, tuple) and len(key) == self.ndim

        # TODO: we should explicitly raise an error if we detect np.ndarray keys with ndim > 1 as this would
        #       require more than one combined axis. Supporting that is impossible (because we cannot know what
        #       the corresponding labels are) so we should either return wildcard axes in that case or raise an
        #       explicit error. Given the probability our internal users ever use that is so close to 0, the easiest
        #       solution should win. I am unsure which it is, but I guess an error should be easier. Note that there
        #       is no such issue with ND Array keys because for those we know the labels already so nothing needs to
        #       be done here.

        # XXX: can we use/factorize with AxisCollection._flat_lookup????
        # TODO: use/factorize with AxisCollection.combine_axes. The problem is that it uses product(*axes_labels)
        #       while here we need zip(*axes_labels)
        ignored_types = (int, np.integer, slice, Array)
        adv_keys = [(axis_key, axis) for axis_key, axis in zip(key, self)
                    if not isinstance(axis_key, ignored_types)]
        if not adv_keys:
            return None

        # axes with a scalar key are not taken, since we want to kill them

        # all anonymous axes => anonymous combined axis
        if all(axis.name is None for axis_key, axis in adv_keys):
            combined_name = None
        else:
            # using axis_id instead of name to allow combining a mix of anonymous & non anonymous axes
            combined_name = sep.join(str(self.axis_id(axis)) for axis_key, axis in adv_keys)

        # explicitly check that all combined keys have the same length
        first_key, first_axis = adv_keys[0]
        combined_axis_len = len(first_key)
        if not all(len(axis_key) == combined_axis_len for axis_key, axis in adv_keys[1:]):
            raise ValueError("all combined keys should have the same length")

        if wildcard:
            combined_axis = Axis(combined_axis_len, combined_name)
        else:
            # TODO: the combined keys should be objects which display as:
            # (axis1_label, axis2_label, ...) but which should also store
            # the axis (names?)
            # Q: Should it be the same object as the NDLGroup?/NDKey?
            # A: yes, probably. On the Pandas backend, we could/should have
            #    separate axes. On the numpy backend we cannot.
            # TODO: only convert if
            if len(adv_keys) == 1:
                # we do not convert to string when there is only a single axis
                axes_labels = [axis.labels[axis_key]
                               for axis_key, axis in adv_keys]
                # Q: if axis is a wildcard axis, should the result be a
                #    wildcard axis (and axes_labels discarded?)
                combined_labels = axes_labels[0]
            else:
                axes_labels = [axis.labels.astype(str, copy=False)[axis_key].tolist()
                               for axis_key, axis in adv_keys]
                sepjoin = sep.join
                combined_labels = [sepjoin(comb) for comb in zip(*axes_labels)]
            combined_axis = Axis(combined_labels, combined_name)
        return AxisCollection(combined_axis)


def align_axis_collections(axis_collections, join='outer', axes=None):
    if join not in {'outer', 'inner', 'left', 'right', 'exact'}:
        raise ValueError("join should be one of 'outer', 'inner', 'left', "
                         "'right' or 'exact'")

    # if axes not specified
    if axes is None:
        # and we have only anonymous axes
        if all(name is None for col in axis_collections
               for name in col.names):
            # use all axes by position
            max_length = max(len(col) for col in axis_collections)
            join_axes_refs = list(range(max_length))
        elif any(name is None for col in axis_collections
                 for name in col.names):
            raise ValueError(
                "axes collections with mixed anonymous/non anonymous axes "
                "are not supported by align without specifying axes "
                "explicitly")
        else:
            assert all(name is not None for col in axis_collections
                       for name in col.names)
            # use all axes by name
            join_axes_refs = OrderedSet(axis_collections[0].names)
            for col in axis_collections[1:]:
                join_axes_refs |= OrderedSet(col.names)
    else:
        if isinstance(axes, (int, str, Axis)):
            axes = [axes]
        join_axes_refs = axes

    # first compute all aligned axes for all collections
    axes_changes = {
        axis_ref: align_axes([axis_col[axis_ref]
                              for axis_col in axis_collections
                              if axis_ref in axis_col],
                             join=join)
        for axis_ref in join_axes_refs
    }

    # then apply the changed axes for the collections where the axis exists
    return tuple(
        axis_col.replace({
            axis_ref: aligned_axis
            for axis_ref, aligned_axis in axes_changes.items()
            if axis_ref in axis_col
        })
        for axis_col in axis_collections
    )


class AxisReference(ABCAxisReference, ExprNode, Axis):
    def __init__(self, name):
        self.name = name
        self._labels = None
        self._iswildcard = False

    def index(self, key):
        raise NotImplementedError("an AxisReference (X.) cannot translate labels")

    def __repr__(self) -> str:
        return f'AxisReference({self.name!r})'

    def evaluate(self, context) -> Axis:
        r"""
        Parameters
        ----------
        context : AxisCollection
            Use axes from this collection
        """
        return context[self.name]

    def can_evaluate_with(self, context) -> set:  # set[int]
        return self.name in context

    # Use the default hash. We have to specify it explicitly because we define __eq__ via ExprNode and
    # ExprNode.__hash__ (which is not set explicitly) takes precedence over Axis.__hash__
    __hash__ = object.__hash__


class AxisReferenceFactory:
    # needed to make pickle work (because we have a __getattr__ which does not return AttributeError on __getstate__)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, key) -> AxisReference:
        return AxisReference(key)

    def __getitem__(self, key) -> AxisReference:
        return AxisReference(key)


X = AxisReferenceFactory()


class DeprecatedAxisReferenceFactory:
    # needed to make pickle work (because we have a __getattr__ which does not return AttributeError on __getstate__)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, key):
        raise AttributeError("variable 'x' is deprecated, please use 'X' instead")

    def __getitem__(self, key):
        raise KeyError("variable 'x' is deprecated, please use 'X' instead")


x = DeprecatedAxisReferenceFactory()
