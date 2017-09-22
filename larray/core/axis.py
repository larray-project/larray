# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

import re
import sys
import warnings
from itertools import product

import numpy as np

from larray.core.abc import ABCAxis, ABCAxisReference, ABCLArray
from larray.core.expr import ExprNode
from larray.core.group import (Group, LGroup, PGroup, PGroupMaker, _to_tick, _to_ticks, _to_key, _seq_summary,
                               _contain_group_ticks, _seq_group_to_name)
from larray.util.oset import *
from larray.util.misc import basestring, PY2, unicode, long, duplicates, array_lookup2, ReprString, index_by_id

__all__ = ['Axis', 'AxisCollection', 'X', 'x']


class Axis(ABCAxis):
    """
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
    # ticks instead of labels?
    def __init__(self, labels, name=None):
        if isinstance(labels, Group) and name is None:
            name = labels.axis
        if isinstance(name, Axis):
            name = name.name
        if isinstance(labels, basestring):
            if '=' in labels:
                name, labels = [o.strip() for o in labels.split('=')]
            elif '..' not in labels and ',' not in labels:
                warnings.warn("Arguments 'name' and 'labels' of Axis constructor have been inverted in "
                              "version 0.22 of larray. Please check you are passing labels first and name "
                              "as second argument.", FutureWarning, stacklevel=2)
                name, labels = labels, name

        # make sure we do not have np.str_ as it causes problems down the
        # line with xlwings. Cannot use isinstance to check that though.
        is_python_str = type(name) is unicode or type(name) is bytes
        assert name is None or isinstance(name, int) or is_python_str, type(name)
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
        >>> from larray import ndrange
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> arr = ndrange([sex, time])
        >>> arr
        sex\\time  2007  2008  2009  2010
               M     0     1     2     3
               F     4     5     6     7
        >>> arr[time.i[0, -1]]
        sex\\time  2007  2010
               M     0     3
               F     4     7
        """
        return PGroupMaker(self)

    @property
    def labels(self):
        """
        labels of the axis.
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
            # LGroup ticks, it does not make a difference since a list of LGroup
            # and an ndarray of LGroup are both arrays of pointers)
            ticks = _to_ticks(labels, parse_single_int=True)
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
        >>> age = Axis(range(10), 'age')
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
        ValueError: Axis to append must (not) be wildcard if self is (not) wildcard
        """
        other = labels if isinstance(labels, Axis) else Axis(labels)
        if self.iswildcard != other.iswildcard:
            raise ValueError ("Axis to append must (not) be wildcard if self is (not) wildcard")
        labels = self._length + other._length if self.iswildcard else np.append(self.labels, other.labels)
        return Axis(labels, self.name)

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
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
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
        raise NotImplementedError('Axis.all is deprecated. Use {}[:] >> {} instead.'
                                  .format(axis_name, repr(group_name)))

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
            Subaxis. If key is a None slice and name is None, the original Axis is returned.
            If key is a LArray, the list of axes is returned.

        Examples
        --------
        >>> age = Axis(range(100), 'age')
        >>> age.subaxis(range(10, 19), 'teenagers')
        Axis([10, 11, 12, 13, 14, 15, 16, 17, 18], 'teenagers')
        """
        if name is None and isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            return self
        # we must NOT modify the axis name, even though this creates a new axis that is independent from the original
        # one because the original name is probably what users will want to use to filter
        if name is None:
            name = self.name
        if isinstance(key, ABCLArray):
            return tuple(key.axes)
        # TODO: compute length for wildcard axes more efficiently
        labels = len(self.labels[key]) if self.iswildcard else self.labels[key]
        return Axis(labels, name)

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

    def equals(self, other):
        """
        Checks if self is equal to another axis.
        Two axes are equal if they have the same name and label(s).

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
        return isinstance(other, Axis) and self.name == other.name and self.iswildcard == other.iswildcard and \
               (len(self) == len(other) if self.iswildcard else np.array_equal(self.labels, other.labels))

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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')

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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')
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
        >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')
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

        key is a label-based key (other axis, slice and fancy indexing are supported)

        Returns
        -------
        Group
            group containing selected label(s)/position(s).

        Notes
        -----
        key is label-based (slice and fancy indexing are supported)
        """
        # if isinstance(key, basestring):
        #     key = to_keys(key)

        def isscalar(k):
            return np.isscalar(k) or (isinstance(k, Group) and np.isscalar(k.key))

        if isinstance(key, Axis):
            key = key.labels

        # the not all(np.isscalar) part is necessary to support axis[a, b, c] and axis[[a, b, c]]
        if isinstance(key, (tuple, list)) and not all(isscalar(k) for k in key):
            # this creates a group for each key if it wasn't and retargets PGroup
            list_res = [self[k] for k in key]
            return list_res if isinstance(key, list) else tuple(list_res)

        name = key.name if isinstance(key, Group) else None
        return LGroup(key, name, self)

    def _ipython_key_completions_(self):
        return list(self.labels)

    def __contains__(self, key):
        return _to_tick(key) in self._mapping

    def __hash__(self):
        return id(self)

    def _is_key_type_compatible(self, key):
        key_kind = np.dtype(type(key)).kind
        label_kind = self.labels.dtype.kind
        # on Python2, ascii-only unicode string can match byte strings (and vice versa), so we shouldn't be more picky
        # here than dict hashing
        str_key = key_kind in ('S', 'U')
        str_label = label_kind in ('S', 'U')
        py2_str_match = PY2 and str_key and str_label
        # object kind can match anything
        return key_kind == label_kind or key_kind == 'O' or label_kind == 'O' or py2_str_match

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
        >>> people = Axis(['John Doe', 'Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')
        >>> people.translate('Waldo')
        3
        >>> people.translate(people.matches('Bruce'))
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
                    return mapping[potential_tick]
            # we must catch TypeError because key might not be hashable (eg slice)
            # IndexError is for when mapping is an ndarray
            except (KeyError, TypeError, IndexError):
                pass

        if isinstance(key, basestring):
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

            # transform "specially formatted strings" for slices, lists, LGroup and PGroup to actual objects
            key = _to_key(key)

        if not PY2 and isinstance(key, range):
            key = list(key)

        if isinstance(key, PGroup):
            assert key.axis is self
            return key.key

        if isinstance(key, LGroup):
            # this can happen when key was passed as a string and converted to an LGroup via _to_key
            if isinstance(key.axis, basestring) and key.axis != self.name:
                raise KeyError(key)

            # at this point we do not care about the axis nor the name
            key = key.key

        if isinstance(key, slice):
            start = mapping[key.start] if key.start is not None else None
            # stop is inclusive in the input key and exclusive in the output !
            stop = mapping[key.stop] + 1 if key.stop is not None else None
            return slice(start, stop, key.step)
        # XXX: bool LArray do not pass through???
        elif isinstance(key, np.ndarray) and key.dtype.kind is 'b' and bool_passthrough:
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
        elif isinstance(key, ABCLArray):
            from .array import LArray
            return LArray(self.translate(key.data), key.axes)
        else:
            # the first mapping[key] above will cover most cases.
            # This code path is only used if the key was given in "non normalized form"
            assert np.isscalar(key), "%s (%s) is not scalar" % (key, type(key))
            # key is scalar (integer, float, string, ...)
            if self._is_key_type_compatible(key):
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
        return 'Axis(%r, %r)' % (labels, self.name)

    def labels_summary(self):
        """
        Returns a short representation of the labels.

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
        """
        Method factory to create binary operators special methods.
        """
        fullname = '__%s__' % opname

        def opmethod(self, other):
            # give a chance to AxisCollection.__rXXX__ ops to trigger
            if isinstance(other, AxisCollection):
                # in this case it is indeed return NotImplemented, not raise NotImplementedError!
                return NotImplemented

            from .array import labels_array
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
        from .array import labels_array
        return labels_array(self)

    def copy(self):
        """
        Returns a copy of the axis.
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

    def replace(self, old, new=None):
        """
        Returns a new axis with some labels replaced.

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
            assert new is not None and np.isscalar(new), "%s is not a scalar but a %s" % (new, type(new).__name__)
            old = [old]
            new = [new]
        else:
            seq = (tuple, list, np.ndarray)
            assert isinstance(old, seq), "%s is not a sequence but a %s" % (old, type(old).__name__)
            assert isinstance(new, seq), "%s is not a sequence but a %s" % (new, type(new).__name__)
            assert len(old) == len(new)
        # using object dtype because new labels length can be larger than the fixed str length in the self.labels array
        labels = self.labels.astype(object)
        indices = self.translate(old)
        labels[indices] = new
        return Axis(labels, self.name)

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

    def _rename(self, name):
        raise TypeError("Axis._rename is deprecated, use Axis.rename instead")

    def union(self, other):
        """Returns axis with the union of this axis labels and other labels.

        Labels relative order will be kept intact, but only unique labels will be returned. Labels from this axis will
        be before labels from other.

        Parameters
        ----------
        other : Axis or any sequence of labels
            other labels

        Returns
        -------
        Axis

        Examples
        --------
        >>> letters = Axis('letters=a,b')
        >>> letters.union(Axis('letters=b,c'))
        Axis(['a', 'b', 'c'], 'letters')
        >>> letters.union(['b', 'c'])
        Axis(['a', 'b', 'c'], 'letters')
        """
        if isinstance(other, Axis):
            other = other.labels
        unique_labels = []
        seen = set()
        unique_list(self.labels, unique_labels, seen)
        unique_list(other, unique_labels, seen)
        return Axis(unique_labels, self.name)

    def intersection(self, other):
        """Returns axis with the (set) intersection of this axis labels and other labels.

        In other words, this will use labels from this axis if they are also in other. Labels relative order will be
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
        >>> letters = Axis('letters=a,b')
        >>> letters.intersection(Axis('letters=b,c'))
        Axis(['b'], 'letters')
        >>> letters.intersection(['b', 'c'])
        Axis(['b'], 'letters')
        """
        if isinstance(other, Axis):
            other = other.labels
        seen = set(other)
        return Axis([l for l in self.labels if l in seen], self.name)

    def difference(self, other):
        """Returns axis with the (set) difference of this axis labels and other labels.

        In other words, this will use labels from this axis if they are not in other. Labels relative order will be
        kept intact, but only unique labels will be returned.

        Parameters
        ----------
        other : Axis or any sequence of labels
            other labels

        Returns
        -------
        Axis

        Examples
        --------
        >>> letters = Axis('letters=a,b')
        >>> letters.difference(Axis('letters=b,c'))
        Axis(['a'], 'letters')
        >>> letters.difference(['b', 'c'])
        Axis(['a'], 'letters')
        """
        if isinstance(other, Axis):
            other = other.labels
        seen = set(other)
        return Axis([l for l in self.labels if l not in seen], self.name)

    def align(self, other, join='outer'):
        """Align axis with other object using specified join method.

        Parameters
        ----------
        other : Axis or label sequence
        join : {'outer', 'inner', 'left', 'right'}, optional
            Defaults to 'outer'.

        Returns
        -------
        Axis
            Aligned axis

        See Also
        --------
        LArray.align

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
        """
        assert join in {'outer', 'inner', 'left', 'right'}
        if join == 'outer':
            return self.union(other)
        elif join == 'inner':
            return self.intersection(other)
        elif join == 'left':
            return self
        elif join == 'right':
            if not isinstance(other, Axis):
                other = Axis(other)
            return other


def _make_axis(obj):
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


# not using OrderedDict because it does not support indices-based getitem
# not using namedtuple because we have to know the fields in advance (it is a one-off class) and we need more
# functionality than just a named tuple
class AxisCollection(object):
    """
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
        elif isinstance(axes, (int, long, Group, Axis)):
            axes = [axes]
        elif isinstance(axes, str):
            axes = [axis.strip() for axis in axes.split(';')]

        axes = [_make_axis(axis) for axis in axes]
        assert all(isinstance(a, Axis) for a in axes)
        # check for duplicate axes
        dupe_axes = list(duplicates(axes))
        if dupe_axes:
            axis = dupe_axes[0]
            raise ValueError("Cannot have multiple occurrences of the same axis object in a collection ('%s' -- %s "\
                             "with id %d). Several axes with the same name are allowed though (but not recommended)."
                             % (axis.name, axis.labels_summary(), id(axis)))
        self._list = axes
        self._map = {axis.name: axis for axis in axes if axis.name is not None}

        # # check dupes on each axis
        # for axis in axes:
        #     axis_dupes = list(duplicates(axis.labels))
        #     if axis_dupes:
        #         dupe_labels = ', '.join(str(l) for l in axis_dupes)
        #         warnings.warn("duplicate labels found for axis %s: %s" % (axis.name, dupe_labels),
        #                       category=UserWarning, stacklevel=2)
        #
        # # check dupes between axes. Using unique to not spot the dupes within the same axis that we just displayed.
        # all_labels = chain(*[np.unique(axis.labels) for axis in axes])
        # dupe_labels = list(duplicates(all_labels))
        # if dupe_labels:
        #     label_axes = [(label, ', '.join(display_name for axis, display_name in zip(axes, self.display_names)
        #                                     if label in axis))
        #                   for label in dupe_labels]
        #     dupes = '\n'.join("{} is valid in {{{}}}".format(label, axes) for label, axes in label_axes)
        #     warnings.warn("ambiguous labels found:\n%s" % dupes, category=UserWarning, stacklevel=5)

    def __dir__(self):
        # called by dir() and tab-completion at the interactive prompt, must return a list of any valid getattr key.
        # dir() takes care of sorting but not uniqueness, so we must ensure that.
        names = set(axis.name for axis in self._list if axis.name is not None)
        return list(set(dir(self.__class__)) | names)

    def __iter__(self):
        return iter(self._list)

    def __getattr__(self, key):
        try:
            return self._map[key]
        except KeyError:
            return self.__getattribute__(key)

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before _map exists, which leads to
    # an infinite recursion)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getitem__(self, key):
        if isinstance(key, Axis):
            try:
                key = self.index(key)
            # transform ValueError to KeyError
            except ValueError:
                if key.name is None:
                    raise KeyError("axis '%s' not found in %s" % (key, self))
                else:
                    # we should NOT check that the object is the same, so that we can use AxisReference objects to
                    # target real axes
                    key = key.name

        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, (tuple, list)):
            if any(axis is Ellipsis for axis in key):
                ellipsis_idx = index_by_id(key, Ellipsis)
                # going through lists (instead of doing self[key[:ellipsis_idx]] to avoid problems with anonymous axes
                before_ellipsis = [self[k] for k in key[:ellipsis_idx]]
                after_ellipsis = [self[k] for k in key[ellipsis_idx + 1:]]
                ellipsis_axes = list(self - before_ellipsis - after_ellipsis)
                return AxisCollection(before_ellipsis + ellipsis_axes + after_ellipsis)
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

    def _ipython_key_completions_(self):
        return list(self._map.keys())

    # XXX: I wonder if this whole positional crap should really be part of AxisCollection or the default behavior.
    # It could either be moved to make_numpy_broadcastable or made non default
    def get_by_pos(self, key, i):
        """
        Returns axis corresponding to a key, or to position i if the key has no name and key object not found.

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
                        raise ValueError("axis %s is not compatible with %s" % (res, key))
                # XXX: KeyError instead?
                raise ValueError("axis %s not found in %s" % (key, self))
        else:
            return self[key]

    def __setitem__(self, key, value):
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
            if isinstance(value, (int, basestring, list, tuple)):
                value = Axis(value)
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
        # TODO: move this to a class method (possibly private) so that we are sure we use same heuristic than in .extend
        def contains(col, i, axis):
            return axis in col or (axis.name is None and i in col)

        return AxisCollection([axis for i, axis in enumerate(self) if contains(other, i, axis)])

    def __eq__(self, other):
        """
        Other collection compares equal if all axes compare equal and in the same order. Works with a list.
        """
        if self is other:
            return True
        if not isinstance(other, list):
            other = list(other)
        return len(self._list) == len(other) and all(a.equals(b) for a, b in zip(self._list, other))

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
        Tests if input is an Axis object or the name of an axis contained in self.

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
        return isinstance(value, Axis) or (isinstance(value, basestring) and value in self)
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
        Returns axis corresponding to key. If not found, the argument `name` is used to create a new Axis.
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

    def get_all(self, key):
        """
        Returns all axes from key if present and length 1 wildcard axes otherwise.

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

    def keys(self):
        """
        Returns list of all axis names.

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

    def pop(self, axis=-1):
        """
        Removes and returns an axis.

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

    def append(self, axis):
        """
        Appends axis at the end of the collection.

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

    def check_compatible(self, axes):
        """
        Checks if axes passed as argument are compatible with those contained in the collection.
        Raises ValueError if not.

        See Also
        --------
        Axis.iscompatible
        """
        for i, axis in enumerate(axes):
            local_axis = self.get_by_pos(axis, i)
            if not local_axis.iscompatible(axis):
                raise ValueError("incompatible axes:\n%r\nvs\n%r" % (axis, local_axis))

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
            raise TypeError("AxisCollection can only be extended by a sequence of Axis, not %s" % type(axes).__name__)
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
                    raise ValueError("incompatible axes:\n%r\nvs\n%r" % (axis, old_axis))
                if replace_wildcards and old_axis.iswildcard:
                    self[old_axis] = axis

    def index(self, axis):
        """
        Returns the index of axis.

        `axis` can be a name or an Axis object (or an index). If the Axis object itself exists in the list, index()
        will return it. Otherwise, it will return the index of the local axis with the same name than the key (whether
        it is compatible or not).

        Parameters
        ----------
        axis : Axis or int or str
            Can be the axis itself or its position (returned if represents a valid index) or its name.

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
        if isinstance(axis, int):
            if -len(self) <= axis < len(self):
                return axis
            else:
                raise ValueError("axis %d is not in collection" % axis)
        elif isinstance(axis, Axis):
            try:
                # first look by id. This avoids testing labels of each axis and makes sure the result is correct even
                # if there are several axes with no name and the same labels.
                return index_by_id(self._list, axis)
            except ValueError:
                name = axis.name
        else:
            name = axis
        if name is None:
            raise ValueError("%r is not in collection" % axis)
        return self.names.index(name)

    # XXX: we might want to return a new AxisCollection (same question for other inplace operations:
    # append, extend, pop, __delitem__, __setitem__)
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

    def copy(self):
        """
        Returns a copy.
        """
        return self[:]

    def replace(self, axes_to_replace=None, new_axis=None, inplace=False, **kwargs):
        """Replace one, several or all axes of the collection.

        Parameters
        ----------
        axes_to_replace : axis ref or dict {axis ref: axis} or list of tuple (axis ref, axis) \
                          or list of Axis or AxisCollection, optional
            Axes to replace. If a single axis reference is given, the `new_axis` argument must be provided.
            If a list of Axis or an AxisCollection is given, all axes will be replaced by the new ones.
            In that case, the number of new axes must match the number of the old ones. Defaults to None.
        new_axis : axis ref, optional
            New axis if `axes_to_replace` contains a single axis reference. Defaults to None.
        inplace : bool, optional
            Whether or not to modify the original object or return a new AxisCollection and leave the original intact.
            Defaults to False.
        **kwargs : Axis
            New axis for each axis to replace given as a keyword argument.

        Returns
        -------
        AxisCollection
            AxisCollection with axes replaced.

        Examples
        --------
        >>> from larray import ndtest, ndrange
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
        >>> arr = ndrange([row, column])
        >>> axes.replace(arr.axes)
        AxisCollection([
            Axis(['r0', 'r1'], 'row'),
            Axis(['c0', 'c1', 'c2'], 'column')
        ])
        """
        if not PY2 and isinstance(axes_to_replace, zip):
            axes_to_replace = list(axes_to_replace)

        if isinstance(axes_to_replace, (list, AxisCollection)) and \
                all([isinstance(axis, Axis) for axis in axes_to_replace]):
            if len(axes_to_replace) != len(self):
                raise ValueError('{} axes given as argument, expected {}'.format(len(axes_to_replace), len(self)))
            axes = axes_to_replace
        else:
            axes = self if inplace else self[:]
            if isinstance(axes_to_replace, dict):
                items = list(axes_to_replace.items())
            elif isinstance(axes_to_replace, list):
                assert all([isinstance(item, tuple) and len(item) == 2 for item in axes_to_replace])
                items = axes_to_replace[:]
            elif isinstance(axes_to_replace, (basestring, Axis, int)):
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
        return AxisCollection([axis for i, axis in enumerate(self) if i not in to_remove])

    def translate_full_key(self, key):
        """
        Translates a label-based key to a positional key.

        Parameters
        ----------
        key : tuple
            A full label-based key. All dimensions must be present and in the correct order.

        Returns
        -------
        tuple
            A full positional key.

        See Also
        --------
        Axis.translate

        Examples
        --------
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
        >>> AxisCollection([age,sex,time]).translate_full_key((':', 'F', 2009))
        (slice(None, None, None), 1, 2)
        """
        assert len(key) == len(self)
        return tuple(axis.translate(axis_key) for axis_key, axis in zip(key, self))

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
        >>> age = Axis(range(10), 'age')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
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
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
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
        >>> a = Axis(2, 'a')
        >>> b = Axis(2)
        >>> c = Axis(2, 'c')
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
    def shape(self):
        """
        Returns the shape of the collection.

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
        >>> age = Axis(range(20), 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
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
        >>> age = Axis(20, 'age')
        >>> sex = Axis('sex=M,F')
        >>> time = Axis([2007, 2008, 2009, 2010], 'time')
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
        axes : tuple, list, AxisCollection of axes or list of combination of those or dict, optional
            axes to combine. Tuple, list or AxisCollection will combine several axes into one. To chain several axes 
            combinations, pass a list of tuple/list/AxisCollection of axes. To set the name(s) of resulting axis(es), 
            use a {(axes, to, combine): 'new_axis_name'} dictionary. Defaults to all axes.
        sep : str, optional
            delimiter to use for combining. Defaults to '_'.
        wildcard : bool, optional
            whether or not to produce a wildcard axis even if the axes to combine are not.
            This is much faster, but loose axes labels.
        front_if_spread : bool, optional
            whether or not to move the combined axis at the front (it will be the first axis) if the combined axes are
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
                    combined_labels = [sep.join(str(l) for l in p)
                                       for p in product(*_axes.labels)]

                combined_axis = Axis(combined_labels, combined_name)
            new_axes = new_axes - _axes
            new_axes.insert(combined_axis_pos, combined_axis)
        return new_axes

    def split_axes(self, axes, sep='_', names=None, regex=None):
        """Split axes and returns a new collection

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
            use regex instead of delimiter to split labels. Defaults to None.

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
        >>> combined.split_axes('a_b')
        AxisCollection([
            Axis(['a0', 'a1'], 'a'),
            Axis(['b0', 'b1', 'b2'], 'b')
        ])

        Split labels using regex

        >>> combined = AxisCollection('a_b = a0b0..a1b2')
        >>> combined
        AxisCollection([
            Axis(['a0b0', 'a0b1', 'a0b2', 'a1b0', 'a1b1', 'a1b2'], 'a_b')
        ])
        >>> combined.split_axes('a_b', regex='(\w{2})(\w{2})')
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
        if axes is None:
            axes = {axis: None for axis in self if sep in axis.name}
        elif isinstance(axes, (int, basestring, Axis)):
            axes = {axes: None}
        elif isinstance(axes, (list, tuple)):
            if all(isinstance(axis, (int, basestring, Axis)) for axis in axes):
                axes = {axis: None for axis in axes}
            else:
                raise ValueError("Expected tuple or list of int, string or Axis instances")
        # axes should be a dict at this time
        assert isinstance(axes, dict)

        new_axes = self[:]
        for axis, names in axes.items():
            axis = new_axes[axis]
            axis_index = new_axes.index(axis)
            if names is None:
                if sep not in axis.name:
                    raise ValueError('{} not found in axis name ({})'.format(sep, axis.name))
                else:
                    _names = axis.name.split(sep)
            elif isinstance(names, str):
                if sep not in names:
                    raise ValueError('{} not found in names ({})'.format(sep, names))
                else:
                    _names = names.split(sep)
            else:
                assert all(isinstance(name, str) for name in names)
                _names = names

            if not regex:
                # gives us an array of lists
                split_labels = np.char.split(axis.labels, sep)
            else:
                rx = re.compile(regex)
                split_labels = [rx.match(l).groups() for l in axis.labels]
            # not using np.unique because we want to keep the original order
            axes_labels = [unique_list(ax_labels) for ax_labels in zip(*split_labels)]
            split_axes = [Axis(axis_labels, name) for axis_labels, name in zip(axes_labels, _names)]
            new_axes = new_axes[:axis_index] + split_axes + new_axes[axis_index + 1:]
        return new_axes

    def align(self, other, join='outer', axes=None):
        """Align this axis collection with another.

        This ensures all common axes are compatible.

        Parameters
        ----------
        other : AxisCollection
        join : {'outer', 'inner', 'left', 'right'}, optional
            Defaults to 'outer'.
        axes : AxisReference or sequence of them, optional
            Axes to align. Need to be valid in both arrays. Defaults to None (all common axes). This must be specified
            when mixing anonymous and non-anonymous axes.

        Returns
        -------
        (left, right) : (AxisCollection, AxisCollection)
            Aligned collections

        See Also
        --------
        LArray.align

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
        if join not in {'outer', 'inner', 'left', 'right'}:
            raise ValueError("join should be one of 'outer', 'inner', 'left' or 'right'")
        other = other if isinstance(other, AxisCollection) else AxisCollection(other)

        # if axes not specified
        if axes is None:
            # and we have only anonymous axes on both sides
            if all(name is None for name in self.names) and all(name is None for name in other.names):
                # use N first axes by position
                join_axes = list(range(min(len(self), len(other))))
            elif any(name is None for name in self.names) or any(name is None for name in other.names):
                raise ValueError("axes collections with mixed anonymous/non anonymous axes are not supported by align"
                                 "without specifying axes explicitly")
            else:
                assert all(name is not None for name in self.names) and all(name is not None for name in other.names)
                # use all common axes
                join_axes = list(OrderedSet(self.names) & OrderedSet(other.names))
        else:
            if isinstance(axes, (int, str, Axis)):
                axes = [axes]
            join_axes = axes
        new_axes = [self_axis.align(other_axis, join=join)
                    for self_axis, other_axis in zip(self[join_axes], other[join_axes])]
        axes_changes = list(zip(join_axes, new_axes))
        return self.replace(axes_changes), other.replace(axes_changes)


class AxisReference(ABCAxisReference, ExprNode, Axis):
    def __init__(self, name):
        self.name = name
        self._labels = None
        self._iswildcard = False

    def translate(self, key):
        raise NotImplementedError("an AxisReference (X.) cannot translate labels")

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

    # needed because ExprNode.__hash__ (which is object.__hash__) takes precedence over Axis.__hash__
    def __hash__(self):
        return id(self)


class AxisReferenceFactory(object):
    # needed to make pickle work (because we have a __getattr__ which does not return AttributeError on __getstate__)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, key):
        return AxisReference(key)

    def __getitem__(self, key):
        return AxisReference(key)


X = AxisReferenceFactory()


class DeprecatedAxisReferenceFactory(object):
    # needed to make pickle work (because we have a __getattr__ which does not return AttributeError on __getstate__)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, key):
        warnings.warn("Special variable 'x' is deprecated, use 'X' instead", FutureWarning, stacklevel=2)
        return AxisReference(key)

    def __getitem__(self, key):
        warnings.warn("Special variable 'x' is deprecated, use 'X' instead", FutureWarning, stacklevel=2)
        return AxisReference(key)


x = DeprecatedAxisReferenceFactory()