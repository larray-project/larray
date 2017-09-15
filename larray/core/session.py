# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import warnings
from collections import OrderedDict

import numpy as np

from larray.core.axis import Axis
from larray.core.array import LArray, larray_nan_equal, get_axes, ndtest, zeros, zeros_like, sequence
from larray.util.misc import float_error_handler_factory, is_interactive_interpreter
from larray.io.session import check_pattern, handler_classes, ext_default_engine


# XXX: inherit from OrderedDict or LArray?
class Session(object):
    """
    Groups several array objects together.

    Parameters
    ----------
    args : str or dict of str, array or iterable of tuples (str, array)
        Name of file to load or dictionary containing couples (name, array).
    kwargs : dict of str, array
        List of arrays to add written as 'name'=array, ...

    Examples
    --------
    >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))

    create a Session by passing a list of pairs (name, array)

    >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])

    create a Session using keyword arguments (but you lose order on Python < 3.6)

    >>> s = Session(arr1=arr1, arr2=arr2, arr3=arr3)

    create a Session by passing a dictionary (but you lose order on Python < 3.6)

    >>> s = Session({'arr1': arr1, 'arr2': arr2, 'arr3': arr3})

    load Session from file

    >>> s = Session('my_session.h5')  # doctest: +SKIP
    """
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, '_objects', OrderedDict())

        if len(args) == 1:
            a0 = args[0]
            if isinstance(a0, str):
                # assume a0 is a filename
                self.load(a0)
            else:
                items = a0.items() if isinstance(a0, dict) else a0
                # assume we have an iterable of tuples
                for k, v in items:
                    self[k] = v
        else:
            self.add(*args, **kwargs)

    # XXX: behave like a dict and return keys instead?
    def __iter__(self):
        return iter(self.values())

    def add(self, *args, **kwargs):
        """
        Adds objects to the current session.

        Parameters
        ----------
        args : array
            List of objects to add. Objects must have an attribute 'name'.
        kwargs : dict of str, array
            List of objects to add written as 'name'=array, ...

        Examples
        --------
        >>> s = Session()
        >>> axis1, axis2 = Axis('x=x0..x2'), Axis('y=y0..y2')
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
        >>> s.add(axis1, axis2, arr1=arr1, arr2=arr2, arr3=arr3)
        >>> # print item's names in sorted order
        >>> s.names
        ['arr1', 'arr2', 'arr3', 'x', 'y']
        """
        for arg in args:
            self[arg.name] = arg
        for k, v in kwargs.items():
            self[k] = v

    def _ipython_key_completions_(self):
        return list(self.keys())

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._objects[self.keys()[key]]
        elif isinstance(key, LArray):
            assert np.issubdtype(key.dtype, np.bool_)
            assert key.ndim == 1
            # only keep True values
            truenames = key[key].axes[0].labels
            return Session([(name, self[name]) for name in truenames])
        elif isinstance(key, (tuple, list)):
            assert all(isinstance(k, str) for k in key)
            return Session([(k, self[k]) for k in key])
        else:
            return self._objects[key]

    def get(self, key, default=None):
        """
        Returns the array object corresponding to the key.
        If the key doesn't correspond to any array object, a default one can be returned.

        Parameters
        ----------
        key : str
            Name of the array.
        default : array, optional
            Returned array if the key doesn't correspond to any array of the current session.

        Returns
        -------
        LArray
            Array corresponding to the given key or a default one if not found.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])
        >>> arr = s.get('arr1')
        >>> arr
        a\\b  b0  b1
         a0   0   1
         a1   2   3
        >>> arr = s.get('arr4', zeros('a=a0,a1;b=b0,b1', dtype=int))
        >>> arr
        a\\b  b0  b1
         a0   0   0
         a1   0   0
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        self._objects[key] = value

    def __delitem__(self, key):
        del self._objects[key]

    def __getattr__(self, key):
        if key in self._objects:
            return self._objects[key]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, key))

    def __setattr__(self, key, value):
        self._objects[key] = value

    def __delattr__(self, key):
        del self._objects[key]

    def __dir__(self):
        return list(set(dir(self.__class__)) | set(self.keys()))

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before ._objects exists, which leads to
    # an infinite recursion)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        # equivalent to self.__dict__ = d (we need this form because __setattr__ is overridden)
        object.__setattr__(self, '__dict__', d)

    def load(self, fname, names=None, engine='auto', display=False, **kwargs):
        """
        Loads array objects from a file, or several .csv files.

        WARNING: never load a file using the pickle engine (.pkl or .pickle) from an untrusted source, as it can lead
        to arbitrary code execution.

        Parameters
        ----------
        fname : str
            This can be either the path to a single file, a path to a directory containing .csv files or a pattern
            representing several .csv files.
        names : list of str, optional
            List of arrays to load. If `fname` is None, list of paths to CSV files.
            Defaults to all valid objects present in the file/directory.
        engine : {'auto', 'pandas_csv', 'pandas_hdf', 'pandas_excel', 'xlwings_excel', 'pickle'}, optional
            Load using `engine`. Defaults to 'auto' (use default engine for the format guessed from the file extension).
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults to False.

        Examples
        --------
        In one module

        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))   # doctest: +SKIP
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])  # doctest: +SKIP
        >>> s.save('input.h5')                                             # doctest: +SKIP

        In another module

        >>> s = Session()                                 # doctest: +SKIP
        >>> s.load('input.h5', ['arr1', 'arr2', 'arr3'])  # doctest: +SKIP
        >>> arr1, arr2, arr3 = s['arr1', 'arr2', 'arr3']  # doctest: +SKIP
        >>> # only if you know the order of arrays stored in session
        >>> arr1, arr2, arr3 = s.values()                 # doctest: +SKIP

        Using .csv files (assuming the same session as above)

        >>> s.save('data')                                # doctest: +SKIP
        >>> s = Session()                                 # doctest: +SKIP
        >>> # load all .csv files starting with "output" in the data directory
        >>> s.load('data')                                # doctest: +SKIP
        >>> # or equivalently in this case
        >>> s.load('data/arr*.csv')                       # doctest: +SKIP
        """
        if display:
            print("opening", fname)
        if fname is None:
            if all([os.path.splitext(name)[1] == '.csv' for name in names]):
                engine = ext_default_engine['csv']
            else:
                raise ValueError("List of paths to only CSV files expected. Got {}".format(names))
        if engine == 'auto':
            _, ext = os.path.splitext(fname)
            ext = ext.strip('.') if '.' in ext else 'csv'
            engine = ext_default_engine[ext]
        handler_cls = handler_classes[engine]
        handler = handler_cls(fname)
        arrays = handler.read_arrays(names, display=display, **kwargs)
        for k, v in arrays.items():
            self[k] = v

    def save(self, fname, names=None, engine='auto', overwrite=True, display=False, **kwargs):
        """
        Dumps all array objects from the current session to a file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. If `fname` is None, list of paths to CSV files.
            Defaults to all objects present in the Session.
        engine : {'auto', 'pandas_csv', 'pandas_hdf', 'pandas_excel', 'xlwings_excel', 'pickle'}, optional
            Dump using `engine`. Defaults to 'auto' (use default engine for the format guessed from the file extension).
        overwrite: bool, optional
            Whether or not to overwrite an existing file, if any. Ignored for CSV files. If False, file is updated.
            Defaults to True.
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))   # doctest: +SKIP
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])  # doctest: +SKIP

        Save all arrays

        >>> s.save('output.h5')  # doctest: +SKIP

        Save only some arrays

        >>> s.save('output.h5', ['arr1', 'arr3'])  # doctest: +SKIP

        Update file

        >>> arr1, arr4 = ndtest((3, 3)), ndtest((2, 3))     # doctest: +SKIP
        >>> s2 = Session([('arr1', arr1), ('arr4', arr4)])  # doctest: +SKIP
        >>> # replace arr1 and add arr4 in file output.h5
        >>> s2.save('output.h5', overwrite=False)           # doctest: +SKIP
        """
        if engine == 'auto':
            _, ext = os.path.splitext(fname)
            ext = ext.strip('.') if '.' in ext else 'csv'
            engine = ext_default_engine[ext]
        if overwrite and engine != ext_default_engine['csv'] and os.path.isfile(fname):
            os.remove(fname)
        handler_cls = handler_classes[engine]
        handler = handler_cls(fname)
        items = self.filter(kind=LArray).items()
        if names is not None:
            names_set = set(names)
            items = [(k, v) for k, v in items if k in names_set]
        handler.dump_arrays(items, display=display, **kwargs)

    def to_globals(self, names=None, depth=0, warn=True, inplace=False):
        """
        Create global variables out of objects in the session.

        Parameters
        ----------
        names : list of str or None, optional
            List of names of objects to convert to globals. Defaults to all objects present in the Session.
        depth : int, optional
            depth of call stack where to create the variables. 0 is where to_globals was called, 1 the caller of
            to_globals, etc. Defaults to 0.
        warn : bool, optional
            Whether or not to warn the user that this method should only be used in an interactive console (see below).
            Defaults to True.
        inplace : bool, optional
            If True, to_globals will assume all arrays already exist and have the same axes and will replace their
            content instead of creating new variables. Non array variables in the session will be ignored.
            Defaults to False.

        Notes
        -----
        This method should usually only be used in an interactive console and not in a script. Code editors are
        confused by this kind of manipulation and will likely consider as invalid the code using variables created in
        this way. Additionally, when using this method auto-completion, "show definition", "go to declaration" and
        other similar code editor features will probably not work for the variables created in this way and any
        variable derived from them.

        Examples
        --------
        >>> s = Session(arr1=ndtest(3), arr2=ndtest((2, 2)))
        >>> s.to_globals()
        >>> arr1
        a  a0  a1  a2
            0   1   2
        >>> arr2
        a\\b  b0  b1
         a0   0   1
         a1   2   3
        """
        # noinspection PyProtectedMember
        if warn and not is_interactive_interpreter():
            warnings.warn("Session.to_globals should usually only be used in interactive consoles and not in scripts. "
                          "Use warn=False to deactivate this warning.",
                          RuntimeWarning, stacklevel=2)
        d = sys._getframe(depth + 1).f_globals
        items = self.items()
        if names is not None:
            names_set = set(names)
            items = [(k, v) for k, v in items if k in names_set]
        if inplace:
            for k, v in items:
                if k not in d:
                    raise ValueError("'{}' not found in current namespace. Session.to_globals(inplace=True) requires "
                                     "all arrays to already exist.".format(k))
                if not isinstance(v, LArray):
                    continue
                if not d[k].axes == v.axes:
                    raise ValueError("Session.to_globals(inplace=True) requires the existing (destination) arrays "
                                     "to have the same axes than those stored in the session and this is not the case "
                                     "for '{}'.\nexisting: {}\nsession: {}".format(k, d[k].info, v.info))
                d[k][:] = v
        else:
            for k, v in items:
                d[k] = v

    def to_pickle(self, fname, names=None, overwrite=True, display=False, **kwargs):
        """
        Dumps all array objects from the current session to a file using pickle.

        WARNING: never load a pickle file (.pkl or .pickle) from an untrusted source, as it can lead to arbitrary code
        execution.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects present in the Session.
        overwrite: bool, optional
            Whether or not to overwrite an existing file, if any.
            If False, file is updated. Defaults to True.
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))   # doctest: +SKIP
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])  # doctest: +SKIP

        Save all arrays

        >>> s.to_pickle('output.pkl')  # doctest: +SKIP

        Save only some arrays

        >>> s.to_pickle('output.pkl', ['arr1', 'arr3'])  # doctest: +SKIP
        """
        self.save(fname, names, ext_default_engine['pkl'], overwrite, display, **kwargs)

    def dump(self, fname, names=None, engine='auto', display=False, **kwargs):
        warnings.warn("Method dump is deprecated. Use method save instead.", FutureWarning, stacklevel=2)
        self.save(fname, names, engine, display, **kwargs)

    def to_hdf(self, fname, names=None, overwrite=True, display=False, **kwargs):
        """
        Dumps all array objects from the current session to an HDF file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects present in the Session.
        overwrite: bool, optional
            Whether or not to overwrite an existing file, if any.
            If False, file is updated. Defaults to True.
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))   # doctest: +SKIP
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])  # doctest: +SKIP

        Save all arrays

        >>> s.to_hdf('output.h5')  # doctest: +SKIP

        Save only some arrays

        >>> s.to_hdf('output.h5', ['arr1', 'arr3'])  # doctest: +SKIP
        """
        self.save(fname, names, ext_default_engine['hdf'], overwrite, display, **kwargs)

    def dump_hdf(self, fname, names=None, *args, **kwargs):
        warnings.warn("Method dump_hdf is deprecated. Use method to_hdf instead.", FutureWarning, stacklevel=2)
        self.to_hdf(fname, names, *args, **kwargs)

    def to_excel(self, fname, names=None, overwrite=True, display=False, **kwargs):
        """
        Dumps all array objects from the current session to an Excel file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects present in the Session.
        overwrite: bool, optional
            Whether or not to overwrite an existing file, if any. If False, file is updated. Defaults to True.
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))   # doctest: +SKIP
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])  # doctest: +SKIP

        Save all arrays

        >>> s.to_excel('output.xlsx')  # doctest: +SKIP

        Save only some arrays

        >>> s.to_excel('output.xlsx', ['arr1', 'arr3'])  # doctest: +SKIP
        """
        self.save(fname, names, ext_default_engine['xlsx'], overwrite, display, **kwargs)

    def dump_excel(self, fname, names=None, *args, **kwargs):
        warnings.warn("Method dump_excel is deprecated. Use method to_excel instead.", FutureWarning, stacklevel=2)
        self.to_excel(fname, names, *args, **kwargs)

    def to_csv(self, fname, names=None, display=False, **kwargs):
        """
        Dumps all array objects from the current session to CSV files.

        Parameters
        ----------
        fname : str
            Path for the directory that will contain CSV files.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects present in the Session.
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))   # doctest: +SKIP
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])  # doctest: +SKIP

        Save all arrays

        >>> s.to_csv('./Output')  # doctest: +SKIP

        Save only some arrays

        >>> s.to_csv('./Output', ['arr1', 'arr3'])  # doctest: +SKIP
        """
        self.save(fname, names, ext_default_engine['csv'], display=display, **kwargs)

    def dump_csv(self, fname, names=None, *args, **kwargs):
        warnings.warn("Method dump_csv is deprecated. Use method to_csv instead.", FutureWarning, stacklevel=2)
        self.to_csv(fname, names, *args, **kwargs)

    def filter(self, pattern=None, kind=None):
        """
        Returns a new session with array objects which match some criteria.

        Parameters
        ----------
        pattern : str, optional
            Only keep arrays whose key match `pattern`.
        kind : type, optional
            Only keep arrays which are instances of type `kind`.

        Returns
        -------
        Session
            The filtered session.

        Examples
        --------
        >>> axis = Axis('a=a0..a2')
        >>> test1, test2, zero1 = ndtest((2, 2)), ndtest(4), zeros((3, 2))
        >>> s = Session([('test1', test1), ('test2', test2), ('zero1', zero1), ('axis', axis)])

        Filter using a pattern argument

        >>> s.filter(pattern='test').names
        ['test1', 'test2']

        Filter using kind argument

        >>> s.filter(kind=Axis).names
        ['axis']
        """
        items = self._objects.items()
        if pattern is not None:
            items = [(k, v) for k, v in items if check_pattern(k, pattern)]
        if kind is not None:
            items = [(k, v) for k, v in items if isinstance(v, kind)]
        return Session(items)

    @property
    def names(self):
        """
        Returns the list of names of the array objects in the session.
        The list is sorted alphabetically and does not follow the internal order.

        Returns
        -------
        list of str

        See Also
        --------
        Session.keys

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
        >>> s = Session([('arr2', arr2), ('arr1', arr1), ('arr3', arr3)])
        >>> # print array's names in the alphabetical order
        >>> s.names
        ['arr1', 'arr2', 'arr3']

        >>> # keys() follows the internal order
        >>> list(s.keys())
        ['arr2', 'arr1', 'arr3']
        """
        return sorted(self._objects.keys())

    def copy(self):
        """Returns a copy of the session.
        """
        # this actually *does* a copy of the internal mapping (the mapping is not reused-as is)
        return Session(self._objects)

    def keys(self):
        """
        Returns a view on the session's keys.

        Returns
        -------
        View on the session's keys.

        See Also
        --------
        Session.names

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
        >>> s = Session([('arr2', arr2), ('arr1', arr1), ('arr3', arr3)])
        >>> # similar to names by follows the internal order
        >>> list(s.keys())
        ['arr2', 'arr1', 'arr3']

        >>> # gives the names of arrays in alphabetical order
        >>> s.names
        ['arr1', 'arr2', 'arr3']
        """
        return self._objects.keys()

    def values(self):
        """
        Returns a view on the session's values.

        Returns
        -------
        View on the session's values.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
        >>> s = Session([('arr2', arr2), ('arr1', arr1), ('arr3', arr3)])
        >>> # assuming you know the order of arrays stored in the session
        >>> arr2, arr1, arr3 = s.values()
        >>> # otherwise, prefer the following syntax
        >>> arr1, arr2, arr3 = s['arr1', 'arr2', 'arr3']
        >>> arr1
        a\\b  b0  b1
         a0   0   1
         a1   2   3
        """
        return self._objects.values()

    def items(self):
        """
        Returns a view of the session’s items ((key, value) pairs).

        Returns
        -------
        View on the session's items.

        Examples
        --------
        >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
        >>> s = Session([('arr2', arr2), ('arr1', arr1), ('arr3', arr3)])
        >>> for k, v in s.items():
        ...     print("{}: {}".format(k, v.info))
        arr2: 4
         a [4]: 'a0' 'a1' 'a2' 'a3'
        arr1: 2 x 2
         a [2]: 'a0' 'a1'
         b [2]: 'b0' 'b1'
        arr3: 3 x 2
         a [3]: 'a0' 'a1' 'a2'
         b [2]: 'b0' 'b1'
        """
        return self._objects.items()

    def __repr__(self):
        return 'Session({})'.format(', '.join(self.keys()))

    def __len__(self):
        return len(self._objects)

    # binary operations are dispatched element-wise to all arrays (we consider Session as an array-like)
    def _binop(opname):
        opfullname = '__%s__' % opname

        def opmethod(self, other):
            self_keys = set(self.keys())
            all_keys = list(self.keys()) + [n for n in other.keys() if n not in self_keys]
            with np.errstate(call=_session_float_error_handler):
                res = []
                for name in all_keys:
                    self_array = self.get(name, np.nan)
                    other_array = other.get(name, np.nan)
                    try:
                        res_array = getattr(self_array, opfullname)(other_array)
                    except TypeError:
                        res_array = np.nan
                    res.append((name, res_array))
            return Session(res)
        opmethod.__name__ = opfullname
        return opmethod

    __add__ = _binop('add')
    __sub__ = _binop('sub')
    __mul__ = _binop('mul')
    __truediv__ = _binop('truediv')

    # element-wise method factory
    # unary operations are (also) dispatched element-wise to all arrays
    def _unaryop(opname):
        opfullname = '__%s__' % opname

        def opmethod(self):
            with np.errstate(call=_session_float_error_handler):
                res = []
                for k, v in self.items():
                    try:
                        res_array = getattr(v, opfullname)()
                    except TypeError:
                        res_array = np.nan
                    res.append((k, res_array))
            return Session(res)
        opmethod.__name__ = opfullname
        return opmethod

    __neg__ = _unaryop('neg')
    __pos__ = _unaryop('pos')
    __abs__ = _unaryop('abs')
    __invert__ = _unaryop('invert')

    # XXX: use _binop (ie elementwise comparison instead of aggregating directly?)
    def __eq__(self, other):
        self_keys = set(self.keys())
        all_keys = list(self.keys()) + [n for n in other.keys() if n not in self_keys]
        res = [larray_nan_equal(self.get(key), other.get(key)) for key in all_keys]
        return LArray(res, [Axis(all_keys, 'name')])

    def __ne__(self, other):
        return ~(self == other)

    def transpose(self, *args):
        """Reorder axes of arrays in session, ignoring missing axes for each array.

        Parameters
        ----------
        *args
            Accepts either a tuple of axes specs or axes specs as `*args`. Omitted axes keep their order.
            Use ... to avoid specifying intermediate axes. Axes missing in an array are ignored.

        Returns
        -------
        Session
            Session with each array with reordered axes where appropriate.

        See Also
        --------
        LArray.transpose

        Examples
        --------
        Let us create a test session and a small helper function to display sessions as a short summary.

        >>> arr1 = ndtest((2, 2, 2))
        >>> arr2 = ndtest((2, 2))
        >>> sess = Session([('arr1', arr1), ('arr2', arr2)])
        >>> def print_summary(s):
        ...     print(s.summary("{name} -> {axes_names}"))
        >>> print_summary(sess)
        arr1 -> a, b, c
        arr2 -> a, b

        Put 'b' axis in front of all arrays

        >>> print_summary(sess.transpose('b'))
        arr1 -> b, a, c
        arr2 -> b, a

        Axes missing on an array are ignored ('c' for arr2 in this case)

        >>> print_summary(sess.transpose('c', 'b'))
        arr1 -> c, b, a
        arr2 -> b, a

        Use ... to move axes to the end

        >>> print_summary(sess.transpose(..., 'a'))   # doctest: +SKIP
        arr1 -> b, c, a
        arr2 -> b, a
        """
        def lenient_transpose(v, axes):
            # filter out axes not in arr.axes
            return v.transpose([a for a in axes if a in v.axes or a is Ellipsis])
        return self.apply(lenient_transpose, args)

    def compact(self, display=False):
        """
        Detects and removes "useless" axes (ie axes for which values are constant over the whole axis) for all array
        objects in session

        Parameters
        ----------
        display : bool, optional
            Whether or not to display a message for each array that is compacted

        Returns
        -------
        Session
            A new session containing all compacted arrays

        Examples
        --------
        >>> arr1 = sequence('b=b0..b2', ndtest(3), zeros_like(ndtest(3)))
        >>> arr1
        a\\b  b0  b1  b2
         a0   0   0   0
         a1   1   1   1
         a2   2   2   2
        >>> compact_ses = Session(arr1=arr1).compact(display=True)
        arr1 was constant over {b}
        >>> compact_ses.arr1
        a  a0  a1  a2
            0   1   2
        """
        new_items = []
        for k, v in self._objects.items():
            compacted = v.compact()
            if compacted is not v and display:
                print(k, "was constant over", get_axes(v) - get_axes(compacted))
            new_items.append((k, compacted))
        return Session(new_items)

    def apply(self, func, *args, **kwargs):
        """
        Apply function `func` on elements of the session and return a new session.

        Parameters
        ----------
        func : function
            Function to apply to each element of the session. It should take a single `element` argument and return
            a single value.
        *args : any
            Any extra arguments are passed to the function
        kind : type or tuple of types, optional
            Type(s) of elements `func` will be applied to. Other elements will be left intact. Use ´kind=object´ to
            apply to all kinds of objects. Defaults to LArray.
        **kwargs : any
            Any extra keyword arguments are passed to the function

        Returns
        -------
        Session
            A new session containing all processed elements

        Examples
        --------
        >>> arr1 = ndtest(2)
        >>> arr1
        a  a0  a1
            0   1
        >>> arr2 = ndtest(3)
        >>> arr2
        a  a0  a1  a2
            0   1   2
        >>> sess1 = Session([('arr1', arr1), ('arr2', arr2)])
        >>> sess1
        Session(arr1, arr2)
        >>> def increment(array):
        ...     return array + 1
        >>> sess2 = sess1.apply(increment)
        >>> sess2.arr1
        a  a0  a1
            1   2
        >>> sess2.arr2
        a  a0  a1  a2
            1   2   3

        You may also pass extra arguments or keyword arguments to the function

        >>> def change(array, increment=1, multiplier=1):
        ...     return (array + increment) * multiplier
        >>> sess2 = sess1.apply(change, 2, 2)
        >>> sess2 = sess1.apply(change, 2, multiplier=2)
        >>> sess2.arr1
        a  a0  a1
            4   6
        >>> sess2.arr2
        a  a0  a1  a2
            4   6   8
        """
        kind = kwargs.pop('kind', LArray)
        return Session([(k, func(v, *args, **kwargs) if isinstance(v, kind) else v) for k, v in self.items()])

    def summary(self, template=None):
        """
        Returns a summary of the content of the session.

        Parameters
        ----------
        template: str
            Template describing how items are summarized (see examples).
            Available arguments are 'name', 'axes_names' and 'title'

        Returns
        -------
        str
            Short representation of the content of the session.
.
        Examples
        --------
        >>> arr1 = ndtest((2, 2), title='array 1')
        >>> arr2 = ndtest(4, title='array 2')
        >>> arr3 = ndtest((3, 2), title='array 3')
        >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])
        >>> print(s.summary())  # doctest: +NORMALIZE_WHITESPACE
        arr1: a, b
            array 1
        arr2: a
            array 2
        arr3: a, b
            array 3
        >>> print(s.summary("{name} -> {axes_names}"))
        arr1 -> a, b
        arr2 -> a
        arr3 -> a, b
        """
        if template is None:
            template = "{name}: {axes_names}\n    {title}\n"
        templ_kwargs = [{'name': k,
                         'axes_names': ', '.join(v.axes.display_names),
                         'title': v.title} for k, v in self.items()]
        return '\n'.join(template.format(**kwargs) for kwargs in templ_kwargs)


def local_arrays(depth=0):
    """
    Returns a session containing all local arrays (sorted in alphabetical order).

    Parameters
    ----------
    depth: int
        depth of call frame to inspect. 0 is where local_arrays was called, 1 the caller of local_arrays, etc.

    Returns
    -------
    Session
    """
    # noinspection PyProtectedMember
    d = sys._getframe(depth + 1).f_locals
    return Session((k, d[k]) for k in sorted(d.keys()) if isinstance(d[k], LArray))


def global_arrays(depth=0):
    """
    Returns a session containing all global arrays (sorted in alphabetical order).

    Parameters
    ----------
    depth: int
        depth of call frame to inspect. 0 is where global_arrays was called,
        1 the caller of global_arrays, etc.

    Returns
    -------
    Session
    """
    # noinspection PyProtectedMember
    d = sys._getframe(depth + 1).f_globals
    return Session((k, d[k]) for k in sorted(d.keys()) if isinstance(d[k], LArray))


_session_float_error_handler = float_error_handler_factory(4)
