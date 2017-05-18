from __future__ import absolute_import, division, print_function

import os
import sys
import warnings
from collections import OrderedDict

import numpy as np

from larray.core.axis import Axis
from larray.core.array import LArray, larray_nan_equal, get_axes
from larray.util.misc import float_error_handler_factory
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
        Adds array objects to the current session.

        Parameters
        ----------
        args : array
            List of arrays to add.
        kwargs : dict of str, array
            List of arrays to add written as 'name'=array, ...
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
        If the key doesn't correspond to any array object,
        a default one can be returned.

        Parameters
        ----------
        key : str
            Name the array.
        default : array, optional
            Returned array if the key doesn't correspond
            to any array of the current session.

        Returns
        -------
        LArray
            Array corresponding to the given key or
            a default one if not found.
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
        Loads array objects from a file.

        WARNING: never load a file using the pickle engine (.pkl or .pickle) from an untrusted source, as it can lead
        to arbitrary code execution.

        Parameters
        ----------
        fname : str
            Path to the file.
        names : list of str, optional
            List of arrays to load. If `fname` is None, list of paths to CSV files.
            Defaults to all valid objects present in the file/directory.
        engine : str, optional
            Load using `engine`. Defaults to 'auto' (use default engine for
            the format guessed from the file extension).
        display : bool, optional
            whether or not to display which file is being worked on. Defaults
            to False.
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

    def save(self, fname, names=None, engine='auto', display=False, **kwargs):
        """
        Dumps all array objects from the current session to a file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. If `fname` is None, list of paths to CSV files.
            Defaults to all objects present in the Session.
        engine : str, optional
            Dump using `engine`. Defaults to 'auto' (use default engine for
            the format guessed from the file extension).
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults
            to False.
        """
        if engine == 'auto':
            _, ext = os.path.splitext(fname)
            ext = ext.strip('.') if '.' in ext else 'csv'
            engine = ext_default_engine[ext]
        handler_cls = handler_classes[engine]
        handler = handler_cls(fname)
        items = self.filter(kind=LArray).items()
        if names is not None:
            names_set = set(names)
            items = [(k, v) for k, v in items if k in names_set]
        handler.dump_arrays(items, display=display, **kwargs)

    def to_pickle(self, fname, names=None, *args, **kwargs):
        """
        Dumps all array objects from the current session to a file using pickle.

        WARNING: never load a pickle file (.pkl or .pickle) from an untrusted source, as it can lead to arbitrary code
        execution.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects
            present in the Session.
        """
        self.save(fname, names, ext_default_engine['pkl'], *args, **kwargs)

    def dump(self, fname, names=None, engine='auto', display=False, **kwargs):
        warnings.warn("Method dump is deprecated. Use method save instead.", DeprecationWarning, stacklevel=2)
        self.save(fname, names, engine, display, **kwargs)

    def to_hdf(self, fname, names=None, *args, **kwargs):
        """
        Dumps all array objects from the current session to an HDF file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects
            present in the Session.
        """
        self.save(fname, names, ext_default_engine['hdf'], *args, **kwargs)

    def dump_hdf(self, fname, names=None, *args, **kwargs):
        warnings.warn("Method dump_hdf is deprecated. Use method to_hdf instead.", DeprecationWarning, stacklevel=2)
        self.to_hdf(fname, names, *args, **kwargs)

    def to_excel(self, fname, names=None, *args, **kwargs):
        """
        Dumps all array objects from the current session to an Excel file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects
            present in the Session.
        """
        self.save(fname, names, ext_default_engine['xlsx'], *args, **kwargs)

    def dump_excel(self, fname, names=None, *args, **kwargs):
        warnings.warn("Method dump_excel is deprecated. Use method to_excel instead.", DeprecationWarning, stacklevel=2)
        self.to_excel(fname, names, *args, **kwargs)

    def to_csv(self, fname, names=None, *args, **kwargs):
        """
        Dumps all array objects from the current session to a CSV file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects
            present in the Session.
        """
        self.save(fname, names, ext_default_engine['csv'], *args, **kwargs)

    def dump_csv(self, fname, names=None, *args, **kwargs):
        warnings.warn("Method dump_csv is deprecated. Use method to_csv instead.", DeprecationWarning, stacklevel=2)
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
        Returns the list of names of the array objects in the session

        Returns
        -------
        list of str
        """
        return sorted(self._objects.keys())

    def copy(self):
        """Returns a copy of the session.
        """
        # this actually *does* a copy of the internal mapping (the mapping is not reused-as is)
        return Session(self._objects)

    def keys(self):
        return self._objects.keys()

    def values(self):
        return self._objects.values()

    def items(self):
        return self._objects.items()

    def __repr__(self):
        return 'Session({})'.format(', '.join(self.keys()))

    def __len__(self):
        return len(self._objects)

    # binary operations are dispatched element-wise to all arrays
    # (we consider Session as an array-like)
    def _binop(opname):
        opfullname = '__%s__' % opname

        def opmethod(self, other):
            self_keys = set(self.keys())
            all_keys = list(self.keys()) + [n for n in other.keys() if
                                            n not in self_keys]
            with np.errstate(call=_session_float_error_handler):
                res = []
                for name in all_keys:
                    self_array = self.get(name, np.nan)
                    other_array = other.get(name, np.nan)
                    res.append((name, getattr(self_array, opfullname)(other_array)))
            return Session(res)
        opmethod.__name__ = opfullname
        return opmethod

    __add__ = _binop('add')
    __sub__ = _binop('sub')
    __mul__ = _binop('mul')
    __truediv__ = _binop('truediv')

    # XXX: use _binop (ie elementwise comparison instead of aggregating
    #      directly?)
    def __eq__(self, other):
        self_keys = set(self.keys())
        all_keys = list(self.keys()) + [n for n in other.keys()
                                        if n not in self_keys]
        res = [larray_nan_equal(self.get(key), other.get(key)) for key in all_keys]
        return LArray(res, [Axis(all_keys, 'name')])

    def __ne__(self, other):
        return ~(self == other)

    def compact(self, display=False):
        """
        Detects and removes "useless" axes (ie axes for which values are
        constant over the whole axis) for all array objects in session

        Parameters
        ----------
        display : bool, optional
            Whether or not to display a message for each array that is compacted

        Returns
        -------
        Session
            A new session containing all compacted arrays
        """
        new_items = []
        for k, v in self._objects.items():
            compacted = v.compact()
            if compacted is not v and display:
                print(k, "was constant over", get_axes(v) - get_axes(compacted))
            new_items.append((k, compacted))
        return Session(new_items)

    def summary(self, template=None):
        if template is None:
            template = "{name}: {axes_names}\n    {title}\n"
        templ_kwargs = [{'name': k,
                         'axes_names': ', '.join(v.axes.display_names),
                         'title': v.title} for k, v in self.items()]
        return '\n'.join(template.format(**kwargs) for kwargs in templ_kwargs)


def local_arrays(depth=0):
    # noinspection PyProtectedMember
    d = sys._getframe(depth + 1).f_locals
    return Session((k, d[k]) for k in sorted(d.keys())
                   if isinstance(d[k], LArray))


_session_float_error_handler = float_error_handler_factory(4)
