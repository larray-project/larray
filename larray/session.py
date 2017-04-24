from __future__ import absolute_import, division, print_function

import os
import sys
from collections import OrderedDict

import numpy as np
from pandas import ExcelWriter, ExcelFile, HDFStore

from .core import LArray, Axis, read_csv, read_hdf, df_aslarray, larray_equal, larray_nan_equal, get_axes
from .excel import open_excel


def check_pattern(k, pattern):
    return k.startswith(pattern)


class FileHandler(object):
    """
    Abstract class defining the methods for
    "file handler" subclasses.

    Parameters
    ----------
    fname : str
        Filename.

    Attributes
    ----------
    fname : str
        Filename.
    """
    def __init__(self, fname):
        self.fname = fname

    def _open_for_read(self):
        raise NotImplementedError()

    def _open_for_write(self):
        raise NotImplementedError()

    def list(self):
        """
        Returns the list of arrays' names.
        """
        raise NotImplementedError()

    def _read_array(self, key, *args, **kwargs):
        raise NotImplementedError()

    def _dump(self, key, value, *args, **kwargs):
        raise NotImplementedError()

    def save(self):
        """
        Saves arrays in file.
        """
        pass

    def close(self):
        """
        Closes file.
        """
        raise NotImplementedError()

    def read_arrays(self, keys, *args, **kwargs):
        """
        Reads file content (HDF, Excel, CSV, ...)
        and returns a dictionary containing
        loaded arrays.

        Parameters
        ----------
        keys : list of str
            List of arrays' names.
        kwargs :
            * display: a small message is displayed to tell when
              an array is started to be read and when it's done.

        Returns
        -------
        dict(str,LArray)
            Dictionary containing names and arrays loaded from a file.
        """
        display = kwargs.pop('display', False)
        self._open_for_read()
        res = {}
        if keys is None:
            keys = self.list()
        for key in keys:
            if display:
                print("loading", key, "...", end=' ')
            dest_key = key.strip('/')
            res[dest_key] = self._read_array(key, *args, **kwargs)
            if display:
                print("done")
        self.close()
        return res

    def dump_arrays(self, key_values, *args, **kwargs):
        """
        Dumps arrays corresponds to keys in file
        in HDF, Excel, CSV, ... format

        Parameters
        ----------
        key_values : dict of paris (str, LArray)
            Dictionary containing arrays to dump.
        kwargs :
            * display: a small message is displayed to tell when
              an array is started to be dump and when it's done.
        """
        display = kwargs.pop('display', False)
        self._open_for_write()
        for key, value in key_values:
            if display:
                print("dumping", key, "...", end=' ')
            self._dump(key, value, *args, **kwargs)
            if display:
                print("done")
        self.save()
        self.close()


class PandasHDFHandler(FileHandler):
    """
    Handler for HDF5 files using Pandas.
    """
    def _open_for_read(self):
        self.handle = HDFStore(self.fname, mode='r')

    def _open_for_write(self):
        self.handle = HDFStore(self.fname)

    def list(self):
        return self.handle.keys()

    def _read_array(self, key, *args, **kwargs):
        return read_hdf(self.handle, key, *args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        value.to_hdf(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()


class PandasExcelHandler(FileHandler):
    """
    Handler for Excel files using Pandas.
    """
    def _open_for_read(self):
        self.handle = ExcelFile(self.fname)

    def _open_for_write(self):
        self.handle = ExcelWriter(self.fname)

    def list(self):
        return self.handle.sheet_names

    def _read_array(self, key, *args, **kwargs):
        df = self.handle.parse(key, *args, **kwargs)
        return df_aslarray(df)

    def _dump(self, key, value, *args, **kwargs):
        kwargs['engine'] = 'xlsxwriter'
        value.to_excel(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()


class XLWingsHandler(FileHandler):
    """
    Handler for Excel files using XLWings.
    """
    def _open_for_read(self):
        self.handle = open_excel(self.fname)

    def _open_for_write(self):
        self.handle = open_excel(self.fname)

    def list(self):
        return self.handle.sheet_names()

    def _read_array(self, key, *args, **kwargs):
        return self.handle[key].load(*args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        self.handle[key] = value.dump(*args, **kwargs)

    def save(self):
        self.handle.save()

    def close(self):
        self.handle.close()


class PandasCSVHandler(FileHandler):
    def _open_for_read(self):
        pass

    def _open_for_write(self):
        try:
            os.makedirs(self.fname)
        except OSError:
            if not os.path.isdir(self.fname):
                raise

    def list(self):
        # strip extension from files
        # FIXME: only take .csv files
        # TODO: also support fname pattern, eg. "dump_*.csv" (using glob)
        return [os.path.splitext(fname)[0] for fname in os.listdir(self.fname)]

    def _read_array(self, key, *args, **kwargs):
        fpath = os.path.join(self.fname, '{}.csv'.format(key))
        return read_csv(fpath, *args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        value.to_csv(os.path.join(self.fname, '{}.csv'.format(key)), *args,
                     **kwargs)

    def close(self):
        pass


handler_classes = {
    'pandas_hdf': PandasHDFHandler,
    'pandas_excel': PandasExcelHandler,
    'xlwings_excel': XLWingsHandler,
    'pandas_csv': PandasCSVHandler
}

ext_default_engine = {
    'h5': 'pandas_hdf', 'hdf': 'pandas_hdf',
    'xls': 'xlwings_excel', 'xlsx': 'xlwings_excel',
    'csv': 'pandas_csv'
}


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

    def _ipython_key_completions_(self):
        return list(self.keys())

    def __getattr__(self, key):
        if key in self._objects:
            return self._objects[key]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, key))

    def __setattr__(self, key, value):
        self._objects[key] = value

    def __dir__(self):
        return list(set(dir(self.__class__)) | set(self.keys()))

    def load(self, fname, names=None, engine='auto', display=False, **kwargs):
        """
        Loads array objects from a file.

        Parameters
        ----------
        fname : str
            Path to the file.
        names : list of str, optional
            List of arrays to load. Defaults to all valid objects present in
            the file/directory.
        engine : str, optional
            Load using `engine`. Defaults to 'auto' (use default engine for
            the format guessed from the file extension).
        display : bool, optional
            whether or not to display which file is being worked on. Defaults
            to False.
        """
        if display:
            print("opening", fname)
        # TODO: support path + *.csv
        if engine == 'auto':
            _, ext = os.path.splitext(fname)
            engine = ext_default_engine[ext.strip('.')]
        handler_cls = handler_classes[engine]
        handler = handler_cls(fname)
        arrays = handler.read_arrays(names, display=display, **kwargs)
        for k, v in arrays.items():
            self[k] = v

    def dump(self, fname, names=None, engine='auto', display=False, **kwargs):
        """
        Dumps all array objects from the current session to a file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            List of names of objects to dump. Defaults to all objects
            present in the Session.
        engine : str, optional
            Dump using `engine`. Defaults to 'auto' (use default engine for
            the format guessed from the file extension).
        display : bool, optional
            Whether or not to display which file is being worked on. Defaults
            to False.
        """
        if engine == 'auto':
            _, ext = os.path.splitext(fname)
            engine = ext_default_engine[ext.strip('.')]
        handler_cls = handler_classes[engine]
        handler = handler_cls(fname)
        filtered = self.filter(kind=LArray)
        # not using .items() so that arrays are sorted
        arrays = [(k, filtered[k]) for k in filtered.names]
        if names is not None:
            names_set = set(names)
            arrays = [(k, v) for k, v in arrays if k in names_set]
        handler.dump_arrays(arrays, display=display, **kwargs)

    def dump_hdf(self, fname, names=None, *args, **kwargs):
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
        self.dump(fname, names, ext_default_engine['hdf'], *args, **kwargs)

    def dump_excel(self, fname, names=None, *args, **kwargs):
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
        self.dump(fname, names, ext_default_engine['xlsx'], *args, **kwargs)

    def dump_csv(self, fname, names=None, *args, **kwargs):
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
        self.dump(fname, names, ext_default_engine['csv'], *args, **kwargs)

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
        if pattern is not None:
            items = [(k, self._objects[k]) for k in self._objects.keys()
                     if check_pattern(k, pattern)]
        else:
            items = self._objects.items()
        if kind is not None:
            return Session([(k, v) for k, v in items if isinstance(v, kind)])
        else:
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
