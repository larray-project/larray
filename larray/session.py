from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
from pandas import ExcelWriter, ExcelFile, HDFStore

from .core import LArray, Axis, read_csv, read_hdf, df_aslarray, larray_equal
from .excel import open_excel


def check_pattern(k, pattern):
    return k.startswith(pattern)


class FileHandler(object):
    def __init__(self, fname):
        self.fname = fname

    def _open_for_read(self):
        raise NotImplementedError()

    def _open_for_write(self):
        raise NotImplementedError()

    def list(self):
        raise NotImplementedError()

    def _read_array(self, key, *args, **kwargs):
        raise NotImplementedError()

    def _dump(self, key, value, *args, **kwargs):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def read_arrays(self, keys, *args, **kwargs):
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
        display = kwargs.pop('display', False)
        self._open_for_write()
        for key, value in key_values:
            if display:
                print("dumping", key, "...", end=' ')
            self._dump(key, value, *args, **kwargs)
            if display:
                print("done")
        self.close()


class PandasHDFHandler(FileHandler):
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
        # TODO: also support fname pattern, eg. "dump_*.csv"
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


class Session(object):
    def __init__(self, *args, **kwargs):
        # self._objects = {}
        object.__setattr__(self, '_objects', {})

        if len(args) == 1:
            a0 = args[0]
            if isinstance(a0, dict):
                self.add(**a0)
            elif isinstance(a0, str):
                # assume a0 is a filename
                self.load(a0)
            else:
                # assume we have an iterable of tuples
                for k, v in a0:
                    self[k] = v
        else:
            self.add(*args, **kwargs)

    # XXX: behave like a dict and return keys instead or return unsorted?
    def __iter__(self):
        return iter(self._objects[k] for k in self.names)

    def add(self, *args, **kwargs):
        for arg in args:
            self[arg.name] = arg
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._objects[self.names[key]]
        elif isinstance(key, LArray):
            assert np.issubdtype(key.dtype, np.bool_)
            assert key.ndim == 1
            # only keep True values
            truenames = key[key].axes[0].labels
            return Session({name: self[name] for name in truenames})
        elif isinstance(key, (tuple, list)):
            assert all(isinstance(k, str) for k in key)
            return Session({k: self[k] for k in key})
        else:
            return self._objects[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        self._objects[key] = value

    def __getattr__(self, key):
        return self._objects[key]

    def __setattr__(self, key, value):
        self._objects[key] = value

    def load(self, fname, names=None, engine='auto', display=False, **kwargs):
        """Load LArray objects from a file.

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
        """Dumps all arrays from session to a file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
            list of names of objects to dump. Defaults to all objects
            present in the Session.
        engine : str, optional
            Dump using `engine`. Defaults to 'auto' (use default engine for
            the format guessed from the file extension).
        display : bool, optional
            whether or not to display which file is being worked on. Defaults
            to False.
        """
        if engine == 'auto':
            _, ext = os.path.splitext(fname)
            engine = ext_default_engine[ext.strip('.')]
        handler_cls = handler_classes[engine]
        handler = handler_cls(fname)
        arrays = self.filter(kind=LArray).items()
        if names is not None:
            names_set = set(names)
            arrays = [(k, v) for k, v in arrays if k in names_set]
        handler.dump_arrays(arrays, display=display, **kwargs)

    def dump_hdf(self, fname, names=None, *args, **kwargs):
        self.dump(fname, names, ext_default_engine['hdf'], *args, **kwargs)

    def dump_excel(self, fname, names=None, *args, **kwargs):
        self.dump(fname, names, ext_default_engine['xlsx'], *args, **kwargs)

    def dump_csv(self, fname, names=None, *args, **kwargs):
        self.dump(fname, names, ext_default_engine['csv'], *args, **kwargs)

    def filter(self, pattern=None, kind=None):
        """Return a new Session with objects which match some criteria.

        Parameters
        ----------
        pattern : str, optional
            Only keep objects whose key match `pattern`.
        kind : type, optional
            Only keep objects which are instances of type `kind`.

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

    # XXX: would having an option/another function for returning this unsorted
    # be any useful?
    @property
    def names(self):
        """Returns the list of names of the objects in the session

        Returns
        -------
        list of str
        """
        return sorted(self._objects.keys())

    def copy(self):
        return Session(self._objects)

    def keys(self):
        return self._objects.keys()

    # XXX: sorted?
    def values(self):
        return self._objects.values()

    # XXX: sorted?
    def items(self):
        return self._objects.items()

    def __repr__(self):
        return 'Session({})'.format(', '.join(self.names))

    def __len__(self):
        return len(self._objects)

    def __eq__(self, other):
        self_names = set(self.names)
        all_names = self.names + [n for n in other.names if n not in self_names]
        res = [larray_equal(self.get(name), other.get(name))
               for name in all_names]
        return LArray(res, [Axis('name', all_names)])

    def __ne__(self, other):
        return ~(self == other)

    # we could implement two(?) very different behavior:
    # set-like behavior: combination of two Session, if same name,
    # check that it is the same. a Session is more ordered-dict-like than
    # set-like, so this might not make a lot of sense. an "update" method
    # might make more sense. However most set-like operations do make sense.

    # intersection: check common are the same or take left?
    # union: check common are the same or take left?
    # difference

    # elementwise add: consider Session as an array-like and try to add the
    # each array individually
    def __add__(self, other):
        self_names = set(self.names)
        all_names = self.names + [n for n in other.names if n not in self_names]
        return Session({name: self.get(name, np.nan) + other.get(name, np.nan)
                       for name in all_names})

    def __sub__(self, other):
        self_names = set(self.names)
        all_names = self.names + [n for n in other.names if n not in self_names]
        return Session({name: self.get(name, np.nan) - other.get(name, np.nan)
                       for name in all_names})

    def __mul__(self, other):
        self_names = set(self.names)
        all_names = self.names + [n for n in other.names if n not in self_names]
        return Session({name: self.get(name, np.nan) * other.get(name, np.nan)
                       for name in all_names})

    def __truediv__(self, other):
        self_names = set(self.names)
        all_names = self.names + [n for n in other.names if n not in self_names]
        return Session({name: self.get(name, np.nan) / other.get(name, np.nan)
                       for name in all_names})


def local_arrays(depth=0):
    # noinspection PyProtectedMember
    d = sys._getframe(depth + 1).f_locals
    return Session((k, v) for k, v in d.items() if isinstance(v, LArray))
