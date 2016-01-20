from __future__ import absolute_import, division, print_function

import os

from pandas import ExcelWriter, ExcelFile, HDFStore
from larray.core import LArray, read_csv, read_hdf, df_aslarray, larray_equal


def check_pattern(k, pattern):
    return k.startswith(pattern)


class FileHandler(object):
    def __init__(self, fname):
        self.fname = fname

    def list(self):
        raise NotImplementedError()

    def _read_array(self, key, **kwargs):
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

    def close(self):
        raise NotImplementedError()


class HDFHandler(FileHandler):
    def _open_for_read(self):
        self.handle = HDFStore(self.fname, mode='r')

    def _open_for_write(self):
        self.handle = HDFStore(self.fname)

    def list(self):
        return self.handle.keys()

    def _read_array(self, key, **kwargs):
        return read_hdf(self.handle, key, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        value.to_hdf(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()


class ExcelHandler(FileHandler):
    def _open_for_read(self):
        self.handle = ExcelFile(self.fname)

    def _open_for_write(self):
        self.handle = ExcelWriter(self.fname)

    def list(self):
        return self.handle.sheet_names

    def _read_array(self, key, **kwargs):
        df = self.handle.parse(key, **kwargs)
        return df_aslarray(df)

    def _dump(self, key, value, *args, **kwargs):
        value.to_excel(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()


class CSVHandler(FileHandler):
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
        return read_csv(os.path.join(self.fname, '{}.csv'.format(key)),
                        *args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        value.to_csv(os.path.join(self.fname, '{}.csv'.format(key)), *args,
                     **kwargs)

    def close(self):
        pass


ext_classes = {'h5': HDFHandler, 'hdf': HDFHandler,
               'xls': ExcelHandler, 'xlsx': ExcelHandler,
               'csv': CSVHandler}


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

    # XXX: behave like a dict and return keys instead?
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
        else:
            return self._objects[key]

    def __setitem__(self, key, value):
        self._objects[key] = value

    def __getattr__(self, key):
        return self._objects[key]

    def __setattr__(self, key, value):
        self._objects[key] = value

    def load(self, fname, names=None, fmt='auto', display=False, **kwargs):
        """Load LArray objects from a file.

        Parameters
        ----------
        fname : str
            Path to the file.
        names : list of str
            List of arrays to load.
        fmt : str
        """
        if display:
            print("opening", fname)
        # TODO: support path + *.csv
        if fmt == 'auto':
            _, ext = os.path.splitext(fname)
            fmt = ext.strip('.')
        handler = ext_classes[fmt](fname)
        arrays = handler.read_arrays(names, display=display, **kwargs)
        for k, v in arrays.items():
            self[k] = v

    def dump(self, fname, names=None, fmt='auto', display=False, **kwargs):
        """Dumps all LArray objects to a file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        names : list of str or None, optional
        fmt : str, optional
            Dump to the `fmt` format. Defaults to 'auto' (guess from
            filename).
        """
        if fmt == 'auto':
            _, ext = os.path.splitext(fname)
            fmt = ext.strip('.')
        handler = ext_classes[fmt](fname)
        arrays = self.filter(kind=LArray).items()
        if names is not None:
            names_set = set(names)
            arrays = [(k, v) for k, v in arrays if k in names_set]
        handler.dump_arrays(arrays, display=display, **kwargs)

    def dump_hdf(self, fname, names=None, *args, **kwargs):
        self.dump(fname, names, 'hdf', *args, **kwargs)

    def dump_excel(self, fname, names=None, *args, **kwargs):
        self.dump(fname, names, 'xlsx', *args, **kwargs)

    def dump_csv(self, fname, names=None, *args, **kwargs):
        self.dump(fname, names, 'csv', *args, **kwargs)

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

    # XXX: sorted?
    def values(self):
        return self._objects.values()

    # XXX: sorted?
    def items(self):
        return self._objects.items()

    def __repr__(self):
        return 'Session({})'.format(', '.join(self.names))

    def __eq__(self, other):
        return all(larray_equal(a0, a1) for a0, a1 in zip(self, other))
