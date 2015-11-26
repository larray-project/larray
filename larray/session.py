from __future__ import absolute_import, division, print_function

import os

from pandas import ExcelWriter
from larray.core import LArray, read_csv, read_excel, read_hdf


def check_pattern(k, pattern):
    return k.startswith(pattern)


def read_multi_csv(fname, name):
    return read_csv(fname)


class Session(object):
    def __init__(self, *args, **kwargs):
        # self._objects = {}
        object.__setattr__(self, '_objects', {})

        if len(args) == 1 and not hasattr(args[0], 'name') and not \
                isinstance(args[0], dict):
            # assume we have an iterable of tuples
            arg = args[0]
            for k, v in arg:
                self[k] = v
        elif len(args) == 1 and isinstance(args[0], dict):
            self.add(**args[0])
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
        return self._objects[key]

    def __setitem__(self, key, value):
        self._objects[key] = value

    def __getattr__(self, key):
        return self._objects[key]

    def __setattr__(self, key, value):
        self._objects[key] = value

    def load(self, fname, names):
        """Load LArray objects from a file.

        Parameters
        ----------
        fname : str
            Path to the file.
        names : list of str
            List of arrays to load.
        """
        # TODO: add support for names=None for .h5 or .xlsx => all in file
        # TODO: support path + *.csv
        funcs = {'.h5': read_hdf, '.xls': read_excel, '.xlsx': read_excel,
                 '.csv': read_multi_csv}
        _, ext = os.path.splitext(fname)
        func = funcs[ext]
        for name in names:
            self[name] = func(fname, name)

    def dump(self, fname, fmt='auto'):
        """Dumps all LArray objects to a file.

        Parameters
        ----------
        fname : str
            Path for the dump.
        fmt : str, optional
            Dump to the `fmt` format. Defaults to 'auto' (guess from
            filename).
        """
        if fmt == 'auto':
            _, ext = os.path.splitext(fname)
            fmt = ext.strip('.')
        funcs = {'h5': self.dump_hdf, 'hdf': self.dump_hdf,
                 'xlsx': self.dump_excel, 'xls': self.dump_excel,
                 'csv': self.dump_csv}
        funcs[fmt](fname)

    def dump_hdf(self, fname, *args, **kwargs):
        for k, v in self.filter(kind=LArray).items():
            v.to_hdf(fname, k, *args, **kwargs)

    def dump_excel(self, fname, *args, **kwargs):
        writer = ExcelWriter(fname)
        for k, v in self.filter(kind=LArray).items():
            v.to_excel(writer, k, *args, **kwargs)
        writer.save()

    def dump_csv(self, path, *args, **kwargs):
        for k, v in self.filter(kind=LArray).items():
            v.to_csv(os.path.join(path, '{}.csv'.format(k)), *args, **kwargs)

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
        return sorted(self._objects.keys())

    # XXX: sorted?
    def values(self):
        return self._objects.values()

    # XXX: sorted?
    def items(self):
        return self._objects.items()

    def __repr__(self):
        return 'Session({})'.format(', '.join(self.names))
