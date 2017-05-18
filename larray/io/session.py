from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict
from pandas import ExcelWriter, ExcelFile, HDFStore

from larray.core.array import df_aslarray, read_csv, read_hdf
from larray.util.misc import pickle
from larray.io.excel import open_excel

try:
    import xlwings as xw
except ImportError:
    xw = None


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
        OrderedDict(str, LArray)
            Dictionary containing the loaded arrays.
        """
        display = kwargs.pop('display', False)
        self._open_for_read()
        res = OrderedDict()
        if keys is None:
            keys = self.list()
        for key in keys:
            if display:
                print("loading", key, "...", end=' ')
            res[key] = self._read_array(key, *args, **kwargs)
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
        key_values : list of (str, LArray) pairs
            Name and data of arrays to dump.
        kwargs :
            * display: whether or not to display when the dump of each array is started/done.
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
        return [key.strip('/') for key in self.handle.keys()]

    def _to_hdf_key(self, key):
        return '/' + key

    def _read_array(self, key, *args, **kwargs):
        return read_hdf(self.handle, self._to_hdf_key(key), *args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        value.to_hdf(self.handle, self._to_hdf_key(key), *args, **kwargs)

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
        if self.fname is not None:
            try:
                os.makedirs(self.fname)
            except OSError:
                if not os.path.isdir(self.fname):
                    raise ValueError("Path {} must represent a directory".format(self.fname))

    def list(self):
        # strip extension from files
        # TODO: also support fname pattern, eg. "dump_*.csv" (using glob)
        if self.fname is not None:
            return sorted([os.path.splitext(fname)[0] for fname in os.listdir(self.fname) if '.csv' in fname])
        else:
            return []

    def _to_filepath(self, key):
        if self.fname is not None:
            return os.path.join(self.fname, '{}.csv'.format(key))
        else:
            return key

    def _read_array(self, key, *args, **kwargs):
        return read_csv(self._to_filepath(key), *args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        value.to_csv(self._to_filepath(key), *args, **kwargs)

    def close(self):
        pass


class PickleHandler(FileHandler):
    def _open_for_read(self):
        with open(self.fname, 'rb') as f:
            self.data = pickle.load(f)

    def _open_for_write(self):
        self.data = OrderedDict()

    def list(self):
        return self.data.keys()

    def _read_array(self, key):
        return self.data[key]

    def _dump(self, key, value):
        self.data[key] = value

    def close(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)


handler_classes = {
    'pickle': PickleHandler,
    'pandas_csv': PandasCSVHandler,
    'pandas_hdf': PandasHDFHandler,
    'pandas_excel': PandasExcelHandler,
    'xlwings_excel': XLWingsHandler,
}

ext_default_engine = {
    'csv': 'pandas_csv',
    'h5': 'pandas_hdf', 'hdf': 'pandas_hdf',
    'pkl': 'pickle', 'pickle': 'pickle',
    'xls': 'xlwings_excel', 'xlsx': 'xlwings_excel',
}
