from __future__ import absolute_import, division, print_function

import os
from glob import glob
from collections import OrderedDict
from pandas import ExcelWriter, ExcelFile, HDFStore

from larray.core.abstractbases import ABCLArray
from larray.util.misc import pickle
from larray.inout.excel import open_excel
from larray.inout.array import df_aslarray, read_csv, read_hdf

try:
    import xlwings as xw
except ImportError:
    xw = None


def check_pattern(k, pattern):
    return k.startswith(pattern)


class FileHandler(object):
    """
    Abstract class defining the methods for "file handler" subclasses.

    Parameters
    ----------
    fname : str
        Filename.

    Attributes
    ----------
    fname : str
        Filename.
    """
    def __init__(self, fname, overwrite_file=False):
        self.fname = fname
        self.original_file_name = None
        self.overwrite_file = overwrite_file

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

    def _get_original_file_name(self):
        if self.overwrite_file and os.path.isfile(self.fname):
            self.original_file_name = self.fname
            self.fname = '{}~{}'.format(*os.path.splitext(self.fname))

    def _update_original_file(self):
        if self.original_file_name is not None and os.path.isfile(self.fname):
            os.remove(self.original_file_name)
            os.rename(self.fname, self.original_file_name)

    def read_arrays(self, keys, *args, **kwargs):
        """
        Reads file content (HDF, Excel, CSV, ...) and returns a dictionary containing loaded arrays.

        Parameters
        ----------
        keys : list of str
            List of arrays' names.
        *args : any
            Any other argument is passed through to the underlying read function.
        display : bool, optional
            Whether or not the function should display a message when starting and ending to load each array.
            Defaults to False.
        ignore_exceptions : bool, optional
            Whether or not an exception should stop the function or be ignored. Defaults to False.
        **kwargs : any
            Any other keyword argument is passed through to the underlying read function.

        Returns
        -------
        OrderedDict(str, LArray)
            Dictionary containing the loaded arrays.
        """
        display = kwargs.pop('display', False)
        ignore_exceptions = kwargs.pop('ignore_exceptions', False)
        self._open_for_read()
        res = OrderedDict()
        if keys is None:
            keys = self.list()
        for key in keys:
            if display:
                print("loading", key, "...", end=' ')
            try:
                res[key] = self._read_array(key, *args, **kwargs)
            except Exception:
                if not ignore_exceptions:
                    raise
            if display:
                print("done")
        self.close()
        return res

    def dump_arrays(self, key_values, *args, **kwargs):
        """
        Dumps arrays corresponds to keys in file in HDF, Excel, CSV, ... format

        Parameters
        ----------
        key_values : list of (str, LArray) pairs
            Name and data of arrays to dump.
        kwargs :
            * display: whether or not to display when the dump of each array is started/done.
        """
        display = kwargs.pop('display', False)
        self._get_original_file_name()
        self._open_for_write()
        for key, value in key_values:
            if isinstance(value, ABCLArray) and value.ndim == 0:
                if display:
                    print('Cannot dump {}. Dumping 0D arrays is currently not supported.'.format(key))
                continue
            if display:
                print("dumping", key, "...", end=' ')
            self._dump(key, value, *args, **kwargs)
            if display:
                print("done")
        self.save()
        self.close()
        self._update_original_file()


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
        return df_aslarray(df, raw=True)

    def _dump(self, key, value, *args, **kwargs):
        kwargs['engine'] = 'xlsxwriter'
        value.to_excel(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()


class XLWingsHandler(FileHandler):
    """
    Handler for Excel files using XLWings.
    """
    def _get_original_file_name(self):
        # for XLWingsHandler, no need to create a temporary file, the job is already done in the Workbook class
        pass

    def _open_for_read(self):
        self.handle = open_excel(self.fname)

    def _open_for_write(self):
        self.handle = open_excel(self.fname, overwrite_file=self.overwrite_file)

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
    def __init__(self, fname, overwrite_file=False):
        super(PandasCSVHandler, self).__init__(fname, overwrite_file)
        if fname is None:
            self.pattern = None
            self.directory = None
        elif '.csv' in fname or '*' in fname or '?' in fname:
            self.pattern = fname
            self.directory = os.path.dirname(fname)
        else:
            # assume fname is a directory.
            # Not testing for os.path.isdir(fname) here because when writing, the directory might not exist.
            self.pattern = os.path.join(fname, '*.csv')
            self.directory = fname

    def _get_original_file_name(self):
        pass

    def _open_for_read(self):
        if self.directory and not os.path.isdir(self.directory):
            raise ValueError("Directory '{}' does not exist".format(self.directory))

    def _open_for_write(self):
        if self.directory is not None:
            try:
                os.makedirs(self.directory)
            except OSError:
                if not os.path.isdir(self.directory):
                    raise ValueError("Path {} must represent a directory".format(self.directory))

    def list(self):
        fnames = glob(self.pattern) if self.pattern is not None else []
        # drop directory
        fnames = [os.path.basename(fname) for fname in fnames]
        # strip extension from files
        # XXX: unsure we should use sorted here
        return sorted([os.path.splitext(fname)[0] for fname in fnames])

    def _to_filepath(self, key):
        if self.directory is not None:
            return os.path.join(self.directory, '{}.csv'.format(key))
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
