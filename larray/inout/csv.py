from __future__ import absolute_import, print_function

import os
import csv
import warnings
from glob import glob
from collections import OrderedDict

import pandas as pd
import numpy as np

from larray.core.axis import Axis
from larray.core.group import Group
from larray.core.array import LArray, ndtest
from larray.util.misc import skip_comment_cells, strip_rows, csv_open, deprecate_kwarg
from larray.inout.session import register_file_handler
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_aslarray, _axes_to_df, _df_to_axes, _groups_to_df, _df_to_groups


__all__ = ['read_csv', 'read_tsv', 'read_eurostat']


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_csv(filepath_or_buffer, nb_axes=None, index_col=None, sep=',', headersep=None, fill_value=np.nan,
             na=np.nan, sort_rows=False, sort_columns=False, wide=True, dialect='larray', **kwargs):
    """
    Reads csv file and returns an array with the contents.

    Notes
    -----
    csv file format:
    arr,ages,sex,nat\time,1991,1992,1993
    A1,BI,H,BE,1,0,0
    A1,BI,H,FO,2,0,0
    A1,BI,F,BE,0,0,1
    A1,BI,F,FO,0,0,0
    A1,A0,H,BE,0,0,0

    Parameters
    ----------
    filepath_or_buffer : str or any file-like object
        Path where the csv file has to be read or a file handle.
    nb_axes : int, optional
        Number of axes of output array. The first `nb_axes` - 1 columns and the header of the CSV file will be used
        to set the axes of the output array. If not specified, the number of axes is given by the position of the
        column header including the character `\` plus one. If no column header includes the character `\`, the array
        is assumed to have one axis. Defaults to None.
    index_col : list, optional
        Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3]). Defaults to None (see nb_axes above).
    sep : str, optional
        Separator.
    headersep : str or None, optional
        Separator for headers.
    fill_value : scalar or LArray, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    wide : bool, optional
        Whether or not to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    dialect : 'classic' | 'larray' | 'liam2', optional
        Name of dialect. Defaults to 'larray'.
    **kwargs

    Returns
    -------
    LArray

    Examples
    --------
    >>> import os
    >>> from larray import EXAMPLE_FILES_DIR
    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'test2d.csv')
    >>> read_csv(fname)
    a\\b  b0  b1
      1   0   1
      2   2   3
      3   4   5

    Missing label combinations

    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'missing_values_3d.csv')
    >>> # let's take a look inside the CSV file.
    >>> # they are missing label combinations: (a=2, b=b0) and (a=3, b=b1)
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    a,b\c,c0,c1,c2
    1,b0,0,1,2
    1,b1,3,4,5
    2,b1,9,10,11
    3,b0,12,13,14
    >>> # by default, cells associated with missing label combinations are filled with NaN.
    >>> # In that case, an int array is converted to a float array.
    >>> read_csv(fname)
    a  b\c    c0    c1    c2
    1   b0   0.0   1.0   2.0
    1   b1   3.0   4.0   5.0
    2   b0   nan   nan   nan
    2   b1   9.0  10.0  11.0
    3   b0  12.0  13.0  14.0
    3   b1   nan   nan   nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_csv(fname, fill_value=0)
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   0   0   0
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1   0   0   0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'missing_axis_name.csv')
    >>> # let's take a look inside the CSV file.
    >>> # The name of the second axis is missing.
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    a,b0,b1,b2
    a0,0,1,2
    a1,3,4,5
    a2,6,7,8
    >>> # read the array stored in the CSV file as is
    >>> read_csv(fname)
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5
       a2   6   7   8
    >>> # using argument 'nb_axes', you can force the number of axes of the output array
    >>> read_csv(fname, nb_axes=2)
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5
       a2   6   7   8

    Read array saved in "narrow" format (wide=False)

    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'narrow_2d.csv')
    >>> # let's take a look inside the CSV file.
    >>> # Here, data are stored in a 'narrow' format.
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    a,b,value
    1,b0,0
    1,b1,1
    2,b0,2
    2,b1,3
    3,b0,4
    3,b1,5
    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_csv
    >>> read_csv(fname, wide=False)
    a\\b  b0  b1
      1   0   1
      2   2   3
      3   4   5
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_csv `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    if dialect == 'liam2':
        # read axes names. This needs to be done separately instead of reading the whole file with Pandas then
        # manipulating the dataframe because the header line must be ignored for the column types to be inferred
        # correctly. Note that to read one line, this is faster than using Pandas reader.
        with csv_open(filepath_or_buffer) as f:
            reader = csv.reader(f, delimiter=sep)
            line_stream = skip_comment_cells(strip_rows(reader))
            axes_names = next(line_stream)

        if nb_axes is not None or index_col is not None:
            raise ValueError("nb_axes and index_col are not compatible with dialect='liam2'")
        if len(axes_names) > 1:
            nb_axes = len(axes_names)
        # use the second data line for column headers (excludes comments and blank lines before counting)
        kwargs['header'] = 1
        kwargs['comment'] = '#'

    index_col = _get_index_col(nb_axes, index_col, wide)

    if headersep is not None:
        if index_col is None:
            index_col = [0]

    df = pd.read_csv(filepath_or_buffer, index_col=index_col, sep=sep, **kwargs)
    if dialect == 'liam2':
        if len(df) == 1:
            df.set_index([[np.nan]], inplace=True)
        if len(axes_names) > 1:
            df.index.names = axes_names[:-1]
        df.columns.name = axes_names[-1]
        raw = False
    else:
        raw = index_col is None

    if headersep is not None:
        combined_axes_names = df.index.name
        df.index = df.index.str.split(headersep, expand=True)
        df.index.names = combined_axes_names.split(headersep)
        raw = False

    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value, raw=raw, wide=wide)


def read_tsv(filepath_or_buffer, **kwargs):
    return read_csv(filepath_or_buffer, sep='\t', **kwargs)


def read_eurostat(filepath_or_buffer, **kwargs):
    """Reads EUROSTAT TSV (tab-separated) file into an array.

    EUROSTAT TSV files are special because they use tabs as data separators but comas to separate headers.

    Parameters
    ----------
    filepath_or_buffer : str or any file-like object
        Path where the tsv file has to be read or a file handle.
    kwargs
        Arbitrary keyword arguments are passed through to read_csv.

    Returns
    -------
    LArray
    """
    return read_csv(filepath_or_buffer, sep='\t', headersep=',', **kwargs)


@register_file_handler('pandas_csv', 'csv')
class PandasCSVHandler(FileHandler):
    def __init__(self, fname, overwrite_file=False, sep=','):
        super(PandasCSVHandler, self).__init__(fname, overwrite_file)
        self.sep = sep
        self.axes = None
        self.groups = None
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

    def _to_filepath(self, key):
        if self.directory is not None:
            return os.path.join(self.directory, '{}.csv'.format(key))
        else:
            return key

    def _load_axes_and_groups(self):
        # load all axes
        filepath_axes = self._to_filepath('__axes__')
        if os.path.isfile(filepath_axes):
            df = pd.read_csv(filepath_axes, sep=self.sep)
            self.axes = _df_to_axes(df)
        else:
            self.axes = OrderedDict()
        # load all groups
        filepath_groups = self._to_filepath('__groups__')
        if os.path.isfile(filepath_groups):
            df = pd.read_csv(filepath_groups, sep=self.sep)
            self.groups = _df_to_groups(df, self.axes)
        else:
            self.groups = OrderedDict()

    def _open_for_read(self):
        if self.directory and not os.path.isdir(self.directory):
            raise ValueError("Directory '{}' does not exist".format(self.directory))
        self._load_axes_and_groups()

    def _open_for_write(self):
        if self.directory is not None:
            try:
                os.makedirs(self.directory)
            except OSError:
                if not os.path.isdir(self.directory):
                    raise ValueError("Path {} must represent a directory".format(self.directory))
        self.axes = OrderedDict()
        self.groups = OrderedDict()

    def list_items(self):
        fnames = glob(self.pattern) if self.pattern is not None else []
        # drop directory
        fnames = [os.path.basename(fname) for fname in fnames]
        # strip extension from files
        # XXX: unsure we should use sorted here
        fnames = sorted([os.path.splitext(fname)[0] for fname in fnames])
        items = []
        try:
            fnames.remove('__axes__')
            items = [(name, 'Axis') for name in sorted(self.axes.keys())]
        except:
            pass
        try:
            fnames.remove('__groups__')
            items += [(name, 'Group') for name in sorted(self.groups.keys())]
        except:
            pass
        items += [(name, 'Array') for name in fnames]
        return items

    def _read_item(self, key, type, *args, **kwargs):
        if type == 'Array':
            return key, read_csv(self._to_filepath(key), *args, **kwargs)
        elif type == 'Axis':
            return key, self.axes[key]
        elif type == 'Group':
            return key, self.groups[key]
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, LArray):
            value.to_csv(self._to_filepath(key), *args, **kwargs)
        elif isinstance(value, Axis):
            self.axes[key] = value
        elif isinstance(value, Group):
            self.groups[key] = value
        else:
            raise TypeError()

    def save(self):
        if len(self.axes) > 0:
            df = _axes_to_df(self.axes.values())
            df.to_csv(self._to_filepath('__axes__'), sep=self.sep, index=False)
        if len(self.groups) > 0:
            df = _groups_to_df(self.groups.values())
            df.to_csv(self._to_filepath('__groups__'), sep=self.sep, index=False)

    def close(self):
        pass
