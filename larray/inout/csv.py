from __future__ import absolute_import, print_function

import os
import csv
import warnings
from glob import glob
from collections import OrderedDict

import pandas as pd
import numpy as np

from larray.core.array import LArray, aslarray, ndtest
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.group import Group
from larray.core.metadata import Metadata
from larray.util.misc import skip_comment_cells, strip_rows, csv_open, deprecate_kwarg
from larray.inout.session import register_file_handler
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_aslarray, _axes_to_df, _df_to_axes, _groups_to_df, _df_to_groups
from larray.example import get_example_filepath


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_csv(filepath_or_buffer, nb_axes=None, index_col=None, sep=',', headersep=None, fill_value=nan,
             na=nan, sort_rows=False, sort_columns=False, wide=True, dialect='larray', **kwargs):
    r"""
    Reads csv file and returns an array with the contents.

    Parameters
    ----------
    filepath_or_buffer : str or any file-like object
        Path where the csv file has to be read or a file handle.
    nb_axes : int or None, optional
        Number of axes of output array. The first ``nb_axes`` - 1 columns and the header of the CSV file will be used
        to set the axes of the output array. If not specified, the number of axes is given by the position of the
        first column header including a ``\`` character plus one. If no column header includes a ``\`` character,
        the array is assumed to have one axis. Defaults to None.
    index_col : list or None, optional
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
    dialect : {'classic', 'larray', 'liam2'}, optional
        Name of dialect. Defaults to 'larray'.
    **kwargs
        Extra keyword arguments are passed on to pandas.read_csv

    Returns
    -------
    LArray

    Notes
    -----
    Without using any argument to tell otherwise, the csv files are assumed to be in this format: ::

        axis0_name,axis1_name\axis2_name,axis2_label0,axis2_label1
        axis0_label0,axis1_label0,value,value
        axis0_label0,axis1_label1,value,value
        axis0_label1,axis1_label0,value,value
        axis0_label1,axis1_label1,value,value

    For example: ::

        country,gender\time,2013,2014,2015
        Belgium,Male,5472856,5493792,5524068
        Belgium,Female,5665118,5687048,5713206
        France,Male,31772665,31936596,32175328
        France,Female,33827685,34005671,34280951
        Germany,Male,39380976,39556923,39835457
        Germany,Female,41142770,41210540,41362080

    Examples
    --------
    >>> csv_dir = get_example_filepath('examples')
    >>> fname = csv_dir + '/pop.csv'

    >>> # The data below is derived from a subset of the demo_pjan table from Eurostat
    >>> read_csv(fname)
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male  31772665  31936596  32175328
     France       Female  33827685  34005671  34280951
    Germany         Male  39380976  39556923  39835457
    Germany       Female  41142770  41210540  41362080

    Missing label combinations

    >>> fname = csv_dir + '/pop_missing_values.csv'
    >>> # let's take a look inside the CSV file.
    >>> # they are missing label combinations: (Paris, male) and (New York, female)
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    country,gender\time,2013,2014,2015
    Belgium,Male,5472856,5493792,5524068
    Belgium,Female,5665118,5687048,5713206
    France,Female,33827685,34005671,34280951
    Germany,Male,39380976,39556923,39835457
    >>> # by default, cells associated with missing label combinations are filled with NaN.
    >>> # In that case, an int array is converted to a float array.
    >>> read_csv(fname)
    country  gender\time        2013        2014        2015
    Belgium         Male   5472856.0   5493792.0   5524068.0
    Belgium       Female   5665118.0   5687048.0   5713206.0
     France         Male         nan         nan         nan
     France       Female  33827685.0  34005671.0  34280951.0
    Germany         Male  39380976.0  39556923.0  39835457.0
    Germany       Female         nan         nan         nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_csv(fname, fill_value=0)
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male         0         0         0
     France       Female  33827685  34005671  34280951
    Germany         Male  39380976  39556923  39835457
    Germany       Female         0         0         0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    >>> fname = csv_dir + '/pop_missing_axis_name.csv'
    >>> # let's take a look inside the CSV file.
    >>> # The name of the last axis is missing.
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    country,gender,2013,2014,2015
    Belgium,Male,5472856,5493792,5524068
    Belgium,Female,5665118,5687048,5713206
    France,Male,31772665,31936596,32175328
    France,Female,33827685,34005671,34280951
    Germany,Male,39380976,39556923,39835457
    Germany,Female,41142770,41210540,41362080
    >>> # read the array stored in the CSV file as is
    >>> arr = read_csv(fname)
    >>> # we expected a 3 x 2 x 3 array with data of type int
    >>> # but we got a 6 x 4 array with data of type object
    >>> arr.info
    6 x 4
     country [6]: 'Belgium' 'Belgium' 'France' 'France' 'Germany' 'Germany'
     {1} [4]: 'gender' '2013' '2014' '2015'
    dtype: object
    memory used: 192 bytes
    >>> # using argument 'nb_axes', you can force the number of axes of the output array
    >>> arr = read_csv(fname, nb_axes=3)
    >>> # as expected, we have a 3 x 2 x 3 array with data of type int
    >>> arr.info
    3 x 2 x 3
     country [3]: 'Belgium' 'France' 'Germany'
     gender [2]: 'Male' 'Female'
     {2} [3]: 2013 2014 2015
    dtype: int64
    memory used: 144 bytes

    Read array saved in "narrow" format (wide=False)

    >>> fname = csv_dir + '/pop_narrow_format.csv'
    >>> # let's take a look inside the CSV file.
    >>> # Here, data are stored in a 'narrow' format.
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    country,time,value
    Belgium,2013,11137974
    Belgium,2014,11180840
    Belgium,2015,11237274
    France,2013,65600350
    France,2014,65942267
    France,2015,66456279
    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_csv
    >>> read_csv(fname, wide=False)
    country\time      2013      2014      2015
         Belgium  11137974  11180840  11237274
          France  65600350  65942267  66456279
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
            df.set_index([[nan]], inplace=True)
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
            fnames.remove('__metadata__')
        except:
            pass
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

    def _read_metadata(self):
        filepath = self._to_filepath('__metadata__')
        if os.path.isfile(filepath):
            meta = read_csv(filepath, wide=False)
            return Metadata.from_array(meta)
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        if len(metadata) > 0:
            meta = aslarray(metadata)
            meta.to_csv(self._to_filepath('__metadata__'), sep=self.sep, wide=False, value_name='')

    def save(self):
        if len(self.axes) > 0:
            df = _axes_to_df(self.axes.values())
            df.to_csv(self._to_filepath('__axes__'), sep=self.sep, index=False)
        if len(self.groups) > 0:
            df = _groups_to_df(self.groups.values())
            df.to_csv(self._to_filepath('__groups__'), sep=self.sep, index=False)

    def close(self):
        pass
