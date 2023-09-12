import os
import csv
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from typing import Dict

from larray.core.array import Array, asarray
from larray.core.constants import nan
from larray.core.metadata import Metadata
from larray.util.misc import skip_comment_cells, strip_rows, deprecate_kwarg
from larray.inout.session import register_file_handler
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_asarray
from larray.example import get_example_filepath         # noqa: F401


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_csv(filepath_or_buffer, nb_axes=None, index_col=None, sep=',', headersep=None, decimal='.', fill_value=nan,
             na=nan, sort_rows=False, sort_columns=False, wide=True, dialect='larray', **kwargs) -> Array:
    r"""
    Read csv file and returns an array with the contents.

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
        Separator to use. Defaults to ','.
    headersep : str or None, optional
        Specific separator to use for headers. Defaults to None (uses `sep`).
    decimal : str, optional
        Character to use as decimal point. Defaults to '.'.
    fill_value : scalar or Array, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    wide : bool, optional
        Whether to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    dialect : {'classic', 'larray', 'liam2'}, optional
        Name of dialect. Defaults to 'larray'.
    **kwargs
        Extra keyword arguments are passed on to pandas.read_csv

    Returns
    -------
    Array

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
        France,Male,31772665,32045129,32174258
        France,Female,33827685,34120851,34283895
        Germany,Male,39380976,39556923,39835457
        Germany,Female,41142770,41210540,41362080

    Examples
    --------
    >>> csv_dir = get_example_filepath('examples')
    >>> fname = csv_dir / 'population.csv'

    >>> # The data below is derived from a subset of the demo_pjan table from Eurostat
    >>> read_csv(fname)
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male  31772665  32045129  32174258
     France       Female  33827685  34120851  34283895
    Germany         Male  39380976  39556923  39835457
    Germany       Female  41142770  41210540  41362080

    Missing label combinations

    >>> fname = csv_dir / 'population_missing_values.csv'
    >>> # let's take a look inside the CSV file.
    >>> # they are missing label combinations: (Paris, male) and (New York, female)
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    country,gender\time,2013,2014,2015
    Belgium,Male,5472856,5493792,5524068
    Belgium,Female,5665118,5687048,5713206
    France,Female,33827685,34120851,34283895
    Germany,Male,39380976,39556923,39835457
    >>> # by default, cells associated with missing label combinations are filled with NaN.
    >>> # In that case, an int array is converted to a float array.
    >>> read_csv(fname)
    country  gender\time        2013        2014        2015
    Belgium         Male   5472856.0   5493792.0   5524068.0
    Belgium       Female   5665118.0   5687048.0   5713206.0
     France         Male         nan         nan         nan
     France       Female  33827685.0  34120851.0  34283895.0
    Germany         Male  39380976.0  39556923.0  39835457.0
    Germany       Female         nan         nan         nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_csv(fname, fill_value=0)
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male         0         0         0
     France       Female  33827685  34120851  34283895
    Germany         Male  39380976  39556923  39835457
    Germany       Female         0         0         0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    >>> fname = csv_dir / 'population_missing_axis_name.csv'
    >>> # let's take a look inside the CSV file.
    >>> # The name of the last axis is missing.
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    country,gender,2013,2014,2015
    Belgium,Male,5472856,5493792,5524068
    Belgium,Female,5665118,5687048,5713206
    France,Male,31772665,32045129,32174258
    France,Female,33827685,34120851,34283895
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

    >>> fname = csv_dir / 'population_narrow_format.csv'
    >>> # let's take a look inside the CSV file.
    >>> # Here, data are stored in a 'narrow' format.
    >>> with open(fname) as f:
    ...     print(f.read().strip())
    country,time,value
    Belgium,2013,11137974
    Belgium,2014,11180840
    Belgium,2015,11237274
    France,2013,65600350
    France,2014,66165980
    France,2015,66458153
    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_csv
    >>> read_csv(fname, wide=False)
    country\time      2013      2014      2015
         Belgium  11137974  11180840  11237274
          France  65600350  66165980  66458153
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_csv `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    if dialect == 'liam2':
        # read axes names. This needs to be done separately instead of reading the whole file with Pandas then
        # manipulating the dataframe because the header line must be ignored for the column types to be inferred
        # correctly. Note that to read one line, this is faster than using Pandas reader.
        with open(filepath_or_buffer, mode='r', newline='', encoding='utf8') as f:
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

    df = pd.read_csv(filepath_or_buffer, index_col=index_col, sep=sep, decimal=decimal, **kwargs)
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

    return df_asarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value, raw=raw, wide=wide)


def read_tsv(filepath_or_buffer, **kwargs) -> Array:
    return read_csv(filepath_or_buffer, sep='\t', **kwargs)


def read_eurostat(filepath_or_buffer, **kwargs) -> Array:
    r"""Read EUROSTAT TSV (tab-separated) file into an array.

    EUROSTAT TSV files are special because they use tabs as data separators but comas to separate headers.

    Parameters
    ----------
    filepath_or_buffer : str or any file-like object
        Path where the tsv file has to be read or a file handle.
    kwargs
        Arbitrary keyword arguments are passed through to read_csv.

    Returns
    -------
    Array
    """
    return read_csv(filepath_or_buffer, sep='\t', headersep=',', **kwargs)


@register_file_handler('pandas_csv', 'csv')
class PandasCSVHandler(FileHandler):
    def __init__(self, fname, overwrite_file=False, sep=','):
        super().__init__(fname, overwrite_file)
        self.sep = sep
        self.axes = None
        self.groups = None
        if self.fname.suffix == '.csv' or '*' in self.fname.name or '?' in self.fname.name:
            self.pattern = self.fname.name
            self.directory = fname.parent
        else:
            # assume fname is a directory.
            # Not testing for fname.is_dir() here because when writing, the directory might not exist.
            self.pattern = '*.csv'
            self.directory = self.fname

    def _get_original_file_name(self):
        pass

    def _to_filepath(self, key) -> Path:
        if self.directory is not None:
            return self.directory / f'{key}.csv'
        else:
            return Path(key)

    def _open_for_read(self):
        if self.directory and not self.directory.is_dir():
            raise ValueError(f"Directory '{self.directory}' does not exist")

    def _open_for_write(self):
        if self.directory is not None:
            try:
                os.makedirs(self.directory)
            except OSError:
                if not self.directory.is_dir():
                    raise ValueError(f"Path {self.directory} must represent a directory")

    def item_types(self) -> Dict[str, str]:
        fnames = self.directory.glob(self.pattern) if self.pattern is not None else []
        # stem = filename without extension
        # FIXME : not sure sorted is required here
        fnames = sorted([fname.stem for fname in fnames])
        return {name: 'Array' for name in fnames if name != '__metadata__'}

    def _read_item(self, key, type, *args, **kwargs) -> Array:
        if type == 'Array':
            return read_csv(self._to_filepath(key), *args, **kwargs)
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, Array):
            value.to_csv(self._to_filepath(key), *args, **kwargs)
        else:
            raise TypeError()

    def _read_metadata(self) -> Metadata:
        filepath = self._to_filepath('__metadata__')
        if filepath.is_file():
            meta = read_csv(filepath, wide=False)
            return Metadata.from_array(meta)
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        if len(metadata) > 0:
            meta = asarray(metadata)
            meta.to_csv(self._to_filepath('__metadata__'), sep=self.sep, wide=False, value_name='')

    def save(self):
        pass

    def close(self):
        pass
