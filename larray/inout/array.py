from __future__ import absolute_import, print_function

import os
import csv
import numpy as np
import pandas as pd
import warnings
from itertools import product

from larray.core.axis import Axis
from larray.core.array import LArray, ndtest
from larray.core.group import _translate_sheet, _translate_key_hdf
from larray.util.misc import (basestring, skip_comment_cells, strip_rows, csv_open, StringIO, decode, unique,
                              deprecate_kwarg)

try:
    import xlwings as xw
except ImportError:
    xw = None

__all__ = ['from_frame', 'read_csv', 'read_tsv', 'read_eurostat', 'read_hdf', 'read_excel', 'read_sas',
           'from_lists', 'from_string']


def parse(s):
    """
    Used to parse the "folded" axis ticks (usually periods).
    """
    # parameters can be strings or numbers
    if isinstance(s, basestring):
        s = s.strip()
        low = s.lower()
        if low == 'true':
            return True
        elif low == 'false':
            return False
        elif s.isdigit():
            return int(s)
        else:
            try:
                return float(s)
            except ValueError:
                return s
    else:
        return s


def df_labels(df, sort=True):
    """
    Returns unique labels for each dimension.
    """
    idx = df.index
    if isinstance(idx, pd.core.index.MultiIndex):
        if sort:
            return list(idx.levels)
        else:
            return [list(unique(idx.get_level_values(l))) for l in idx.names]
    else:
        assert isinstance(idx, pd.core.index.Index)
        # use .values if needed
        return [idx]


def cartesian_product_df(df, sort_rows=False, sort_columns=False, **kwargs):
    labels = df_labels(df, sort=sort_rows)
    if sort_rows:
        new_index = pd.MultiIndex.from_product(labels)
    else:
        new_index = pd.MultiIndex.from_tuples(list(product(*labels)))
    columns = sorted(df.columns) if sort_columns else list(df.columns)
    # the prodlen test is meant to avoid the more expensive array_equal test
    prodlen = np.prod([len(axis_labels) for axis_labels in labels])
    if prodlen == len(df) and columns == list(df.columns) and np.array_equal(df.index.values, new_index.values):
        return df, labels
    return df.reindex(new_index, columns, **kwargs), labels


def from_series(s, sort_rows=False):
    """
    Converts Pandas Series into 1D LArray.

    Parameters
    ----------
    s : Pandas Series
        Input Pandas Series.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically. Defaults to False.

    Returns
    -------
    LArray
    """
    name = s.name if s.name is not None else s.index.name
    if name is not None:
        name = str(name)
    if sort_rows:
        s = s.sort_index()
    return LArray(s.values, Axis(s.index.values, name))


def from_frame(df, sort_rows=False, sort_columns=False, parse_header=False, unfold_last_axis_name=False, **kwargs):
    """
    Converts Pandas DataFrame into LArray.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe. By default, name and labels of the last axis are defined by the name and labels of the
        columns Index of the dataframe unless argument unfold_last_axis_name is set to True.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    parse_header : bool, optional
        Whether or not to parse columns labels. Pandas treats column labels as strings.
        If True, column labels are converted into int, float or boolean when possible. Defaults to False.
    unfold_last_axis_name : bool, optional
        Whether or not to extract the names of the last two axes by splitting the name of the last index column of the
        dataframe using ``\\``. Defaults to False.

    Returns
    -------
    LArray

    See Also
    --------
    LArray.to_frame

    Examples
    --------
    >>> df = ndtest((2, 2, 2)).to_frame()
    >>> df                                                                             # doctest: +NORMALIZE_WHITESPACE
    c      c0  c1
    a  b
    a0 b0   0   1
       b1   2   3
    a1 b0   4   5
       b1   6   7
    >>> from_frame(df)
     a  b\\c  c0  c1
    a0   b0   0   1
    a0   b1   2   3
    a1   b0   4   5
    a1   b1   6   7

    Names of the last two axes written as ``before_last_axis_name\\last_axis_name``

    >>> df = ndtest((2, 2, 2)).to_frame(fold_last_axis_name=True)
    >>> df                                                                             # doctest: +NORMALIZE_WHITESPACE
            c0  c1
    a  b\\c
    a0 b0    0   1
       b1    2   3
    a1 b0    4   5
       b1    6   7
    >>> from_frame(df, unfold_last_axis_name=True)
     a  b\\c  c0  c1
    a0   b0   0   1
    a0   b1   2   3
    a1   b0   4   5
    a1   b1   6   7
    """
    axes_names = [decode(name, 'utf8') for name in df.index.names]

    # handle 2 or more dimensions with the last axis name given using \
    if unfold_last_axis_name:
        if isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]:
            last_axes = [name.strip() for name in axes_names[-1].split('\\')]
            axes_names = axes_names[:-1] + last_axes
        else:
            axes_names += [None]
    else:
        axes_names += [df.columns.name]

    df, axes_labels = cartesian_product_df(df, sort_rows=sort_rows, sort_columns=sort_columns, **kwargs)

    # Pandas treats column labels as column names (strings) so we need to convert them to values
    last_axis_labels = [parse(cell) for cell in df.columns.values] if parse_header else list(df.columns.values)
    axes_labels.append(last_axis_labels)
    axes_names = [str(name) if name is not None else name
                  for name in axes_names]

    axes = [Axis(labels, name) for labels, name in zip(axes_labels, axes_names)]
    data = df.values.reshape([len(axis) for axis in axes])
    return LArray(data, axes)


def df_aslarray(df, sort_rows=False, sort_columns=False, raw=False, parse_header=True, **kwargs):
    """
    Prepare Pandas DataFrame and then convert it into LArray.

    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    raw : bool, optional
        Whether or not to consider the input dataframe as a raw dataframe, i.e. read without index at all.
        If True, build the first N-1 axes of the output array from the first N-1 dataframe columns. Defaults to False.
    parse_header : bool, optional
        Whether or not to parse columns labels. Pandas treats column labels as strings.
        If True, column labels are converted into int, float or boolean when possible. Defaults to True.

    Returns
    -------
    LArray
    """
    # we could inline df_aslarray into the functions that use it, so that the original (non-cartesian) df is freed from
    # memory at this point, but it would be much uglier and would not lower the peak memory usage which happens during
    # cartesian_product_df.reindex

    # raw = True: the dataframe was read without index at all (ie 2D dataframe),
    # irrespective of the actual data dimensionality
    if raw:
        columns = df.columns.values.tolist()
        try:
            # take the first column which contains '\'
            pos_last = next(i for i, v in enumerate(columns) if isinstance(v, basestring) and '\\' in v)
        except StopIteration:
            # we assume first column will not contain data
            pos_last = 0

        # This is required to handle int column names (otherwise we can simply use column positions in set_index).
        # This is NOT the same as df.columns[list(range(...))] !
        index_columns = [df.columns[i] for i in range(pos_last + 1)]
        # TODO: we should pass a flag to df_aslarray so that we can use inplace=True here
        # df.set_index(index_columns, inplace=True)
        df = df.set_index(index_columns)

    # handle 1D
    if len(df) == 1 and (pd.isnull(df.index.values[0]) or
                             (isinstance(df.index.values[0], basestring) and df.index.values[0].strip() == '')):
        if parse_header:
            df.columns = pd.Index([parse(cell) for cell in df.columns.values], name=df.columns.name)
        series = df.iloc[0]
        series.name = df.index.name
        if sort_rows:
            raise ValueError('sort_rows=True is not valid for 1D arrays. Please use sort_columns instead.')
        return from_series(series, sort_rows=sort_columns)
    else:
        axes_names = [decode(name, 'utf8') for name in df.index.names]
        unfold_last_axis_name = isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]
        return from_frame(df, sort_rows=sort_rows, sort_columns=sort_columns, parse_header=parse_header,
                          unfold_last_axis_name=unfold_last_axis_name, **kwargs)


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_csv(filepath_or_buffer, nb_axes=None, index_col=None, sep=',', headersep=None, fill_value=np.nan,
             na=np.nan, sort_rows=False, sort_columns=False, dialect='larray', **kwargs):
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
    dialect : 'classic' | 'larray' | 'liam2', optional
        Name of dialect. Defaults to 'larray'.
    **kwargs

    Returns
    -------
    LArray

    Examples
    --------
    >>> from larray import ndrange
    >>> tmpdir = getfixture('tmpdir')
    >>> fname = os.path.join(tmpdir.strpath, 'test.csv')
    >>> a = ndtest('nat=BE,FO;sex=M,F')

    >>> a.to_csv(fname)
    >>> read_csv(fname)
    nat\\sex  M  F
         BE  0  1
         FO  2  3
    >>> read_csv(fname, sort_columns=True)
    nat\\sex  F  M
         BE  1  0
         FO  3  2
    >>> fname = 'no_axis_name.csv'
    >>> a.to_csv(fname, dialect='classic')
    >>> read_csv(fname, nb_axes=2)
    nat\\{1}  M  F
         BE  0  1
         FO  2  3
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

    if nb_axes is not None and index_col is not None:
        raise ValueError("cannot specify both nb_axes and index_col")
    elif nb_axes is not None:
        index_col = list(range(nb_axes - 1))
    elif isinstance(index_col, int):
        index_col = [index_col]

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

    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value, raw=raw)


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


def read_hdf(filepath_or_buffer, key, fill_value=np.nan, na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """Reads an array named key from a HDF5 file in filepath (path+name)

    Parameters
    ----------
    filepath_or_buffer : str or pandas.HDFStore
        Path and name where the HDF5 file is stored or a HDFStore object.
    key : str or Group
        Name of the array.
    fill_value : scalar or LArray, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.

    Returns
    -------
    LArray
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_hdf `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    key = _translate_key_hdf(key)
    df = pd.read_hdf(filepath_or_buffer, key, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value, parse_header=False)


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_excel(filepath, sheetname=0, nb_axes=None, index_col=None, fill_value=np.nan, na=np.nan,
               sort_rows=False, sort_columns=False, engine=None, **kwargs):
    """
    Reads excel file from sheet name and returns an LArray with the contents

    Parameters
    ----------
    filepath : str
        Path where the Excel file has to be read.
    sheetname : str, Group or int, optional
        Name or index of the Excel sheet containing the array to be read.
        By default the array is read from the first sheet.
    nb_axes : int, optional
        Number of axes of output array. The first `nb_axes` - 1 columns and the header of the Excel sheet will be used
        to set the axes of the output array. If not specified, the number of axes is given by the position of the
        column header including the character `\` plus one. If no column header includes the character `\`, the array
        is assumed to have one axis. Defaults to None.
    index_col : list, optional
        Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3]). Defaults to None (see nb_axes above).
    fill_value : scalar or LArray, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    engine : {'xlrd', 'xlwings'}, optional
        Engine to use to read the Excel file. If None (default), it will use 'xlwings' by default if the module is
        installed and relies on Pandas default reader otherwise.
    **kwargs
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_excel `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    sheet = _translate_sheet(sheet)

    if engine is None:
        engine = 'xlwings' if xw is not None else None

    if nb_axes is not None and index_col is not None:
        raise ValueError("cannot specify both nb_axes and index_col")
    elif nb_axes is not None:
        index_col = list(range(nb_axes - 1))
    elif isinstance(index_col, int):
        index_col = [index_col]

    if engine == 'xlwings':
        if kwargs:
            raise TypeError("'{}' is an invalid keyword argument for this function when using the xlwings backend"
                            .format(list(kwargs.keys())[0]))
        from larray.inout.excel import open_excel
        with open_excel(filepath) as wb:
            return wb[sheetname].load(index_col=index_col, fill_value=fill_value, sort_rows=sort_rows,
                                      sort_columns=sort_columns)
    else:
        df = pd.read_excel(filepath, sheetname, index_col=index_col, engine=engine, **kwargs)
        return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, raw=index_col is None,
                           fill_value=fill_value)


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_sas(filepath, nb_axes=None, index_col=None, fill_value=np.nan, na=np.nan, sort_rows=False, sort_columns=False,
             **kwargs):
    """
    Reads sas file and returns an LArray with the contents
        nb_axes: number of axes of the output array
    or
        index_col: Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3])
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_sas `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    if nb_axes is not None and index_col is not None:
        raise ValueError("cannot specify both nb_axes and index_col")
    elif nb_axes is not None:
        index_col = list(range(nb_axes - 1))
    elif isinstance(index_col, int):
        index_col = [index_col]

    df = pd.read_sas(filepath, index=index_col, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value)


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def from_lists(data, nb_axes=None, index_col=None, fill_value=np.nan, sort_rows=False, sort_columns=False):
    """
    initialize array from a list of lists (lines)

    Parameters
    ----------
    data : sequence (tuple, list, ...)
        Input data. All data is supposed to already have the correct type (e.g. strings are not parsed).
    nb_axes : int, optional
        Number of axes of output array. The first `nb_axes` - 1 columns and the header will be used
        to set the axes of the output array. If not specified, the number of axes is given by the position of the
        column header including the character `\` plus one. If no column header includes the character `\`, the array
        is assumed to have one axis. Defaults to None.
    index_col : list, optional
        Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3]). Defaults to None (see nb_axes above).
    fill_value : scalar or LArray, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.

    Returns
    -------
    LArray

    Examples
    --------
    >>> from_lists([['sex', 'M', 'F'],
    ...             ['',      0,   1]])
    sex  M  F
         0  1
    >>> from_lists([['sex\\\\year', 1991, 1992, 1993],
    ...             [ 'M',           0,    1,    2],
    ...             [ 'F',           3,    4,    5]])
    sex\\year  1991  1992  1993
           M     0     1     2
           F     3     4     5
    >>> from_lists([['sex', 'nat\\\\year', 1991, 1992, 1993],
    ...             [  'M', 'BE',           1,    0,    0],
    ...             [  'M', 'FO',           2,    0,    0],
    ...             [  'F', 'BE',           0,    0,    1]])
    sex  nat\\year  1991  1992  1993
      M        BE   1.0   0.0   0.0
      M        FO   2.0   0.0   0.0
      F        BE   0.0   0.0   1.0
      F        FO   nan   nan   nan
    >>> from_lists([['sex', 'nat', 1991, 1992, 1993],
    ...             [  'M', 'BE',     1,    0,    0],
    ...             [  'M', 'FO',     2,    0,    0],
    ...             [  'F', 'BE',     0,    0,    1]], nb_axes=3)
    sex  nat\\{2}  1991  1992  1993
      M       BE   1.0   0.0   0.0
      M       FO   2.0   0.0   0.0
      F       BE   0.0   0.0   1.0
      F       FO   nan   nan   nan
    >>> from_lists([['sex', 'nat\\\\year', 1991, 1992, 1993],
    ...             [  'M', 'BE',           1,    0,    0],
    ...             [  'M', 'FO',           2,    0,    0],
    ...             [  'F', 'BE',           0,    0,    1]], fill_value=42)
    sex  nat\\year  1991  1992  1993
      M        BE     1     0     0
      M        FO     2     0     0
      F        BE     0     0     1
      F        FO    42    42    42
    """
    if nb_axes is not None and index_col is not None:
        raise ValueError("cannot specify both nb_axes and index_col")
    elif nb_axes is not None:
        index_col = list(range(nb_axes - 1))
    elif isinstance(index_col, int):
        index_col = [index_col]

    df = pd.DataFrame(data[1:], columns=data[0])
    if index_col is not None:
        df.set_index([df.columns[c] for c in index_col], inplace=True)

    return df_aslarray(df, raw=index_col is None, parse_header=False, sort_rows=sort_rows, sort_columns=sort_columns,
                       fill_value=fill_value)


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def from_string(s, nb_axes=None, index_col=None, sep=' ', **kwargs):
    """Create an array from a multi-line string.

    Parameters
    ----------
    s : str
        input string.
    nb_axes : int, optional
        Number of axes of output array. The first `nb_axes` - 1 columns and the header will be used
        to set the axes of the output array. If not specified, the number of axes is given by the position of the
        column header including the character `\` plus one. If no column header includes the character `\`, the array
        is assumed to have one axis. Defaults to None.
    index_col : list, optional
        Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3]). Defaults to None (see nb_axes above).
    sep : str
        delimiter used to split each line into cells.
    \**kwargs
        See arguments of Pandas read_csv function.

    Returns
    -------
    LArray

    Examples
    --------
    >>> # to create a 1D array using the default separator ' ', a tabulation character \t must be added in front
    >>> # of the data line
    >>> from_string("sex  M  F\\n\\t  0  1")
    sex  M  F
         0  1
    >>> from_string("nat\\\\sex  M  F\\nBE  0  1\\nFO  2  3")
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> from_string("period  a  b\\n2010  0  1\\n2011  2  3")
    period\{1}  a  b
          2010  0  1
          2011  2  3

    Each label is stripped of leading and trailing whitespace, so this is valid too:

    >>> from_string('''nat\\\\sex  M  F
    ...                BE        0  1
    ...                FO        2  3''')
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> from_string('''age  nat\\\\sex  M  F
    ...                0    BE        0  1
    ...                0    FO        2  3
    ...                1    BE        4  5
    ...                1    FO        6  7''')
    age  nat\sex  M  F
      0       BE  0  1
      0       FO  2  3
      1       BE  4  5
      1       FO  6  7

    Empty lines at the beginning or end are ignored, so one can also format the string like this:

    >>> from_string('''
    ... nat\\\\sex  M  F
    ... BE        0  1
    ... FO        2  3
    ... ''')
    nat\sex  M  F
         BE  0  1
         FO  2  3
    """
    return read_csv(StringIO(s), nb_axes=nb_axes, index_col=index_col, sep=sep, skipinitialspace=True, **kwargs)
