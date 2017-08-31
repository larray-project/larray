from __future__ import absolute_import, print_function

import csv
import numpy as np
import pandas as pd
from itertools import product

from larray.core.axis import Axis
from larray.core.array import LArray
from larray.util.misc import basestring, unique, decode, skip_comment_cells, strip_rows, csv_open, StringIO

try:
    import xlwings as xw
except ImportError:
    xw = None


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
    if prodlen == len(df) and columns == list(df.columns) and \
            np.array_equal(df.index.values, new_index.values):
        return df, labels
    return df.reindex(new_index, columns, **kwargs), labels


def df_aslarray(df, sort_rows=False, sort_columns=False, raw=False, parse_header=True, **kwargs):
    # the dataframe was read without index at all (ie 2D dataframe), irrespective of the actual data dimensionality
    if raw:
        columns = df.columns.values.tolist()
        try:
            # take the first column which contains '\'
            # pos_last = next(i for i, v in enumerate(columns) if '\\' in str(v))
            pos_last = next(i for i, v in enumerate(columns) if isinstance(v, basestring) and '\\' in v)
            onedim = False
        except StopIteration:
            # we assume first column will not contain data
            pos_last = 0
            onedim = True

        axes_names = columns[:pos_last + 1]
        if onedim:
            df = df.iloc[:, 1:]
        else:
            # This is required to handle int column names (otherwise we can simply use column positions in set_index).
            # This is NOT the same as df.columns[list(range(...))] !
            index_columns = [df.columns[i] for i in range(pos_last + 1)]
            # TODO: we should pass a flag to df_aslarray so that we can use inplace=True here
            # df.set_index(index_columns, inplace=True)
            df = df.set_index(index_columns)
    else:
        axes_names = [decode(name, 'utf8') for name in df.index.names]

    # handle 2 or more dimensions with the last axis name given using \
    if isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]:
        last_axes = [name.strip() for name in axes_names[-1].split('\\')]
        axes_names = axes_names[:-1] + last_axes
    # handle 1D
    elif len(df) == 1 and axes_names == [None]:
        axes_names = [df.columns.name]
    # handle 2 or more dimensions with the last axis name given as the columns index name
    elif len(df) > 1:
        axes_names += [df.columns.name]

    if len(axes_names) > 1:
        df, axes_labels = cartesian_product_df(df, sort_rows=sort_rows, sort_columns=sort_columns, **kwargs)
    else:
        axes_labels = []

    # we could inline df_aslarray into the functions that use it, so that the
    # original (non-cartesian) df is freed from memory at this point, but it
    # would be much uglier and would not lower the peak memory usage which
    # happens during cartesian_product_df.reindex

    # Pandas treats column labels as column names (strings) so we need to convert them to values
    last_axis_labels = [parse(cell) for cell in df.columns.values] if parse_header else list(df.columns.values)
    axes_labels.append(last_axis_labels)
    axes_names = [str(name) if name is not None else name
                  for name in axes_names]

    axes = [Axis(labels, name) for labels, name in zip(axes_labels, axes_names)]
    data = df.values.reshape([len(axis) for axis in axes])
    return LArray(data, axes)


def read_csv(filepath_or_buffer, nb_index=None, index_col=None, sep=',', headersep=None, na=np.nan,
             sort_rows=False, sort_columns=False, dialect='larray', **kwargs):
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
    nb_index : int, optional
        Number of leading index columns (ex. 4).
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]).
    sep : str, optional
        Separator.
    headersep : str or None, optional
        Separator for headers.
    na : scalar, optional
        Value for NaN (Not A Number). Defaults to NumPy NaN.
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
    >>> from larray.tests.common import abspath
    >>> from larray import ndrange
    >>> fpath = abspath('test.csv')
    >>> a = ndrange('nat=BE,FO;sex=M,F')

    >>> a.to_csv(fpath)
    >>> read_csv(fpath)
    nat\\sex  M  F
         BE  0  1
         FO  2  3
    >>> read_csv(fpath, sort_columns=True)
    nat\\sex  F  M
         BE  1  0
         FO  3  2
    >>> fpath = abspath('no_axis_name.csv')
    >>> a.to_csv(fpath, dialect='classic')
    >>> read_csv(fpath, nb_index=1)
    nat\\{1}  M  F
         BE  0  1
         FO  2  3
    """
    if dialect == 'liam2':
        # read axes names. This needs to be done separately instead of reading the whole file with Pandas then
        # manipulating the dataframe because the header line must be ignored for the column types to be inferred
        # correctly. Note that to read one line, this is faster than using Pandas reader.
        with csv_open(filepath_or_buffer) as f:
            reader = csv.reader(f, delimiter=sep)
            line_stream = skip_comment_cells(strip_rows(reader))
            axes_names = next(line_stream)

        if nb_index is not None or index_col is not None:
            raise ValueError("nb_index and index_col are not compatible with dialect='liam2'")
        if len(axes_names) > 1:
            nb_index = len(axes_names) - 1
        # use the second data line for column headers (excludes comments and blank lines before counting)
        kwargs['header'] = 1
        kwargs['comment'] = '#'

    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    if headersep is not None:
        if index_col is None:
            index_col = [0]

    df = pd.read_csv(filepath_or_buffer, index_col=index_col, sep=sep, **kwargs)
    if dialect == 'liam2':
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

    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=na, raw=raw)


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


def read_hdf(filepath_or_buffer, key, na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """Reads an array named key from a HDF5 file in filepath (path+name)

    Parameters
    ----------
    filepath_or_buffer : str or pandas.HDFStore
        Path and name where the HDF5 file is stored or a HDFStore object.
    key : str
        Name of the array.

    Returns
    -------
    LArray
    """
    df = pd.read_hdf(filepath_or_buffer, key, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=na, parse_header=False)


def read_excel(filepath, sheetname=0, nb_index=None, index_col=None, na=np.nan, sort_rows=False, sort_columns=False,
               engine=None, **kwargs):
    """
    Reads excel file from sheet name and returns an LArray with the contents

    Parameters
    ----------
    filepath : str
        Path where the Excel file has to be read.
    sheetname : str or int, optional
        Name or index of the Excel sheet containing the array to be read.
        By default the array is read from the first sheet.
    nb_index : int, optional
        Number of leading index columns (ex. 4). Defaults to 1.
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]).
        Default to [0].
    na : scalar, optional
        Value for NaN (Not A Number). Defaults to NumPy NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    engine : {'xlrd', 'xlwings'}, optional
        Engine to use to read the Excel file. If None (default), it will use 'xlwings' by default if the module is
        installed and relies on Pandas default reader otherwise.
    **kwargs
    """
    if engine is None:
        engine = 'xlwings' if xw is not None else None

    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    if engine == 'xlwings':
        if kwargs:
            raise TypeError("'{}' is an invalid keyword argument for this function when using the xlwings backend"
                            .format(list(kwargs.keys())[0]))
        if not np.isnan(na):
            raise NotImplementedError("na argument is not currently supported with the (default) "
                                      "xlwings engine")
        if sort_rows:
            raise NotImplementedError("sort_rows argument is not currently supported with the (default) "
                                      "xlwings engine")
        if sort_columns:
            raise NotImplementedError("sort_columns argument is not currently supported with the (default) "
                                      "xlwings engine")
        from larray.io.excel import open_excel
        with open_excel(filepath) as wb:
            return wb[sheetname].load(index_col=index_col)
    else:
        df = pd.read_excel(filepath, sheetname, index_col=index_col, engine=engine, **kwargs)
        return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, raw=index_col is None, fill_value=na)


def read_sas(filepath, nb_index=None, index_col=None, na=np.nan, sort_rows=False, sort_columns=False, **kwargs):
    """
    Reads sas file and returns an LArray with the contents
        nb_index: number of leading index columns (e.g. 4)
    or
        index_col: list of columns for the index (e.g. [0, 1, 3])
    """
    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    df = pd.read_sas(filepath, index=index_col, **kwargs)
    return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=na)


def from_lists(data, nb_index=None, index_col=None):
    """
    initialize array from a list of lists (lines)

    Parameters
    ----------
    data : sequence (tuple, list, ...)
        Input data. All data is supposed to already have the correct type (e.g. strings are not parsed).
    nb_index : int, optional
        Number of leading index columns (ex. 4). Defaults to None, in which case it guesses the number of index columns
        by using the position of the first '\' in the first line.
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]). Defaults to None (see nb_index above).

    Returns
    -------
    LArray

    Examples
    --------
    >>> from_lists([['sex', 'M', 'F'],
    ...             ['',      0,   1]])
    sex  M  F
         0  1
    >>> from_lists([['sex\\year', 1991, 1992, 1993],
    ...             [ 'M',           0,    1,    2],
    ...             [ 'F',           3,    4,    5]])
    sex\\year  1991  1992  1993
           M     0     1     2
           F     3     4     5
    >>> from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
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
    ...             [  'F', 'BE',     0,    0,    1]], nb_index=2)
    sex  nat\\{2}  1991  1992  1993
      M       BE   1.0   0.0   0.0
      M       FO   2.0   0.0   0.0
      F       BE   0.0   0.0   1.0
      F       FO   nan   nan   nan
    """
    if nb_index is not None and index_col is not None:
        raise ValueError("cannot specify both nb_index and index_col")
    elif nb_index is not None:
        index_col = list(range(nb_index))
    elif isinstance(index_col, int):
        index_col = [index_col]

    df = pd.DataFrame(data[1:], columns=data[0])
    if index_col is not None:
        df.set_index([df.columns[c] for c in index_col], inplace=True)

    return df_aslarray(df, raw=index_col is None, parse_header=False)


def from_string(s, nb_index=None, index_col=None, sep=' ', **kwargs):
    """Create an array from a multi-line string.

    Parameters
    ----------
    s : str
        input string.
    nb_index : int, optional
        Number of leading index columns (ex. 4). Defaults to None, in which case it guesses the number of index columns
        by using the position of the first '\' in the first line.
    index_col : list, optional
        List of columns for the index (ex. [0, 1, 2, 3]). Defaults to None (see nb_index above).
    sep : str
        delimiter used to split each line into cells.
    \**kwargs
        See arguments of Pandas read_csv function.

    Returns
    -------
    LArray

    Examples
    --------
    >>> # if one dimension array and default separator ' ', a - must be added in front of the data line  
    >>> from_string("sex  M  F\\n-  0  1")
    sex  M  F
         0  1
    >>> from_string("nat\\sex  M  F\\nBE  0  1\\nFO  2  3")
    nat\sex  M  F
         BE  0  1
         FO  2  3

    Each label is stripped of leading and trailing whitespace, so this is valid too:

    >>> from_string('''nat\\sex  M  F
    ...                BE        0  1
    ...                FO        2  3''')
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> from_string('''age  nat\\sex  M  F
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
    ... nat\\sex  M  F
    ... BE        0  1
    ... FO        2  3
    ... ''')
    nat\sex  M  F
         BE  0  1
         FO  2  3
    """
    return read_csv(StringIO(s), nb_index=nb_index, index_col=index_col, sep=sep, skipinitialspace=True, **kwargs)
