from io import StringIO

from pandas import DataFrame, Index

from larray.core.array import Array
from larray.core.constants import nan
from larray.util.misc import deprecate_kwarg
from larray.inout.common import _get_index_col
from larray.inout.pandas import df_asarray, set_dataframe_index_by_position
from larray.inout.csv import read_csv


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def from_lists(data, nb_axes=None, index_col=None, fill_value=nan, sort_rows=False, sort_columns=False,
               wide=True) -> Array:
    r"""
    initialize array from a list of lists (lines).

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

    Returns
    -------
    Array

    Examples
    --------
    >>> from_lists([['sex', 'M', 'F'],
    ...             ['',      0,   1]])
    sex  M  F
         0  1
    >>> from_lists([['sex\\year', 1991, 1992, 1993],
    ...             [ 'M',           0,    1,    2],
    ...             [ 'F',           3,    4,    5]])
    sex\year  1991  1992  1993
           M     0     1     2
           F     3     4     5

    Read array with missing values + `fill_value` argument

    >>> from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
    ...             [  'M', 'BE',           1,    0,    0],
    ...             [  'M', 'FO',           2,    0,    0],
    ...             [  'F', 'BE',           0,    0,    1]])
    sex  nat\year  1991  1992  1993
      M        BE   1.0   0.0   0.0
      M        FO   2.0   0.0   0.0
      F        BE   0.0   0.0   1.0
      F        FO   nan   nan   nan

    >>> from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
    ...             [  'M', 'BE',           1,    0,    0],
    ...             [  'M', 'FO',           2,    0,    0],
    ...             [  'F', 'BE',           0,    0,    1]], fill_value=42)
    sex  nat\year  1991  1992  1993
      M        BE     1     0     0
      M        FO     2     0     0
      F        BE     0     0     1
      F        FO    42    42    42

    Specify the number of axes of the array to be read

    >>> from_lists([['sex', 'nat', 1991, 1992, 1993],
    ...             [  'M', 'BE',     1,    0,    0],
    ...             [  'M', 'FO',     2,    0,    0],
    ...             [  'F', 'BE',     0,    0,    1]], nb_axes=3)
    sex  nat\{2}  1991  1992  1993
      M       BE   1.0   0.0   0.0
      M       FO   2.0   0.0   0.0
      F       BE   0.0   0.0   1.0
      F       FO   nan   nan   nan

    Read array saved in "narrow" format (wide=False)

    >>> from_lists([['sex', 'nat', 'year', 'value'],
    ...             [  'M', 'BE',  1991,    1     ],
    ...             [  'M', 'BE',  1992,    0     ],
    ...             [  'M', 'BE',  1993,    0     ],
    ...             [  'M', 'FO',  1991,    2     ],
    ...             [  'M', 'FO',  1992,    0     ],
    ...             [  'M', 'FO',  1993,    0     ],
    ...             [  'F', 'BE',  1991,    0     ],
    ...             [  'F', 'BE',  1992,    0     ],
    ...             [  'F', 'BE',  1993,    1     ]], wide=False)
    sex  nat\year  1991  1992  1993
      M        BE   1.0   0.0   0.0
      M        FO   2.0   0.0   0.0
      F        BE   0.0   0.0   1.0
      F        FO   nan   nan   nan
    """
    index_col = _get_index_col(nb_axes, index_col, wide)

    columns = data[0]
    # issue #950: avoid pandas interpreting [None, 0, 1] as a Float64Index([nan, 0.0, 1.0], dtype='float64')
    if None in columns:
        columns = Index(columns, dtype=object)

    df = DataFrame(data[1:], columns=columns)
    if index_col is not None:
        df = set_dataframe_index_by_position(df, index_col)

    return df_asarray(df, raw=index_col is None, parse_header=False, sort_rows=sort_rows, sort_columns=sort_columns,
                      fill_value=fill_value, wide=wide)


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def from_string(s, nb_axes=None, index_col=None, sep=' ', wide=True, **kwargs) -> Array:
    r"""Create an array from a multi-line string.

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
    wide : bool, optional
        Whether to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    **kwargs
        See arguments of Pandas read_csv function.

    Returns
    -------
    Array

    Examples
    --------
    >>> # to create a 1D array using the default separator ' ', a tabulation character \t must be added in front
    >>> # of the data line
    >>> from_string("sex  M  F\n\t  0  1")
    sex  M  F
         0  1
    >>> from_string("nat\\sex  M  F\nBE  0  1\nFO  2  3")
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> from_string("period  a  b\n2010  0  1\n2011  2  3")
    period\{1}  a  b
          2010  0  1
          2011  2  3

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
    return read_csv(StringIO(s), nb_axes=nb_axes, index_col=index_col, sep=sep, skipinitialspace=True,
                    wide=wide, **kwargs)
