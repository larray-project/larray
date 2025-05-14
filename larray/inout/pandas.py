from itertools import product

import numpy as np
import pandas as pd

from larray.core.array import Array
from larray.core.axis import Axis, AxisCollection
from larray.core.constants import nan
from larray.util.misc import unique_list


def decode(s, encoding='utf-8', errors='strict'):
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    else:
        assert s is None or isinstance(s, str), "unexpected " + str(type(s))
        return s


def parse(s):
    r"""
    Used to parse the "folded" axis ticks (usually periods).
    """
    # parameters can be strings or numbers
    if isinstance(s, str):
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


def index_to_labels(idx, sort=True):
    r"""
    Return unique labels for each dimension.
    """
    if isinstance(idx, pd.MultiIndex):
        if sort:
            return list(idx.levels)
        else:
            return [unique_list(idx.get_level_values(label)) for label in range(idx.nlevels)]
    else:
        assert isinstance(idx, pd.Index)
        labels = list(idx.values)
        return [sorted(labels) if sort else labels]


def cartesian_product_df(df, sort_rows=False, sort_columns=False, fill_value=nan, **kwargs):
    idx = df.index
    labels = index_to_labels(idx, sort=sort_rows)
    if isinstance(idx, pd.MultiIndex):
        if sort_rows:
            new_index = pd.MultiIndex.from_product(labels)
        else:
            new_index = pd.MultiIndex.from_tuples(list(product(*labels)))
    else:
        if sort_rows:
            new_index = pd.Index(labels[0], name=idx.name)
        else:
            new_index = idx
    columns = sorted(df.columns) if sort_columns else list(df.columns)
    # the prodlen test is meant to avoid the more expensive array_equal test
    prodlen = np.prod([len(axis_labels) for axis_labels in labels])
    if prodlen == len(df) and columns == list(df.columns) and np.array_equal(idx.values, new_index.values):
        return df, labels
    return df.reindex(index=new_index, columns=columns, fill_value=fill_value, **kwargs), labels


def from_series(s, sort_rows=False, fill_value=nan, meta=None, **kwargs) -> Array:
    r"""
    Convert Pandas Series into Array.

    Parameters
    ----------
    s : Pandas Series
        Input Pandas Series.
    sort_rows : bool, optional
        Whether to sort the rows alphabetically. Defaults to False.
    fill_value : scalar, optional
        Value used to fill cells corresponding to label combinations which are not present in the input Series.
        Defaults to NaN.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    See Also
    --------
    Array.to_series

    Examples
    --------
    >>> from larray import ndtest
    >>> s = ndtest((2, 2, 2), dtype=float).to_series()
    >>> s                                                                             # doctest: +NORMALIZE_WHITESPACE
    a   b   c
    a0  b0  c0    0.0
            c1    1.0
        b1  c0    2.0
            c1    3.0
    a1  b0  c0    4.0
            c1    5.0
        b1  c0    6.0
            c1    7.0
    dtype: float64
    >>> from_series(s)
     a  b\c   c0   c1
    a0   b0  0.0  1.0
    a0   b1  2.0  3.0
    a1   b0  4.0  5.0
    a1   b1  6.0  7.0
    """
    if isinstance(s.index, pd.MultiIndex):
        # TODO: use argument sort=False when it will be available
        # (see https://github.com/pandas-dev/pandas/issues/15105)
        df = s.unstack(level=-1, fill_value=fill_value)
        # pandas (un)stack and pivot(_table) methods return a Dataframe/Series with sorted index and columns
        if not sort_rows:
            labels = index_to_labels(s.index, sort=False)
            if isinstance(df.index, pd.MultiIndex):
                index = pd.MultiIndex.from_tuples(list(product(*labels[:-1])), names=s.index.names[:-1])
            else:
                index = labels[0]
            columns = labels[-1]
            df = df.reindex(index=index, columns=columns, fill_value=fill_value)
        return from_frame(df, sort_rows=sort_rows, sort_columns=sort_rows, fill_value=fill_value, meta=meta, **kwargs)
    else:
        name = decode(s.name, 'utf8') if s.name is not None else decode(s.index.name, 'utf8')
        if sort_rows:
            s = s.sort_index()
        return Array(s.values, Axis(s.index.values, name), meta=meta)


def from_frame(df, sort_rows=False, sort_columns=False, parse_header=False, unfold_last_axis_name=False,
               fill_value=nan, meta=None, cartesian_prod=True, **kwargs) -> Array:
    r"""
    Convert Pandas DataFrame into Array.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe. By default, name and labels of the last axis are defined by the name and labels of the
        columns Index of the dataframe unless argument unfold_last_axis_name is set to True.
    sort_rows : bool, optional
        Whether to sort the rows alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    sort_columns : bool, optional
        Whether to sort the columns alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    parse_header : bool, optional
        Whether to parse columns labels. Pandas treats column labels as strings.
        If True, column labels are converted into int, float or boolean when possible. Defaults to False.
    unfold_last_axis_name : bool, optional
        Whether to extract the names of the last two axes by splitting the name of the last index column of the
        dataframe using ``\``. Defaults to False.
    fill_value : scalar, optional
        Value used to fill cells corresponding to label combinations which are not present in the input DataFrame.
        Defaults to NaN.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.
    cartesian_prod : bool, optional
        Whether to expand the dataframe to a cartesian product dataframe as needed by Array.
        This is an expensive operation but is absolutely required if you cannot guarantee your dataframe is already
        well formed. If True, arguments `sort_rows` and `sort_columns` must be set to False.
        Defaults to True.

    Returns
    -------
    Array

    See Also
    --------
    Array.to_frame

    Examples
    --------
    >>> from larray import ndtest
    >>> df = ndtest((2, 2, 2)).to_frame()
    >>> df                                                                             # doctest: +NORMALIZE_WHITESPACE
    c      c0  c1
    a  b
    a0 b0   0   1
       b1   2   3
    a1 b0   4   5
       b1   6   7
    >>> from_frame(df)
     a  b\c  c0  c1
    a0   b0   0   1
    a0   b1   2   3
    a1   b0   4   5
    a1   b1   6   7
    """
    axes_names = [decode(name, 'utf8') if isinstance(name, bytes) else name
                  for name in df.index.names]

    # handle 2 or more dimensions with the last axis name given using \
    if unfold_last_axis_name:
        if isinstance(axes_names[-1], str) and '\\' in axes_names[-1]:
            last_axes = [name.strip() for name in axes_names[-1].split('\\')]
            axes_names = axes_names[:-1] + last_axes
        else:
            axes_names += [None]
    else:
        axes_names += [df.columns.name]

    if cartesian_prod:
        df, axes_labels = cartesian_product_df(df, sort_rows=sort_rows, sort_columns=sort_columns,
                                               fill_value=fill_value, **kwargs)
    else:
        if sort_rows or sort_columns:
            raise ValueError('sort_rows and sort_columns cannot not be used when cartesian_prod is set to False. '
                             'Please call the method sort_labels on the returned array to sort rows or columns')
        axes_labels = index_to_labels(df.index, sort=False)

    # Pandas treats column labels as column names (strings) so we need to convert them to values
    last_axis_labels = [parse(cell) for cell in df.columns.values] if parse_header else list(df.columns.values)
    axes_labels.append(last_axis_labels)

    axes = AxisCollection([Axis(labels, name) for labels, name in zip(axes_labels, axes_names)])
    data = df.values.reshape(axes.shape)
    return Array(data, axes, meta=meta)


def set_dataframe_index_by_position(df, index_col_indices):
    """
    equivalent to Dataframe.set_index but with column indices, not column labels.

    This is necessary to support creating an index from columns without a name or with duplicate names.

    Return a new Dataframe
    """
    if not isinstance(index_col_indices, list):
        index_col_indices = [index_col_indices]
    index_col_indices_set = set(index_col_indices)
    index_col_values = [df.iloc[:, i] for i in index_col_indices]
    non_index_col_indices = [i for i in range(len(df.columns)) if i not in index_col_indices_set]
    # drop the index columns from the "normal" columns of the dataframe
    df = df.iloc[:, non_index_col_indices]
    # add them back as index columns
    df.set_index(index_col_values, inplace=True)
    return df


def df_asarray(df, sort_rows=False, sort_columns=False, raw=False, parse_header=True, wide=True, cartesian_prod=True,
               **kwargs) -> Array:
    r"""
    Prepare Pandas DataFrame and then convert it into Array.

    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe.
    sort_rows : bool, optional
        Whether to sort the rows alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    sort_columns : bool, optional
        Whether to sort the columns alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    raw : bool, optional
        Whether to consider the input dataframe as a raw dataframe, i.e. read without index at all.
        If True, build the first N-1 axes of the output array from the first N-1 dataframe columns. Defaults to False.
    parse_header : bool, optional
        Whether to parse columns labels. Pandas treats column labels as strings.
        If True, column labels are converted into int, float or boolean when possible. Defaults to True.
    wide : bool, optional
        Whether to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    cartesian_prod : bool, optional
        Whether to expand the dataframe to a cartesian product dataframe as needed by Array.
        This is an expensive operation but is absolutely required if you cannot guarantee your dataframe is already
        well formed. If True, arguments `sort_rows` and `sort_columns` must be set to False.
        Defaults to True.

    Returns
    -------
    Array
    """
    # we could inline df_asarray into the functions that use it, so that the original (non-cartesian) df is freed from
    # memory at this point, but it would be much uglier and would not lower the peak memory usage which happens during
    # cartesian_product_df.reindex

    # raw = True: the dataframe was read without index at all (ie 2D dataframe),
    # irrespective of the actual data dimensionality
    if raw:
        columns = df.columns.values.tolist()
        if wide:
            try:
                # take the first column which contains '\'
                pos_last = next(i for i, v in enumerate(columns) if isinstance(v, str) and '\\' in v)
            except StopIteration:
                # we assume first column will not contain data
                pos_last = 0

            # This is required to handle int column names (otherwise we can simply use column positions in set_index).
            # This is NOT the same as df.columns[list(range(...))] !
            df = set_dataframe_index_by_position(df, list(range(pos_last + 1)))
        else:
            df = set_dataframe_index_by_position(df, list(range(len(df.columns) - 1)))
            series = df.iloc[:, -1]
            series.name = df.index.name
            return from_series(series, sort_rows=sort_columns, **kwargs)

    # handle 1D arrays
    if len(df) == 1 and (pd.isnull(df.index.values[0])
                         or (isinstance(df.index.values[0], str) and df.index.values[0].strip() == '')):
        if parse_header:
            df.columns = pd.Index([parse(cell) for cell in df.columns.values], name=df.columns.name)
        series = df.iloc[0]
        series.name = df.index.name
        if sort_rows:
            raise ValueError('sort_rows=True is not valid for 1D arrays. Please use sort_columns instead.')
        res = from_series(series, sort_rows=sort_columns)
    else:
        def parse_axis_name(name):
            if isinstance(name, bytes):
                name = decode(name, 'utf8')
            if not name:
                name = None
            return name
        axes_names = [parse_axis_name(name) for name in df.index.names]
        unfold_last_axis_name = isinstance(axes_names[-1], str) and '\\' in axes_names[-1]
        res = from_frame(df, sort_rows=sort_rows, sort_columns=sort_columns, parse_header=parse_header,
                         unfold_last_axis_name=unfold_last_axis_name, cartesian_prod=cartesian_prod, **kwargs)

    # ugly hack to avoid anonymous axes converted as axes with name 'Unnamed: x' by pandas
    # we also take the opportunity to change axes with empty name to real anonymous axes (name is None) to
    # make them roundtrip correctly, based on the assumption that in an in-memory LArray an anonymouse axis is more
    # likely and useful than an Axis with an empty name.
    # TODO : find a more robust and elegant solution
    res = res.rename({axis: None for axis in res.axes if (isinstance(axis.name, str)
                                                          and (axis.name == '' or 'Unnamed:' in axis.name))})
    return res
