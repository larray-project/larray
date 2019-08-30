from __future__ import absolute_import, print_function

import re
from itertools import product
from collections import OrderedDict

import numpy as np
import pandas as pd

from larray.core.array import LArray
from larray.core.axis import Axis, AxisCollection
from larray.core.group import LGroup
from larray.core.constants import nan
from larray.util.misc import basestring, decode, unique


def parse(s):
    r"""
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


def index_to_labels(idx, sort=True):
    r"""
    Returns unique labels for each dimension.
    """
    if isinstance(idx, pd.core.index.MultiIndex):
        if sort:
            return list(idx.levels)
        else:
            return [list(unique(idx.get_level_values(l))) for l in idx.names]
    else:
        assert isinstance(idx, pd.core.index.Index)
        labels = list(idx.values)
        return [sorted(labels) if sort else labels]


def cartesian_product_df(df, sort_rows=False, sort_columns=False, fill_value=nan, **kwargs):
    idx = df.index
    labels = index_to_labels(idx, sort=sort_rows)
    if isinstance(idx, pd.core.index.MultiIndex):
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


def from_series(s, sort_rows=False, fill_value=nan, meta=None, **kwargs):
    r"""
    Converts Pandas Series into LArray.

    Parameters
    ----------
    s : Pandas Series
        Input Pandas Series.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically. Defaults to False.
    fill_value : scalar, optional
        Value used to fill cells corresponding to label combinations which are not present in the input Series.
        Defaults to NaN.
    meta : list of pairs or dict or OrderedDict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    LArray

    See Also
    --------
    LArray.to_series

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
    if isinstance(s.index, pd.core.index.MultiIndex):
        # TODO: use argument sort=False when it will be available
        # (see https://github.com/pandas-dev/pandas/issues/15105)
        df = s.unstack(level=-1, fill_value=fill_value)
        # pandas (un)stack and pivot(_table) methods return a Dataframe/Series with sorted index and columns
        if not sort_rows:
            labels = index_to_labels(s.index, sort=False)
            if isinstance(df.index, pd.core.index.MultiIndex):
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
        return LArray(s.values, Axis(s.index.values, name), meta=meta)


_anonymous_axis_pattern = re.compile(r'\{(\d+|\??)\}\*?')


def from_frame(df, sort_rows=False, sort_columns=False, parse_header=False, unfold_last_axis_name=False,
               fill_value=nan, meta=None, cartesian_prod=True, **kwargs):
    r"""
    Converts Pandas DataFrame into LArray.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe. By default, name and labels of the last axis are defined by the name and labels of the
        columns Index of the dataframe unless argument unfold_last_axis_name is set to True.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    parse_header : bool, optional
        Whether or not to parse columns labels. Pandas treats column labels as strings.
        If True, column labels are converted into int, float or boolean when possible. Defaults to False.
    unfold_last_axis_name : bool, optional
        Whether or not to extract the names of the last two axes by splitting the name of the last index column of the
        dataframe using ``\``. Defaults to False.
    fill_value : scalar, optional
        Value used to fill cells corresponding to label combinations which are not present in the input DataFrame.
        Defaults to NaN.
    meta : list of pairs or dict or OrderedDict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.
    cartesian_prod : bool, optional
        Whether or not to expand the dataframe to a cartesian product dataframe as needed by LArray.
        This is an expensive operation but is absolutely required if you cannot guarantee your dataframe is already
        well formed. If True, arguments `sort_rows` and `sort_columns` must be set to False.
        Defaults to True.

    Returns
    -------
    LArray

    See Also
    --------
    LArray.to_frame

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

    Names of the last two axes written as ``before_last_axis_name\\last_axis_name``

    >>> df = ndtest((2, 2, 2)).to_frame(fold_last_axis_name=True)
    >>> df                                                                             # doctest: +NORMALIZE_WHITESPACE
            c0  c1
    a  b\c
    a0 b0    0   1
       b1    2   3
    a1 b0    4   5
       b1    6   7
    >>> from_frame(df, unfold_last_axis_name=True)
     a  b\c  c0  c1
    a0   b0   0   1
    a0   b1   2   3
    a1   b0   4   5
    a1   b1   6   7
    """
    axes_names = [decode(name, 'utf8') if isinstance(name, basestring) else name
                  for name in df.index.names]

    # handle 2 or more dimensions with the last axis name given using \
    if unfold_last_axis_name:
        if isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]:
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
                             'Please call the method sort_axes on the returned array to sort rows or columns')
        axes_labels = index_to_labels(df.index, sort=False)

    # Pandas treats column labels as column names (strings) so we need to convert them to values
    last_axis_labels = [parse(cell) for cell in df.columns.values] if parse_header else list(df.columns.values)
    axes_labels.append(last_axis_labels)
    axes_names = [str(name) if name is not None else name
                  for name in axes_names]

    def _to_axis(labels, name):
        if name is not None:
            if name[-1] == '*':
                labels = len(labels)
                name = name[:-1]
            if _anonymous_axis_pattern.match(name):
                name = None
        return Axis(labels, name)

    axes = AxisCollection([_to_axis(labels, name) for labels, name in zip(axes_labels, axes_names)])
    data = df.values.reshape(axes.shape)
    return LArray(data, axes, meta=meta)


def df_aslarray(df, sort_rows=False, sort_columns=False, raw=False, parse_header=True, wide=True, cartesian_prod=True,
                **kwargs):
    r"""
    Prepare Pandas DataFrame and then convert it into LArray.

    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Must be False if `cartesian_prod` is set to True.
        Defaults to False.
    raw : bool, optional
        Whether or not to consider the input dataframe as a raw dataframe, i.e. read without index at all.
        If True, build the first N-1 axes of the output array from the first N-1 dataframe columns. Defaults to False.
    parse_header : bool, optional
        Whether or not to parse columns labels. Pandas treats column labels as strings.
        If True, column labels are converted into int, float or boolean when possible. Defaults to True.
    wide : bool, optional
        Whether or not to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    cartesian_prod : bool, optional
        Whether or not to expand the dataframe to a cartesian product dataframe as needed by LArray.
        This is an expensive operation but is absolutely required if you cannot guarantee your dataframe is already
        well formed. If True, arguments `sort_rows` and `sort_columns` must be set to False.
        Defaults to True.

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
        if wide:
            try:
                # take the first column which contains '\'
                pos_last = next(i for i, v in enumerate(columns) if isinstance(v, basestring) and '\\' in v)
            except StopIteration:
                # we assume first column will not contain data
                pos_last = 0

            # This is required to handle int column names (otherwise we can simply use column positions in set_index).
            # This is NOT the same as df.columns[list(range(...))] !
            index_columns = [df.columns[i] for i in range(pos_last + 1)]
            df.set_index(index_columns, inplace=True)
        else:
            index_columns = [df.columns[i] for i in range(len(df.columns) - 1)]
            df.set_index(index_columns, inplace=True)
            series = df[df.columns[-1]]
            series.name = df.index.name
            return from_series(series, sort_rows=sort_columns, **kwargs)

    # handle 1D arrays
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
        axes_names = [decode(name, 'utf8') if isinstance(name, basestring) else name
                      for name in df.index.names]
        unfold_last_axis_name = isinstance(axes_names[-1], basestring) and '\\' in axes_names[-1]
        return from_frame(df, sort_rows=sort_rows, sort_columns=sort_columns, parse_header=parse_header,
                          unfold_last_axis_name=unfold_last_axis_name, cartesian_prod=cartesian_prod, **kwargs)


# #################################### #
#    SERIES <--> AXIS, GROUP, META     #
# #################################### #

def _extract_labels_from_series(series):
    # remove trailing NaN or None values
    # (multiple Axis or Group objects of different lengths
    # are stored in the same DataFrame leading to trailing
    # NaNs or None values when split into series)
    series = series.loc[:series.last_valid_index()]

    labels = np.asarray(series.values)
    # integer labels of axes or groups may have been converted to float values
    # because of trailing NaNs
    if labels.dtype.kind == 'f' and all([label.is_integer() for label in labels]):
        labels = labels.astype(int)
    # if dtype is still object, we assume values are strings
    if labels.dtype.kind == 'O':
        labels = labels.astype(str)
    return labels


def _axis_to_series(key, axis, dtype=None):
    name = '{}:{}'.format(key, str(axis))
    labels = len(axis) if axis.iswildcard else axis.labels
    return pd.Series(data=labels, name=name, dtype=dtype)


def _series_to_axis(series):
    name = str(series.name)
    labels = _extract_labels_from_series(series)
    if ':' in name:
        key, axis_name = name.split(':')
        if axis_name[-1] == '*':
            labels = labels[0]
        if _anonymous_axis_pattern.match(axis_name):
            axis_name = None
    else:
        # for backward compatibility
        key = axis_name = name
    return key, Axis(labels=labels, name=axis_name)


def _group_to_series(key, group, dtype=None):
    if group.axis.name is None:
        raise ValueError("Cannot save a group with an anonymous associated axis")
    name = '{}:{}@{}'.format(key, group.name, group.axis.name)
    return pd.Series(data=group.eval(), name=name, dtype=dtype)


def _series_to_group(series, axes):
    key, name = str(series.name).split(':')
    group_name, axis_name = name.split('@')
    if group_name == 'None':
        group_name = None
    axis = axes[axis_name]
    group_key = _extract_labels_from_series(series)
    return key, LGroup(key=group_key, name=group_name, axis=axis)


# ######################################## #
#    DATAFRAME <--> AXES, GROUPS, META     #
# ######################################## #

def _df_to_axes(df):
    return OrderedDict([_series_to_axis(df[col_name]) for col_name in df.columns.values])


def _axes_to_df(axes):
    # set dtype to np.object otherwise pd.concat below may convert an int row/column as float
    # if trailing NaN need to be added
    return pd.concat([_axis_to_series(key, axis, dtype=np.object) for key, axis in axes.items()], axis=1)


def _df_to_groups(df, axes):
    return OrderedDict([_series_to_group(df[col_name], axes) for col_name in df.columns.values])


def _groups_to_df(groups):
    # set dtype to np.object otherwise pd.concat below may convert an int row/column as float
    # if trailing NaN need to be added
    return pd.concat([_group_to_series(key, group, dtype=np.object) for key, group in groups.items()], axis=1)
