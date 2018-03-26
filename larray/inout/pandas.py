from __future__ import absolute_import, print_function

from itertools import product

import numpy as np
import pandas as pd

from larray.core.axis import Axis
from larray.core.array import LArray
from larray.util.misc import basestring, decode, unique


__all__ = ['from_frame', 'from_series']


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


def df_aslarray(df, sort_rows=False, sort_columns=False, raw=False, parse_header=True, wide=True, **kwargs):
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
    wide : bool, optional
        Whether or not to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
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
            if isinstance(series.index, pd.core.index.MultiIndex):
                fill_value = kwargs.get('fill_value', np.nan)
                # TODO: use argument sort=False when it will be available
                # (see https://github.com/pandas-dev/pandas/issues/15105)
                df = series.unstack(level=-1, fill_value=fill_value)
                # pandas (un)stack and pivot(_table) methods return a Dataframe/Series with sorted index and columns
                labels = df_labels(series, sort=False)
                index = pd.MultiIndex.from_tuples(list(product(*labels[:-1])), names=series.index.names[:-1])
                columns = labels[-1]
                df = df.reindex(index=index, columns=columns, fill_value=fill_value)
            else:
                series.name = series.index.name
                if sort_rows:
                    raise ValueError('sort_rows=True is not valid for 1D arrays. Please use sort_columns instead.')
                return from_series(series, sort_rows=sort_columns)

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
