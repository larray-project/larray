from __future__ import absolute_import, print_function

import pandas as pd

from larray.inout.pandas import from_frame

__all__ = ['read_stata']


def read_stata(filepath_or_buffer, index_col=None, sort_rows=False, sort_columns=False,
               **kwargs):
    """
    Reads Stata .dta file and returns an LArray with the contents

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        Path to .dta file or a file handle.
    index_col : str or None, optional
        Name of column to set as index. Defaults to None.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting).
        This only makes sense in combination with index_col. Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.

    Returns
    -------
    LArray
    """
    df = pd.read_stata(filepath_or_buffer, index_col=index_col, **kwargs)
    return from_frame(df, sort_rows=sort_rows, sort_columns=sort_columns)
