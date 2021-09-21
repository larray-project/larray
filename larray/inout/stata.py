import pandas as pd

from larray.core.array import Array
from larray.inout.pandas import from_frame

__all__ = ['read_stata']


def read_stata(filepath_or_buffer, index_col=None, sort_rows=False, sort_columns=False, **kwargs) -> Array:
    r"""
    Reads Stata .dta file and returns an Array with the contents

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
    Array

    See Also
    --------
    Array.to_stata

    Notes
    -----
    The round trip to Stata (Array.to_stata followed by read_stata) loose the name of the "column" axis.

    Examples
    --------
    >>> read_stata('test.dta')                   # doctest: +SKIP
    {0}\{1}  row  country  sex
          0    0       BE    F
          1    1       FR    M
          2    2       FR    F
    >>> read_stata('test.dta', index_col='row')  # doctest: +SKIP
    row\{1}  country  sex
          0       BE    F
          1       FR    M
          2       FR    F
    """
    df = pd.read_stata(filepath_or_buffer, index_col=index_col, **kwargs)
    return from_frame(df, sort_rows=sort_rows, sort_columns=sort_columns)
