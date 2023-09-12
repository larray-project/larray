import warnings

import numpy as np
import pandas as pd

from larray.core.array import Array
from larray.core.constants import nan
from larray.inout.pandas import df_asarray
from larray.util.misc import deprecate_kwarg


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
def read_sas(filepath, nb_axes=None, index_col=None, fill_value=nan, na=nan, sort_rows=False, sort_columns=False,
             **kwargs) -> Array:
    r"""
    Read sas file and returns an Array with the contents
        nb_axes: number of axes of the output array
    or
        index_col: Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3]).
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
    return df_asarray(df, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value)
