from __future__ import absolute_import, print_function

import warnings

import numpy as np
import pandas as pd
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.core.group import _translate_sheet_name
from larray.util.misc import deprecate_kwarg
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_aslarray
from larray.inout.xw_excel import open_excel

__all__ = ['read_excel']


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
@deprecate_kwarg('sheetname', 'sheet')
def read_excel(filepath, sheet=0, nb_axes=None, index_col=None, fill_value=np.nan, na=np.nan,
               sort_rows=False, sort_columns=False, wide=True, engine=None, **kwargs):
    """
    Reads excel file from sheet name and returns an LArray with the contents

    Parameters
    ----------
    filepath : str
        Path where the Excel file has to be read or use -1 to refer to the currently active workbook.
    sheet : str, Group or int, optional
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
    wide : bool, optional
        Whether or not to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    engine : {'xlrd', 'xlwings'}, optional
        Engine to use to read the Excel file. If None (default), it will use 'xlwings' by default if the module is
        installed and relies on Pandas default reader otherwise.
    **kwargs

    Returns
    -------
    LArray

    Examples
    --------
    >>> import os
    >>> from larray import EXAMPLE_FILES_DIR
    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'test.xlsx')

    Read array from first sheet

    >>> read_excel(fname)
    a  a0  a1  a2
        0   1   2

    Read array from a specific sheet

    >>> read_excel(fname, '3d')
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   6   7   8
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1  15  16  17

    Missing label combinations

    >>> # let's take a look inside the sheet 'missing_values'.
    >>> # they are missing label combinations: (a=2, b=b0) and (a=3, b=b1):

    a  b\c  c0  c1  c2
    1  b0   0   1   2
    1  b1   3   4   5
    2  b1   9   10  11
    3  b0   12  13  14

    >>> # by default, cells associated with missing label combinations are filled with NaN.
    >>> # In that case, an int array is converted to a float array.
    >>> read_excel(fname, 'missing_values')
    a  b\c    c0    c1    c2
    1   b0   0.0   1.0   2.0
    1   b1   3.0   4.0   5.0
    2   b0   nan   nan   nan
    2   b1   9.0  10.0  11.0
    3   b0  12.0  13.0  14.0
    3   b1   nan   nan   nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_excel(fname, 'missing_values', fill_value=0)
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   0   0   0
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1   0   0   0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    >>> # read the array stored in the CSV file as it
    >>> read_excel(fname, '2d_classic')
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5
       a2   6   7   8
    >>> # using argument 'nb_axes', you can force the number of axes of the output array
    >>> read_excel(fname, '2d_classic', nb_axes=2)
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5
       a2   6   7   8

    Sort rows and columns

    >>> # let's first read the arrays from sheet 'unsorted' as it:
    >>> read_excel(fname, 'unsorted')
    a  b\c  c2  c1  c0
    3   b1   0   1   2
    3   b0   3   4   5
    2   b1   6   7   8
    2   b0   9  10  11
    1   b1  12  13  14
    1   b0  15  16  17
    >>> # by setting arguments 'sort_rows' and 'sort_columns' to True,
    >>> # the output array has rows and columns sorted.
    >>> read_excel(fname, 'unsorted', sort_rows=True, sort_columns=True)
    a  b\c  c0  c1  c2
    1   b0  17  16  15
    1   b1  14  13  12
    2   b0  11  10   9
    2   b1   8   7   6
    3   b0   5   4   3
    3   b1   2   1   0

    Read array saved in "narrow" format (wide=False)

    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'test_narrow.xlsx')
    >>> # let's take a look inside the sheet '3d'.
    >>> # The data are stored in a 'narrow' format:

    a  b   c   value
    1  b0  c0  0
    1  b0  c1  1
    1  b0  c2  2
    1  b1  c0  3
    1  b1  c1  4
    1  b1  c2  5
    2  b0  c0  6
    2  b0  c1  7
    2  b0  c2  8
    2  b1  c0  9
    2  b1  c1  10
    2  b1  c2  11
    3  b0  c0  12
    3  b0  c1  13
    3  b0  c2  14
    3  b1  c0  15
    3  b1  c1  16
    3  b1  c2  17

    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_excel
    >>> read_excel(fname, '3d', wide=False)
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   6   7   8
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1  15  16  17
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_excel `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    sheet = _translate_sheet_name(sheet)

    if engine is None:
        engine = 'xlwings' if xw is not None else None

    index_col = _get_index_col(nb_axes, index_col, wide)

    if engine == 'xlwings':
        if kwargs:
            raise TypeError("'{}' is an invalid keyword argument for this function when using the xlwings backend"
                            .format(list(kwargs.keys())[0]))
        from larray.inout.xw_excel import open_excel
        with open_excel(filepath) as wb:
            return wb[sheet].load(index_col=index_col, fill_value=fill_value, sort_rows=sort_rows,
                                  sort_columns=sort_columns, wide=wide)
    else:
        df = pd.read_excel(filepath, sheet, index_col=index_col, engine=engine, **kwargs)
        return df_aslarray(df, sort_rows=sort_rows, sort_columns=sort_columns, raw=index_col is None,
                           fill_value=fill_value, wide=wide)


class PandasExcelHandler(FileHandler):
    """
    Handler for Excel files using Pandas.
    """
    def _open_for_read(self):
        self.handle = pd.ExcelFile(self.fname)

    def _open_for_write(self):
        self.handle = pd.ExcelWriter(self.fname)

    def list(self):
        return self.handle.sheet_names

    def _read_item(self, key, *args, **kwargs):
        df = self.handle.parse(key, *args, **kwargs)
        return key, df_aslarray(df, raw=True)

    def _dump(self, key, value, *args, **kwargs):
        kwargs['engine'] = 'xlsxwriter'
        value.to_excel(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()


class XLWingsHandler(FileHandler):
    """
    Handler for Excel files using XLWings.
    """
    def _get_original_file_name(self):
        # for XLWingsHandler, no need to create a temporary file, the job is already done in the Workbook class
        pass

    def _open_for_read(self):
        self.handle = open_excel(self.fname)

    def _open_for_write(self):
        self.handle = open_excel(self.fname, overwrite_file=self.overwrite_file)

    def list(self):
        return self.handle.sheet_names()

    def _read_item(self, key, *args, **kwargs):
        return key, self.handle[key].load(*args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        self.handle[key] = value.dump(*args, **kwargs)

    def save(self):
        self.handle.save()

    def close(self):
        self.handle.close()