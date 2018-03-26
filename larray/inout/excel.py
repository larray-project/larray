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


# TODO : add examples
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