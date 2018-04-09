from __future__ import absolute_import, print_function

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.core.axis import Axis
from larray.core.group import Group, _translate_sheet_name
from larray.core.array import LArray
from larray.util.misc import deprecate_kwarg
from larray.inout.session import register_file_handler
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_aslarray, _axes_to_df, _df_to_axes, _groups_to_df, _df_to_groups
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
    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'examples.xlsx')

    Read array from first sheet

    >>> read_excel(fname)
    a  a0  a1  a2
        0   1   2

    Read array from a specific sheet

    >>> read_excel(fname, '2d')
    a\\b  b0  b1
      1   0   1
      2   2   3
      3   4   5

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
    >>> read_excel(fname, sheet='missing_values')
    a  b\c    c0    c1    c2
    1   b0   0.0   1.0   2.0
    1   b1   3.0   4.0   5.0
    2   b0   nan   nan   nan
    2   b1   9.0  10.0  11.0
    3   b0  12.0  13.0  14.0
    3   b1   nan   nan   nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_excel(fname, sheet='missing_values', fill_value=0)
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   0   0   0
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1   0   0   0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    >>> # read the array stored in the CSV file as it
    >>> read_excel(fname, sheet='missing_axis_name')
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5
       a2   6   7   8
    >>> # using argument 'nb_axes', you can force the number of axes of the output array
    >>> read_excel(fname, sheet='missing_axis_name', nb_axes=2)
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5
       a2   6   7   8

    Read array saved in "narrow" format (wide=False)

    >>> # let's take a look inside the sheet 'narrow_2d'.
    >>> # The data are stored in a 'narrow' format:

    a  b   value
    1  b0  0
    1  b1  1
    2  b0  2
    2  b1  3
    3  b0  4
    3  b1  5

    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_excel
    >>> read_excel(fname, 'narrow_2d', wide=False)
    a\\b  b0  b1
      1   0   1
      2   2   3
      3   4   5
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


@register_file_handler('pandas_excel')
class PandasExcelHandler(FileHandler):
    """
    Handler for Excel files using Pandas.
    """
    def __init__(self, fname, overwrite_file=False):
        super(PandasExcelHandler, self).__init__(fname, overwrite_file)
        self.axes = None
        self.groups = None

    def _load_axes_and_groups(self):
        # load all axes
        sheet_axes = '__axes__'
        if sheet_axes in self.handle.sheet_names:
            df = pd.read_excel(self.handle, sheet_axes)
            self.axes = _df_to_axes(df)
        else:
            self.axes = OrderedDict()
        # load all groups
        sheet_groups = '__groups__'
        if sheet_groups in self.handle.sheet_names:
            df = pd.read_excel(self.handle, sheet_groups)
            self.groups = _df_to_groups(df, self.axes)
        else:
            self.groups = OrderedDict()

    def _open_for_read(self):
        self.handle = pd.ExcelFile(self.fname)
        self._load_axes_and_groups()

    def _open_for_write(self):
        self.handle = pd.ExcelWriter(self.fname)
        self.axes = OrderedDict()
        self.groups = OrderedDict()

    def list_items(self):
        sheet_names = self.handle.sheet_names
        items = []
        try:
            sheet_names.remove('__axes__')
            items = [(name, 'Axis') for name in sorted(self.axes.keys())]
        except:
            pass
        try:
            sheet_names.remove('__groups__')
            items += [(name, 'Group') for name in sorted(self.groups.keys())]
        except:
            pass
        items += [(name, 'Array') for name in sheet_names]
        return items

    def _read_item(self, key, type, *args, **kwargs):
        if type == 'Array':
            df = self.handle.parse(key, *args, **kwargs)
            return key, df_aslarray(df, raw=True)
        elif type == 'Axis':
            return key, self.axes[key]
        elif type == 'Group':
            return key, self.groups[key]
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        kwargs['engine'] = 'xlsxwriter'
        if isinstance(value, LArray):
            value.to_excel(self.handle, key, *args, **kwargs)
        elif isinstance(value, Axis):
            self.axes[key] = value
        elif isinstance(value, Group):
            self.groups[key] = value
        else:
            raise TypeError()

    def save(self):
        if len(self.axes) > 0:
            df = _axes_to_df(self.axes.values())
            df.to_excel(self.handle, '__axes__', engine='xlsxwriter')
        if len(self.groups) > 0:
            df = _groups_to_df(self.groups.values())
            df.to_excel(self.handle, '__groups__', engine='xlsxwriter')

    def close(self):
        self.handle.close()


@register_file_handler('xlwings_excel', ['xls', 'xlsx'])
class XLWingsHandler(FileHandler):
    """
    Handler for Excel files using XLWings.
    """
    def __init__(self, fname, overwrite_file=False):
        super(XLWingsHandler, self).__init__(fname, overwrite_file)
        self.axes = None
        self.groups = None

    def _get_original_file_name(self):
        # for XLWingsHandler, no need to create a temporary file, the job is already done in the Workbook class
        pass

    def _load_axes_and_groups(self):
        # load all axes
        sheet_axes = '__axes__'
        if sheet_axes in self.handle:
            df = self.handle[sheet_axes][:].options(pd.DataFrame, index=False).value
            self.axes = _df_to_axes(df)
        else:
            self.axes = OrderedDict()
        # load all groups
        sheet_groups = '__groups__'
        if sheet_groups in self.handle:
            df = self.handle[sheet_groups][:].options(pd.DataFrame, index=False).value
            self.groups = _df_to_groups(df, self.axes)
        else:
            self.groups = OrderedDict()

    def _open_for_read(self):
        self.handle = open_excel(self.fname)
        self._load_axes_and_groups()

    def _open_for_write(self):
        self.handle = open_excel(self.fname, overwrite_file=self.overwrite_file)
        self._load_axes_and_groups()

    def list_items(self):
        sheet_names = self.handle.sheet_names()
        items = []
        try:
            sheet_names.remove('__axes__')
            items = [(name, 'Axis') for name in sorted(self.axes.keys())]
        except:
            pass
        try:
            sheet_names.remove('__groups__')
            items += [(name, 'Group') for name in sorted(self.groups.keys())]
        except:
            pass
        items += [(name, 'Array') for name in sheet_names]
        return items

    def _read_item(self, key, type, *args, **kwargs):
        if type == 'Array':
            return key, self.handle[key].load(*args, **kwargs)
        elif type == 'Axis':
            return key, self.axes[key]
        elif type == 'Group':
            return key, self.groups[key]
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, LArray):
            self.handle[key] = value.dump(*args, **kwargs)
        elif isinstance(value, Axis):
            self.axes[key] = value
        elif isinstance(value, Group):
            self.groups[key] = value
        else:
            raise TypeError()

    def save(self):
        if len(self.axes) > 0:
            df = _axes_to_df(self.axes.values())
            self.handle['__axes__'] = ''
            self.handle['__axes__'][:].options(pd.DataFrame, index=False).value = df
        if len(self.groups) > 0:
            df = _groups_to_df(self.groups.values())
            self.handle['__groups__'] = ''
            self.handle['__groups__'][:].options(pd.DataFrame, index=False).value = df
        self.handle.save()

    def close(self):
        self.handle.close()