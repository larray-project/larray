from __future__ import absolute_import, print_function

import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.core.array import LArray, aslarray
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.group import Group, _translate_sheet_name
from larray.core.metadata import Metadata
from larray.util.misc import deprecate_kwarg
from larray.inout.session import register_file_handler
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_aslarray, _axes_to_df, _df_to_axes, _groups_to_df, _df_to_groups
from larray.inout.xw_excel import open_excel
from larray.example import get_example_filepath


__all__ = ['read_excel']


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
@deprecate_kwarg('sheetname', 'sheet')
def read_excel(filepath, sheet=0, nb_axes=None, index_col=None, fill_value=nan, na=nan,
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
    >>> fname = get_example_filepath('examples.xlsx')

    Read array from first sheet

    >>> # The data below is derived from a subset of the demo_pjan table from Eurostat
    >>> read_excel(fname)
        geo  gender\\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male  31772665  31936596  32175328
     France       Female  33827685  34005671  34280951
    Germany         Male  39380976  39556923  39835457
    Germany       Female  41142770  41210540  41362080

    Read array from a specific sheet

    >>> # The data below is derived from a subset of the demo_fasec table from Eurostat
    >>> read_excel(fname, 'births')
        geo  gender\\time    2013    2014    2015
    Belgium         Male   64371   64173   62561
    Belgium       Female   61235   60841   59713
     France         Male  415762  418721  409145
     France       Female  396581  400607  390526
    Germany         Male  349820  366835  378478
    Germany       Female  332249  348092  359097

    Missing label combinations

    >>> # let's take a look inside the sheet 'pop_missing_values'.
    >>> # they are missing label combinations: (Paris, male) and (New York, female):

    geo       gender\\time  2013      2014      2015
    Belgium   Male          5472856   5493792   5524068
    Belgium   Female        5665118   5687048   5713206
    France    Female        33827685  34005671  34280951
    Germany   Male          39380976  39556923  39835457

    >>> # by default, cells associated with missing label combinations are filled with NaN.
    >>> # In that case, an int array is converted to a float array.
    >>> read_excel(fname, sheet='pop_missing_values')
        geo  gender\\time        2013        2014        2015
    Belgium         Male   5472856.0   5493792.0   5524068.0
    Belgium       Female   5665118.0   5687048.0   5713206.0
     France         Male         nan         nan         nan
     France       Female  33827685.0  34005671.0  34280951.0
    Germany         Male  39380976.0  39556923.0  39835457.0
    Germany       Female         nan         nan         nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_excel(fname, sheet='pop_missing_values', fill_value=0)
        geo  gender\\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male         0         0         0
     France       Female  33827685  34005671  34280951
    Germany         Male  39380976  39556923  39835457
    Germany       Female         0         0         0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    The content of the sheet 'missing_axis_name' is:

    geo      gender  2013      2014      2015
    Belgium  Male    5472856   5493792   5524068
    Belgium  Female  5665118   5687048   5713206
    France   Male    31772665  31936596  32175328
    France   Female  33827685  34005671  34280951
    Germany  Male    39380976  39556923  39835457
    Germany  Female  41142770  41210540  41362080

    >>> # read the array stored in the sheet 'pop_missing_axis_name' as is
    >>> arr = read_excel(fname, sheet='pop_missing_axis_name')
    >>> # we expected a 3 x 2 x 3 array with data of type int
    >>> # but we got a 6 x 4 array with data of type object
    >>> arr.info
    6 x 4
     geo [6]: 'Belgium' 'Belgium' 'France' 'France' 'Germany' 'Germany'
     {1} [4]: 'gender' '2013' '2014' '2015'
    dtype: object
    memory used: 192 bytes
    >>> # using argument 'nb_axes', you can force the number of axes of the output array
    >>> arr = read_excel(fname, sheet='pop_missing_axis_name', nb_axes=3)
    >>> # as expected, we have a 3 x 2 x 3 array with data of type int
    >>> arr.info
    3 x 2 x 3
     geo [3]: 'Belgium' 'France' 'Germany'
     gender [2]: 'Male' 'Female'
     {2} [3]: 2013 2014 2015
    dtype: int64
    memory used: 144 bytes

    Read array saved in "narrow" format (wide=False)

    >>> # let's take a look inside the sheet 'pop_narrow'.
    >>> # The data are stored in a 'narrow' format:

    geo      time  value
    Belgium  2013  11137974
    Belgium  2014  11180840
    Belgium  2015  11237274
    France   2013  65600350
    France   2014  65942267
    France   2015  66456279

    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_excel
    >>> read_excel(fname, 'pop_narrow_format', wide=False)
    geo\\time      2013      2014      2015
     Belgium  11137974  11180840  11237274
      France  65600350  65942267  66456279
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
            df = pd.read_excel(self.handle, sheet_axes, index_col=None)
            self.axes = _df_to_axes(df)
        else:
            self.axes = OrderedDict()
        # load all groups
        sheet_groups = '__groups__'
        if sheet_groups in self.handle.sheet_names:
            df = pd.read_excel(self.handle, sheet_groups, index_col=None)
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
            sheet_names.remove('__metadata__')
        except:
            pass
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

    def _read_metadata(self):
        sheet_meta = '__metadata__'
        if sheet_meta in self.handle.sheet_names:
            meta = read_excel(self.handle, sheet_meta, engine='xlrd', wide=False)
            return Metadata.from_array(meta)
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        if len(metadata) > 0:
            metadata = aslarray(metadata)
            metadata.to_excel(self.handle, '__metadata__', engine='xlsxwriter', wide=False, value_name='')

    def save(self):
        if len(self.axes) > 0:
            df = _axes_to_df(self.axes.values())
            df.to_excel(self.handle, '__axes__', index=False, engine='xlsxwriter')
        if len(self.groups) > 0:
            df = _groups_to_df(self.groups.values())
            df.to_excel(self.handle, '__groups__', index=False, engine='xlsxwriter')

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
            sheet_names.remove('__metadata__')
        except:
            pass
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

    def _read_metadata(self):
        sheet_meta = '__metadata__'
        if sheet_meta in self.handle:
            meta = self.handle[sheet_meta].load(wide=False)
            return Metadata.from_array(meta)
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        if len(metadata) > 0:
            metadata = aslarray(metadata)
            self.handle['__metadata__'] = metadata.dump(wide=False, value_name='')

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
