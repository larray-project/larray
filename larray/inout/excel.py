import warnings

import numpy as np
import pandas as pd
try:
    import xlwings as xw
except ImportError:
    xw = None
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None

from typing import Dict

from larray.core.array import Array, asarray
from larray.core.constants import nan
from larray.core.group import _translate_sheet_name
from larray.core.metadata import Metadata
from larray.util.misc import deprecate_kwarg
from larray.inout.session import register_file_handler
from larray.inout.common import _get_index_col, FileHandler
from larray.inout.pandas import df_asarray
from larray.inout.xw_excel import open_excel
from larray.example import get_example_filepath             # noqa: F401


__all__ = ['read_excel']


@deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
@deprecate_kwarg('sheetname', 'sheet')
# We use "# doctest: +SKIP" for all tests because they work only if openpyxl (an *optional* dependency) is installed
def read_excel(filepath, sheet=0, nb_axes=None, index_col=None, fill_value=nan, na=nan,
               sort_rows=False, sort_columns=False, wide=True, engine=None, range=slice(None), **kwargs) -> Array:
    r"""
    Read excel file from sheet name and returns an Array with the contents.

    Parameters
    ----------
    filepath : str or Path
        Path where the Excel file has to be read or use -1 to refer to the currently active workbook.
    sheet : str, Group or int, optional
        Name or index of the Excel sheet containing the array to be read.
        By default the array is read from the first sheet.
    nb_axes : int, optional
        Number of axes of output array. The first ``nb_axes`` - 1 columns and the header of the Excel sheet will be used
        to set the axes of the output array. If not specified, the number of axes is given by the position of the
        first column header including a ``\`` character plus one. If no column header includes a ``\`` character, the
        array is assumed to have one axis. Defaults to None.
    index_col : list, optional
        Positions of columns for the n-1 first axes (ex. [0, 1, 2, 3]). Defaults to None (see nb_axes above).
    fill_value : scalar or Array, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    wide : bool, optional
        Whether to assume the array is stored in "wide" format.
        If False, the array is assumed to be stored in "narrow" format: one column per axis plus one value column.
        Defaults to True.
    engine : {'xlwings', 'openpyxl', 'xlrd'}, optional
        Engine to use to read the Excel file.
        The 'xlrd' engine must be used to read Excel files with the old '.xls' extension.
        Either 'xlwings' or 'openpyxl' can be used to read Excel files with the standard '.xlsx' extension.
        Defaults to 'xlwings' if the module is installed, 'openpyxl' otherwise.
    range : str, optional
        Range to load the array from (only supported for the 'xlwings' engine). Defaults to slice(None) which loads
        the whole sheet, ignoring blank cells in the bottom right corner.
    **kwargs

    Returns
    -------
    Array

    Examples
    --------
    >>> fname = get_example_filepath('examples.xlsx')

    Read array from first sheet

    >>> # The data below is derived from a subset of the demo_pjan table from Eurostat
    >>> read_excel(fname)                                                               # doctest: +SKIP
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male  31772665  32045129  32174258
     France       Female  33827685  34120851  34283895
    Germany         Male  39380976  39556923  39835457
    Germany       Female  41142770  41210540  41362080

    Read array from a specific sheet

    >>> # The data below is derived from a subset of the demo_fasec table from Eurostat
    >>> read_excel(fname, 'births')                                                     # doctest: +SKIP
    country  gender\time    2013    2014    2015
    Belgium         Male   64371   64173   62561
    Belgium       Female   61235   60841   59713
     France         Male  415762  418721  409145
     France       Female  396581  400607  390526
    Germany         Male  349820  366835  378478
    Germany       Female  332249  348092  359097

    Missing label combinations

    Let us take a look inside the sheet 'population_missing_values'. Note the missing label combinations:
    (Paris, male) and (New York, female): ::

        country  gender\time      2013      2014      2015
        Belgium         Male   5472856   5493792   5524068
        Belgium       Female   5665118   5687048   5713206
         France       Female  33827685  34120851  34283895
        Germany         Male  39380976  39556923  39835457

    By default, cells associated with missing label combinations are filled with NaN. In that case, an int array
    is converted to a float array.

    >>> read_excel(fname, sheet='population_missing_values')                            # doctest: +SKIP
    country  gender\time        2013        2014        2015
    Belgium         Male   5472856.0   5493792.0   5524068.0
    Belgium       Female   5665118.0   5687048.0   5713206.0
     France         Male         nan         nan         nan
     France       Female  33827685.0  34120851.0  34283895.0
    Germany         Male  39380976.0  39556923.0  39835457.0
    Germany       Female         nan         nan         nan

    Using the ``fill_value`` argument, you can choose another value to use to fill missing cells.

    >>> read_excel(fname, sheet='population_missing_values', fill_value=0)              # doctest: +SKIP
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male         0         0         0
     France       Female  33827685  34120851  34283895
    Germany         Male  39380976  39556923  39835457
    Germany       Female         0         0         0

    Specify the number of axes of the output array (useful when the name of the last axis is implicit)

    The content of the sheet 'missing_axis_name' is: ::

        country  gender      2013      2014      2015
        Belgium    Male   5472856   5493792   5524068
        Belgium  Female   5665118   5687048   5713206
         France    Male  31772665  32045129  32174258
         France  Female  33827685  34120851  34283895
        Germany    Male  39380976  39556923  39835457
        Germany  Female  41142770  41210540  41362080

    >>> # read the array stored in the sheet 'population_missing_axis_name' as is
    >>> arr = read_excel(fname, sheet='population_missing_axis_name')                   # doctest: +SKIP
    >>> # we expected a 3 x 2 x 3 array with data of type int
    >>> # but we got a 6 x 4 array with data of type object
    >>> arr.info                                                                        # doctest: +SKIP
    6 x 4
     country [6]: 'Belgium' 'Belgium' 'France' 'France' 'Germany' 'Germany'
     {1} [4]: 'gender' '2013' '2014' '2015'
    dtype: object
    memory used: 192 bytes
    >>> # using argument 'nb_axes', you can force the number of axes of the output array
    >>> arr = read_excel(fname, sheet='population_missing_axis_name', nb_axes=3)        # doctest: +SKIP
    >>> # as expected, we have a 3 x 2 x 3 array with data of type int
    >>> arr.info                                                                        # doctest: +SKIP
    3 x 2 x 3
     country [3]: 'Belgium' 'France' 'Germany'
     gender [2]: 'Male' 'Female'
     {2} [3]: 2013 2014 2015
    dtype: int64
    memory used: 144 bytes

    Read array saved in "narrow" format (wide=False)

    Let us take a look inside the sheet 'population_narrow' where the data is stored in a 'narrow' format: ::

        country  time     value
        Belgium  2013  11137974
        Belgium  2014  11180840
        Belgium  2015  11237274
         France  2013  65600350
         France  2014  66165980
         France  2015  66458153

    >>> # to read arrays stored in 'narrow' format, you must pass wide=False to read_excel
    >>> read_excel(fname, 'population_narrow_format', wide=False)                       # doctest: +SKIP
    country\time      2013      2014      2015
         Belgium  11137974  11180840  11237274
          France  65600350  66165980  66458153

    Extract array from a given range (xlwings only)

    >>> read_excel(fname, 'population_births_deaths', range='A9:E15')                   # doctest: +SKIP
    country  gender\time    2013    2014    2015
    Belgium         Male   64371   64173   62561
    Belgium       Female   61235   60841   59713
     France         Male  415762  418721  409145
     France       Female  396581  400607  390526
    Germany         Male  349820  366835  378478
    Germany       Female  332249  348092  359097
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_excel `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    sheet = _translate_sheet_name(sheet)

    if engine is None:
        engine = 'xlwings' if xw is not None else 'openpyxl'

    index_col = _get_index_col(nb_axes, index_col, wide)

    if engine == 'xlwings':
        if kwargs:
            first_kwarg = list(kwargs.keys())[0]
            raise TypeError(f"'{first_kwarg}' is an invalid keyword argument for this function when using the xlwings "
                            f"backend")
        from larray.inout.xw_excel import open_excel
        with open_excel(filepath) as wb:
            return wb[sheet][range].load(index_col=index_col, fill_value=fill_value, sort_rows=sort_rows,
                                         sort_columns=sort_columns, wide=wide)
    else:
        # TODO: add support for range argument (using usecols, skiprows and nrows arguments of pandas.read_excel)
        df = pd.read_excel(filepath, sheet, index_col=index_col, engine=engine, **kwargs)
        return df_asarray(df, sort_rows=sort_rows, sort_columns=sort_columns, raw=index_col is None,
                          fill_value=fill_value, wide=wide)


@register_file_handler('pandas_excel', ['xls', 'xlsx'] if xw is None else None)
class PandasExcelHandler(FileHandler):
    r"""
    Handler for Excel files using Pandas.
    """

    def _open_for_read(self):
        self.handle = pd.ExcelFile(self.fname)

    def _open_for_write(self):
        engine = 'xlsxwriter' if (self.fname.suffix == '.xlsx' and xlsxwriter is not None) else None
        self.handle = pd.ExcelWriter(self.fname, engine=engine)

    def item_types(self) -> Dict[str, str]:
        return {name: 'Array' for name in self.handle.sheet_names if name != '__metadata__'}

    def _read_item(self, key, type, *args, **kwargs) -> Array:
        if type == 'Array':
            df = self.handle.parse(key, *args, **kwargs)
            return df_asarray(df, raw=True)
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        kwargs['engine'] = self.handle.engine
        if isinstance(value, Array):
            value.to_excel(self.handle, key, *args, **kwargs)
        else:
            raise TypeError()

    def _read_metadata(self) -> Metadata:
        sheet_meta = '__metadata__'
        if sheet_meta in self.handle.sheet_names:
            meta = read_excel(self.handle, sheet_meta, engine=self.handle.engine, wide=False)
            return Metadata.from_array(meta)
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        if len(metadata) > 0:
            metadata = asarray(metadata)
            metadata.to_excel(self.handle, '__metadata__', engine=self.handle.engine, wide=False, value_name='')

    def save(self):
        pass

    def close(self):
        self.handle.close()


@register_file_handler('xlwings_excel', ['xls', 'xlsx'] if xw is not None else None)
class XLWingsHandler(FileHandler):
    r"""
    Handler for Excel files using XLWings.
    """

    def _get_original_file_name(self):
        # for XLWingsHandler, no need to create a temporary file, the job is already done in the Workbook class
        pass

    def _open_for_read(self):
        self.handle = open_excel(self.fname)

    def _open_for_write(self):
        self.handle = open_excel(self.fname, overwrite_file=self.overwrite_file)

    def item_types(self) -> Dict[str, str]:
        return {name: 'Array' for name in self.handle.sheet_names() if name != '__metadata__'}

    def _read_item(self, key, type, *args, **kwargs) -> Array:
        if type == 'Array':
            return self.handle[key].load(*args, **kwargs)
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, Array):
            self.handle[key] = value.dump(*args, **kwargs)
        else:
            raise TypeError()

    def _read_metadata(self) -> Metadata:
        sheet_meta = '__metadata__'
        if sheet_meta in self.handle:
            meta = self.handle[sheet_meta].load(wide=False)
            return Metadata.from_array(meta)
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        if len(metadata) > 0:
            metadata = asarray(metadata)
            self.handle['__metadata__'] = metadata.dump(wide=False, value_name='')

    def save(self):
        self.handle.save()

    def close(self):
        self.handle.close()
