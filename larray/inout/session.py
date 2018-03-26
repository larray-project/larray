from __future__ import absolute_import, division, print_function

from larray.inout.csv import PandasCSVHandler
from larray.inout.excel import PandasExcelHandler, XLWingsHandler
from larray.inout.hdf import PandasHDFHandler
from larray.inout.pickle import PickleHandler


def check_pattern(k, pattern):
    return k.startswith(pattern)


handler_classes = {
    'pickle': PickleHandler,
    'pandas_csv': PandasCSVHandler,
    'pandas_hdf': PandasHDFHandler,
    'pandas_excel': PandasExcelHandler,
    'xlwings_excel': XLWingsHandler,
}

ext_default_engine = {
    'csv': 'pandas_csv',
    'h5': 'pandas_hdf', 'hdf': 'pandas_hdf',
    'pkl': 'pickle', 'pickle': 'pickle',
    'xls': 'xlwings_excel', 'xlsx': 'xlwings_excel',
}
