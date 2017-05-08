from __future__ import absolute_import, division, print_function

from unittest import TestCase

import pytest
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray import open_excel


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
class TestExcel(TestCase):
    def test_setitem(self):
        with open_excel(visible=False) as wb:
            wb['sheet_name'] = 'sheet content'
            assert wb.sheet_names() == ['sheet_name']

    def test_delitem(self):
        with open_excel(visible=False) as wb:
            wb['sheet1'] = 'sheet1 content'
            wb['sheet2'] = 'sheet2 content'
            del wb['sheet1']
            assert wb.sheet_names() == ['sheet2']

    def test_rename(self):
        with open_excel(visible=False) as wb:
            wb['sheet_name'] = 'sheet content'
            wb['sheet_name'].name = 'renamed'
            assert wb.sheet_names() == ['renamed']

