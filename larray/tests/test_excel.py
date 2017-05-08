from __future__ import absolute_import, division, print_function

import pytest
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray import open_excel


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
class TestExcel(object):
    def test_setitem(self):
        with open_excel(visible=False) as wb:
            # sheet did not exist, str value
            wb['sheet1'] = 'sheet1 content'
            wb['sheet2'] = 'sheet2 content'
            assert wb.sheet_names() == ['sheet1', 'sheet2']

            # sheet did exist, str value
            wb['sheet2'] = 'sheet2 content v2'
            assert wb.sheet_names() == ['sheet1', 'sheet2']
            assert wb['sheet2']['A1'].value == 'sheet2 content v2'

            # sheet did not exist, Sheet value
            wb['sheet3'] = wb['sheet1']
            assert wb.sheet_names() == ['sheet1', 'sheet2', 'sheet3']
            assert wb['sheet3']['A1'].value == 'sheet1 content'

            # sheet did exist, Sheet value
            wb['sheet2'] = wb['sheet1']
            assert wb.sheet_names() == ['sheet1', 'sheet2', 'sheet3']
            assert wb['sheet2']['A1'].value == 'sheet1 content'

            with open_excel(visible=False) as wb2:
                with pytest.raises(ValueError) as e_info:
                    wb2['sheet1'] = wb['sheet1']
                assert e_info.value.args[0] == "cannot copy a sheet from one instance of Excel to another"

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

