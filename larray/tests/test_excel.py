from __future__ import absolute_import, division, print_function

import pytest

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray import open_excel
from larray.io import excel


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
class TestExcel(object):
    def test_open_excel(self):
        # not using context manager because we call .quit manually
        wb1 = open_excel(visible=False)
        app1 = wb1.app
        wb1.close()
        # anything using wb1 will fail
        with pytest.raises(Exception):
            wb1.sheet_names()
        wb2 = open_excel(visible=False)
        app2 = wb2.app
        assert app1 == app2 == excel.global_app
        # this effectively close all workbooks but leaves the instance intact (this is probably due to us keeping a
        # reference to it).
        app1.quit()
        # anything using wb2 will fail
        with pytest.raises(Exception):
            wb2.sheet_names()

        # in any case, this should work
        with open_excel(visible=False) as wb:
            wb['test'] = 'content'

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

            with open_excel(visible=False, app="new") as wb2:
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


if __name__ == "__main__":
    pytest.main()