from __future__ import absolute_import, division, print_function

import re

import pytest
import numpy as np

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray import ndtest, ndrange, larray_equal, open_excel, aslarray
from larray.io import excel


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
class TestWorkbook(object):
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

    def test_repr(self):
        with open_excel(visible=False) as wb:
            assert re.match('<larray.io.excel.Workbook \[Book\d+\]>', repr(wb))

    def test_getitem(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]
            assert isinstance(sheet, excel.Sheet)
            # this might not be true on non-English locale
            assert sheet.name == 'Sheet1'

            # this might not work on non-English locale
            sheet = wb['Sheet1']
            assert isinstance(sheet, excel.Sheet)
            assert sheet.name == 'Sheet1'

            with pytest.raises(KeyError) as e_info:
                wb['this_sheet_does_not_exist']
            assert e_info.value.args[0] == "Workbook has no sheet named this_sheet_does_not_exist"

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

            # group key
            arr = ndtest((3, 3))
            for label in arr.b:
                wb[label] = arr[label].dump()
                assert larray_equal(wb[label].load(), arr[label])

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


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
class TestSheet(object):
    def test_get_and_set_item(self):
        arr = ndtest((2, 3))

        with open_excel(visible=False) as wb:
            sheet = wb[0]
            # set a few values
            sheet['A1'] = 1.5
            sheet['A2'] = 2
            sheet['A3'] = True
            sheet['A4'] = 'toto'
            # array without header
            sheet['A5'] = arr
            # array with header
            sheet['A8'] = arr.dump()

            # read them back
            assert sheet['A1'].value == 1.5
            assert sheet['A2'].value == 2
            assert sheet['A3'].value == True
            assert sheet['A4'].value == 'toto'
            # array without header
            assert np.array_equal(sheet['A5:C6'].value, arr.data)
            # array with header
            assert larray_equal(sheet['A8:D10'].load(), arr)

    def test_asarray(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndtest((2, 3))
            # no header so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res1 = np.asarray(sheet)
            assert np.array_equal(res1, arr1.data)
            assert res1.dtype == arr1.dtype

    def test_array_method(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            # normal test array
            arr1 = ndtest((2, 3))
            sheet['A1'] = arr1.dump()
            res1 = sheet.array('B2:D3', 'A2:A3', 'B1:D1', names=['a', 'b'])
            assert larray_equal(res1, arr1)

            # array with int labels
            arr2 = ndrange('0..1;0..2')
            sheet['A1'] = arr2.dump()
            res2 = sheet.array('B2:D3', 'A2:A3', 'B1:D1')
            # larray_equal passes even if the labels are floats...
            assert larray_equal(res2, arr2)
            # so we check the dtype explicitly
            assert res2.axes[0].labels.dtype == arr2.axes[0].labels.dtype
            assert res2.axes[1].labels.dtype == arr2.axes[1].labels.dtype

    def test_repr(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]
            assert re.match('<larray.io.excel.Sheet \[Book\d+\]Sheet1>', repr(sheet))


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
class TestRange(object):
    def test_scalar_convert(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]
            # set a few values
            sheet['A1'] = 1
            rng = sheet['A1']
            assert int(rng) == 1
            assert float(rng) == 1.0
            assert rng.__index__() == 1

            sheet['A2'] = 1.0
            rng = sheet['A2']
            assert int(rng) == 1
            assert float(rng) == 1.0
            # Excel stores everything as float so we cannot really make the difference between 1 and 1.0
            assert rng.__index__() == 1

            sheet['A3'] = 1.5
            rng = sheet['A3']
            assert int(rng) == 1
            assert float(rng) == 1.5
            with pytest.raises(TypeError) as e_info:
                rng.__index__()
            assert e_info.value.args[0] == "only integer scalars can be converted to a scalar index"

    def test_asarray(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndtest((2, 3))
            # no header so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res1 = np.asarray(sheet['A1:C2'])
            assert np.array_equal(res1, arr1.data)
            assert res1.dtype == arr1.dtype

    def test_aslarray(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndrange((2, 3))
            # no header so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res1 = aslarray(sheet['A1:C2'])
            assert larray_equal(res1, arr1)
            assert res1.dtype == arr1.dtype

    # this tests Range.__getattr__ with an LArray attribute
    def test_aggregate(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndrange((2, 3))
            # no header so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res = sheet['A1:C2'].sum()
            assert res == 15

    def test_repr(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndrange((2, 3))
            sheet['A1'] = arr1
            res = repr(sheet['A1:C2'])
            assert res == """\
{0}*\{1}*  0  1  2
        0  0  1  2
        1  3  4  5"""

if __name__ == "__main__":
    pytest.main()
