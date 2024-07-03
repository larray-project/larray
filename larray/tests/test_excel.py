import re
import faulthandler
import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np

from larray.tests.common import needs_xlwings, needs_pytables, must_warn, must_raise
from larray import ndtest, open_excel, asarray, Array, Axis, nan, ExcelReport, read_excel
from larray.inout import xw_excel
from larray.example import load_example_data, EXAMPLE_EXCEL_TEMPLATES_DIR


TEST_DATA_PATH = Path(__file__).parent / 'data'


@needs_xlwings
class TestWorkbook:
    def test_open_excel(self):
        import pywintypes

        # not using context manager because we call .quit manually
        wb1 = open_excel(visible=False)
        app1 = wb1.app
        # close workbook but leave app instance open (anything using wb1 will fail now)
        wb1.close()

        # disable faulthandler to avoid annoying "Windows fatal exception" messages in the console
        # See https://github.com/pywinauto/pywinauto/issues/858
        # and https://stackoverflow.com/questions/57523762/pytest-windows-fatal-exception-code-0x8001010d
        faulthandler_enabled = faulthandler.is_enabled()
        if faulthandler_enabled:
            faulthandler.disable()
        disconnected_msg = r"The object invoked has disconnected from its clients."
        with must_raise(pywintypes.com_error, match=rf"\(.*, '{disconnected_msg}', .*\)"):
            wb1.sheet_names()
        if faulthandler_enabled:
            faulthandler.enable()

        wb2 = open_excel(visible=False)
        app2 = wb2.app
        assert app1 == app2 == xw_excel.global_app
        # this effectively close all workbooks but leaves the instance intact (this is probably due to us keeping a
        # reference to it).
        app1.quit()
        # anything using wb2 will fail
        if faulthandler_enabled:
            faulthandler.disable()
        # from xlwings 0.30.2 onward, the message is a "call failed" error, while it was
        # "disconnected" error in previous versions
        call_failed_msg = "The remote procedure call failed."
        error_pattern = fr"\(.*, '({disconnected_msg})|({call_failed_msg})', .*\)"
        with must_raise(pywintypes.com_error, match=error_pattern):
            wb2.sheet_names()
        if faulthandler_enabled:
            faulthandler.enable()

        # in any case, this should work
        with open_excel(visible=False) as wb:
            wb['test'] = 'content'

    def test_links(self):
        test_data_dir = TEST_DATA_PATH / 'excel_with_links'
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for book in ('BookA.xlsx', 'BookB.xlsx', 'BookC.xlsx'):
                shutil.copy2(test_data_dir / book, tmp_dir)
            fpath1 = tmp_dir / 'BookA.xlsx'
            fpath2 = tmp_dir / 'BookB.xlsx'
            fpath3 = tmp_dir / 'BookC.xlsx'
            assert read_excel(fpath1)['link to cell with link to other workbook', 'value from link'] == 4
            assert read_excel(fpath2)['cell with link to other workbook', 'formula value'] == 4
            assert read_excel(fpath3).i[0, 0] == 3

            with open_excel(fpath1) as a, open_excel(fpath2) as b, open_excel(fpath3) as c:
                c[0]['B2'] = 41
                a.save()
                b.save()
                c.save()
            assert read_excel(fpath1)['link to cell with link to other workbook', 'value from link'] == 42
            assert read_excel(fpath2)['cell with link to other workbook', 'formula value'] == 42
            assert read_excel(fpath3).i[0, 0] == 41
            with open_excel(fpath1) as a, open_excel(fpath2) as b, open_excel(fpath3) as c:
                c[0]['B2'] = 3
                a.save()
                b.save()
                c.save()
            assert read_excel(fpath1)['link to cell with link to other workbook', 'value from link'] == 4
            assert read_excel(fpath2)['cell with link to other workbook', 'formula value'] == 4
            assert read_excel(fpath3).i[0, 0] == 3

    def test_repr(self):
        with open_excel(visible=False) as wb:
            assert re.match(r'<larray.inout.xw_excel.Workbook \[Book\d+]>', repr(wb))

    def test_getitem(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]
            assert isinstance(sheet, xw_excel.Sheet)
            # this might not be true on non-English locale
            assert sheet.name == 'Sheet1'

            # this might not work on non-English locale
            sheet = wb['Sheet1']
            assert isinstance(sheet, xw_excel.Sheet)
            assert sheet.name == 'Sheet1'

            with must_raise(KeyError, msg="'Workbook has no sheet named this_sheet_does_not_exist'"):
                _ = wb['this_sheet_does_not_exist']

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

            # reset sheet 3 content (for next test)
            wb['sheet3'] = 'sheet3 content'
            assert wb['sheet3']['A1'].value == 'sheet3 content'

            # sheet did exist, Sheet value (int key)
            wb[1] = wb['sheet3']
            assert wb.sheet_names() == ['sheet1', 'sheet3 (2)', 'sheet3']
            assert wb[1]['A1'].value == 'sheet3 content'

            with open_excel(visible=False, app="new") as wb2:
                assert wb.app != wb2.app
                with must_raise(ValueError, msg="cannot copy a sheet from one instance of Excel to another"):
                    wb2['sheet1'] = wb['sheet1']

            # group key
            arr = ndtest((3, 3))
            for label in arr.b:
                wb[label] = arr[label].dump()
                assert arr[label].equals(wb[label].load())

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


@needs_xlwings
class TestSheet:
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

            # object array with a *numpy* NaN (dc2019 infamous 65535 bug)
            obj_arr = ndtest((2, 3)).astype(object)
            obj_arr['a0', 'b1'] = np.float64('nan')

            assert type(obj_arr['a0', 'b1']) is np.float64

            obj_arr_dump = obj_arr.dump()
            # [['a\\b', 'b0', 'b1', 'b2'], ['a0', 0, nan, 2], ['a1', 3, 4, 5]]

            # float and *not* np.float64, otherwise it gets converted to 65535 when written to Excel
            assert type(obj_arr_dump[1][2]) is float

            sheet['A12'] = obj_arr_dump

            # read them back
            assert sheet['A1'].value == 1.5
            assert sheet['A2'].value == 2
            assert sheet['A3'].value is True
            assert sheet['A4'].value == 'toto'
            # array without header
            assert np.array_equal(sheet['A5:C6'].value, arr.data)
            # array with header
            assert sheet['A8:D10'].load().equals(arr)
            assert sheet['A12:D14'].load().equals(obj_arr, nans_equal=True)

    def test_asarray(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            # empty sheet
            res = asarray(sheet)
            assert res.equals(Array(None))

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
            assert arr1.equals(res1)

            # array with int labels
            arr2 = ndtest('0..1;0..2')
            sheet['A1'] = arr2.dump()
            res2 = sheet.array('B2:D3', 'A2:A3', 'B1:D1')
            # larray_equal passes even if the labels are floats...
            assert arr2.equals(res2)
            # so we check the dtype explicitly
            assert res2.axes[0].labels.dtype == arr2.axes[0].labels.dtype
            assert res2.axes[1].labels.dtype == arr2.axes[1].labels.dtype

    def test_repr(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]
            assert re.match(r'<larray.inout.xw_excel.Sheet \[Book\d+]Sheet1>', repr(sheet))


@needs_xlwings
class TestRange:
    def test_get_and_set_item(self):
        arr = ndtest((4, 5))

        with open_excel(visible=False) as wb:
            sheet = wb[0]
            sheet['A1'] = arr
            rng = sheet['B1:D3']
            sliced_rng = rng[1:3, :2]
            assert sliced_rng.shape == (2, 2)
            asarray(sliced_rng).equals(Array([[6, 7], [11, 12]]))

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
            with must_raise(TypeError, msg="only integer scalars can be converted to a scalar index"):
                rng.__index__()

    def test_asarray(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndtest((2, 3))
            # dump only the data (no labels/headers) so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res1 = np.asarray(sheet['A1:C2'])
            assert np.array_equal(res1, arr1.data)
            assert res1.dtype == arr1.dtype

            arr1 = ndtest([Axis(2), Axis(3)])
            # no header so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res1 = asarray(sheet['A1:C2'])
            assert res1.equals(arr1)
            assert res1.dtype == arr1.dtype

    # this tests Range.__getattr__ with an Array attribute
    def test_aggregate(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndtest((2, 3))
            # no header so that we have an uniform dtype for the whole sheet
            sheet['A1'] = arr1
            res = sheet['A1:C2'].sum()
            assert res == 15

    def test_repr(self):
        with open_excel(visible=False) as wb:
            sheet = wb[0]

            arr1 = ndtest((2, 3))
            sheet['A1'] = arr1
            res = repr(sheet['A1:C2'])
            assert res == """\
{0}*\\{1}*  0  1  2
        0  0  1  2
        1  3  4  5"""


# ================ #
# Test ExcelReport #
# ================ #


@needs_xlwings
def test_excel_report_init():
    # No argument
    ExcelReport()
    # with template dir
    ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)
    # with graphs_per_row
    ExcelReport(graphs_per_row=2)


@needs_xlwings
def test_excel_report_setting_template():
    excel_report = ExcelReport()

    # test setting template dir
    # 1) wrong template dir
    wrong_template_dir = r"C:\Wrong\Directory\Path"
    with must_raise(ValueError, msg=f"The directory {wrong_template_dir} could not be found."):
        excel_report.template_dir = wrong_template_dir
    # 2) correct path
    excel_report.template_dir = EXAMPLE_EXCEL_TEMPLATES_DIR
    assert excel_report.template_dir == EXAMPLE_EXCEL_TEMPLATES_DIR

    # test setting template file
    # 1) wrong extension
    template_file = 'wrong_extension.txt'
    msg = "Extension for the excel template file must be '.crtx' instead of .txt"
    with must_raise(ValueError, msg=msg):
        excel_report.template = template_file
    # 2) add .crtx extension if no extension
    template_name = 'Line'
    excel_report.template = template_name
    assert excel_report.template == EXAMPLE_EXCEL_TEMPLATES_DIR / (template_name + '.crtx')


@needs_xlwings
def test_excel_report_sheets():
    report = ExcelReport()
    # test adding new sheets
    report.new_sheet('Population')
    report.new_sheet('Births')
    report.new_sheet('Deaths')
    # test warning if sheet already exists
    with must_warn(UserWarning, msg="Sheet 'Population' already exists in the report and will be reset"):
        sheet_population2 = report.new_sheet('Population')  # noqa: F841
    # test sheet_names()
    assert report.sheet_names() == ['Population', 'Births', 'Deaths']


@needs_xlwings
def test_excel_report_titles():
    excel_report = ExcelReport()

    # test dumping titles
    sheet_titles = excel_report.new_sheet('Titles')
    # 1) default
    sheet_titles.add_title('Default title')
    # 2) specify width and height
    width, height = 1100, 100
    sheet_titles.add_title(f'Width = {width} and Height = {height}',
                           width=width, height=height)
    # 3) specify fontsize
    fontsize = 13
    sheet_titles.add_title(f'Fontsize = {fontsize}', fontsize=fontsize)

    # generate Excel file
    fpath = 'test_excel_report_titles.xlsx'
    excel_report.to_excel(fpath)


@needs_xlwings
@needs_pytables
def test_excel_report_arrays():
    excel_report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)
    demo = load_example_data('demography_eurostat')
    population = demo.population
    population_be = population['Belgium']
    population_be_nan = population_be.astype(float)
    population_be_nan[2013] = nan

    # fake simple user defined function to test if any user defined function (and their arguments)
    # is actually called and correctly handled.
    def customize_func(obj_chart, alternative_title):
        obj_chart.HasTitle = True
        obj_chart.ChartTitle.Caption = alternative_title

    # test dumping arrays
    sheet_graphs = excel_report.new_sheet('Graphs')
    # 1) default
    sheet_graphs.add_title('No template')
    sheet_graphs.add_graph(population_be['Female'], 'Pop Belgium Female')
    sheet_graphs.add_graph(population_be, 'Pop Belgium')
    sheet_graphs.add_graph(population_be_nan, 'Pop Belgium with nans')
    # 2) no title
    sheet_graphs.add_title('No title graph')
    sheet_graphs.add_graph(population_be)
    # 3) specify width and height
    sheet_graphs.add_title('Alternative Width and Height')
    width, height = 500, 300
    sheet_graphs.add_graph(population_be, f'width = {width} and Height = {height}',
                           width=width, height=height)
    # 4) specify template
    template_name = 'Line_Marker'
    sheet_graphs.add_title(f'Template = {template_name}')
    sheet_graphs.add_graph(population_be, f'Template = {template_name}', template_name)
    # 5) specify min_y and max_y
    sheet_graphs.add_graph(population_be, 'min (max) Y axis is 5e6 (6e6)', min_y=5e6, max_y=6e6)
    # 6) specify the space between 2 ticks
    sheet_graphs.add_graph(population_be, 'periodicity = every 2 years', xticks_spacing=2)
    # 7) pass a user defined function
    sheet_graphs.add_graph(population_be, customize_func=customize_func,
                           customize_kwargs={'alternative_title': 'alternative title'})

    # test setting default size
    # 1) pass a not registered kind of item
    item_type = 'unknown_item'
    msg = f"Item type {item_type} is not registered. Please choose in list ['graph', 'title']"
    with must_raise(ValueError, msg=msg):
        sheet_graphs.set_item_default_size(item_type, width, height)
    # 2) update default size for graphs
    sheet_graphs.set_item_default_size('graph', width, height)
    sheet_graphs.add_title('Using Defined Sizes For All Graphs')
    sheet_graphs.add_graph(population_be, 'Pop Belgium')

    # test setting default number of graphs per row
    sheet_graphs = excel_report.new_sheet('Graphs2')
    sheet_graphs.graphs_per_row = 2
    sheet_graphs.add_title('graphs_per_row = 2')
    for combined_labels, subset in population.items(('country', 'gender')):
        title = ' - '.join([str(label) for label in combined_labels])
        sheet_graphs.add_graph(subset, title)

    # testing add_graphs
    sheet_graphs = excel_report.new_sheet('Graphs3')
    # 1) default
    sheet_graphs.add_title('add_graphs')
    sheet_graphs.add_graphs({'Population for {country} - {gender}': population},
                            {'gender': population.gender, 'country': population.country},
                            template='line', width=350, height=250, graphs_per_row=3)
    # 2) specify min_y and max_y
    sheet_graphs.add_title('min_y = 0 and max_y = 50e6')
    sheet_graphs.add_graphs({'Population for {country} - {gender}': population},
                            {'gender': population.gender, 'country': population.country},
                            template='line', width=350, height=250, graphs_per_row=3,
                            min_y=0, max_y=50e6)
    # 3) specify the space between 2 ticks
    sheet_graphs.add_title('periodicity = every 2 years')
    sheet_graphs.add_graphs({'Population for {country} - {gender}': population},
                            {'gender': population.gender, 'country': population.country},
                            template='line', width=350, height=250, graphs_per_row=3,
                            xticks_spacing=2)
    # 4) pass a user defined function
    sheet_graphs.add_title('user defined function')
    sheet_graphs.add_graphs({'Population for {country} - {gender}': population},
                            {'gender': population.gender, 'country': population.country},
                            template='line', width=350, height=250, graphs_per_row=3,
                            customize_func=customize_func,
                            customize_kwargs={'alternative_title': 'alternative title'})

    # generate Excel file
    fpath = 'test_excel_report_arrays.xlsx'
    excel_report.to_excel(fpath)


if __name__ == "__main__":
    pytest.main()
