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
    def test_rename(self):
        with open_excel(visible=False) as wb:
            # create sheet
            wb['new'] = 'hello world'
            wb['new'].name = 'renamed'
            assert wb.sheet_names() == ['renamed']
