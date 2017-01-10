from __future__ import absolute_import, division, print_function

from larray.core import *
from larray.session import *
from larray.ufuncs import *
from larray.excel import open_excel

try:
    from larray.viewer import view, edit, compare
except ImportError:
    def view(*args, **kwargs):
        raise Exception('view() is not available because Qt is not installed')

    def edit(*args, **kwargs):
        raise Exception('edit() is not available because Qt is not installed')

    def compare(*args, **kwargs):
        raise Exception('compare() is not available because Qt is not installed')
