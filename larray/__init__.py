from __future__ import absolute_import, division, print_function

from larray.core import *
from larray.session import *
from larray.ufuncs import *

try:
    import sys

    from PyQt4 import QtGui, QtCore

    from larray.view import view, edit

    orig_hook = sys.displayhook

    def qt_display_hook(value):
        if isinstance(value, LArray):
            view(value)
        else:
            orig_hook(value)

    sys.displayhook = qt_display_hook

    # cleanup namespace
    del QtGui, QtCore, sys
except ImportError:
    def view(array):
        raise Exception('view() is not available because Qt is not installed')

    def edit(array):
        raise Exception('edit() is not available because Qt is not installed')
