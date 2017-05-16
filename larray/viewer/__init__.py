from __future__ import absolute_import, division, print_function

try:
    from larray.viewer.api import *
except ImportError:
    def view(*args, **kwargs):
        raise Exception('view() is not available because Qt is not installed')

    def edit(*args, **kwargs):
        raise Exception('edit() is not available because Qt is not installed')

    def compare(*args, **kwargs):
        raise Exception('compare() is not available because Qt is not installed')