from __future__ import absolute_import, division, print_function

try:
    from larray_editor import *
except ImportError:
    def view(*args, **kwargs):
        raise Exception('view() is not available because the larray_editor package is not installed')

    def edit(*args, **kwargs):
        raise Exception('edit() is not available because the larray_editor package is not installed')

    def compare(*args, **kwargs):
        raise Exception('compare() is not available because the larray_editor package is not installed')
