from __future__ import absolute_import, division, print_function

from larray.core import *
from larray.io import *
from larray.util import *
from larray.example import *
from larray.extra import *
from larray.viewer import *

__version__ = "0.24.1"
try:
    from larray.viewer import view, edit, compare, animate
except ImportError:
    def view(*args, **kwargs):
        raise Exception('view() is not available because Qt is not installed')

    def edit(*args, **kwargs):
        raise Exception('edit() is not available because Qt is not installed')

    def compare(*args, **kwargs):
        raise Exception('compare() is not available because Qt is not installed')

    def animate(*args, **kwargs):
        raise Exception('animate() is not available because Qt is not installed')
