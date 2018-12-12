from __future__ import absolute_import, division, print_function

from larray.util.misc import basestring


__all__ = ['set_printoptions']


DISPLAY_PRECISION = 'precision'
DISPLAY_WIDTH = 'display_width'
MAXLINES = 'maxlines'
EDGEITEMS = 'edgeitems'


OPTIONS = {
    DISPLAY_PRECISION: None,
    DISPLAY_WIDTH: 80,
    MAXLINES: None,
    EDGEITEMS: 5,
}


def _positive_integer(value):
    if not (isinstance(value, int) and value > 0):
        raise ValueError("Expected positive integer")


def _positive_integer_or_none(value):
    if value is None:
        return
    else:
        _positive_integer(value)


_VALIDATORS = {
    DISPLAY_PRECISION: _positive_integer_or_none,
    DISPLAY_WIDTH: _positive_integer,
    MAXLINES: _positive_integer_or_none,
    EDGEITEMS: _positive_integer,
}


# idea taken from xarray. See xarray/core/options.py module for original implementation.
class set_printoptions(object):
    r"""Set options for printing arrays in a controlled context.

    Currently supported options:

    - ``precision``: number of digits of precision for floating point output.
    - ``display_width``: maximum display width for ``repr`` on larray objects. Defaults to 80.
    - ``maxlines``: Maximum number of lines to show. Default behavior shows all lines.
    - ``edgeitems`` : if number of lines to display is greater than ``maxlines``, only the first and last
      ``edgeitems`` lines are displayed. Only active if ``maxlines`` is not None. Defaults to 5.

    Examples
    --------
    >>> from larray import *
    >>> arr = ndtest((500, 100), dtype=float) / 3

    You can use ``set_options`` either as a context manager:

    >>> with set_printoptions(display_width=60, edgeitems=2, precision=2):
    ...     print(arr)
     a\b        b0        b1  ...       b98       b99
      a0      0.00      0.33  ...     32.67     33.00
      a1     33.33     33.67  ...     66.00     66.33
     ...       ...       ...  ...       ...       ...
    a498  16600.00  16600.33  ...  16632.67  16633.00
    a499  16633.33  16633.67  ...  16666.00  16666.33

    Or to set global options:

    >>> set_printoptions(display_width=40, maxlines=10, precision=2) # doctest: +SKIP
    >>> print(arr) # doctest: +SKIP
     a\b        b0  ...       b99
      a0      0.00  ...     33.00
      a1     33.33  ...     66.33
      a2     66.67  ...     99.67
      a3    100.00  ...    133.00
      a4    133.33  ...    166.33
     ...       ...  ...       ...
    a495  16500.00  ...  16533.00
    a496  16533.33  ...  16566.33
    a497  16566.67  ...  16599.67
    a498  16600.00  ...  16633.00
    a499  16633.33  ...  16666.33
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError('Argument {} is not in the set of valid options {}'.format(k, set(OPTIONS)))
            if k in _VALIDATORS:
                _VALIDATORS[k](v)
            self.old[k] = OPTIONS[k]
        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.update(self.old)
