from __future__ import absolute_import, division, print_function

from larray.util.misc import basestring


__all__ = ['set_options']


DISPLAY_PRECISION = 'display_precision'
DISPLAY_WIDTH = 'display_width'
DISPLAY_MAXLINES = 'display_maxlines'
DISPLAY_EDGEITEMS = 'display_edgeitems'


OPTIONS = {
    DISPLAY_PRECISION: None,
    DISPLAY_WIDTH: 80,
    DISPLAY_MAXLINES: 200,
    DISPLAY_EDGEITEMS: 5,
}


def _positive_integer(value):
    if not (isinstance(value, int) and value > 0):
        raise ValueError("Expected positive integer")


def _non_negative_integer(value):
    if not (isinstance(value, int) and value >= 0):
        raise ValueError("Expected non-negative integer")


def _positive_integer_or_none(value):
    if value is None:
        return
    else:
        _positive_integer(value)


_VALIDATORS = {
    DISPLAY_PRECISION: _positive_integer_or_none,
    DISPLAY_WIDTH: _positive_integer,
    DISPLAY_MAXLINES: _non_negative_integer,
    DISPLAY_EDGEITEMS: _positive_integer,
}


# idea taken from xarray. See xarray/core/options.py module for original implementation.
class set_options(object):
    r"""Set options for larray in a controlled context.

    Currently supported options:

    - ``display_precision``: number of digits of precision for floating point output.
    - ``display_width``: maximum display width for ``repr`` on larray objects. Defaults to 80.
    - ``display_maxlines``: Maximum number of lines to show. If 0 all lines are shown.
      Defaults to 200.
    - ``display_edgeitems`` : if number of lines to display is greater than ``display_maxlines``,
      only the first and last ``display_edgeitems`` lines are displayed.
      Only active if ``display_maxlines`` is not None. Defaults to 5.

    Examples
    --------
    >>> from larray import *
    >>> arr = ndtest((500, 100), dtype=float) / 3

    You can use ``set_options`` either as a context manager:

    >>> with set_options(display_width=100, display_edgeitems=2):
    ...     print(arr)
     a\b                  b0                  b1  ...                 b98                 b99
      a0                 0.0  0.3333333333333333  ...  32.666666666666664                33.0
      a1  33.333333333333336  33.666666666666664  ...                66.0   66.33333333333333
     ...                 ...                 ...  ...                 ...                 ...
    a498             16600.0  16600.333333333332  ...  16632.666666666668             16633.0
    a499  16633.333333333332  16633.666666666668  ...             16666.0  16666.333333333332

    Or to set global options:

    >>> set_options(display_maxlines=10, display_precision=2) # doctest: +SKIP
    >>> print(arr) # doctest: +SKIP
     a\b        b0        b1        b2  ...       b97       b98       b99
      a0      0.00      0.33      0.67  ...     32.33     32.67     33.00
      a1     33.33     33.67     34.00  ...     65.67     66.00     66.33
      a2     66.67     67.00     67.33  ...     99.00     99.33     99.67
      a3    100.00    100.33    100.67  ...    132.33    132.67    133.00
      a4    133.33    133.67    134.00  ...    165.67    166.00    166.33
     ...       ...       ...       ...  ...       ...       ...       ...
    a495  16500.00  16500.33  16500.67  ...  16532.33  16532.67  16533.00
    a496  16533.33  16533.67  16534.00  ...  16565.67  16566.00  16566.33
    a497  16566.67  16567.00  16567.33  ...  16599.00  16599.33  16599.67
    a498  16600.00  16600.33  16600.67  ...  16632.33  16632.67  16633.00
    a499  16633.33  16633.67  16634.00  ...  16665.67  16666.00  16666.33
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
