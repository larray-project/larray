from larray.util.misc import _positive_integer

DISPLAY_PRECISION = 'display_precision'
DISPLAY_WIDTH = 'display_width'
DISPLAY_MAXLINES = 'display_maxlines'
DISPLAY_EDGEITEMS = 'display_edgeitems'


_OPTIONS = {
    DISPLAY_PRECISION: None,
    DISPLAY_WIDTH: 80,
    DISPLAY_MAXLINES: 200,
    DISPLAY_EDGEITEMS: 5,
}


def _integer_maxlines(value):
    if not isinstance(value, int) and value >= -1:
        raise ValueError("Expected integer")


def _positive_integer_or_none(value):
    if value is None:
        return
    else:
        _positive_integer(value)


_VALIDATORS = {
    DISPLAY_PRECISION: _positive_integer_or_none,
    DISPLAY_WIDTH: _positive_integer,
    DISPLAY_MAXLINES: _integer_maxlines,
    DISPLAY_EDGEITEMS: _positive_integer,
}


# idea taken from xarray. See xarray/core/options.py module for original implementation.
class set_options:
    r"""Set options for larray in a controlled context.

    Currently supported options:

    - ``display_precision``: number of digits of precision for floating point output.
      Print as many digits as necessary to uniquely specify the value by default (None).
    - ``display_width``: maximum display width for ``repr`` on larray objects. Defaults to 80.
    - ``display_maxlines``: Maximum number of lines to show. All lines are shown if -1.
      Defaults to 200.
    - ``display_edgeitems`` : if number of lines to display is greater than ``display_maxlines``,
      only the first and last ``display_edgeitems`` lines are displayed.
      Only active if ``display_maxlines`` is not -1. Defaults to 5.

    Examples
    --------
    >>> from larray import *
    >>> arr = ndtest((500, 100), dtype=float) + 0.123456

    You can use ``set_options`` either as a context manager:

    >>> with set_options(display_width=100, display_edgeitems=2):
    ...     print(arr)
     a\b            b0            b1            b2  ...           b97           b98           b99
      a0      0.123456      1.123456      2.123456  ...     97.123456     98.123456     99.123456
      a1    100.123456    101.123456    102.123456  ...    197.123456    198.123456    199.123456
     ...           ...           ...           ...  ...           ...           ...           ...
    a498  49800.123456  49801.123456  49802.123456  ...  49897.123456  49898.123456  49899.123456
    a499  49900.123456  49901.123456  49902.123456  ...  49997.123456  49998.123456  49999.123456

    Or to set global options:

    >>> set_options(display_maxlines=10, display_precision=2) # doctest: +SKIP
    >>> print(arr) # doctest: +SKIP
     a\b        b0        b1        b2  ...       b97       b98       b99
      a0      0.12      1.12      2.12  ...     97.12     98.12     99.12
      a1    100.12    101.12    102.12  ...    197.12    198.12    199.12
      a2    200.12    201.12    202.12  ...    297.12    298.12    299.12
      a3    300.12    301.12    302.12  ...    397.12    398.12    399.12
      a4    400.12    401.12    402.12  ...    497.12    498.12    499.12
     ...       ...       ...       ...  ...       ...       ...       ...
    a495  49500.12  49501.12  49502.12  ...  49597.12  49598.12  49599.12
    a496  49600.12  49601.12  49602.12  ...  49697.12  49698.12  49699.12
    a497  49700.12  49701.12  49702.12  ...  49797.12  49798.12  49799.12
    a498  49800.12  49801.12  49802.12  ...  49897.12  49898.12  49899.12
    a499  49900.12  49901.12  49902.12  ...  49997.12  49998.12  49999.12

    To put back the default options, you can use:

    >>> set_options(display_precision=None, display_width=80, display_maxlines=200, display_edgeitems=5)
    ... # doctest: +SKIP
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in _OPTIONS:
                options_str = ', '.join(sorted(_OPTIONS.keys()))
                raise ValueError(f'Argument {k} is not in the set of valid options: {options_str}')
            if k in _VALIDATORS:
                _VALIDATORS[k](v)
            self.old[k] = _OPTIONS[k]
        _OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        _OPTIONS.update(self.old)


def get_options():
    r"""
    Return the current options.

    Returns
    -------
    Dictionary of current print options with keys

        - display_precision: int or None
        - display_width: int
        - display_maxlines: int
        - display_edgeitems : int

    For a full description of these options, see :py:obj:`set_options`.

    See Also
    --------
    set_options

    Examples
    --------
    >>> get_options() # doctest: +SKIP
    {'display_precision': None, 'display_width': 80, 'display_maxlines': 200, 'display_edgeitems': 5}
    """
    return _OPTIONS.copy()
