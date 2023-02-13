from larray.tests.common import must_raise

import larray


def test_invalid_option_raises():
    msg = "Argument not_a_valid_option is not in the set of valid options: " \
          "display_edgeitems, display_maxlines, display_precision, " \
          "display_width"
    with must_raise(ValueError, msg=msg):
        larray.set_options(not_a_valid_option=True)


def test_set_options_as_global():
    original_ops = larray.get_options()
    arr = larray.ndtest((500, 100))
    larray.set_options(display_width=40, display_maxlines=10)
    expected = """\
 a\\b     b0     b1  ...    b98    b99
  a0      0      1  ...     98     99
  a1    100    101  ...    198    199
  a2    200    201  ...    298    299
  a3    300    301  ...    398    399
  a4    400    401  ...    498    499
 ...    ...    ...  ...    ...    ...
a495  49500  49501  ...  49598  49599
a496  49600  49601  ...  49698  49699
a497  49700  49701  ...  49798  49799
a498  49800  49801  ...  49898  49899
a499  49900  49901  ...  49998  49999"""
    assert str(arr) == expected
    larray.set_options(**original_ops)
