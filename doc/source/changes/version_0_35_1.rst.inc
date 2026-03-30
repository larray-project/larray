.. py:currentmodule:: larray


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* added warning when setting an array subset causes an unsafe type conversion
  (which could lead to information loss). For example, when setting an integer
  array with a float value, the float value is truncated to an integer
  value, loosing the decimal part. The behavior did not change (yet) but
  a warning is now emitted (whether there is actual loss of information or
  not). Closes :issue:`1164`.

    >>> arr = ndtest(3)
    >>> arr
    a  a0  a1  a2
        0   1   2
    >>> arr.dtype
    dtype('int64')
    >>> arr['a1'] = 3.9
    Setting a subset of an array with int64 dtype with a value with float dtype.

    It will be converted to int64 like in previous larray versions but this is
    not a safe operation (some information could be lost in the conversion).

    If you want to keep doing this conversion and silence this warning, please
    convert the value explicitly using value.astype(<target_array_type>).

    This warning will become an error in a future version of larray.
    >>> arr
    a  a0  a1  a2
        0   3   2

* added a similar warning when doing unsafe type conversions when setting
  CheckedSession fields (closes :issue:`1168`).

    >>> a = Axis('a=a0,a1')
    >>> class MySession(CheckedSession):
    ...     my_field: CheckedArray(a, dtype=int)
    >>> good_value = ndtest(a)
    >>> good_value
    a  a0  a1
        0   1
    >>> s = MySession(my_field=good_value)
    >>> bad_value = good_value * 1.6
    >>> bad_value
    a   a0   a1
       0.0  1.6
    >>> s.my_field = bad_value
    Array 'my_field' was declared with dtype int64 but <...> (rest of warning
    omitted for brevity)


Fixes
^^^^^

* fixed a critical error: arithmetic operations using Group instances silently
  produced bad results in some cases when using Numpy 2.1 or later
  (closes :issue:`1173`).

* fixed most data reading functions returning readonly Arrays when using
  Pandas >= 3.0 (closes :issue:`1170`).

* fixed loading string data from H5 files written with Pandas >= 3.0 (whether
  pyarrow is installed or not). Closes :issue:`1180`.

* fixed loading data files with all the following conditions when using
  Pandas >= 3.0: mixed type columns, some combination of labels in index
  columns not being present in the data and a non-NaN fill_value
  (closes :issue:`1166`).

* fixed `sequence()` when both `inc` and `mult` are defined.
