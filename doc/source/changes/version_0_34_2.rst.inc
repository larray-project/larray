.. py:currentmodule:: larray


New features
^^^^^^^^^^^^

* added support for evaluating expressions using X.axis_name when calling some
  built-in functions, most notably `where()`. For example, the following code
  now works (previously it seemed to work but produced the wrong result -- see the
  fixes section below): ::

    >>> arr = ndtest("age=0..3")
    >>> arr
    age  0  1  2  3
         0  1  2  3
    >>> where(X.age == 2, 42, arr)
    age  0  1   2  3
         0  1  42  3


Fixes
^^^^^

* fixed Array.reindex when using an axis object from the array as `axes_to_reindex` (closes :issue:`1088`).

* fixed Array.reindex({axis: list_of_labels}) (closes :issue:`1068`).

* Array.split_axes now raises an explicit error when some labels contain
  more separators than others, instead of silently dropping part of those
  labels, or even some data (closes :issue:`1089`).

* a boolean condition including only `X.axis_name` and scalars (e.g. X.age == 0)
  raises an error when Python needs to know whether it is True or not (because
  there is no array to extract the axis labels from) instead of always
  evaluating to True. This was especially dangerous in the context of a
  where() function, which always evaluated to its left side
  (e.g. `where(X.age > 0, arr, 0)` evaluated to `arr` for all ages).
  Closes :issue:`1083`.

* expressions using `X.axis_name` and an Array now evaluate correctly when
  the Array is not involved in the first operation. For example, this already
  worked: ::

    >>> arr = ndtest("age=0..3")
    >>> arr
    age  0  1  2  3
         0  1  2  3
    >>> arr * (X.age != 2)
    age  0  1  2  3
         0  1  0  3

  but this did not: ::

    >>> (X.age != 2) * arr

* fixed plots with fewer than 6 integer labels in the x axis. In that case,
  it interpolated the values, which usually looks wrong for integer labels (e.g. year).
  Closes :issue:`1076`.
