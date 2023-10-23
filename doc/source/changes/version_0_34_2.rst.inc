.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed ``Array.old_method_name()`` to :py:obj:`Array.new_method_name()` (closes :issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`Array.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* other backward incompatible changes


New features
^^^^^^^^^^^^

* added a feature (see the :ref:`miscellaneous section <misc>` for details). It works on :ref:`api-axis` and
  :ref:`api-group` objects.

  Here is an example of the new feature:

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5

  And it can also be used like this:

    >>> arr = ndtest("a=a0..a2")
    >>> arr
    a  a0  a1  a2
        0   1   2

* added another feature in the editor (closes :editor_issue:`1`).

  .. note::

     - It works for foo bar !
     - It does not work for foo baz !


.. _misc:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved something.


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
  (e.g. where(X.age > 0, arr, 0) evaluated to 'arr' for all ages).
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
