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

* added warning when doing unsafe implicit type conversions (for example from
  float to integers) when setting CheckedSession fields (closes :issue:`1168`).

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
    Array 'my_field' was declared with dtype int64 but got value with float64 dtype.

    It will be converted to int64 like in previous larray versions but this is
    not a safe operation (some information could be lost in the conversion).

    If you want to keep doing this conversion and silence this warning, please
    convert the value explicitly using value.astype(<declared_type>).

    This warning will become an error in a future version of larray.


Fixes
^^^^^

* fixed loading data files with all the following conditions when using
  Pandas >= 3.0: mixed type columns, some combination of labels in index
  columns not being present in the data and a non-NaN fill_value
  (closes :issue:`1166`).
