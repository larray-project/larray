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

* using a boolean array as a filter to take a subset of another array now raise an error when the
  two arrays have incompatible axes instead of producing wrong result (closes :issue:`1085`).

* fixed converting a scalar Array (an Array with 0 dimensions) to string with numpy 1.22+.

* fixed copying a sheet from one Excel workbook to another when the destination
  sheet is given by position (closes :issue:`1092`).

* fixed :py:obj:`Array.values()` and :py:obj:`Array.items()` on the first axis given by position.
  (e.g. `my_array.values(axes=0)`). Closes :issue:`1093`.
