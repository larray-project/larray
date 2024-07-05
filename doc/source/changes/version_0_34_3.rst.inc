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

* added support for Python 3.12 (closes :issue:`1109`).

.. _misc:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved something.


Fixes
^^^^^

* using a boolean array as a filter to take a subset of another array now raise an error when the
  two arrays have incompatible axes instead of producing wrong result (closes :issue:`1085`).

* fixed converting a scalar Array (an Array with 0 dimensions) to string with numpy 1.22+.

* fixed copying a sheet from one Excel workbook to another when the destination sheet
  is given by position (closes :issue:`1092`).

* fixed :py:obj:`Array.values()` and :py:obj:`Array.items()` on the first axis given by position.
  (e.g. `my_array.values(axes=0)`). Closes :issue:`1093`.

* fixed :py:obj:`Array.dump()` ``axes_names`` argument for 1D arrays (closes :issue:`1094`).

* avoid warnings and errors with recent versions of our dependencies (Numpy 2+, Pandas 2.2+ and
  xlwings 0.30.2+). Closes :issue:`1100`, :issue:`1107` and :issue:`1108`.
