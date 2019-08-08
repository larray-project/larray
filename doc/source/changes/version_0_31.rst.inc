.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed ``LArray.old_method_name()`` to :py:obj:`LArray.new_method_name()` (closes :issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`LArray.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* other backward incompatible changes


New features
^^^^^^^^^^^^

* added the :py:obj:`ExcelReport` class allowing to generate multiple graphs in an
  Excel file at once (closes :issue:`676`).


.. _misc:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved the performance of a few LArray methods.


Fixes
^^^^^

* fixed taking a subset of an array with boolean labels for an axis if the user explicitly specify the axis
  (closes :issue:`735`). When the user does not specify the axis, it currently fails but it is unclear what to do in
  that case (see :issue:`794`).
