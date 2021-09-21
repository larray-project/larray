.. py:currentmodule:: larray


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved something.


Fixes
^^^^^

* fixed CheckedSession with pydantic version >1.5 (closes :issue:`958`).

* removed the constraint on pydantic version in larrayenv, making it actually installable.

* fixed using labels for x and y in :py:obj:`Array.plot()` and :py:obj:`Array.plot.scatter()` functions, as well as
  :py:obj:`Array.plot.pie()` (closes :issue:`969`).

* fixed wrong "source code line" in "field is not declared" warning in CheckedSession.__init__()
  (closes :issue:`968`).