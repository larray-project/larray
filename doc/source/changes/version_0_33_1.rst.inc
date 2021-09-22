.. py:currentmodule:: larray


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* added type hints for many Array methods (see :issue:`864`) which improves autocompletion in editors (such as PyCharm).


Fixes
^^^^^

* fixed CheckedSession with pydantic version >1.5 (closes :issue:`958`).

* removed the constraint on pydantic version in larrayenv, making it actually installable.

* fixed using labels for x and y in :py:obj:`Array.plot()` and :py:obj:`Array.plot.scatter()` functions, as well as
  :py:obj:`Array.plot.pie()` (closes :issue:`969`).

* fixed wrong "source code line" in "field is not declared" warning in CheckedSession.__init__()
  (closes :issue:`968`).

* fixed :py:obj:`Array.growth_rate()` returning nans instead of zeros when consecutive values are zeros
  (closes :issue:`903`).