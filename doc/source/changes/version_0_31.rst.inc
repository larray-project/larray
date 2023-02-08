.. py:currentmodule:: larray


New features
^^^^^^^^^^^^

* added the :py:obj:`ExcelReport` class allowing to generate multiple graphs in an
  Excel file at once (closes :issue:`676`).


Fixes
^^^^^

* fixed binary operations (+, -, \*, etc.) between an LArray and a (scalar) Group which silently gave a wrong
  result (closes :issue:`797`).

* fixed taking a subset of an array with boolean labels for an axis if the user explicitly specify the axis
  (closes :issue:`735`). When the user does not specify the axis, it currently fails but it is unclear what to do in
  that case (see :issue:`794`).

* fixed a regression in 0.30: X.axis_name[groups] failed when groups were originally defined on axes with the same name
  (i.e. when the operation was not actually needed). Closes :issue:`787`.
