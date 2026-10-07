.. py:currentmodule:: larray


Fixes
^^^^^

* fixed indexing an array which has an axis with the object dtype using an
  array key with more than one dimension (closes :issue:`1203`).
  This was a regression in larray 0.35.1.
