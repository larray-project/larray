.. py:currentmodule:: larray


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* the larrayenv metapackage now allows pandas 3.0 and numpy 2.4 and 2.5.


Fixes
^^^^^

* fixed importing larray when pydantic 1.x is installed. Only CheckedArray,
  CheckedSession and CheckedParameters require pydantic (>= 2). This was a
  regression in larray 0.35.0.

* fixed setting a subset of an array with an array with the same shape but
  only anonymous wildcard axes, such as one created from a numpy array
  (e.g. ``Array(ndarray)``) or by calling ``.astype()`` on an Excel range
  (closes :issue:`1202`).

* fixed binary operations (e.g. ``arr + other``) when the right operand has the
  same shape as the left one but only anonymous wildcard axes
  (see :issue:`1202`).

* fixed the warning emitted when setting a subset of an array with a value
  which can be converted to a numpy array (e.g. an Excel range) but is not one
  already. It mentioned the type of the value (e.g. "Range dtype") instead of
  the dtype of its data, even if converting it was safe (see :issue:`1202`).

* fixed indexing an array which has an axis with the object dtype using an
  array key with more than one dimension (closes :issue:`1203`).
  This was a regression in larray 0.35.1.
