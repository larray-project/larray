.. py:currentmodule:: larray


New features
^^^^^^^^^^^^

* added support for Python 3.11.

* added support for stacking all arrays of a Session by simply doing: `stack(my_session)` instead of
  `stack(my_session.items())` (closes :issue:`1057`).


Fixes
^^^^^

* avoid warnings with recent versions of Pandas or Numpy (closes :issue:`1061`).
