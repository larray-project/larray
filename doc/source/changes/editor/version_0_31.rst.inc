.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* implemented :py:obj:`run_editor_on_exception()` function. If you call this function in your code (for example at the
  top of your main script), Python will open an larray editor if any unexpected error happens anywhere in your script
  (closes :editor_issue:`180`).

Fixes
^^^^^

* fixed exception raised when loading a session with non array objects
  from a file (closes :editor_issue:`179`).

* fixed passing a simple array to the :py:obj:`view()` and :py:obj:`edit()` functions
  (closes :editor_issue:`183`).
