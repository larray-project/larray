.. py:currentmodule:: larray_editor

Fixes
^^^^^

* fixed the "Cancel" button of the confirmation dialog when trying to quit the editor with unsaved modifications.
  It was equivalent to discard, potentially leading to data loss.

* fixed (harmless) error messages appearing when trying to display any variable via the console when using
  matplotlib 3.1+
