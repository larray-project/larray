.. py:currentmodule:: larray_editor

Fixes
^^^^^

* fixed the viewer being unusable after showing a matplotlib plot (closes :editor_issue:`261`).

* silence spurious debugger warning on Python 3.11 (closes :editor_issue:`263`).

* when code in the interactive console creates *and shows* a plot window, avoid showing it
  a second time (closes :editor_issue:`265`).

* depending on the system regional settings, comparator tolerance sometimes did not allow simple
  fractional numbers (e.g. 0.1). The only way to specify the tolerance was the scientific notation
  (closes :editor_issue:`260`).
