.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* added support for displaying Polars Series (closes :editor_issue:`323`).


Fixes
^^^^^

* fixed displaying and filtering datetime columns in Pandas DataFrame
  (closes :editor_issue:`328`).

* remove no longer displayable variables from the variable list (for example,
  when a variable represents an open file and the file is closed).
  Closes :editor_issue:`327`).

* fixed displaying Pandas 3+ DataFrame and Series when pyarrow is not installed
  (closes :editor_issue:`322`).

* fixed displaying Polars *Frame when pyarrow is not installed
  (closes :editor_issue:`324`).