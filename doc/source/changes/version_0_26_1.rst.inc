﻿Miscellaneous improvements
--------------------------

* Made handling Excel sheets with many blank columns/rows after the data much faster (but still slower than sheets
  without such blank cells).

Fixes
-----

* fixed reading from and writing to Excel sheets with 16384 columns or 1048576 rows (Excel's maximum).

* fixed LArray.split_axes using a custom separator and not using sort=True or when the split labels are
  ambiguous with labels from other axes (closes :issue:`485`).

* fixed reading 1D arrays with non-string labels (closes :issue:`495`).

* fixed read_csv(sort_columns=True) for 1D arrays (closes :issue:`497`).
