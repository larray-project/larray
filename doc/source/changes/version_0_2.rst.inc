New features
------------

* added to_clipboard.

* added embryonic documentation.

* added sort_columns and na arguments to read_hdf.

* added sort_rows, sort_columns and na arguments to read_excel.

* added setup.py to install the module.

Miscellaneous improvements
--------------------------

* IO functions (to_*/read_*) now support unnamed axes. The set of
  supported operations is very limited with such arrays though.

* to_excel sheet_name defaults to "Sheet1" like in Pandas.

* reorganised files.

* automated somewhat releases (added a rudimentary release script).

Fixes
-----

* column titles are no longer converted to lowercase.
