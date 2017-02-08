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

* viewer: view and edit functions accept path to file containing
  arrays to be loaded as argument.

* plot function from viewer updated: can plot several rows/columns.
  Draw a new curve for each row except if only one column has been
  selected.

* allowed to get axis from array using syntax a.axis_name
  (instead of a.axes.axis_name)

Fixes
-----

* column titles are no longer converted to lowercase.

* 'by' aggregation functions accept both axes and groups.

* nb_index is no longer required in read_excel when using xlrd engine

* viewer: selection of entire rows/columns will load all data

* viewer: filtering on anonymous axes is now possible