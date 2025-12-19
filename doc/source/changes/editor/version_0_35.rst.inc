.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* allow displaying *many* more different kinds of objects, and not only arrays
  from larray. One specific goal when developing this new feature was speed.
  Most of these viewers should be fast (when at all possible), even on (very)
  large datasets. We only support displaying (not editing) all the new types.

  The following types are supported so far (but adding more is relatively easy):

  * Python builtin objects:
    - tuple (including named tuple), list (sequences), dict (mappings),
      dict views, memoryview and array
    - text and binary files
  * Python stdlib objects:
    - pathlib.Path
      * if the path points to a directory, it will display the content of the
        directory
      * if the path points to a file, it will try to display it, if we
        implemented support for that file type (see below for the list
        of supported types).
    - sqlite3.Connection (and their tables)
    - pstats.Stats (results of Python's profiler)
    - zipfile.ZipFile and zipfile.Path
  * new objects from LArray: Axis, Excel Workbook (what you get from
    larray.open_excel()), Sheets and Range
  * IODE "collections" objects: Comments, Equations, Identities, Lists, Tables,
      Scalars and Variables, as well as Table objects
  * Pandas: DataFrame, Series and DataFrameGroupBy
  * Polars: DataFrame and LazyFrame
  * Numpy: ndarray
  * PyArrow: Array, Table, RecordBatchFileReader (reader object for feather
    files) and ParquetFile
  * Narwhals: DataFrame and LazyFrame
  * PyTables: File, Group (with special support for Pandas DataFrames written
    in HDF files), Array and Table
  * IBIS: Table
  * DuckDB: DuckDBPyConnection and DuckDBPyRelation (what you receive from any
    query)

  File types (extensions) currently supported:
    - Iode files: .ac, .ae, .ai, .al, .as, .at, .av, .cmt, .eqs, .idt, .lst,
                  .scl, .tbl, .var
    - Text files: .bat, .c, .cfg, .cpp, .h, .htm, .html, .ini, .log, .md,
                  .py, .pyx, .pxd, .rep, .rst, .sh, .sql, .toml, .txt, .wsgi,
                  .yaml, .yml
    - HDF5 files: .h5, .hdf
    - Parquet files: .parquet
    - Stata files: .dta
    - Feather files: .feather
    - SAS files: .sas7bdat
      It is limited to the first few thousand rows (the exact number depends on
      the number of columns), because reading later rows get increasingly slow,
      to the point of being unusable.
    - CSV files: .csv
    - Gzipped CSV files: .csv.gz
    - Excel files: .xls, .xlsx
    - Zip files: .zip
    - DuckDB files: .ddb, .duckdb

* the editor now features a new "File Explorer" (accessible from the "File"
  menu) so that one can more easily make use of all the above file viewers.

* added a new SQL Console (next to the iPython console) for querying Polars
  structures (DataFrame, LazyFrame and Series) as SQL tables. The console
  features auto-completion for SQL keywords, table names and column names
  and stores the last 1000 queries (even across sessions). Recalling a query
  from history is done with the up and down arrows and like in the iPython
  console, it searches through history with the current command as prefix.
  This console will only be present if the polars module is installed.

* allow sorting some objects by column by pressing on an horizontal label.
  This is currently implemented for the following objects:
    - python built-in sequences (e.g. tuples and lists)
    - python pathlib.Path objects representing directories
    - LArray (only for 2D arrays)
    - Pandas DataFrame
    - Polars DataFrame and LazyFrame
    - Narwhals LazyFrame
    - SQLite tables
    - DuckDB relations

* allow filtering some objects by pressing on an horizontal label.
  This is currently implemented for the following objects:
    - Pandas DataFrame
    - Polars DataFrame and LazyFrame
    - DuckDB relations

* allow comparing arrays/sessions with different axes in :py:obj:`compare()`.
  The function gained ``align`` and ``fill_value`` arguments and the interface
  has a new combobox to change the alignment method during the comparison:
      - outer: will use a label if it is in any array (ordered like the first array).
               This is the default as it results in no information loss.
      - inner: will use a label if it is in all arrays (ordered like the first array).
      - left: will use the first array axis labels.
      - right: will use the last array axis labels.
      - exact: raise an error when axes are not equal.
  Closes :editor_issue:`214` and :editor_issue:`251`.

* double-clicking on a name in the variable list will open it in a new window
  (closes :editor_issue:`143`).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* made the editor interruptible by an outside program (i.e. made PyCharm stop &
  restart buttons work directly instead of only when the editor receives the
  focus again). Closes :editor_issue:`257`.

* resize axes and vertical label columns automatically

* string values are left aligned instead of right aligned

* when comparing sessions via :py:obj:`compare()`, the color of arrays in the
  list is now updated depending on the tolerance. To reflect that the tolerance
  widget moved to the top of the interface. Closes :editor_issue:`201`.

* :py:obj:`compare()` max difference is colored red when the difference is not 0

* typing the name of a variable holding a matplotlib figure (or axes) in the
  console shows it (previously, only expressions were displayed and *not*
  simple variables). For example: ::

     >>> arr.plot()

  already displayed a plot, but: ::

     >>> ax = arr.plot()
     >>> ax

  did not.

* display matplotlib Figure objects in addition to Axes objects


Fixes
^^^^^

* fixed :py:obj:`compare()` colors when the only difference is nans on either
  side.

* fixed :py:obj:`compare()` colors and max difference label when the only
  differences are for rows where the value is 0 in the first array.

* fixed single column plot in viewer when ticks are not strings