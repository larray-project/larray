.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* added explicit support for Python 3.14.

* the editor is now associated with some file extensions on Windows, which
  means that double-clicking on files with these extensions in the Windows File
  Explorer will open them in the editor. The following extensions are
  associated: .h5, .hdf, .feather, .parquet, .ddb and .duckdb
  (closes :editor_issue:`302`).

  Known issues
  ------------
  * This does not work when multiple users share a single Python environment
    because the file association is done only for the user which installed the
    editor.
  * The reported file type is very long and weird. For example, for .h5 files,
    it is ".h5 larray-editor.AssocFile.h5 file" instead of just "H5 File". This
    is harmless though.
  * We choose not to associate the editor with file formats we support but
    which likely have a better viewer installed on the computer (.xlsx, .csv,
    .dta, .sas7bdat, text files, IODE files and zip files)


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Files passed as command-line arguments to the editor are now displayed via
  the "File Explorer" mechanism instead of being loaded as a Session
  (closes :editor_issue:`303`).
  In practice, this means we can support many more file formats (.feather,
  .parquet, .dta, .sas7bdat, ...) and it is possible to open files which do not
  fit entirely in memory and much faster to open other large files.

  On the other hand, we loose the ability to *directly* manipulate the data as
  arrays, unless the user loads the file explicitly. Assuming the format is
  supported by Session (.h5, larray-formatted .xlsx or directory of .csv), this
  can be done either via the File->Open Data menu or via this command in the
  console:

    Session(name_of_the_variable_with_the_filepath).to_globals()


Fixes
^^^^^

* fixed displaying DataFrames with Pandas >= 3 (closes :editor_issue:`308`).

* fixed some objects not being immediately visually updated after they are
  modified in the console. This includes updates to Pandas DataFrames done via
  `df.loc[key] = value` and `df.iloc[position] = value`
  (closes :editor_issue:`310`).

* fixed File Explorer on Python 3.9 (closes :editor_issue:`300`).

* fixed displaying "Collection" objects which raise an error on computing
  their length and/or iterating them, such as scalar Groups (closes
  :editor_issue:`313`).

* avoid warnings when displaying data with any column entirely non-numeric
  (including NaN). Closes :editor_issue:`311`.

* fixed `compare()` not working for object or string arrays (closes
  :editor_issue:`163`). The background color and "maximum absolute
  relative difference" label were always wrong and the comparison failed
  completely when the first array was an object array containing any 0 value.

* do not try to display Path objects which point to a file or directory which
  does not exist (closes :editor_issue:`317`).

* fixed (Excel) Sheet objects being only partially displayed when
  the first value contains a number but other values contain text.

* fixed the mechanism writing warning/error messages happening during the
  editor initialization. The errors are now correctly written in the user
  TEMP directory / larray-editor-stderr.log