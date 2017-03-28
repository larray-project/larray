Fixes
-----

* fixed support for loading arrays without axe names from Excel files (in that case index_col/nb_index are necessary)
* fixed using a single int for index_col in read_excel() and sheet.load()
* fixed loading empty Excel sheets via xlwings correctly (ie do not crash)
* fixed dumping a session loaded from an H5 file to Excel
