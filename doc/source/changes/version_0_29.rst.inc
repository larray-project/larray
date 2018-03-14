Syntax changes
--------------

* new syntax


Backward incompatible changes
-----------------------------

* backward incompatible changes


New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.


.. _misc:

Miscellaneous improvements
--------------------------

* improved something.


Fixes
-----

* fixed Excel add-ins not loaded when opening an Excel Workbook by calling the method
  ``to_excel`` with no path on an array or via "Copy to Excel (CTRL+E)" in the editor
  (closes :issue:`154`).

* fixed current active Excel Workbook closed when calling the method ``to_excel`` on an array
  with -1 as filepath argument (closes :issue:`473`).
