.. py:currentmodule:: larray


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dropped support for Python 2 (closes :issue:`567`).


New features
^^^^^^^^^^^^

* added support for Python 3.8 (closes :issue:`850`).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* scalar objects (i.e of type int, float, bool, string, date, time or datetime) belonging to a session
  are now also saved and loaded when using the HDF5 or pickle format (closes :issue:`842`).

* implemented :py:obj:`Axis.astype()` method (closes :issue:`880`).

* added `min_y`, `max_y` and `xticks_spacing` keyword arguments to the :py:obj:`ReportSheet.add_graph()` and
  :py:obj:`ReportSheet.add_graphs()` methods (closes :issue:`901`).

* implemented :py:obj:`isscalar()` function (closes :issue:`872`).

* implemented the :py:obj:`Array.allclose()` method (closes :issue:`871`).

* implemented :py:obj:`Axis.min()` and :py:obj:`Axis.max()` methods (closes :issue:`874`).


Fixes
^^^^^

* fixed an edge case for group aggregates and labels in reverse order (closes :issue:`868`).

* fixed \*_by functions returning the same value as if no axis was passed when specifying all axes
  (closes :issue:`913`)
