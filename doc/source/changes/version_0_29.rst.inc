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

* saving or loading a session from a file now includes `Axis` and `Group` objects in addition to arrays
  (closes :issue:`578`):

    Create a session containing axes, groups and arrays

    >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")
    >>> a01 = a['a0,a1'] >> 'a01'
    >>> arr1, arr2 = ndtest((a, b)), ndtest(a)
    >>> s = Session([('a', a), ('b', b), ('a01', a01), ('arr1', arr1), ('arr2', arr2)])

    Saving a session will save axes, groups and arrays

    >>> s.save('session.h5')

    Loading a session will load axes, groups and arrays

    >>> s2 = s.load('session.h5')
    >>> s2
    Session(arr1, arr2, a, b, a01)


Fixes
-----

* fixed Excel add-ins not loaded when opening an Excel Workbook by calling the method
  ``to_excel`` with no path on an array or via "Copy to Excel (CTRL+E)" in the editor
  (closes :issue:`154`).

* fixed current active Excel Workbook closed when calling the method ``to_excel`` on an array
  with -1 as filepath argument (closes :issue:`473`).
