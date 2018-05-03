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

  Note: all axes (groups) of a session are stored in the same CSV file/Excel sheet/HDF group
  named __axes__ (__groups__).

* added examples for `read_excel` and `read_hdf` functions (closes :issue:`617`).

* updated `Session.summary` so as to display all kinds of objects and allowed to pass a function
  returning a string representation of an object instead of passing a pre-defined string template
  (closes :issue:`608`):

    >>> axis1 = Axis("a=a0..a2")
    >>> group1 = axis1['a0,a1'] >> 'a01'
    >>> arr1 = ndtest((2, 2), title='array 1', dtype=np.int64)
    >>> arr2 = ndtest(4, title='array 2', dtype=np.int64)
    >>> arr3 = ndtest((3, 2), title='array 3', dtype=np.int64)
    >>> s = Session([('axis1', axis1), ('group1', group1), ('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])

    Default template

    >>> print(s.summary())
    axis1: a ['a0' 'a1' 'a2'] (3)
    group1: a01@a ['a0', 'a1'] (2)
    arr1: a, b (2 x 2) [int64]
        array 1
    arr2: a (4) [int64]
        array 2
    arr3: a, b (3 x 2) [int64]
        array 3

    Using a specific template

    >>> def print_array(key, array):
    ...     axes_names = ', '.join(array.axes.display_names)
    ...     shape = ' x '.join(str(i) for i in array.shape)
    ...     return "{} -> {} ({})\\n  title = {}\\n  dtype = {}".format(key, axes_names, shape,
    ...                                                                 array.title, array.dtype)
    >>> template = {Axis:  "{key} -> {name} [{labels}] ({length})",
    ...             Group: "{key} -> {name}:{axis_name} {labels} ({length})",
    ...             LArray: print_array}
    >>> print(s.summary(template))
    axis1 -> a ['a0' 'a1' 'a2'] (3)
    group1 -> a01:a ['a0', 'a1'] (2)
    arr1 -> a, b (2 x 2)
      title = array 1
      dtype = int64
    arr2 -> a (4)
      title = array 2
      dtype = int64
    arr3 -> a, b (3 x 2)
      title = array 3
      dtype = int64


Fixes
-----

* fixed Excel add-ins not loaded when opening an Excel Workbook by calling the method
  ``to_excel`` with no path on an array or via "Copy to Excel (CTRL+E)" in the editor
  (closes :issue:`154`).

* fixed current active Excel Workbook closed when calling the method ``to_excel`` on an array
  with -1 as filepath argument (closes :issue:`473`).
