.. py:currentmodule:: larray

Syntax changes
--------------

* deprecated ``title`` attribute of ``LArray`` objects and ``title`` argument of array creation functions.
  A title is now considered as a metadata and must be added as:

    >>> # add title at array creation
    >>> arr = ndtest((3, 3), meta=[('title', 'array for testing')])

    >>> # or after array creation
    >>> arr = ndtest((3, 3))
    >>> arr.meta.title = 'array for testing'

  See below for more information about metadata handling.

* renamed ``LArray.drop_labels()`` to :py:obj:`LArray.ignore_labels()` to avoid confusion with the new
  :py:obj:`LArray.drop()` method (closes :issue:`672`).

* renamed ``Session.array_equals()`` to :py:obj:`Session.element_equals()` because this method now also compares
  axes and groups in addition to arrays.

* renamed ``Sheet.load()`` and ``Range.load()`` ``nb_index`` argument to ``nb_axes`` to be consistent with all other
  input functions (read_*). ``Sheet`` and ``Range`` are the objects one gets when taking subsets
  of the excel :py:obj:`Workbook` objects obtained via :py:obj:`open_excel()` (closes :issue:`648`).

* deprecated the ``element_equal()`` function in favor of the :py:obj:`LArray.eq()` method (closes :issue:`630`) to be
  consistent with other future methods for operations between two arrays.

* renamed ``nan_equals`` argument of :py:obj:`LArray.equals()` and :py:obj:`LArray.eq()` methods to ``nans_equal``
  because it is grammatically more correct and is explained more naturally as "whether two nans should be considered
  equal".

* :py:obj:`LArray.insert()` ``pos`` and ``axis`` arguments are deprecated because those were only useful for very
  specific cases and those can easily be rewritten by using an indices group (``axis.i[pos]``) for the ``before``
  argument instead (closes :issue:`652`).


New features
------------

* allowed arrays to have metadata (e.g. title, description, authors, ...).

  Metadata can be added when creating arrays:

    >>> # for Python <= 3.5
    >>> arr = ndtest((3, 3), meta=[('title', 'array for testing'), ('author', 'John Smith')])

    >>> # for Python >= 3.6
    >>> arr = ndtest((3, 3), meta=Metadata(title='array for testing', author='John Smith'))

  To access all existing metadata, use ``array.meta``, for example:

    >>> arr.meta
    title: array for testing
    author: John Smith

  To access some specific existing metadata, use ``array.meta.<name>``, for example:

    >>> arr.meta.author
    'John Smith'

  Updating some existing metadata, or creating new metadata (the metadata is added if there was no metadata using that
  name) should be done using ``array.meta.<name> = <value>``. For example:

    >>> arr.meta.city = 'London'

  To remove some metadata, use ``del array.meta.<name>``, for example:

    >>> del arr.meta.city

  .. note::

     - Currently, only the HDF (.h5) file format supports saving and loading array metadata.
     - Metadata is not kept when actions or methods are applied on an array
       except for operations modifying the object in-place, such as `pop[age < 10] = 0`,
       and when the method `copy()` is called. Do not add metadata to an array if you know
       you will apply actions or methods on it before dumping it.

  Closes :issue:`78` and :issue:`79`.

* allowed sessions to have metadata. Session metadata is created and accessed **using the same syntax than for arrays**
  (``session.meta.<name>``), for example to add metadata to a session at creation:

    >>> # Python <= 3.5
    >>> s = Session([('arr1', ndtest(2)), ('arr2', ndtest(3)], meta=[('title', 'my title'), ('author', 'John Smith')])

    >>> # Python 3.6+
    >>> s = Session(arr1=ndtest(2), arr2=ndtest(3), meta=Metadata(title='my title', author='John Smith'))

  .. note::

     - Contrary to array metadata, saving and loading session metadata is supported for all current session file
       formats: Excel, CSV and HDF (.h5)
     - Metadata is not kept when actions or methods are applied on a session
       except for operations modifying a specific array, such as: `s['arr1'] = 0`.
       Do not add metadata to a session if you know you will apply actions or methods
       on it before dumping it.

  Closes :issue:`640`.

* implemented :py:obj:`LArray.drop()` to return an array without some labels or indices along an axis
  (closes :issue:`506`).

    >>> arr1 = ndtest((2, 4))
    >>> arr1
    a\b  b0  b1  b2  b3
     a0   0   1   2   3
     a1   4   5   6   7
    >>> a, b = arr1.axes

  Dropping a single label

    >>> arr1.drop('b1')
    a\b  b0  b2  b3
     a0   0   2   3
     a1   4   6   7

  Dropping multiple labels

    >>> # arr1.drop('b1,b3')
    >>> arr1.drop(['b1', 'b3'])
    a\b  b0  b2
     a0   0   2
     a1   4   6

  Dropping a slice

    >>> # arr1.drop('b1:b3')
    >>> arr1.drop(b['b1':'b3'])
    a\b  b0
     a0   0
     a1   4

  Dropping labels by position requires to specify the axis

    >>> # arr1.drop('b.i[1]')
    >>> arr1.drop(b.i[1])
    a\b  b0  b2  b3
     a0   0   2   3
     a1   4   6   7

* added new module to create arrays with values generated randomly following a few different distributions, or shuffle
  an existing array along an axis:

    >>> from larray.random import *

  Generate integers between two bounds (0 and 10 in this example)

    >>> randint(0, 10, axes='a=a0..a2')
    a  a0  a1  a2
        3   6   2

  Generate values following a uniform distribution

    >>> uniform(axes='a=a0..a2')
    a                   a0                  a1                  a2
       0.33293756929238394  0.5331412592583252  0.6748786766763107

  Generate values following a normal distribution (:math:`\mu` = 1 and :math:`\sigma` = 2 in this example)

    >>> normal(1, scale=2, axes='a=a0..a2')
    a                   a0                 a1                  a2
       -0.9216651561025018  5.119734598931103  4.4467876992838935

  Randomly shuffle an existing array along one axis

    >>> arr = ndtest((3, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
     a2   6   7   8
    >>> permutation(arr, axis='b')
    a\b  b1  b2  b0
     a0   1   2   0
     a1   4   5   3
     a2   7   8   6

  Generate values by randomly choosing between specified values (5, 10 and 15 in this example), potentially with a
  specified probability for each value (respectively a 30%, 50%, 20% probability of occurring in this example).

    >>> choice([5, 10, 15], p=[0.3, 0.5, 0.2], axes='a=a0,a1;b=b0..b2')
    a\b  b0  b1  b2
     a0  15  10  10
     a1  10   5  10

  Same as above with labels and probabilities given as a one dimensional LArray

    >>> proba = LArray([0.3, 0.5, 0.2], Axis([5, 10, 15], 'outcome'))
    >>> proba
    outcome    5   10   15
             0.3  0.5  0.2
    >>> choice(p=proba, axes='a=a0,a1;b=b0..b2')
    a\b  b0  b1  b2
     a0  10  15   5
     a1  10   5  10

* made a few useful constants accessible directly from the larray module: ``nan``, ``inf``, ``pi``, ``e`` and
  ``euler_gamma``. Like for any Python functionality, you can choose how to import and use them. For example, for
  ``pi``:

    >>> from larray import *
    >>> pi
    3.141592653589793
    OR
    >>> from larray import pi
    >>> pi
    3.141592653589793
    OR
    >>> import larray as la
    >>> la.pi
    3.141592653589793

* added :py:obj:`Group.equals()` method which compares group names, associated axis names and labels between two groups:

    >>> a = Axis('a=a0..a3')
    >>> a02 = a['a0:a2'] >> 'group_a'
    >>> # different group name
    >>> a02.equals(a['a0:a2'])
    False
    >>> # different axis name
    >>> other_axis = a.rename('other_name')
    >>> a02.equals(other_axis['a0:a2'] >> 'group_a')
    False
    >>> # different labels
    >>> a02.equals(a['a1:a3'] >> 'group_a')
    False


Miscellaneous improvements
--------------------------

* completely rewritten the 'Load And Dump Arrays, Sessions, Axes And Groups' section of the tutorial
  (closes :issue:`645`)

* saving or loading a session from a file now includes :ref:`api-axis` and :ref:`api-group` objects in addition to
  arrays (closes :issue:`578`).

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

  .. note::

     All axes and groups of a session are stored in the same CSV file/Excel sheet/HDF group named respectively
     ``__axes__`` and ``__groups__``.

* updated :py:obj:`Session.summary()` so as to display all kinds of objects and allowed to pass a function
  returning a string representation of an object instead of passing a pre-defined string template
  (closes :issue:`608`):

    >>> axis1 = Axis("a=a0..a2")
    >>> group1 = axis1['a0,a1'] >> 'a01'
    >>> arr1 = ndtest((2, 2), title='array 1', dtype=np.int64)
    >>> arr2 = ndtest(4, title='array 2', dtype=np.int64)
    >>> arr3 = ndtest((3, 2), title='array 3', dtype=np.int64)
    >>> s = Session([('axis1', axis1), ('group1', group1), ('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])

  Using the default template

    >>> print(s.summary())
    axis1: a ['a0' 'a1' 'a2'] (3)
    group1: a['a0', 'a1'] >> a01 (2)
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
    ...             Group: "{key} -> {name}: {axis_name} {labels} ({length})",
    ...             LArray: print_array}
    >>> print(s.summary(template))
    axis1 -> a ['a0' 'a1' 'a2'] (3)
    group1 -> a01: a ['a0', 'a1'] (2)
    arr1 -> a, b (2 x 2)
      title = array 1
      dtype = int64
    arr2 -> a (4)
      title = array 2
      dtype = int64
    arr3 -> a, b (3 x 2)
      title = array 3
      dtype = int64

* methods :py:obj:`Session.equals()` and :py:obj:`Session.element_equals()` now also compare axes and groups in
  addition to arrays (closes :issue:`610`):

    >>> a = Axis('a=a0..a2')
    >>> a01 = a['a0,a1'] >> 'a01'
    >>> s1 = Session([('a', a), ('a01', a01), ('arr1', ndtest(2)), ('arr2', ndtest((2, 2)))])
    >>> s2 = Session([('a', a), ('a01', a01), ('arr1', ndtest(2)), ('arr2', ndtest((2, 2)))])

  Identical sessions

    >>> s1.element_equals(s2)
    name     a   a01  arr1  arr2
          True  True  True  True

  Different value(s) between two arrays

    >>> s2.arr1['a1'] = 0
    >>> s1.element_equals(s2)
    name     a   a01   arr1  arr2
          True  True  False  True

  Different label(s)

    >>> s2.arr2 = ndtest("b=b0,b1; a=a0,a1")
    >>> s2.a = Axis('a=a0,a1')
    >>> s1.element_equals(s2)
    name      a   a01   arr1   arr2
          False  True  False  False

  Extra/missing objects

    >>> s2.arr3 = ndtest((3, 3))
    >>> del s2.a
    >>> s1.element_equals(s2)
    name      a   a01   arr1   arr2   arr3
          False  True  False  False  False

* added arguments ``wide`` and ``value_name`` to methods :py:obj:`LArray.as_table()` and :py:obj:`LArray.dump()` like
  in :py:obj:`LArray.to_excel()` and :py:obj:`LArray.to_csv()` (closes :issue:`653`).

* the :py:obj:`from_series()` function supports Pandas series with a MultiIndex (closes :issue:`465`)

* the :py:obj:`stack()` function supports any array-like object instead of only LArray objects.

    >>> stack(a0=[1, 2, 3], a1=[4, 5, 6], axis='a')
    {0}*\a  a0  a1
         0   1   4
         1   2   5
         2   3   6

* made some operations on Excel Workbooks a bit faster by telling Excel to avoid updating the screen when the Excel
  instance is not visible anyway. This affects all workbooks opened via :py:obj:`open_excel()` as well as
  :py:obj:`read_excel()` and :py:obj:`LArray.to_excel()` when using the default ``xlwings`` engine.

* made the documentation link in Windows start menu version-specific (instead of always pointing to the
  latest release) so that users do not inadvertently use the latest release syntax when using an older version of
  larray (closes :editor_issue:`142`).

* added menu bar with undo/redo when editing single arrays (as a byproduct of :editor_issue:`133`).


Fixes
-----

* fixed Copy(to Excel)/Paste/Plot in the editor not working for 1D and 2D arrays (closes :editor_issue:`140`).

* fixed Excel add-ins not loaded when opening an Excel Workbook by calling the :py:obj:`LArray.to_excel()` method
  with no path or via "Copy to Excel (CTRL+E)" in the editor (closes :issue:`154`).

* made LArray support Pandas versions >= 0.21 (closes :issue:`569`)

* fixed current active Excel Workbook being closed when calling the :py:obj:`LArray.to_excel()` method on an array with
  ``-1`` as ``filepath`` argument (closes :issue:`473`).

* fixed :py:obj:`LArray.split_axes()` when splitting a single axis and using the names argument
  (e.g. ``arr.split_axes('bd', names=('b', 'd'))``).

* fixed splitting an anonymous axis without specifying the names argument.

  >>> combined = ndtest('a0_b0,a0_b1,a0_b2,a1_b0,a1_b1,a1_b2')
  >>> combined
  {0}  a0_b0  a0_b1  a0_b2  a1_b0  a1_b1  a1_b2
           0      1      2      3      4      5
  >>> combined.split_axes(0)
  {0}\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5

* fixed :py:obj:`LArray.combine_axes()` with ``wildcard=True``.

* fixed taking a subset of an array by giving an index along a specific axis using a string (strings
  like ``"axisname.i[pos]"``).

* fixed the editor not working with Python 2 or recent Qt4 versions.
