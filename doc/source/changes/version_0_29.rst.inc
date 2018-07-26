Syntax changes
--------------

* renamed LArray.drop_labels to ignore_labels to avoid confusion with the new LArray.drop (closes :issue:`672`).

* LArray.insert pos and axis arguments are deprecated because those were only useful for very specific cases and those
  can easily be rewritten by using an indices group (axis.i[pos]) for the "before" argument instead
  (closes :issue:`652`).


Backward incompatible changes
-----------------------------

* backward incompatible changes


New features
------------

* added new module to create arrays with values generated randomly following a few different distributions, or shuffle
  an existing array along an axis:

    >>> from larray.random import *

    generate integers between two bounds (0 and 10 in this example)

    >>> randint(0, 10, axes='a=a0..a2')
    a  a0  a1  a2
        3   6   2

    generate values following a uniform distribution

    >>> uniform(axes='a=a0..a2')
    a                   a0                  a1                  a2
       0.33293756929238394  0.5331412592583252  0.6748786766763107

    generate values following a normal distribution (:math:`\mu` = 1 and :math:`\sigma` = 2 in this example)

    >>> normal(1, scale=2, axes='a=a0..a2')
    a                   a0                 a1                  a2
       -0.9216651561025018  5.119734598931103  4.4467876992838935

    randomly shuffle an existing array along one axis

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

    generate values by randomly choosing between specified values (5, 10 and 15 in this example), potentially with a
    specified probability for each value (respectively a 30%, 50%, 20% probability of occurring in this example).

    >>> choice([5, 10, 15], p=[0.3, 0.5, 0.2], axes='a=a0,a1;b=b0..b2')
    a\b  b0  b1  b2
     a0  15  10  10
     a1  10   5  10

    same as above with labels and probabilities given as a one dimensional LArray

    >>> proba = LArray([0.3, 0.5, 0.2], Axis([5, 10, 15], 'outcome'))
    >>> proba
    outcome    5   10   15
             0.3  0.5  0.2
    >>> choice(p=proba, axes='a=a0,a1;b=b0..b2')
    a\b  b0  b1  b2
     a0  10  15   5
     a1  10   5  10

* added method `equals` to Group which compares group names, associated axis names
  and labels between two groups:

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

* allowed arrays to have metadata (e.g. title, description, authors, ...).
  Array metadata can be accessed using the syntax ``array.meta.name``.
  See below for use cases:

    >>> # add metadata to/from an array
    >>> # for Python <= 3.5
    >>> arr = ndtest((3, 3), meta=[('title', 'array for testing'), ('author', 'John Smith')])
    >>> # for Python >= 3.6
    >>> arr = ndtest((3, 3), meta=Metadata(title='array for testing', author='John Smith'))
    >>> # access an item
    >>> arr.meta.author
    'John Smith'
    >>> # update/add an item (the item is added if not found in existing metadata)
    >>> arr.meta.city = 'London'
    >>> # delete an item
    >>> del arr.meta.city

  Warnings:

  - Currently, only the HDF (.h5) file format supports saving and loading array metadata.
  - Metadata is not kept when actions or methods are applied on an array
    except for operations modifying the object in-place, such as `pop[age < 10] = 0`,
    and when the method `copy()` is called. Do not add metadata to an array if you know
    you will apply actions or methods on it before dumping it.

  Closes :issue:`78` and :issue:`79`.

* deprecated attribute `title` of array objects. A title is now considered as a metadata
  and must be added as:

    >>> # add title at array creation
    >>> arr = ndtest((3, 3), meta=[('title', 'array for testing')])

    >>> # or after array creation
    >>> arr = ndtest((3, 3))
    >>> arr.meta.title = 'array for testing'

* made a few useful constants accessible directly from the larray module: nan, inf, pi, e and euler_gamma. Like for
  any Python functionality, you can choose how to import and use them. For example, for pi:

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

* allowed sessions to have metadata.
  Like for arrays, session metadata can be accessed using the syntax ``session.meta.name``.
  See below for use cases:

    >>> # add metadata to a session at creation
    >>> # Python <= 3.5
    >>> s = Session([('arr1', ndtest(2)), ('arr2', ndtest(3)], meta=[('title', 'my title'), ('author', 'John Smith')])
    >>> # Python 3.6+
    >>> s = Session(arr1=ndtest(2), arr2=ndtest(3), meta=Metadata(title='my title', author='John Smith'))
    >>> # show/access metadata
    >>> s.meta
    title: my title
    author: John Smith

  Warnings:

  - Contrary to array metadata, saving and loading session metadata is supported for
    all current session file formats: Excel, CSV and HDF (.h5)
  - Metadata is not kept when actions or methods are applied on a session
    except for operations modifying a specific array, such as: `s['arr1'] = 0`.
    Do not add metadata to a session if you know you will apply actions or methods
    on it before dumping it.

  Closes :issue:`640`.

* implemented LArray.drop to return the array without some labels or indices along an axis (closes :issue:`506`).

    >>> arr1 = ndtest((2, 4))
    >>> arr1
    a\b  b0  b1  b2  b3
     a0   0   1   2   3
     a1   4   5   6   7
    >>> a, b = arr1.axes

    dropping a single label

    >>> arr1.drop('b1')
    a\b  b0  b2  b3
     a0   0   2   3
     a1   4   6   7

    dropping multiple labels or a slice

    >>> # arr1.drop('b1,b3')
    >>> arr1.drop(['b1', 'b3'])
    a\b  b0  b2
     a0   0   2
     a1   4   6

    dropping a slice

    >>> # arr1.drop('b1:b3')
    >>> arr1.drop(b['b1':'b3'])
    a\b  b0
     a0   0
     a1   4

    dropping labels by position requires to specify the axis

    >>> # arr1.drop('b.i[1]')
    >>> arr1.drop(b.i[1])
    a\b  b0  b2  b3
     a0   0   2   3
     a1   4   6   7


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

* made some operations on Excel Workbooks a bit faster by telling Excel to avoid updating the screen when the Excel
  instance is not visible anyway. This affects all workbooks opened via open_excel as well as read_excel and
  LArray.to_excel when using the default "xlwings" engine.

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

* renamed method `array_equals` of Session as `element_equals`.

* methods `equals` and `element_equals` of Session also compare axes and groups in addition to arrays
  (closes :issue:`610`):

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

* updated documentation of argument 'keepaxes' of aggregation functions (closes :issue:`607`).

* deprecated argument `nb_index` in favor of `nb_axes` in load method used
  in the context of ``open_excel`` (closes :issue:`648`).

* added missing arguments `wide` and `value_name` to methods `LArray.as_table` and `LArray.dump`
  (closes :issue:`653`).

* stack accepts any array-like object instead of only LArray

    >>> stack(a0=[1, 2, 3], a1=[4, 5, 6], axis='a')
    {0}*\a  a0  a1
         0   1   4
         1   2   5
         2   3   6

* replaced the `element_equal` function by the `eq` method of LArray (closes :issue:`630`).

* renamed `nan_equals` argument of `equals` and `eq` methods of LArray as `nans_equal`.

* made the `from_series` function support series with multiindex (closes :issue:`465`)


Fixes
-----

* fixed Excel add-ins not loaded when opening an Excel Workbook by calling the method
  ``to_excel`` with no path on an array or via "Copy to Excel (CTRL+E)" in the editor
  (closes :issue:`154`).

* fixed current active Excel Workbook closed when calling the method ``to_excel`` on an array
  with -1 as filepath argument (closes :issue:`473`).

* fixed LArray.split_axes when splitting a single axis and using the names argument
  (e.g. arr.split_axes('bd', names=('b', 'd'))).

* fixed splitting an anonymous axis without specifying the names argument.

  >>> combined = ndtest('a0_b0,a0_b1,a0_b2,a1_b0,a1_b1,a1_b2')
  >>> combined
  {0}  a0_b0  a0_b1  a0_b2  a1_b0  a1_b1  a1_b2
           0      1      2      3      4      5
  >>> combined.split_axes(0)
  {0}\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5

* fixed LArray.combine_axes(wildcard=True).

* fixed taking a subset of an array by giving an index along a specific axis using a string (strings
  like 'axisname.i[pos]').

* made LArray support pandas versions >= 0.21 (closes :issue:`569`)
