Backward incompatible changes
-----------------------------

* changed behavior of operators `session1 == session2` and `session1 != session2`: returns a session
  of boolean arrays (closes:issue:`516`):

    >>> s1 = Session([('arr1', ndtest(2)), ('arr2', ndtest((2, 2)))])
    >>> s2 = Session([('arr1', ndtest(2)), ('arr2', ndtest((2, 2)))])
    >>> (s1 == s2).arr1
    a    a0    a1
       True  True
    >>> s2.arr1['a1'] = 0
    >>> (s1 == s2).arr1
    a    a0     a1
       True  False
    >>> (s1 != s2).arr1
    a     a0    a1
       False  True


New features
------------

* added methods `array_equals` and `equals` to `Session` object to compare arrays from two sessions.
  The method `array_equals` return a boolean value for each array while the method `equals` returns a unique
  boolean value (True if all arrays of both sessions are equal, False otherwise):

    >>> s1 = Session([('arr1', ndtest(2)), ('arr2', ndtest((2, 2)))])
    >>> s2 = Session([('arr1', ndtest(2)), ('arr2', ndtest((2, 2)))])
    >>> s1.array_equals(s2)
    name  arr1  arr2
          True  True
    >>> s1.equals(s2)
    True

    Different value(s)

    >>> s2.arr1['a1'] = 0
    >>> s1.array_equals(s2)
    name   arr1  arr2
          False  True
    >>> s1.equals(s2)
    False

    Different label(s)

    >>> from larray import ndrange
    >>> s2.arr2 = ndrange("b=b0,b1; a=a0,a1")
    >>> s1.array_equals(s2)
    name   arr1   arr2
          False  False
    >>> s1.equals(s2)
    False

    Extra/missing array(s)

    >>> s2.arr3 = ndtest((3, 3))
    >>> s1.array_equals(s2)
    name   arr1   arr2   arr3
          False  False  False
    >>> s1.equals(s2)
    False

  Closes:issue:`517`.

* added method `equals` to `LArray` object to compare two arrays (closes :issue:`518`):

    >>> arr1 = ndtest((2, 3))
    >>> arr1
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr2 = arr1.copy()
    >>> arr1.equals(arr2)
    True
    >>> arr2['b1'] += 1
    >>> arr1.equals(arr2)
    False
    >>> arr3 = arr1.set_labels('a', ['x0', 'x1'])
    >>> arr1.equals(arr3)
    False

    Arrays with nan values

    >>> arr1 = ndtest((2, 3), dtype=float)
    >>> arr1['a1', 'b1'] = nan
    >>> arr1
    a\b   b0   b1   b2
     a0  0.0  1.0  2.0
     a1  3.0  nan  5.0
    >>> arr2 = arr1.copy()
    >>> # By default, an array containing nan values is never equal to another array,
    >>> # even if that other array also contains nan values at the same positions.
    >>> # The reason is that a nan value is different from *anything*, including itself.
    >>> arr1.equals(arr2)
    False
    >>> # set flag nan_equal to True to override this behavior
    >>> arr1.equals(arr2, nan_equal=True)
    True


Miscellaneous improvements
--------------------------

* functions `local_arrays`, `global_arrays` and `arrays` returns a session excluding arrays starting by `_`
  by default. To include them, set the flag `include_private` to True (closes :issue:`513`):

    >>> global_arr1 = ndtest((2, 2))
    >>> _global_arr2 = ndtest((3, 3))
    >>> def foo():
    ...     local_arr1 = ndtest(2)
    ...     _local_arr2 = ndtest(3)
    ...
    ...     # exclude arrays starting with '_' by default
    ...     s = arrays()
    ...     print(s.names)
    ...
    ...     # use flag 'include_private' to include arrays starting with '_'
    ...     s = arrays(include_private=True)
    ...     print(s.names)
    >>> foo()
    ['global_arr1', 'local_arr1']
    ['_global_arr2', '_local_arr2', 'global_arr1', 'local_arr1']

* implemented sessions binary operations with non sessions objects (closes :issue:`514` and :issue:`515`):

    >>> s = Session(arr1=ndtest((2, 2)), arr2=ndtest((3, 3)))
    >>> s.arr1
    a\b  b0  b1
     a0   0   1
     a1   2   3
    >>> s.arr2
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
     a2   6   7   8

    Add a scalar to all arrays

    >>> # equivalent to s2 = 3 + s
    >>> s2 = s + 3
    >>> s2.arr1
    a\b  b0  b1
     a0   3   4
     a1   5   6
    >>> s2.arr2
    a\b  b0  b1  b2
     a0   3   4   5
     a1   6   7   8
     a2   9  10  11

    Apply binary operations between two sessions

    >>> sdiff = (s2 - s) / s
    >>> sdiff.arr1
    a\b   b0   b1
     a0  inf  3.0
     a1  1.5  1.0
    >>> sdiff.arr2
    a\b   b0    b1     b2
     a0  inf   3.0    1.5
     a1  1.0  0.75    0.6
     a2  0.5  0.43  0.375

* added possibility to call the method `reindex` with a group (closes :issue:`531`):

    >>> arr = ndtest((2, 2))
    >>> arr
    a\b  b0  b1
     a0   0   1
     a1   2   3
    >>> b = Axis("b=b2..b0")
    >>> arr.reindex('b', b['b1':])
    a\b  b1  b0
     a0   1   0
     a1   3   2

* added possibility to call the methods `diff` and `growth_rate` with a group (closes :issue:`532`):

    >>> data = [[2, 4, 5, 4, 6], [4, 6, 3, 6, 9]]
    >>> a = LArray(data, "sex=M,F; year=2016..2020")
    >>> a
    sex\year  2016  2017  2018  2019  2020
           M     2     4     5     4     6
           F     4     6     3     6     9
    >>> a.diff(a.year[2017:])
    sex\year  2018  2019  2020
           M     1    -1     2
           F    -3     3     3
    >>> a.growth_rate(a.year[2017:])
    sex\year  2018  2019  2020
           M  0.25  -0.2   0.5
           F  -0.5   1.0   0.5

* function `ndrange` has been deprecated in favor of `sequence` or `ndtest`.
  Also, an Axis or a list/tuple/collection of axes can be passed to the `ndtest` function (closes :issue:`534`):

    >>> ndtest("nat=BE,FO;sex=M,F")
    nat\sex  M  F
         BE  0  1
         FO  2  3

* allowed to pass a group for argument `axis` of `stack` function (closes :issue:`535`):

    >>> b = Axis('b=b0..b2')
    >>> stack(b0=ndtest(2), b1=ndtest(2), axis=b[:'b1'])
    a\b  b0  b1
     a0   0   0
     a1   1   1

* renamed argument `nb_index` of `read_csv`, `read_excel`, `read_sas`, `from_lists` and `from_string` functions
  as `nb_axes`. The relation between `nb_index` and `nb_axes` is given by `nb_axes = nb_index + 1`:

  For a given file 'arr.csv' with content ::

  a,b\c,c0,c1
  a0,b0,0,1
  a0,b1,2,3
  a1,b0,4,5
  a1,b1,6,7

  previous code to read this array such as :

    >>> # deprecated
    >>> arr = read_csv('arr.csv', nb_index=2)

  must be updated as follow :

    >>> arr = read_csv('arr.csv', nb_axes=3)

  Closes :issue:`548`.

* added the relative tolerance `rtol` and the absolute tolerance `atol` arguments to the `LArray.equals` method.
  These two arguments can be used to test the equality between two arrays within a given relative or
  absolute tolerance:

      >>> arr1 = LArray([6., 8.], "a=a0,a1")
      >>> arr1
      a   a0   a1
         6.0  8.0
      >>> arr2 = LArray([5.999, 8.001], "a=a0,a1")
      >>> arr2
      a     a0     a1
         5.999  8.001
      >>> arr1.equals(arr2)
      False
      >>> # equals returns True if abs(array1 - array2) <= (atol + rtol * abs(array2))
      >>> arr1.equals(arr2, atol=0.01)
      True
      >>> arr1.equals(arr2, rtol=0.01)
      True

  Closes :issue:`488`.

* deprecated `nan_equal` function in favor of `element_equal` function. The `element_equal` function has the
  same optional arguments as the `LArray.equals` method but compares two arrays element-wise and
  returns an array of booleans:

    >>> arr1 = LArray([6., np.nan, 8.], "a=a0..a2")
    >>> arr1
    a   a0   a1   a2
       6.0  nan  8.0
    >>> arr2 = LArray([5.999, np.nan, 8.001], "a=a0..a2")
    >>> arr2
    a     a0   a1     a2
       5.999  nan  8.001
    >>> element_equal(arr1, arr2)
    a     a0     a1     a2
       False  False  False
    >>> element_equal(arr1, arr2, nan_equals=True)
    a     a0    a1     a2
       False  True  False
    >>> element_equal(arr1, arr2, atol=0.01, nan_equals=True)
    a    a0    a1    a2
       True  True  True
    >>> element_equal(arr1, arr2, rtol=0.01, nan_equals=True)
    a    a0    a1    a2
       True  True  True

  Closes :issue:`593`.

* renamed argument `transpose` by `wide` in `to_csv` method.

* added argument `wide` in `to_excel` method. When argument `wide` is set to False, the array is exported
  in "narrow" format, i.e. one column per axis plus one value column:

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5

  Default behavior (`wide=True`):

    >>> arr.to_excel('my_file.xlsx')

    a\b  b0  b1  b2
    a0    0   1   2
    a1    3   4   5

  With `wide=False`:

    >>> arr.to_excel('my_file.xlsx', wide=False)

     a   b  value
    a0  b0      0
    a0  b1      1
    a0  b2      2
    a1  b0      3
    a1  b1      4
    a1  b2      5

  Argument `transpose` has a different purpose than `wide` and is mainly useful to allow multiple axes as header
  when exporting arrays with more than 2 dimensions. Closes :issue:`575` and :issue:`371`.

* added argument `wide` to `read_csv` and `read_excel` functions. If False, the array to be loaded is assumed to
  be stored in "narrow" format:

    >>> # assuming the array was saved using command: arr.to_excel('my_file.xlsx', wide=False)
    >>> read_excel('my_file.xlsx', wide=False)
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5

  Closes :issue:`574`.

* added argument `name` to `to_series` method allowing to set a name to the Pandas Series returned by the method.

* added argument `value_name` to `to_csv` and `to_excel` allowing to change the default name ('value') to
  the column containg the values when the argument `wide` is set to False:

    >>> arr.to_csv('my_file.csv', wide=False, value_name='data')
    a,b,data
    a0,b0,0
    a0,b1,1
    a0,b2,2
    a1,b0,3
    a1,b1,4
    a1,b2,5

  Closes :issue:`549`.

* renamed argument `sheetname` of `read_excel` function as `sheet` (closes :issue:`587`).

* Renamed sheet_name of LArray.to_excel to sheet since it can also be an index (closes :issue:580).

* allowed to create axes with zero padded string labels (closes :issue:`533`):

    >>> Axis('zero_padding=01,02,03,10,11,12')
    Axis(['01', '02', '03', '10', '11', '12'], 'zero_padding')


Fixes
-----

* fixed passing a scalar group from an external axis to get a subset of an array (closes :issue:`178`):

    >>> arr = ndtest((3, 2))
    >>> arr['a1']
    b  b0  b1
        2   3
    >>> alt_a = Axis("alt_a=a1..a2")
    >>> arr[alt_a['a1']]
    b  b0  b1
        2   3
    >>> arr[alt_a.i[0]]
    b  b0  b1
        2   3

* fixed subscript a string LGroup key (closes :issue:`437`):

    >>> axis = Axis("a=a0,a1")
    >>> axis['a0'][0]
    'a'

* fixed `Axis.union`, `Axis.intersection` and `Axis.difference` when passed value is a single string
  (closes :issue:`489`):

    >>> a = Axis('a=a0..a2')
    >>> a.union('a1')
    Axis(['a0', 'a1', 'a2'], 'a')
    >>> a.union('a3')
    Axis(['a0', 'a1', 'a2', 'a3'], 'a')
    >>> a.union('a1..a3')
    Axis(['a0', 'a1', 'a2', 'a3'], 'a')
    >>> a.intersection('a1..a3')
    Axis(['a1', 'a2'], 'a')
    >>> a.difference('a1..a3')
    Axis(['a0'], 'a')

* fixed `to_excel` applied on >= 2D arrays using `transpose=True` (closes :issue:`579`)

    >>> arr = ndtest((2, 3))
    >>> arr.to_excel('my_file.xlsx', transpose=True)

    b\a  a0  a1
    b0    0   3
    b1    1   4
    b2    2   5

* fixed aggregation on arrays containing zero padded string labels (closes :issue:`522`):

    >>> arr = ndtest(Axis('zero_padding=01,02,03,10,11,12'))
    >>> arr
    zero_padding  01  02  03  10  11  12
                   0   1   2   3   4   5
    >>> arr.sum('01,02,03 >> 01_03; 10')
    zero_padding  01_03  10
                      3   3
