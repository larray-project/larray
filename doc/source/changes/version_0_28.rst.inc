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

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

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
  Also, an Axis or a list/tuple/collection of axes can be passed to the `ndtest` function (closes:issue:`534`):

    >>> ndtest("nat=BE,FO;sex=M,F")
    nat\sex  M  F
         BE  0  1
         FO  2  3

Fixes
-----

* fixed something (closes :issue:`1`).