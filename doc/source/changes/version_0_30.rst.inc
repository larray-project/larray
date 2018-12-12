.. py:currentmodule:: larray

Syntax changes
--------------

* new syntax


Backward incompatible changes
-----------------------------

* :py:obj:`LArray.equals()` now returns True for arrays even when axes are in a different order or some axes are
  missing on either side (but the data is constant over that axis on the other side). To get back the old behavior, use
  check_axes=True. Closes :issue:`237`.

    >>> a = Axis('a=a0,a1')
    >>> arr1 = ndtest(a)
    >>> arr1
    a  a0  a1
        0   1

    Identical arrays are (still) considered equal

    >>> arr2 = arr1.copy()
    >>> arr2.equals(arr1)
    True

    Arrays with different labels (for the same axes), are (still) not equal

    >>> arr3 = arr1.set_labels('a', 'a8,a9')
    >>> arr3
    a  a8  a9
        0   1
    >>> arr3.equals(arr1)
    False

    Arrays with the same axes but different data, are (still) not equal

    >>> arr4 = arr1.copy()
    >>> arr4['a1'] = 42
    >>> arr4
    a  a0  a1
        0  42
    >>> arr4.equals(arr1)
    False

    Arrays with extra axes but the same data are now considered equal

    >>> arr5 = arr1.expand('b=b0..b2')
    >>> arr5
    a\b  b0  b1  b2
     a0   0   0   0
     a1   1   1   1
    >>> arr5.equals(arr1)
    True

    Unless check_axes is True

    >>> arr5.equals(arr1, check_axes=True)
    False

    Arrays with axes in a different order (but the same data) are also equal...

    >>> arr6 = arr5.transpose()
    >>> arr6
    b\a  a0  a1
     b0   0   1
     b1   0   1
     b2   0   1
    >>> arr6.equals(arr5)
    True

    Unless check_axes is True

    >>> arr3.equals(arr4, check_axes=True)
    False

* modified the behavior of the ``pattern`` argument of :py:obj:`Session.filter()` to work as the ``pattern``
  argument of :py:obj:`Group.matching()`:

    >>> axis = Axis('a=a0..a2')
    >>> group = axis['a0,a1'] >> 'a01'
    >>> test1, zero1 = ndtest((2, 2)), zeros((3, 2))
    >>> s = Session([('test1', test1), ('zero1', zero1), ('axis', axis), ('group', group)])

    >>> # get all items with names ending with '1'
    >>> s.filter(pattern='*1').names
    ['test1', 'zero1']

    >>> # get all items with names starting with letter in range a-k
    >>> s.filter(pattern='[a-k]*').names
    ['axis', 'group']

  Warning: to retrieve the previous behavior, add the character ``*`` to your pattern
  (e.g. ``s.filter('test')`` becomes ``s.filter('test*')``).

  Closes :issue:`703`.


New features
------------

* added :py:obj:`LArray.isin()` method to check whether each element of an array is contained in a list (or array) of
  values.

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr.isin([1, 5, 7])
    a\b     b0     b1     b2
     a0  False   True  False
     a1  False  False   True
    >>> arr[arr.isin([1, 5, 7])]
    a_b  a0_b1  a1_b2
             1      5

* added a feature (see the :ref:`miscellaneous section <misc>` for details).


* implemented :py:obj:`Axis.apply()` method to transform an axis labels by a function and return a new Axis.

    >>> sex = Axis('sex=MALE,FEMALE')
    >>> sex.apply(str.capitalize)
    Axis(['Male', 'Female'], 'sex')


Miscellaneous improvements
--------------------------

* implemented a simpler pattern language in :py:obj:`Axis.matching()` and :py:obj:`Group.matching()`. In addition to
  regular expressions (which now require using the ``regexp`` argument), the two methods support the following simpler
  patterns:

    * `?`     matches any single character
    * `*`     matches any number of characters
    * [seq]   matches any character in seq
    * [!seq]  matches any character not in seq

  For example, assuming the following axis:

    >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Waldo', 'Arthur Dent', 'Harvey Dent'], 'people')

  All labels starting with "A" and ending with "t" are given by:

    >>> people.matching(pattern='A*t')
    people['Arthur Dent']

  All labels containing "W" and ending with "s":

    >>> people.matching(pattern='*W*s')
    people['Bruce Willis']

  All labels with exactly 5 characters:

    >>> people.matching(pattern='?????')
    people['Waldo']

  All labels starting with either "A" or "B":

    >>> people.matching(pattern='[AB]*')
    people['Bruce Wayne', 'Bruce Willis', 'Arthur Dent']

* :py:obj:`LArray.set_labels()` can now take functions to transform axes labels (closes :issue:`536`).

    >>> arr = ndtest((2, 2))
    >>> arr
    a\b  b0  b1
     a0   0   1
     a1   2   3
    >>> arr.set_labels('a', str.upper)
    a\b  b0  b1
     A0   0   1
     A1   2   3

* added option ``exact`` to ``join`` argument of :py:obj:`Axis.align()` and :py:obj:`LArray.align()` methods.
  Instead of aligning, passing ``join='exact'`` to the ``align`` method will raise an error when axes are not equal.
  Closes :issue:`338`.

* made :py:obj:`Axis.by()` and :py:obj:`Group.by()` return a list of named groups instead of anonymous groups.
  By default, group names are defined as ``<start>:<end>``. This can be changed via the new ``template`` argument:

    >>> age = Axis('age=0..6')
    >>> age
    Axis([0, 1, 2, 3, 4, 5, 6], 'age')
    >>> age.by(3)
    (age.i[0:3] >> '0:2', age.i[3:6] >> '3:5', age.i[6:7] >> '6')
    >>> age.by(3, step=2)
    (age.i[0:3] >> '0:2', age.i[2:5] >> '2:4', age.i[4:7] >> '4:6', age.i[6:7] >> '6')
    >>> age.by(3, template='{start}-{end}')
    (age.i[0:3] >> '0-2', age.i[3:6] >> '3-5', age.i[6:7] >> '6')

  Closes :issue:`669`.

* allowed to specify an axis by its postion when selecting a subset of an array using the string notation:

    >>> pop_mouv = ndtest('geo_from=BE,FR,UK;geo_to=BE,FR,UK')
    >>> pop_mouv
    geo_from\geo_to  BE  FR  UK
                 BE   0   1   2
                 FR   3   4   5
                 UK   6   7   8
    >>> pop_mouv['0[BE, UK]']   # equivalent to pop_mouv[pop_mouv.geo_from['BE,UK']]
    geo_from\geo_to  BE  FR  UK
                 BE   0   1   2
                 UK   6   7   8
    >>> pop_mouv['1.i[0, 2]']   # equivalent to pop_mouv[pop_mouv.geo_to.i[0, 2]]
    geo_from\geo_to  BE  UK
                 BE   0   2
                 FR   3   5
                 UK   6   8

  Closes :issue:`671`.

* added documentation and examples for :py:obj:`where()`, :py:obj:`maximum()`
  and :py:obj:`minimum()` functions (closes :issue:`700`)

* updated the ``Working With Sessions`` section of the tutorial (closes :issue:`568`).

* renamed `a_min` and `a_max` arguments of :py:obj:`LArray.clip()` as `minval` and `maxval` respectively
  and made them optional (closes :issue:`747`).

* implemented :py:obj:`LArray.reverse()` method to reverse one or several axes of an array (closes :issue:`631`).


* added :py:obj:`set_options` allowing to set options for larray within a ``with`` block or globally:

    >>> from larray import *
    >>> arr = ndtest((500, 100), dtype=float) / 3

    You can use ``set_options`` either as a context manager:

    >>> with set_options(display_width=100, display_edgeitems=2):
    ...     print(arr)
     a\b                  b0                  b1  ...                 b98                 b99
      a0                 0.0  0.3333333333333333  ...  32.666666666666664                33.0
      a1  33.333333333333336  33.666666666666664  ...                66.0   66.33333333333333
     ...                 ...                 ...  ...                 ...                 ...
    a498             16600.0  16600.333333333332  ...  16632.666666666668             16633.0
    a499  16633.333333333332  16633.666666666668  ...             16666.0  16666.333333333332

    Or to set global options:

    >>> set_options(display_maxlines=10, display_precision=2) # doctest: +SKIP
    >>> print(arr) # doctest: +SKIP
     a\b        b0        b1        b2  ...       b97       b98       b99
      a0      0.00      0.33      0.67  ...     32.33     32.67     33.00
      a1     33.33     33.67     34.00  ...     65.67     66.00     66.33
      a2     66.67     67.00     67.33  ...     99.00     99.33     99.67
      a3    100.00    100.33    100.67  ...    132.33    132.67    133.00
      a4    133.33    133.67    134.00  ...    165.67    166.00    166.33
     ...       ...       ...       ...  ...       ...       ...       ...
    a495  16500.00  16500.33  16500.67  ...  16532.33  16532.67  16533.00
    a496  16533.33  16533.67  16534.00  ...  16565.67  16566.00  16566.33
    a497  16566.67  16567.00  16567.33  ...  16599.00  16599.33  16599.67
    a498  16600.00  16600.33  16600.67  ...  16632.33  16632.67  16633.00
    a499  16633.33  16633.67  16634.00  ...  16665.67  16666.00  16666.33

  Closes :issue:`274`.

Fixes
-----

* fixed :py:obj:`LArray.divnot0()` being slow when the divisor has many axes and many zeros (closes :issue:`705`).

* fixed maximum length of sheet names (31 characters instead of 30 characters) when adding a new sheet
  to an Excel Workbook (closes :issue:`713`).

* fixed missing documentation of many functions in :ref:`Utility Functions <api-ufuncs>` section
  of the API Reference (closes :issue:`698`).

* fixed arithmetic operations between two sessions returning a nan value for each axis and group (closes :issue:`725`).
