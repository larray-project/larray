.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* new syntax


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^

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

* implemented :py:obj:`Axis.apply()` method to transform an axis labels by a function and return a new Axis.

    >>> sex = Axis('sex=MALE,FEMALE')
    >>> sex.apply(str.capitalize)
    Axis(['Male', 'Female'], 'sex')

* added :py:obj:`Session.update()` method to add and modify items from an existing session by passing
  either another session or a dict-like object or an iterable object with (key, value) pairs (closes :issue:`754`).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

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

* added dtype argument to LArray to set the type of the array explicitly instead of relying on auto-detection.

* implemented :py:obj:`LArray.reverse()` method to reverse one or several axes of an array (closes :issue:`631`).

* added :py:obj:`set_options` allowing to set options for larray within a ``with`` block or globally:

    >>> from larray import *
    >>> arr = ndtest((500, 100), dtype=float) + 0.123456

    You can use ``set_options`` either as a context manager:

    >>> # display_width defines the maximum display width when printing an array
    >>> # display_edgeitems defines the number of lines of the header and tail to display
    >>> with set_options(display_width=100, display_edgeitems=2):
    ...     print(arr)
     a\b            b0            b1            b2  ...           b97           b98           b99
      a0      0.123456      1.123456      2.123456  ...     97.123456     98.123456     99.123456
      a1    100.123456    101.123456    102.123456  ...    197.123456    198.123456    199.123456
     ...           ...           ...           ...  ...           ...           ...           ...
    a498  49800.123456  49801.123456  49802.123456  ...  49897.123456  49898.123456  49899.123456
    a499  49900.123456  49901.123456  49902.123456  ...  49997.123456  49998.123456  49999.123456

    Or to set global options:

    >>> # display_maxlines defines the maximum number of lines to show
    >>> # display_precision defines the number of digits of precision for floating point output
    >>> set_options(display_maxlines=10, display_precision=2)
    >>> print(arr)
     a\b        b0        b1        b2  ...       b97       b98       b99
      a0      0.12      1.12      2.12  ...     97.12     98.12     99.12
      a1    100.12    101.12    102.12  ...    197.12    198.12    199.12
      a2    200.12    201.12    202.12  ...    297.12    298.12    299.12
      a3    300.12    301.12    302.12  ...    397.12    398.12    399.12
      a4    400.12    401.12    402.12  ...    497.12    498.12    499.12
     ...       ...       ...       ...  ...       ...       ...       ...
    a495  49500.12  49501.12  49502.12  ...  49597.12  49598.12  49599.12
    a496  49600.12  49601.12  49602.12  ...  49697.12  49698.12  49699.12
    a497  49700.12  49701.12  49702.12  ...  49797.12  49798.12  49799.12
    a498  49800.12  49801.12  49802.12  ...  49897.12  49898.12  49899.12
    a499  49900.12  49901.12  49902.12  ...  49997.12  49998.12  49999.12

    To put back the default options, you can use:

    >>> set_options(display_precision=None, display_width=80, display_maxlines=200, display_edgeitems=5)

  The :py:obj:`get_options` function returns a view of the current options as a dictionary:

    >>> get_options()
    {'display_precision': None, 'display_width': 80, 'display_maxlines': 200, 'display_edgeitems': 5}

  Closes :issue:`274`.

* improved speed of :py:obj:`read_hdf()` function when reading a stored LArray object dumped with
  the current and future version of larray. To get benefit of the speedup of reading arrays dumped
  with older versions of larray, please read and re-dump them. Closes :issue:`563`.

* allowed to not specifiy the axes in :py:obj:`LArray.set_labels()` (closes :issue:`634`):

    >>> a = ndtest('nat=BE,FO;sex=M,F')
    >>> a
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> a.set_labels({'M': 'Men', 'BE': 'Belgian'})
    nat\sex  Men  F
    Belgian    0  1
         FO    2  3

* allowed to pass a single axis or group as ``axes_to_reindex`` argument
  of the :py:obj:`LArray.reindex()` method (closes :issue:`712`).


Fixes
^^^^^

* fixed :py:obj:`LArray.divnot0()` being slow when the divisor has many axes and many zeros (closes :issue:`705`).

* fixed maximum length of sheet names (31 characters instead of 30 characters) when adding a new sheet
  to an Excel Workbook (closes :issue:`713`).

* fixed missing documentation of many functions in :ref:`Utility Functions <api-ufuncs>` section
  of the API Reference (closes :issue:`698`).

* fixed arithmetic operations between two sessions returning a nan value for each axis and group (closes :issue:`725`).

* fixed dumping sessions with metadata in HDF format (closes :issue:`702`).

* fixed minimum version of pandas to install. The minimum version is now 0.20.0.

* fixed from_frame for dataframes with non string index names.

* fixed creating an LSet from an IGroup with a (single) scalar key

    >>> a = Axis('a=a0,a1,a2')
    >>> a.i[1].set()
    a['a1'].set()
