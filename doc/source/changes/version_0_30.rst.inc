.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* :py:obj:`LArray.as_table()` is deprecated. Please use :py:obj:`LArray.dump()` instead.

* :py:obj:`stack()` ``axis`` argument was renamed to ``axes`` to reflect the fact that the function can now stack
  along multiple axes at once (see below).


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

* implemented :py:obj:`read_stata()` and :py:obj:`LArray.to_stata()` to read arrays from and write arrays to Stata .dta
  files.

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

* implemented :py:obj:`LArray.keys()`, :py:obj:`LArray.values()` and :py:obj:`LArray.items()`
  methods to respectively loop on an array labels, values or (key, value) pairs.

* implemented :py:obj:`LArray.unique()` method to compute unique values (or sub-arrays) for an array,
  optionally along axes.

* implemented :py:obj:`LArray.apply()` method to apply a python function to all values of an array or to all sub-arrays
  along some axes of an array and return the result. This is an extremely versatile method as it can be used both with
  aggregating functions or element-wise functions.

  First let us define a test array

    >>> arr = LArray([[0, 2, 1],
    ...               [3, 1, 5]], 'a=a0,a1;b=b0..b2')
    >>> arr
    a\b  b0  b1  b2
     a0   0   2   1
     a1   3   1   5

  Here is a simple function we would like to apply to each element of the array. Note that this particular example
  should rather be written as: arr ** 2 as it is both more concise and much faster.

    >>> def square(x):
    ...     return x ** 2
    >>> arr.apply(square)
    a\b  b0  b1  b2
     a0   0   4   1
     a1   9   1  25

  Functions can also be applied along some axes:

    >>> # this is equivalent to (but much slower than): arr.sum_by('a')
    ... arr.apply(sum, by='a')
    a  a0  a1
        3   9

  Applying the function along some axes will return an array with the
  union of those axes and the axes of the returned values. For example,
  let us define a function which returns the k highest values of an array.

    >>> def topk(a, k=2):
    ...     return a.sort_values(ascending=False).ignore_labels().i[:k]
    >>> arr.apply(topk, by='a')
    a\b*  0  1
      a0  2  1
      a1  5  3

  Extra positional arguments given to apply are passed to the function:

    >>> arr.apply(topk, 3, by='a')
    a\b*  0  1  2
      a0  2  1  0
      a1  5  3  1

  as well as (unknown) keyword arguments:

    >>> arr.apply(topk, by='a', k=3)
    a\b*  0  1  2
      a0  2  1  0
      a1  5  3  1

  If the function returns several values (as a tuple), the result will be a tuple of arrays. For example,
  let use define a function which decompose an array in its mean and the difference to that mean :

    >>> def mean_decompose(a):
    ...     mean = a.mean()
    ...     return mean, a - mean
    >>> mean_by_a, diff_to_mean = arr.apply(mean_decompose, by='a')
    >>> mean_by_a
    a   a0   a1
       1.0  3.0
    >>> diff_to_mean
    a\b    b0    b1   b2
     a0  -1.0   1.0  0.0
     a1   0.0  -2.0  2.0

* implemented :py:obj:`LArray.apply_map()` method to apply a transformation mapping to array elements. For example, this
  can be used to transform some numeric codes to labels.

* implemented :py:obj:`Axis.apply()` method to transform an axis labels by a function and return a new Axis.

    >>> sex = Axis('sex=MALE,FEMALE')
    >>> sex.apply(str.capitalize)
    Axis(['Male', 'Female'], 'sex')

* added :py:obj:`Session.update()` method to add and modify items from an existing session by passing
  either another session or a dict-like object or an iterable object with (key, value) pairs (closes :issue:`754`).

* implemented :py:obj:`wrap_elementwise_array_func()` function to make a function defined in another library work with
  LArray arguments instead of with numpy arrays.

* implemented :py:obj:`LArray.roll()` to roll the cells of an array n-times to the right along an axis. This is similar
  to :py:obj:`LArray.shift()`, except that cells which are pushed "outside of the axis" are reintroduced on the opposite
  side of the axis instead of being dropped.

* implemented :py:obj:`AxisCollection.rename()` to rename axes of an AxisCollection, independently of any array.

* implemented :py:obj:`zip_array_values()` and :py:obj:`zip_array_items()` to loop respectively on several arrays values
  or (key, value) pairs.

* implemented :py:obj:`AxisCollection.iter_labels()` to iterate over all (possible combinations of) labels of the axes
  of the collection.


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

* py:obj:`stack()` can now stack along several axes at once (closes :issue:`56`).

    >>> country = Axis('country=BE,FR,DE')
    >>> gender = Axis('gender=M,F')
    >>> stack({('BE', 'M'): 0,
    ...        ('BE', 'F'): 1,
    ...        ('FR', 'M'): 2,
    ...        ('FR', 'F'): 3,
    ...        ('DE', 'M'): 4,
    ...        ('DE', 'F'): 5},
    ...       (country, gender))
    country\gender  M  F
                BE  0  1
                FR  2  3
                DE  4  5

* py:obj:`stack()` using a dictionary as elements can now use a simple axis name instead of requiring a full axis
  object. This will print a warning on Python < 3.7 though because the ordering of labels is not guaranteed in
  that case. Closes :issue:`755` and :issue:`581`.

* py:obj:`stack()` using keyword arguments can now use a simple axis name instead of requiring a full axis
  object, even on Python < 3.6. This will print a warning though because the ordering of labels is not guaranteed in
  that case.

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

* added dtype argument to stack to set the type of the resulting array explicitly instead of relying on auto-detection.

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

* :py:obj:`LArray.dump()` gained a few extra arguments to further customize output :
  - axes_names : to specify whether or not the output should contain the axes names (and which)
  - maxlines and edgeitems : to dump only the start and end of large arrays
  - light : to output axes labels only when they change instead of repeating them on each line
  - na_repr : to specify how to represent N/A (NaN) values

* substantially improved performance of creating, iterating, and doing a few other operations over larray objects.
  This solves a few pathological cases of slow operations, especially those involving many small-ish arrays but sadly
  the overall performance improvement is negligible over most of the real-world models using larray that we tested these
  changes on.


Fixes
^^^^^

* fixed dumping to Excel arrays of "object" dtype containing NaN values using numpy float types (fixes the
  infamous 65535 bug).

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
