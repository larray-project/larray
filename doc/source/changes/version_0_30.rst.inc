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

* implemented :py:obj:`read_stata()` and :py:obj:`LArray.to_stata()` to read arrays from and write arrays to Stata .dta
  files.

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr.to_stata('test.dta')
    >>> read_stata('test.dta')
    {0}\{1}   a  b0  b1  b2
          0  a0   0   1   2
          1  a1   3   4   5
    >>> read_stata('test.dta', index_col='a')
    a\{1}  b0  b1  b2
       a0   0   1   2
       a1   3   4   5

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

* implemented :py:obj:`LArray.unique()` method to compute unique values for an array, optionally
  along axes.

    >>> arr = LArray([[0, 2, 0, 0],
    ...               [1, 1, 1, 0]], 'a=a0,a1;b=b0..b3')
    >>> arr
    a\b  b0  b1  b2  b3
     a0   0   2   0   0
     a1   1   1   1   0

    By default unique() returns the first occurrence of each unique value in the order it appears:

    >>> arr.unique()
    a_b  a0_b0  a0_b1  a1_b0
             0      2      1

    To sort the unique values, use the sort argument:

    >>> arr.unique(sort=True)
    a_b  a0_b0  a1_b0  a0_b1
             0      1      2

    One can also compute unique sub-arrays (i.e. combination of values) along axes. In our example the a0=0, a1=1
    combination appears twice along the 'b' axis, so 'b2' is not returned:

    >>> arr.unique('b')
    a\b  b0  b1  b3
     a0   0   2   0
     a1   1   1   0
    >>> arr.unique('b', sort=True)
    a\b  b3  b0  b1
     a0   0   0   2
     a1   0   1   1

* implemented :py:obj:`LArray.apply()` method to apply a python function or mapping to all
  elements of an array or to all sub-arrays along some axes of an array and return the result. This is an extremely
  versatile method as it can be used both with aggregating functions or element-wise functions.

  First let us define a test array

    >>> arr = LArray([[0, 2, 1],
    ...               [3, 1, 5]], 'a=a0,a1;b=b0..b2')
    >>> arr
    a\b  b0  b1  b2
     a0   0   2   1
     a1   3   1   5

  Here is a simple function we would like to apply to each element of the array.
  Note that this particular example should rather be written as: arr ** 2
  as it is both more concise and much faster.

    >>> def square(x):
    ...     return x ** 2
    >>> arr.apply(square)
    a\b  b0  b1  b2
     a0   0   4   1
     a1   9   1  25

  Now, assuming for a moment that the values of our test array above were in fact some numeric representation of
  names and we had the correspondence to the actual names stored in a dictionary:

    >>> code_to_names = {0: 'foo', 1: 'bar', 2: 'baz',
    ...                  3: 'boo', 4: 'far', 5: 'faz'}

  We could get back an array with the actual names by using:

    >>> arr.apply(code_to_names)
    a\b   b0   b1   b2
     a0  foo  baz  bar
     a1  boo  bar  faz

  Functions can also be applied along some axes:

    >>> # this is equivalent to (but much slower than): arr.sum_by('a')
    ... arr.apply(sum, 'a')
    a  a0  a1
        3   9

  Applying the function along some axes will return an array with the
  union of those axes and the axes of the returned values. For example,
  let us define a function which returns the k highest values of an array.

    >>> def topk(a, k=2):
    ...     return a.sort_values(ascending=False).ignore_labels().i[:k]
    >>> arr.apply(topk, 'a')
    a\b*  0  1
      a0  2  1
      a1  5  3

  Other arguments can be passed to the function as a tuple in the "args" argument:

    >>> arr.apply(topk, axes='a', args=(3,))
    a\b*  0  1  2
      a0  2  1  0
      a1  5  3  1

  or by using keyword arguments:

    >>> arr.apply(topk, axes='a', k=3)
    a\b*  0  1  2
      a0  2  1  0
      a1  5  3  1

* implemented :py:obj:`LArray.keys()` :py:obj:`LArray.values()` and :py:obj:`LArray.items()`
  methods to iterate (loop) on an array labels (keys), values or (key, value) pairs.

    >>> arr = ndtest((2, 2))
    >>> arr
    a\b  b0  b1
     a0   0   1
     a1   2   3

    By default they iterates on all axes, in the order they are in the array.

    >>> for value in arr.values():
    ...     print(value)
    0
    1
    2
    3
    >>> for key in arr.keys():
    ...     print(key)
    (a.i[0], b.i[0])
    (a.i[0], b.i[1])
    (a.i[1], b.i[0])
    (a.i[1], b.i[1])
    >>> for key, value in arr.items():
    ...     print(key, "->", value)
    (a.i[0], b.i[0]) -> 0
    (a.i[0], b.i[1]) -> 1
    (a.i[1], b.i[0]) -> 2
    (a.i[1], b.i[1]) -> 3

    You can iterate in reverse order.

    >>> for value in arr.values(ascending=False):
    ...     print(value)
    3
    2
    1
    0

    or specify another axis order:

    >>> for value in arr.values(('b', 'a')):
    ...     print(value)
    0
    2
    1
    3

    You can also iterate on less axes than the array has.

    >>> # iterate on the "b" axis, that is, return each label along the "b" axis in turn
    >>> for key in arr.keys('b'):
    ...     print(key)
    (b.i[0],)
    (b.i[1],)
    >>> # iterate on the array values along the "b" axis.
    >>> # That is, for each label along the "b" axis, return the corresponding (sub)array
    >>> for value in arr.values('b'):
    ...     print(value)
    a  a0  a1
        0   2
    a  a0  a1
        1   3

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

* py:obj:`stack()` can now stack along several axes at once.

    >>> gender = Axis('gender=M,F')
    >>> country = Axis('country=BE,FR,DE')
    >>> stack({('BE', 'M'): 0,
    ...        ('FR', 'F'): 2,
    ...        ('BE', 'F'): 2,
    ...        ('FR', 'M'): 2,
    ...        ('DE', 'M'): 2,
    ...        ('DE', 'F'): 2},
    ...       (gender, country))
    FIXME: this is not correct
    sex  nat\test   T1   T2
      M        BE  1.0  0.0
      M        FO  0.0  1.0
      F        BE  1.0  0.0
      F        FO  0.0  1.0


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


Fixes
-----

* fixed :py:obj:`LArray.divnot0()` being slow when the divisor has many axes and many zeros (closes :issue:`705`).

* fixed maximum length of sheet names (31 characters instead of 30 characters) when adding a new sheet
  to an Excel Workbook (closes :issue:`713`).

* fixed missing documentation of many functions in :ref:`Utility Functions <api-ufuncs>` section
  of the API Reference (closes :issue:`698`).

* fixed arithmetic operations between two sessions returning a nan value for each axis and group (closes :issue:`725`).
