Viewer 
------

* implemented "Copy to Excel" in context menu (Ctrl+E), to open the selection in a new Excel sheet directly, without
  the need to use paste. If nothing is selected, copies the whole array.
* when nothing is selected, Ctrl C selects & copies the whole array to the clipboard.
* when nothing is selected, Ctrl V paste at top-left corner
* implemented view(dict_with_array_values)

    >>> view({'a': array1, 'b': array2})

* fixed copy (ctrl-C) when viewing a 2D array: it did not include labels from the first axis in that case


Core
----

* implemented LArray.growth_rate to compute the growth along an axis

    >>> sex = Axis('sex', ['M', 'F'])
    >>> year = Axis('year', [2015, 2016, 2017])
    >>> a = ndrange([sex, year]).cumsum(x.year)
    >>> a
    sex\year | 2015 | 2016 | 2017
           M |    0 |    1 |    3
           F |    3 |    7 |   12
    >>> a.growth_rate()
    sex\year |          2016 |           2017
           M |           inf |            2.0
           F | 1.33333333333 | 0.714285714286
    >>> a.growth_rate(d=2)
    sex\year | 2017
           M |  inf
           F |  3.0

* implemented LArray.diff (difference along an axis)

    >>> sex = Axis('sex', ['M', 'F'])
    >>> xtype = Axis('type', ['type1', 'type2', 'type3'])
    >>> a = ndrange([sex, xtype]).cumsum(x.type)
    >>> a
    sex\type | type1 | type2 | type3
           M |     0 |     1 |     3
           F |     3 |     7 |    12
    >>> a.diff()
    sex\type | type2 | type3
           M |     1 |     2
           F |     4 |     5
    >>> a.diff(n=2)
    sex\type | type3
           M |     1
           F |     1
    >>> a.diff(x.sex)
    sex\type | type1 | type2 | type3
           F |     3 |     6 |     9

* implemented round() (as a nicer alias to around() and round_())

    >>> a = ndrange(5) + 0.5
    >>> a
    axis0 |   0 |   1 |   2 |   3 |   4
          | 0.5 | 1.5 | 2.5 | 3.5 | 4.5
    >>> round(a)
    axis0 |   0 |   1 |   2 |   3 |   4
          | 0.0 | 2.0 | 2.0 | 4.0 | 4.0

* implemented Session[['list', 'of', 'str']] to get a subset of a Session

    >>> s = Session({'a': ndrange(3), 'b': ndrange(4), 'c': ndrange(5)})
    >>> s
    Session(a, b, c)
    >>> s['a', 'c']
    Session(a, c)

* implemented LArray.points to do pointwise indexing instead of the default orthogonal indexing when indexing several
  dimensions at the same time.

    >>> a = Axis('a', ['a1', 'a2', 'a3'])
    >>> b = Axis('b', ['b1', 'b2', 'b3'])
    >>> arr = ndrange((a, b))
    >>> arr
    a\b | b1 | b2 | b3
    a1 |  0 |  1 |  2
    a2 |  3 |  4 |  5
    >>> arr[['a1', 'a3'], ['b1', 'b2']]
    a\b | b1 | b2
    a1 |  0 |  1
    a3 |  6 |  7
    # this selects the points ('a1', 'b1') and ('a3', 'b2')
    >>> arr.points[['a1', 'a3'], ['b1', 'b2']]
    a,b* | 0 | 1
         | 0 | 7

  Note that .ipoints (to do pointwise indexing with positions instead of labels – aka numpy indexing) is planned but not
  functional yet.

* made "arr1.drop_labels() * arr2" use the labels from arr2 if any

    >>> a = Axis('a', ['a1', 'a2'])
    >>> b = Axis('b', ['b1', 'b2'])
    >>> b2 = Axis('b', ['b2', 'b3'])
    >>> arr1 = ndrange([a, b])
    >>> arr1
    a\b | b1 | b2
    a1 |  0 |  1
    a2 |  2 |  3
    >>> arr1.drop_labels(b)
    a\b* | 0 | 1
      a1 | 0 | 1
      a2 | 2 | 3
    >>> arr1.drop_labels([a, b])
    a*\b* | 0 | 1
        0 | 0 | 1
        1 | 2 | 3
    >>> arr2 = ndrange([a, b2])
    >>> arr2
    a\b | b2 | b3
    a1 |  0 |  1
    a2 |  2 |  3
    >>> arr1 * arr2
    Traceback (most recent call last):
    ...
    ValueError: incompatible axes:
    Axis('b', ['b2', 'b3'])
    vs
    Axis('b', ['b1', 'b2'])
    >>> arr1 * arr2.drop_labels()
    a\b | b1 | b2
    a1 |  0 |  1
    a2 |  4 |  9
    # in versions < 0.11, it used to return:
    # >>> arr1.drop_labels() * arr2
    # a*\b* | 0 | 1
    #    0 | 0 | 1
    #    1 | 2 | 3
    >>> arr1.drop_labels() * arr2
    a\b | b2 | b3
    a1 |  0 |  1
    a2 |  4 |  9
    >>> arr1.drop_labels('a') * arr2.drop_labels('b')
    a\b | b1 | b2
    a1 |  0 |  1
    a2 |  4 |  9

* made .plot a property, like in Pandas, so that we can do stuff like:

    >>> a.plot.bar()
    # instead of
    >>> a.plot(kind='bar')

* made labels from different types not match against each other even if their value is the same. This might break some
  code but it is both more efficient and more convenient in some cases, so let us see how it goes:

    >>> a = ndrange(4)
    >>> a
    axis0 | 0 | 1 | 2 | 3
          | 0 | 1 | 2 | 3
    >>> a[1]
    1
    >>> # This used to "work" (and return 1)
    >>> a[True]
    …
    ValueError: True is not a valid label for any axis

    >>> a[1.0]
    …
    ValueError: 1.0 is not a valid label for any axis

* implemented read_csv(dialect='liam2') to read .csv files formatted like in LIAM2 (with the axes names on a separate
  line than the last axis labels)

* implemented Session[boolean LArray]

    >>> a = ndrange(3)
    >>> b = ndrange(4)
    >>> s1 = Session({'a': a, 'b': b})
    >>> s2 = Session({'a': a + 1, 'b': b})
    >>> s1 == s2
    name |     a |    b
         | False | True
    >>> s1[s1 == s2]
    Session(b)
    >>> s1[s1 != s2]
    Session(a)

* implemented experimental support for creating an array sequentially. Comments on the name of the function and syntax
  (especially compared to ndrange) would be appreciated.

    >>> year = Axis('year', range(2016, 2020))
    >>> sex = Axis('sex', ['M', 'F'])
    >>> create_sequential(year)
    year | 2016 | 2017 | 2018 | 2019
         |    0 |    1 |    2 |    3
    >>> create_sequential(year, 1.0, 0.1)
    year | 2016 | 2017 | 2018 | 2019
         |  1.0 |  1.1 |  1.2 |  1.3
    >>> create_sequential(year, 1.0, mult=1.1)
    year | 2016 | 2017 | 2018 |  2019
         |  1.0 |  1.1 | 1.21 | 1.331
    >>> inc = LArray([1, 2], [sex])
    >>> inc
    sex | M | F
        | 1 | 2
    >>> create_sequential(year, 1.0, inc)
    sex\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  3.0 |  4.0
           F |  1.0 |  3.0 |  5.0 |  7.0
    >>> mult = LArray([2, 3], [sex])
    >>> mult
    sex | M | F
        | 2 | 3
    >>> create_sequential(year, 1.0, mult=mult)
    sex\year | 2016 | 2017 | 2018 | 2019
           M |  1.0 |  2.0 |  4.0 |  8.0
           F |  1.0 |  3.0 |  9.0 | 27.0
    >>> initial = LArray([3, 4], [sex])
    >>> initial
    sex | M | F
        | 3 | 4
    >>> create_sequential(year, initial, inc, mult)
    sex\year | 2016 | 2017 | 2018 | 2019
           M |    3 |    7 |   15 |   31
           F |    4 |   14 |   44 |  134
    >>> def modify(prev_value):
    ...     return prev_value / 2
    >>> create_sequential(year, 8, func=modify)
    year | 2016 | 2017 | 2018 | 2019
         |    8 |    4 |    2 |    1
    >>> create_sequential(3)
    axis0* | 0 | 1 | 2
           | 0 | 1 | 2
    >>> create_sequential(x.year, axes=(sex, year))
    sex\year | 2016 | 2017 | 2018 | 2019
           M |    0 |    1 |    2 |    3
           F |    0 |    1 |    2 |    3

* implemented full and full_like to create arrays initialize to something else than zeros or ones

    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> full([nat, sex], 42.0)
    nat\sex |    M |    F
         BE | 42.0 | 42.0
         FO | 42.0 | 42.0
    >>> initial_value = ndrange([sex])
    >>> initial_value
    sex | M | F
        | 0 | 1
    >>> full([nat, sex], initial_value)
    nat\sex | M | F
         BE | 0 | 1
         FO | 0 | 1

* performance improvements when using label keys: a[key] is faster, especially if key is large


Fixes
-----

* to_excel(filepath) only closes the file if it was not open before
* removed code which forced labels from .csv files to be strings (as it caused problems in many cases, e.g. ages in
  LIAM2 files)


Misc. stuff for completeness
----------------------------

* made LGroups usable in Python's builtin range() and convertible to int and float
* implemented AxisCollection.union (equivalent to AxisCollection | Axis)
* fixed boolean array keys (boolean filter) in combination with scalar keys (for other dimensions)
* fixed support for older numpy
* fixed LArray.shift(n=0)
* still more work on making arrays with anonymous axes usable (not there yet) 
* added more tests
* better docstrings/error messages…
* misc. code cleanup/simplification/improved comments
