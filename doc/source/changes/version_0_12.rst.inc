New features
------------

* implemented boolean indexing by using axes objects:

    >>> sex = Axis('sex', 'M,F')
    >>> age = Axis('age', range(5))
    >>> a = ndrange((sex, age))
    >>> a
    sex\age | 0 | 1 | 2 | 3 | 4
          M | 0 | 1 | 2 | 3 | 4
          F | 5 | 6 | 7 | 8 | 9

    >>> a[age < 3]
    sex\age | 0 | 1 | 2
          M | 0 | 1 | 2
          F | 5 | 6 | 7

  This new syntax is equivalent to (but currently much slower than):

    >>> a[age[:2]]
    sex\age | 0 | 1 | 2
          M | 0 | 1 | 2
          F | 5 | 6 | 7

  However, the power of this new syntax comes from the fact that you are not limited to scalar constants

    >>> age_limit = LArray([2, 3], sex)
    >>> age_limit
    sex | M | F
        | 2 | 3

    >>> a[age < age_limit]
    sex,age | M,0 | M,1 | F,0 | F,1 | F,2
            |   0 |   1 |   5 |   6 |   7

  Notice that the concerned axes are merged, so you cannot do much as much with them. For example,
  `a[age < age_limit].sum(x.age)` would not work since there is no "age" axis anymore.

  To keep axes intact, one can often set the values of the corresponding cells to 0 or nan instead.

    >>> a[age < age_limit] = 0
    >>> a
    sex\age | 0 | 1 | 2 | 3 | 4
          M | 0 | 0 | 2 | 3 | 4
          F | 0 | 0 | 0 | 8 | 9
    >>> # in this case, the sum *is* valid (but the mean would not -- one should use nan for that)
    >>> a.sum(x.age)
    sex | M |  F
        | 9 | 17

  To keep axes intact, this idiom is also often useful:

    >>> b = a * (age >= age_limit)
    >>> b
    sex\age | 0 | 1 | 2 | 3 | 4
          M | 0 | 0 | 2 | 3 | 4
          F | 0 | 0 | 0 | 8 | 9

  This also works with axes references (x.axis_name), though this is experimental and the filter value is only computed
  as late as possible (during []), so you cannot display it before that, like you can with "real" axes.

  Using "real" axes:

    >>> filter1 = age < age_limit
    >>> filter1
    age\sex |     M |     F
          0 |  True |  True
          1 |  True |  True
          2 | False |  True
          3 | False | False
          4 | False | False
    >>> a[filter1]
    sex,age | M,0 | M,1 | F,0 | F,1 | F,2
            |   0 |   1 |   5 |   6 |   7

  With axes references:

    >>> filter2 = x.age < age_limit
    >>> filter2
    <larray.core.BinaryOp at 0x1332ae3b588>
    >>> a[filter2]
    sex,age | M,0 | M,1 | F,0 | F,1 | F,2
            |   0 |   1 |   5 |   6 |   7
    >>> a * ~filter2
    sex\age | 0 | 1 | 2 | 3 | 4
          M | 0 | 0 | 2 | 3 | 4
          F | 0 | 0 | 0 | 8 | 9

* implemented LArray.divnot0

    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> a = ndrange((nat, sex))
    >>> a
    nat\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> b = ndrange(sex)
    >>> b
    sex | M | F
        | 0 | 1
    >>> a / b
    nat\sex |   M |   F
         BE | nan | 1.0
         FO | inf | 3.0
    >>> a.divnot0(b)
    nat\sex |   M |   F
         BE | 0.0 | 1.0
         FO | 0.0 | 3.0

* implemented .named() on groups to name groups after the fact

    >>> a = ndrange(Axis('age', range(100)))
    >>> a
    age | 0 | 1 | 2 | 3 | 4 | 5 | 6 | ... | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99
        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | ... | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99
    >>> a.sum((x.age[10:19].named('teens'), x.age[20:29].named('twenties')))
    age | 'teens' (10:19) | 'twenties' (20:29)
        |             145 |                245

* made all array creation functions (ndrange, zeros, ones, full, LArray, …) more flexible:

  They accept a single Axis argument instead of requiring a tuple/list of them

    >>> sex = Axis('sex', 'M,F')
    >>> a = ndrange(sex)
    >>> a
    sex | M | F
        | 0 | 1

  Shortcut definition for axes work

    >>> ndrange("a,b,c")
    {0} | a | b | c
        | 0 | 1 | 2
    >>> ndrange(["1:3", "d,e"])
    {0}\{1} | d | e
          1 | 0 | 1
          2 | 2 | 3
          3 | 4 | 5
    >>> LArray([1, 5, 7], "a,b,c")
    {0} | a | b | c
        | 1 | 5 | 7

  One can mix Axis objects and ints (for axes without labels)

    >>> sex = Axis('sex', 'M,F')
    >>> ndrange([sex, 3])
    sex\{1}* | 0 | 1 | 2
           M | 0 | 1 | 2
           F | 3 | 4 | 5

* made it possible to iterate on labels of a group (eg a slice of an Axis):

    >>> for year in a.axes.year[2010:]:
    ...     # do stuff

* changed representation of anonymous axes from "axisN" (where N is the position of the axis) to "{N}". 
  The problem was that "axisN" was not recognizable enough as an anonymous axis, and it was thus misleading. For
  example "a[x.axis0[...]]" would not work.

* better overall support for arrays with anonymous axes or several axes with the same name

* fixed all output functions (to_csv, to_excel, to_hdf, …) when the last axis has no name but other axes have one

* implemented eye() which creates 2D arrays with ones on the diagonal and zeros elsewhere.

    >>> eye(sex)
    sex\sex |   M |   F
          M | 1.0 | 0.0
          F | 0.0 | 1.0

* implemented the @ operator to do matrix multiplication (Python3.5+ only)

* implemented inverse() to return the (matrix) inverse of a (square) 2D array

    >>> a = eye(sex) * 2
    >>> a
    sex\sex |   M |   F
          M | 2.0 | 0.0
          F | 0.0 | 2.0

    >>> a @ inverse(a)
    sex\sex |   M |   F
          M | 1.0 | 0.0
          F | 0.0 | 1.0

* implemented diag() to extract a diagonal or construct a diagonal array.

    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> a = ndrange([nat, sex], start=1)
    >>> a
    nat\sex | M | F
         BE | 1 | 2
         FO | 3 | 4
    >>> d = diag(a)
    >>> d
    nat,sex | BE,M | FO,F
            |    1 |    4
    >>> diag(d)
    nat\sex | M | F
         BE | 1 | 0
         FO | 0 | 4
    >>> a = ndrange(sex, start=1)
    >>> a
    sex | M | F
        | 1 | 2
    >>> diag(a)
    sex\sex | M | F
          M | 1 | 0
          F | 0 | 2


For completeness
----------------

* added Axis.rename method which returns a copy of the axis with a different name and deprecate Axis._rename

* added labels_array as a generalized version of identity (which is deprecated)

* implemented LArray.ipoints[...] to do point selection using coordinates instead of labels (aka numpy indexing)

* raise an error when trying to do a[key_with_more_axes_than_a] = value instead of silently ignoring extra axes.

* allow using a single int for index_col in read_csv in addition to a list of ints 

* implemented __getitem__ for "x". You can now write stuff like:

    >>> a = ndrange((3, 4))
    >>> a[x[0][1:]]
    {0}\{1}* | 0 | 1 |  2 |  3
           1 | 4 | 5 |  6 |  7
           2 | 8 | 9 | 10 | 11
    >>> a[x[1][2:]]
    {0}*\{1} |  2 |  3
           0 |  2 |  3
           1 |  6 |  7
           2 | 10 | 11
    >>> a.sum(x[0])
    {0}* |  0 |  1 |  2 |  3
         | 12 | 15 | 18 | 21

* produce normal axes instead of wildcard axes on LArray.points[...]. This is (much) slower but more correct/informative.

* changed the way we store axes internally, which has several consequences

    - better overall support for anonymous axes
    - better support for arrays with several axes with the same name
    - small performance improvement
    - the same axis object cannot be added twice in an array (one should use axis.copy() if that need arises)
    - changes the way groups with an axis are displayed

* fixed sum, min, max functions on non-LArray arguments

* changed __repr__ for wildcard axes to not display their labels but their length

    >>> ndrange(3).axes[0]
    Axis(None, 3)

* fixed aggregates on several groups "forgetting" the name of groups which had been created using axis.all()

* allow Axis(..., long) in addition to int (Python2 only)

* better docstrings/tests/comments/error messages/thoughts/…
