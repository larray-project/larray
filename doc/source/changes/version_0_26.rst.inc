New features
------------

* added global_arrays() function which returns a Session containing all arrays defined in global variables. This
  complements the local_arrays() function. When used outside of a function, these two functions should have the same
  results, but inside a function local_arrays will return only arrays local to the function, while global_arrays will
  return only arrays defined globally.


Miscellaneous improvements
--------------------------

* view() and edit() without argument now display global arrays in addition to local ones (closes :editor_issue:`54`).

* renamed special variable `x` as `X` (closes :issue:`167`).

* allowed to pass an array of labels as `new_axis` argument to `reindex` method (closes :issue:`384`):

    >>> arr = ndrange('a=v0..v1;b=v0..v2')
    >>> arr
    a\b  v0  v1  v2
     v0   0   1   2
     v1   3   4   5
    >>> arr.reindex('a', arr.b.labels)
    a\b   v0   v1   v2
     v0    0    1    2
     v1    3    4    5
     v2  nan  nan  nan

* allowed to call the `reindex` method using a differently named axis for labels (closes :issue:`386`):

    >>> arr = ndrange('a=v0..v1;b=v0..v2')
    >>> arr
    a\b  v0  v1  v2
     v0   0   1   2
     v1   3   4   5
    >>> arr.reindex('a', arr.b)
    a\b   v0   v1   v2
     v0    0    1    2
     v1    3    4    5
     v2  nan  nan  nan

* replaced `na` argument of `read_csv`, `read_excel`, `read_hdf` and `read_sas` functions by `fill_value`
  (closes :issue:`394`).

* arguments `fill_value`, `sort_rows` and `sort_columns` of `read_excel` function are also supported by the default
  `xlwings` engine (closes :issue:`393`).

* allowed to pass a label or group as `sheet_name` argument of the method `to_excel` or to a Workbook (`open_excel`).
  Same for `key` argument of the method `to_hdf`. Closes :issue:`328`.

    >>> arr = ndtest((4, 4, 4))

    >>> # iterate over labels of a given axis
    >>> with open_excel('my_file.xlsx') as wb:
    >>>     for label in arr.a:
    ...         wb[label] = arr[label].dump()
    ...     wb.save()
    >>> for label in arr.a:
    ...     arr[label].to_hdf('my_file.h5', label)

    >>> # create and use a group
    >>> even = arr.a['a0,a2'] >> 'even'
    >>> arr[even].to_excel('my_file.xlsx', even)
    >>> arr[even].to_hdf('my_file.h5', even)

    >>> # special characters : \ / ? * [ or ] in labels or groups are replaced by an _ when exporting to excel
    >>> # sheet names cannot exceed 31 characters
    >>> g = arr.a['a1,a3,a4'] >> '?name:with*special\/[char]'
    >>> arr[g].to_excel('my_file.xlsx', g)
    >>> print(open_excel('my_file.xlsx').sheet_names())
    ['_name_with_special___char_']
    >>> # special characters \ or / in labels or groups are replaced by an _ when exporting to HDF file

* allowed setting the name of a Group using another Group or Axis (closes :issue:`341`):

    >>> arr = ndrange('axis=a,a0..a3,b,b0..b3,c,c0..c3')
    >>> arr
    axis  a  a0  a1  a2  a3  b  b0  b1  b2  b3   c  c0  c1  c2  c3
          0   1   2   3   4  5   6   7   8   9  10  11  12  13  14
    >>> # matches('^.$') will select labels with only one character: 'a', 'b' and 'c'
    >>> groups = tuple(arr.axis.startswith(code) >> code for code in arr.axis.matches('^.$'))
    >>> groups
    (axis['a', 'a0', 'a1', 'a2', 'a3'] >> 'a',
     axis['b', 'b0', 'b1', 'b2', 'b3'] >> 'b',
     axis['c', 'c0', 'c1', 'c2', 'c3'] >> 'c')
    >>> arr.sum(groups)
    axis   a   b   c
          10  35  60

* allowed to test if an array contains a label using the `in` operator (closes :issue:`343`):

    >>> arr = ndrange('age=0..99;sex=M,F')
    >>> 'M' in arr
    True
    >>> 'Male' in arr
    False
    >>> # this can be useful for example in an 'if' statement
    >>> if 102 not in arr:
    ...     # with 'reindex', we extend 'age' axis to 102
    ...     arr = arr.reindex('age', Axis('age=0..102'), fill_value=0)
    >>> arr.info
    103 x 2
     age [103]: 0 1 2 ... 100 101 102
     sex [2]: 'M' 'F'

* allowed to create a group on an axis using labels of another axis (closes :issue:`362`):

    >>> year = Axis('year=2000..2017')
    >>> even_year = Axis(range(2000, 2017, 2), 'even_year')
    >>> group_even_year = year[even_year]
    >>> group_even_year
    year[2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016]

* renamed `split_axis` as `split_axes` which now allows to split several axes at once (closes :issue:`366`):

    >>> combined = ndrange('a_b = a0_b0..a1_b1; c_d = c0_d0..c1_d1')
    >>> combined
    a_b\c_d  c0_d0  c0_d1  c1_d0  c1_d1
      a0_b0      0      1      2      3
      a0_b1      4      5      6      7
      a1_b0      8      9     10     11
      a1_b1     12     13     14     15
    >>> combined.split_axes(['a_b', 'c_d'])
     a   b  c\d  d0  d1
    a0  b0   c0   0   1
    a0  b0   c1   2   3
    a0  b1   c0   4   5
    a0  b1   c1   6   7
    a1  b0   c0   8   9
    a1  b0   c1  10  11
    a1  b1   c0  12  13
    a1  b1   c1  14  15
    >>> combined.split_axes({'a_b': ('A', 'B'), 'c_d': ('C', 'D')})
     A   B  C\D  d0  d1
    a0  b0   c0   0   1
    a0  b0   c1   2   3
    a0  b1   c0   4   5
    a0  b1   c1   6   7
    a1  b0   c0   8   9
    a1  b0   c1  10  11
    a1  b1   c0  12  13
    a1  b1   c1  14  15

* argument `axes` of `split_axes` has become optional: defaults to all axes whose name contains the specified delimiter
  (closes :issue:`365`):

    >>> combined = ndrange('a_b = a0_b0..a1_b1; c_d = c0_d0..c1_d1')
    >>> combined
    a_b\c_d  c0_d0  c0_d1  c1_d0  c1_d1
      a0_b0      0      1      2      3
      a0_b1      4      5      6      7
      a1_b0      8      9     10     11
      a1_b1     12     13     14     15
    >>> combined.split_axes()
     a   b  c\d  d0  d1
    a0  b0   c0   0   1
    a0  b0   c1   2   3
    a0  b1   c0   4   5
    a0  b1   c1   6   7
    a1  b0   c0   8   9
    a1  b0   c1  10  11
    a1  b1   c0  12  13
    a1  b1   c1  14  15

* allowed to perform several axes combinations at once with the `combine_axes()` method (closes :issue:`382`):

    >>> arr = ndtest((2, 2, 2, 2))
    >>> arr
     a   b  c\d  d0  d1
    a0  b0   c0   0   1
    a0  b0   c1   2   3
    a0  b1   c0   4   5
    a0  b1   c1   6   7
    a1  b0   c0   8   9
    a1  b0   c1  10  11
    a1  b1   c0  12  13
    a1  b1   c1  14  15
    >>> arr.combine_axes([('a', 'c'), ('b', 'd')])
    a_c\b_d  b0_d0  b0_d1  b1_d0  b1_d1
      a0_c0      0      1      4      5
      a0_c1      2      3      6      7
      a1_c0      8      9     12     13
      a1_c1     10     11     14     15
    >>> # set output axes names by passing a dictionary
    >>> arr.combine_axes({('a', 'c'): 'ac', ('b', 'd'): 'bd'})
    ac\bd  b0_d0  b0_d1  b1_d0  b1_d1
    a0_c0      0      1      4      5
    a0_c1      2      3      6      7
    a1_c0      8      9     12     13
    a1_c1     10     11     14     15

* allowed to use keyword arguments in `set_labels` (closes :issue:`383`):

    >>> a = ndrange('nat=BE,FO;sex=M,F')
    >>> a
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> a.set_labels(sex='Men,Women', nat='Belgian,Foreigner')
      nat\sex  Men  Women
      Belgian    0      1
    Foreigner    2      3

* allowed passing an axis to `set_labels` as 'labels' argument (closes :issue:`408`).

Fixes
-----

* fixed array creation with axis(es) given as string containing only one label (axis name and label were inversed).
