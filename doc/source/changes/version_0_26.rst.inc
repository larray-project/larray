Syntax changes
--------------

* renamed special variable `x` to `X` to let users define an `x` variable in their code without breaking all
  subsequent code using that special variable (closes :issue:`167`).

* renamed Axis.startswith, endswith and matches to startingwith, endingwith and matching to avoid a possible confusion
  with str.startswith and endswith which return booleans (closes :issue:`432`).

* renamed `na` argument of `read_csv`, `read_excel`, `read_hdf` and `read_sas` functions to `fill_value` to avoid
  confusion as to what the argument does and to be consistent with `reindex` and `align` (closes :issue:`394`).

* renamed `split_axis` to `split_axes` to reflect the fact that it can now split several axes at once (see below).

* renamed PGroup as IGroup (I for Index).

* renamed several methods with more explicit names (closes :issue:`50`):

    - `sort_axis` to `sort_axes`
    - `argmax`, `argmin`, `argsort` to `labelofmax`, `labelofmin`, `labelsofsorted`
    - `posargmax`, `posargmin`, `posargsort` to `indexofmax`, `indexofmin`, `indicesofsorted`

New features
------------

* added global_arrays() and arrays() functions to complement the local_arrays() function. They return a Session
  containing respectively all arrays defined in global variables and all available arrays (whether they are defined in
  local or global variables).

  When used outside of a function, these three functions should have the same results, but inside a function
  local_arrays() will return only arrays local to the function, global_arrays() will return only arrays defined
  globally and arrays() will return arrays defined either locally or globally. Closes :issue:`416`.

* a `*` symbol is appended to the window title when unsaved changes are detected in the viewer (closes :editor_issue:`21`).

* implemented Axis.containing to create a Group with all labels of an axis containing some substring (closes
  :issue:`402`).

    >>> people = Axis(['Bruce Wayne', 'Bruce Willis', 'Arthur Dent'], 'people')
    >>> people.containing('Will')
    people['Bruce Willis']

* implemented Group.containing, startingwith, endingwith and matching to create a group with all labels of a group
  matching some criterion (closes :issue:`108`).

    >>> group = people.startingwith('Bru')
    >>> group
    people['Bruce Wayne', 'Bruce Willis']
    >>> group.containing('Will')
    people['Bruce Willis']

* implemented nan_equal() function to create an array of booleans telling whether each cell of the first array is
  equal to the corresponding cell in the other array, even in the presence of NaN.

    >>> arr1 = ndtest(3, dtype=float)
    >>> arr1['a1'] = nan
    >>> arr1
    a   a0   a1   a2
       0.0  nan  2.0
    >>> arr2 = arr1.copy()
    >>> arr1 == arr2
    a    a0     a1    a2
       True  False  True
    >>> nan_equal(arr1, arr2)
    a    a0    a1    a2
       True  True  True

* implemented from_frame() to convert a Pandas DataFrame to an array:

    >>> df = ndtest((2, 2, 2)).to_frame()
    >>> df
    c      c0  c1
    a  b
    a0 b0   0   1
       b1   2   3
    a1 b0   4   5
       b1   6   7
    >>> from_frame(df)
     a  b\\c  c0  c1
    a0   b0   0   1
    a0   b1   2   3
    a1   b0   4   5
    a1   b1   6   7


Miscellaneous improvements
--------------------------

* view() and edit() without argument now display global arrays in addition to local ones (closes :editor_issue:`54`).

* inverted background colors in the viewer (red for low values and blue for high values). Closes :editor_issue:`18`.

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

* allowed to pass a Group to `read_excel`/`read_hdf` as `sheetname`/`key` argument (closes :issue:`439`).

    >>> a, b, c = arr.a, arr.b, arr.c

    >>> # For Excel
    >>> new_from_excel = zeros((a, b, c), dtype=int)
    >>> for label in a:
    ...     new_from_excel[label] = read_excel('my_file.xlsx', label)
    >>> # But, to avoid loading the file in Excel repeatedly (which is very inefficient),
    >>> # this particular example should rather be written like this:
    >>> new_from_excel = zeros((a, b, c), dtype=int)
    >>> with open_excel('my_file.xlsx') as wb:
    ...     for label in a:
    ...         new_from_excel[label] = wb[label].load()

    >>> # For HDF
    >>> new_from_hdf = zeros((a, b, c), dtype=int)
    >>> for label in a:
    ...     new_from_hdf[label] = read_hdf('my_file.h5', label)

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

* `split_axes` (formerly `split_axis`) now allows to split several axes at once (closes :issue:`366`):

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

* added data type (dtype) to array.info (closes :issue:`454`):

    >>> arr = ndtest((2, 2), dtype=float)
    >>> arr
    a\b   b0   b1
     a0  0.0  1.0
     a1  2.0  3.0
    >>> arr.info
    2 x 2
     a [2]: 'a0' 'a1'
     b [2]: 'b0' 'b1'
    dtype: float64

* To create a 1D array using from_string() and the default separator ' ', a tabulation character `\t`
  (instead of - previously) must be added in front of the data line:

    >>> from_string('''sex  M  F
    ...                \t   0  1''')
    sex  M  F
         0  1

* title in the viewer also includes the dtype of the current displayed array (closes :editor_issue:`85`)


Fixes
-----

* fixed array creation with axis(es) given as string containing only one label (axis name and label were inverted).

* fixed reading an array from a CSV or Excel file when the columns axis is not explicitly named (via `\`).
  For example, let's say we want to read a CSV file 'pop.csv' with the following content (indented for clarity) ::

  sex, 2015, 2016
    F,   11,   13
    M,   12,   10

  The result of function `read_csv` is:

    >>> pop = read_csv('pop.csv')
    >>> pop
    sex\{1}  2015  2016
          F    11    13
          M    12    10

  Closes :issue:`372`.

* fixed converting a 1xN Pandas DataFrame to an array using `aslarray` (closes :issue:`427`):

    >>> df = pd.DataFrame([[1, 2, 3]], index=['a0'], columns=['b0', 'b1', 'b2'])
    >>> df
        b0  b1  b2
    a0   1   2   3
    >>> aslarray(df)
    {0}\{1}  b0  b1  b2
         a0   1   2   3

    >>> # setting name to index and columns
    >>> df.index.name = 'a'
    >>> df.columns.name = 'b'
    >>> df
    b   b0  b1  b2
    a
    a0   1   2   3
    >>> aslarray(df)
    a\b  b0  b1  b2
     a0   1   2   3

* fixed original file being deleted when trying to overwrite a file via `Session.save` or `open_excel` failed
  (closes :issue:`441`)

* fixed loading arrays from Excel sheets containing blank cells below or right of the array to read
  (closes :issue:`443`)

* fixed unary and binary operations between sessions failing entirely when the operation failed/was invalid on any of
  the array. Now the result will be nan for that array but the operation will carry on for other arrays.

* fixed stacking arrays with anonymous axes.

* fixed background color in the viewer when using filters in the `compare()` dialog (closes :editor_issue:`66`)

* fixed autoresize of columns by double clicking between column headers (closes :editor_issue:`43`)

* fixed representing a 0D array (scalar) in the viewer (closes :editor_issue:`71`)

* fixed viewer not displaying an error message when saving or loading a file failed (closes :editor_issue:`75`)
