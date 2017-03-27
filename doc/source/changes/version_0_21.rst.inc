New features
------------

* implemented set_axes() method to replace one, several or all axes of an array (closes :issue:`67`).
  The method with_axes() is now deprecated (set_axes() must be used instead).

    >>> arr = ndtest((2, 3))
    >>> arr
    a\\b | b0 | b1 | b2
     a0 |  0 |  1 |  2
     a1 |  3 |  4 |  5
    >>> row = Axis('row', ['r0', 'r1'])
    >>> column = Axis('column', ['c0', 'c1', 'c2'])

    Replace one axis (second argument `new_axis` must be provided)

    >>> arr.set_axes(x.a, row)
    row\\b | b0 | b1 | b2
       r0 |  0 |  1 |  2
       r1 |  3 |  4 |  5

    Replace several axes (keywords, list of tuple or dictionary)

    >>> arr.set_axes(a=row, b=column)
    or
    >>> arr.set_axes([(x.a, row), (x.b, column)])
    or
    >>> arr.set_axes({x.a: row, x.b: column})
    row\\column | c0 | c1 | c2
            r0 |  0 |  1 |  2
            r1 |  3 |  4 |  5

    Replace all axes (list of axes or AxisCollection)

    >>> arr.set_axes([row, column])
    row\\column | c0 | c1 | c2
            r0 |  0 |  1 |  2
            r1 |  3 |  4 |  5
    >>> arr2 = ndrange([row, column])
    >>> arr.set_axes(arr2.axes)
    row\\column | c0 | c1 | c2
            r0 |  0 |  1 |  2
            r1 |  3 |  4 |  5

* implemented Axis.replace to replace some labels from an axis:

    >>> sex = Axis('sex', ['M', 'F'])
    >>> sex
    Axis('sex', ['M', 'F'])
    >>> sex.replace('M', 'Male')
    Axis('sex', ['Male', 'F'])
    >>> sex.replace({'M': 'Male', 'F': 'Female'})
    Axis('sex', ['Male', 'Female'])

* implemented from_string() method to create an array from a string (closes :issue:`96`).

    >>> from_string('''age,nat\\sex, M, F
    ...                0,  BE,       0, 1
    ...                0,  FO,       2, 3
    ...                1,  BE,       4, 5
    ...                1,  FO,       6, 7''')
    age | nat\sex | M | F
      0 |      BE | 0 | 1
      0 |      FO | 2 | 3
      1 |      BE | 4 | 5
      1 |      FO | 6 | 7

* allowed to use a regular expression in split_axis method (closes :issue:`106`):

    >>> combined = ndrange('a_b = a0b0..a1b2')
    >>> combined
        a_b | a0b0 | a0b1 | a0b2 | a1b0 | a1b1 | a1b2
            |    0 |    1 |    2 |    3 |    4 |    5
    >>> combined.split_axis(x.a_b, regex='(\w{2})(\w{2})')
        a\\b | b0 | b1 | b2
         a0 |  0 |  1 |  2
         a1 |  3 |  4 |

* one can assign a new axis to several groups at the same time by using axis[groups]:

    >>> group1 = year[2001:2004]
    >>> group2 = year[2008,2009]
    # let us change the year axis by time
    >>> x.time[group1, group2]
    (x.time[2001:2004], x.time[2008, 2009])

* implemented Axis.by() which is equivalent to axis[:].by() and divides the axis into several groups of specified length

    >>> year = Axis('year', '2010..2016')
    >>> year.by(3)
    (year.i[0:3], year.i[3:6], year.i[6:7])
    # which is equivalent to (year[2010:2012], year[2013:2015], year[2016])

    like for groups, the optional second argument specifies the step between groups

    >>> year.by(3, step=4)
    (year.i[0:3], year.i[4:7])
    # which is equivalent to (year[2010:2012], year[2014:2016])

    and if step is smaller than length, we get overlapping groups, which can be useful for example for moving averages.

    >>> year.by(3, 2)
    (year.i[0:3], year.i[2:5], year.i[4:7], year.i[6:7])
    # which is equivalent to (year[2010:2012], year[2012:2014], year[2014:2016], year[2016])

* implemented larray_nan_equal to test whether two arrays are identical even in the presence of nan values.
  Two arrays are considered identical by larray_equal if they have exactly the same axes and data. However, since a nan
  value has the odd property of not being equal to itself, larray_equal returns False if either array contains a nan
  value. larray_nan_equal returns True if all not-nan data is equal and both arrays have nans at the same place.

    >>> arr1 = ndtest((2, 3), dtype=float)
    >>> arr1['a1', 'b1'] = nan
    >>> arr1
    a\\b |  b0 |  b1 |  b2
     a0 | 0.0 | 1.0 | 2.0
     a1 | 3.0 | nan | 5.0
    >>> arr2 = arr1.copy()
    >>> arr2
    a\\b |  b0 |  b1 |  b2
     a0 | 0.0 | 1.0 | 2.0
     a1 | 3.0 | nan | 5.0
    >>> larray_equal(arr1, arr2)
    False
    >>> larray_nan_equal(arr1, arr2)
    True
    >>> arr2['b1'] = 0.0
    >>> larray_nan_equal(arr1, arr2)
    False


Miscellaneous improvements
--------------------------

* viewer: make keyboard shortcuts work even when the focus is not on the array editor widget. It means that,
  for example, plotting an array (via Ctrl-P) or opening it in Excel (Ctrl-E) can be done directly even when
  interacting with the list of arrays or within the interactive console (closes :issue:`102`).

* viewer: automatically display plots done in the viewer console in a separate window (see example below), unless
  "%matplotlib inline" is used.

    >>> arr = ndtest((3, 3))
    >>> arr.plot()

* viewer: when calling view(an_array), the new window opened does not block the initial window, which means you can
  have several windows open at the same time. view() without argument can still result in odd behavior though.

* improved LArray.set_labels to make it possible to replace only some labels of an axis, instead of all of them
  and to replace labels from several axes at the same time.

    >>> a = ndrange('nat=BE,FO;sex=M,F')
    >>> a
    nat\\sex | M | F
         BE | 0 | 1
         FO | 2 | 3

    to replace only some labels, one must give a mapping giving the new label for each label to replace

    >>> a.set_labels(x.sex, {'M': 'Men'})
    nat\\sex | Men | F
         BE |   0 | 1
         FO |   2 | 3

    to replace labels for several axes at the same time, one should give a mapping giving the new labels for each
    changed axis

    >>> a.set_labels({'sex': 'Men,Women', 'nat': 'Belgian,Foreigner'})
      nat\\sex | Men | Women
      Belgian |   0 |     1
    Foreigner |   2 |     3

    one can also replace some labels in several axes by giving a mapping of mappings

    >>> a.set_labels({'sex': {'M': 'Men'}, 'nat': {'BE': 'Belgian'}})
    nat\\sex | Men | F
    Belgian |   0 | 1
         FO |   2 | 3

* allowed matrix multiplication (@ operator) between arrays with dimension != 2 (closes :issue:`122`).

* improved LArray.plot to get nicer plots by default. The axes are transposed compared to what they used to, because
  the last axis is often used for time series. Also it considers a 1D array like a single series, not N series of 1
  point.

* added installation instructions (closes :issue:`101`).

* Axis.group and Axis.all are now deprecated (closes :issue:`148`).

    >>> city.group(['London', 'Brussels'], name='capitals')
    # should be written as:
    >>> city[['London', 'Brussels']] >> 'capitals'

    and

    >>> city.all()
    # should be written as:
    >>> city[:] >> 'all'


Fixes
-----

* viewer: allow changing the number of displayed digits even for integer arrays as that makes sense when using
  scientific notation (closes :issue:`100`).

* viewer: fixed opening a viewer via view() edit() or compare() from within the viewer
  (closes :issue:`109`)

* viewer: fixed compare() colors when arrays have values which are very close but not exactly equal
  (closes :issue:`123`)

* viewer: fixed legend when plotting arbitrary rows (it always displayed the labels of the first rows)
  (closes :issue:`136`).

* viewer: fixed labels on the x axis when zooming on a plot (closes :issue:`143`)

* fixed indexing an array using a positional group with a position which corresponds to a label on that axis.

* fixed posargsort labels (closes :issue:`137`).

* fixed labels when doing group aggregates using positional groups. Previously, it used the positions as labels. This
  was most visible when using the Group.by() method (which creates positional groups).
