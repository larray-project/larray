Miscellaneous improvements
--------------------------

* Deprecated methods display a warning message when they are still used
  (replaced DeprecationWarning by FutureWarning. Closes :issue:`310`)

Fixes
-----

* fixed error raised in viewer when switching between arrays when a filter was set.

* fixed displaying empty array when starting the viewer or a new session in it.

* fixed the `view()`, `edit()` and `compare()` functions not being available in the viewer console.

* fixed row and column resizing by double clicking on the edge of an header cell.

* setting values of a subset using another LArray is allowed only if axes and labels are compatible
  (closes :issue:`269`). For example:

    >>> arr = ndtest(4)
    >>> arr
    a  a0  a1  a2  a3
        0   1   2   3
    >>> arr['a2:'] = arr[':a1']
    ValueError: incompatible axes:
    Axis(['a2', 'a3'], 'a')
    vs
    Axis(['a0', 'a1'], 'a')

  is not allowed. To set values using a LArray with incompatible labels, use the method
  `set_labels` or `drop_labels`:

    >>> arr['a2:'] = arr[':a1'].set_labels('a', 'a2,a3')
    >>> arr['a2:'] = arr[':a1'].drop_labels('a')

* fixed getting a subset of an array by mixing boolean filters and other filters (closes :issue:`246`):

    >>> arr = ndrange('a=a0..a2;b=0..3')
    >>> arr
    a\b  0  1   2   3
     a0  0  1   2   3
     a1  4  5   6   7
     a2  8  9  10  11
    >>> arr['a0,a2', x.b < 2]
    a\b  0  1
     a0  0  1
     a2  8  9

  Warning: when mixed with other filters, boolean filters are limited to one dimension.

* fixed setting an array values using `array.points[key] = value` when value is an LArray (closes :issue:`368`).

* fixed using syntax 'int..int' in a selection (closes :issue:`350`):

    >>> arr = ndrange('a=2017..2012')
    >>> arr
    a  2017  2016  2015  2014  2013  2012
          0     1     2     3     4     5
    >>> arr['2012..2015']
    a  2012  2013  2014  2015
          5     4     3     2
