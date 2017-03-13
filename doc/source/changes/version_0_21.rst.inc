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

* implemented from_string() method to create an array from a string:

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

* allowed the syntax axis[groups]:

    >>> groups = year[2001:2004], year[2008,2009]
    >>> groups
    (year[2001:2004], year[2008, 2009])
    >>> x.time[groups]
    (x.time[2001:2004], x.time[2008, 2009])

Miscellaneous improvements
--------------------------

* added installation instructions (closes :issue:`101`).

* viewer : make shortcuts work even when the focus is not on the array editor widget
  (ie it is on the array list, or on the interactive console) (closes :issue:`102`).

* allowed matrix multiplication (@ operator) between arrays with dimension != 2 (closes :issue:`122`).

* viewer : automatically display plots done in the viewer console in a separate window
  (unless "%matplotlib inline" is used). Plots can be done via the array widget
  (using shortcut CTRL+P or right click) or from the console:

    >>> arr = ndtest((3, 3))
    >>> arr.plot()

  Now both methods generate a plot in separate window.

* improved LArray.plot to get nicer plots by default.
  The axes are transposed compared to what they used to, because the last axis is often used for time series.
  Also it considers a 1D array like a single series, not N series of 1 point.

* Axis.group is now deprecated (Syntax "age[10:19] >> 'teens'" must be used instead)
  (closes :issue:`148`).

Fixes
-----

* viewer: allow changing the number of displayed digits even for integer arrays as that makes sense when using
  scientific notation (closes :issue:`100`).

* viewer : fixed opening a viewer via view() edit() or compare() from within the viewer
  (closes :issue:`109`)

* viewer : fixed compare() colors when arrays have values which are very close but not exactly equal
  (closes :issue:`123`)

* viewer : fixed legend when plotting arbitrary rows (it always displayed the labels of the first rows)
  (closes :issue:`136`).

* fixed posargsort labels (closes :issue:`137`).

* viewer : fixed xticklabels when zooming on a plot (closes :issue:`143`)