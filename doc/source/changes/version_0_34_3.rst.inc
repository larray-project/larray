.. py:currentmodule:: larray


New features
^^^^^^^^^^^^

* added support for Python 3.12 (closes :issue:`1109`).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved the error message when selecting several labels at the same time and some of them were wrong.
  In that case, it was hard to know which labels were wrong. This was especially annoying if the axis was long and
  thus not shown in its entirety in the error message. For example, given the following array: ::

    >>> arr = la.ndtest('a=a0,a1;b=b0..b2,b4..b7')
    >>> arr
    a\b  b0  b1  b2  b4  b5  b6  b7
     a0   0   1   2   3   4   5   6
     a1   7   8   9  10  11  12  13

  This code: ::

    >>> arr['b0,b2,b3,b7']

  used to produce the following error: ::

    ValueError: 'b0,b2,b3,b7' is not a valid label for any axis:
     a [2]: 'a0' 'a1'
     b [7]: 'b0' 'b1' 'b2' ... 'b5' 'b6' 'b7'

  which did not contain enough information to determine the problem was with 'b3'. It now produces this instead: ::

    ValueError: 'b0,b2,b3,b7' is not a valid subset for any axis:
     a [2]: 'a0' 'a1'
     b [7]: 'b0' 'b1' 'b2' ... 'b5' 'b6' 'b7'
    Some of those labels are valid though:
     * axis 'b' contains 3 out of 4 labels (missing labels: 'b3')

  Closes :issue:`1101`.


Fixes
^^^^^

* using a boolean array as a filter to take a subset of another array now raise an error when the
  two arrays have incompatible axes instead of producing wrong result (closes :issue:`1085`).

* fixed copying a sheet from one Excel workbook to another when the destination sheet
  is given by position (closes :issue:`1092`).

* fixed :py:obj:`Array.values()` and :py:obj:`Array.items()` on the first axis given by position.
  (e.g. `my_array.values(axes=0)`). Closes :issue:`1093`.

* fixed :py:obj:`Array.dump()` ``axes_names`` argument for 1D arrays (closes :issue:`1094`).

* fixed :py:obj:`Axis.difference()`, :py:obj:`Axis.intersection()` and :py:obj:`Axis.union()`
  with a Group argument (closes :issue:`1104`).

* fixed converting a scalar Array (an Array with 0 dimensions) to string with numpy 1.22+.

* avoid warnings and errors with recent versions of our dependencies (Numpy 2+, Pandas 2.2+ and
  xlwings 0.30.2+). Closes :issue:`1100`, :issue:`1107` and :issue:`1108`.
