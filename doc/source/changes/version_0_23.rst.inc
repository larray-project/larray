New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* changed display of arrays (closes :issue:`243`):

    >>> ndtest((2, 3))
    a\\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5

  instead of

    >>> ndtest((2, 3))
    a\\b | b0 | b1 | b2
     a0 |  0 |  1 |  2
     a1 |  3 |  4 |  5

* `..` can now be used within keys (between []). Previously it could only be used to define new axes. As a reminder, it
  generates increasing values between the two bounds. It is slightly different from : which takes everything between
  the two bounds **in the axis order**.

    >>> arr = ndrange('a=a1,a0,a2,a3')
    >>> arr
    a  a1  a0  a2  a3
        0   1   2   3
    >>> arr['a1..a3']
    a  a1  a2  a3
        0   2   3

  this is different from `:` which takes everything in between the two bounds :

    >>> arr['a1:a3']
    a  a1  a0  a2  a3
        0   1   2   3

* set `sep` argument of from_string function to ' ' by default (closes :issue:`271`).
  For 1D array, a "-" must be added in front of the data line.

    >>> from_string('''sex  M  F
                       -    0  1''')
    sex  M  F
         0  1
    >>> from_string('''nat\\sex  M  F
                       BE        0  1
                       FO        2  3''')
    nat\sex  M  F
         BE  0  1
         FO  2  3

Fixes
-----

* fixed using open_excel("inexisting file") yielding to ununderstandble error.
  The function returns an error if passing a non existing Excel file and overwrite_file=False
  (closes :issue:`265`).
