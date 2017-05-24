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

* in both axes definitions and keys (within []) `..` can now be mixed with `,` and other `..` :

    >>> arr = ndrange('code=A,C..E,G,X..Z')
    >>> arr
    code  A  C  D  E  G  X  Y  Z
          0  1  2  3  4  5  6  7
    >>> arr['A,Z..X,G']
    code  A  Z  Y  X  G
          0  7  6  5  4

* within .. extra zeros are only padded to numbers if zeros are present in the pattern.

    >>> ndrange('code=A1..A12')
    code  A1  A2  A3  A4  A5  A6  A7  A8  A9  A10  A11  A12
           0   1   2   3   4   5   6   7   8    9   10   11

    >>> ndrange('code=A01..A12')
    code  A01  A02  A03  A04  A05  A06  A07  A08  A09  A10  A11  A12
            0    1    2    3    4    5    6    7    8    9   10   11

  in previous larray versions, the two above definitions returned the second array.

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

* improved error message when trying to access nonexistent sheet in an Excel workbook (closes :issue:`266`).

* when creating an Axis from a Group and no explicit name was given, reuse the name of the group axis.

    >>> a = Axis('a=a0..a2')
    >>> Axis(a[:'a1'])
    Axis(['a0', 'a1'], 'a')

* allowed to create an array using a single group as if it was an Axis.

    >>> a = Axis('a=a0..a2')
    >>> ndrange(a)
    a  a0  a1  a2
        0   1   2
    >>> # using a group as an axis
    >>> ndrange(a[:'a1'])
    a  a0  a1
        0   1

* improved string representation of Excel workbooks and sheets (they mention the actual file/sheet they correspond to).
  This is mostly useful in the interactive console to check what an object corresponds to.

    >>> wb = open_excel()
    >>> wb
    <larray.io.excel.Workbook [Book1]>
    >>> wb[0]
    <larray.io.excel.Sheet [Book1]Sheet1>

Fixes
-----

* open_excel('non existent file') will raise an explicit error immediately when overwrite_file is False, instead of
  failing at a seemingly random point later on (closes :issue:`265`).

* integer-like strings in axis definition strings using `,` are converted to integers to be consistent with string
  definitions using `..`. In other words, ndrange('a=1,2,3') did not create the same array than ndrange('a=1..3').

* fixed reading a single cell from an Excel sheet.

* fixed script execution not resuming after quitting the viewer when it was called using view(a_single_array).

* fixed opening the viewer after showing a plot window.

* do not display an error when setting the value of an element of a non LArray sequence in the viewer console

    >>> l = [1, 2, 3]
    >>> l[0] = 42
