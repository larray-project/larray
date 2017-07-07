New features
------------

* installing larray-editor (or larrayenv) from conda environment creates a new menu 'LArray' in the Windows start menu.
  It contains a link to open the documentation, a shortcut to launch the user interface in edition mode
  and a shortcut to update larrayenv. Closes :issue:`281`.

* implemented array.align(other_array) which makes two arrays compatible with each other (by making all common axes
  compatible). This is done by adding, removing or reordering labels for each common axis according to the join method
  used:

    - outer: will use a label if it is in either arrays axis (ordered like the first array). This is the default as it
             results in no information loss.
    - inner: will use a label if it is in both arrays axis (ordered like the first array)
    - left: will use the first array axis labels
    - right: will use the other array axis labels

  The fill value for missing labels defaults to nan.

    >>> arr1 = ndtest((2, 3))
    >>> arr1
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr2 = -ndtest((3, 2))
    >>> # reorder array to make the test more interesting
    >>> arr2 = arr2[['b1', 'b0']]
    >>> arr2
    a\\b  b1  b0
     a0  -1   0
     a1  -3  -2
     a2  -5  -4

  Align arr1 and arr2

    >>> aligned1, aligned2 = arr1.align(arr2)
    >>> aligned1
    a\b   b0   b1   b2
     a0  0.0  1.0  2.0
     a1  3.0  4.0  5.0
     a2  nan  nan  nan
    >>> aligned2
    a\b    b0    b1   b2
     a0   0.0  -1.0  nan
     a1  -2.0  -3.0  nan
     a2  -4.0  -5.0  nan

  After aligning all common axes, one can then do operations between the two arrays

    >>> aligned1 + aligned2
    a\b   b0   b1   b2
     a0  0.0  0.0  nan
     a1  1.0  1.0  nan
     a2  nan  nan  nan

  The fill value for missing labels defaults to nan but can be changed to any compatible value.

    >>> aligned1, aligned2 = arr1.align(arr2, fill_value=0)
    >>> aligned1
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
     a2   0   0   0
    >>> aligned2
    a\b  b0  b1  b2
     a0   0  -1   0
     a1  -2  -3   0
     a2  -4  -5   0
    >>> aligned1 + aligned2
    a\b  b0  b1  b2
     a0   0   0   2
     a1   1   1   5
     a2  -4  -5   0

* implemented Axis.align(other_axis) and AxisCollection.align(other_collection) which makes two axes / axis collections
  compatible with each other, see LArray.align above.


Miscellaneous improvements
--------------------------

* implemented Session.to_globals(inplace=True) which will update the content of existing arrays instead of creating new
  variables and overwriting them. This ensures the arrays have the same axes in the session than the existing variables.

* added icon to display in Windows start menu and editor windows.

* added 'Getting Started' section in documentation.


Fixes
-----

* fixed disambiguating an ambiguous key by adding the axis within the string, for example
  arr['axis_name[ambiguouslabel]'] (closes :issue:`331`).
