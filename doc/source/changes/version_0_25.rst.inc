New features
------------

* viewer functions (`view`, `edit` and `compare`) have been moved to the separate `larray-editor` package,
  which needs to be installed separately, unless you are using `larrayenv`.
  Closes :issue:`332`.

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

* implemented Session.transpose(axes) to reorder axes of all arrays within a session, ignoring missing axes for each
  array. For example, let us first create a test session and a small helper function to display sessions as a short
  summary.

    >>> arr1 = ndtest((2, 2, 2))
    >>> arr2 = ndtest((2, 2))
    >>> sess = Session([('arr1', arr1), ('arr2', arr2)])
    >>> def print_summary(s):
    ...     print(s.summary("{name} -> {axes_names}"))
    >>> print_summary(sess)
    arr1 -> a, b, c
    arr2 -> a, b

  Put the 'b' axis in front of all arrays

    >>> print_summary(sess.transpose('b'))
    arr1 -> b, a, c
    arr2 -> b, a

  Axes missing on an array are ignored ('c' for arr2 in this case)

    >>> print_summary(sess.transpose('c', 'b'))
    arr1 -> c, b, a
    arr2 -> b, a

  Use ... to move axes to the end

    >>> print_summary(sess.transpose(..., 'a'))
    arr1 -> b, c, a
    arr2 -> b, a

* implemented unary operations on Session, which means one can negate all arrays in a Session or take the absolute
  value of all arrays in a Session without writing an explicit loop for that.

    >>> arr1 = ndtest(2)
    >>> arr1
    a  a0  a1
        0   1
    >>> arr2 = ndtest(4) - 1
    >>> arr2
    a  a0  a1  a2  a3
       -1   0   1   2
    >>> sess1 = Session([('arr1', arr1), ('arr2', arr2)])
    >>> sess2 = -sess1
    >>> sess2.arr1
    a  a0  a1
        0  -1
    >>> sess2.arr2
    a  a0  a1  a2  a3
        1   0  -1  -2
    >>> sess3 = abs(sess1)
    >>> sess3.arr2
    a  a0  a1  a2  a3
        1   0   1   2

* implemented Axis.align(other_axis) and AxisCollection.align(other_collection) which makes two axes / axis collections
  compatible with each other, see LArray.align above.


Miscellaneous improvements
--------------------------

* added icon to display in Windows start menu and editor windows.

* added 'Getting Started' section in documentation.

* implemented axes argument to ipfp to specify on which axes the fitting procedure should be applied (closes
  :issue:`185`). For example, let us assume you have a 3D array, such as:

    >>> initial = ndrange('a=a0..a9;b=b0..b9;year=2000..2016')

  and you want to apply a 2D fitting procedure for each value of the year axis. Previously, you had to loop on that
  year axis explicitly and call ipfp within the loop, like:

    >>> result = zeros(initial.axes)
    >>> for year in initial.year:
    ...     current = initial[year]
    ...     # assume you have some targets for each year
    ...     current_targets = [current.sum(x.a) + 1, current.sum(x.b) + 1]
    ...     result[year] = ipfp(current_targets, current)

  Now you can apply the procedure on all years at once, by telling you want to do the fitting procedure on the other
  axes. This is a bit shorter to type, but this is also *much* faster.

    >>> all_targets = [initial.sum(x.a) + 1, initial.sum(x.b) + 1]
    >>> result = ipfp(all_targets, initial, axes=(x.a, x.b))

* made ipfp 10 to 20% faster (even without using the axes argument).

* implemented Session.to_globals(inplace=True) which will update the content of existing arrays instead of creating new
  variables and overwriting them. This ensures the arrays have the same axes in the session than the existing variables.

* stack can be used with keyword arguments when labels are "simple strings" (i.e. no integers, no punctuation,
  no string starting with integers, etc.). This is an attractive alternative but as it only works in the usual case and
  not in all cases, it is not recommended to use it except in the interactive console.

    >>> arr1 = ones('nat=BE,FO')
    >>> arr1
    nat   BE   FO
         1.0  1.0
    >>> arr2 = zeros('nat=BE,FO')
    >>> arr2
    nat   BE   FO
         0.0  0.0
    >>> stack(M=arr1, F=arr2, axis='sex=M,F')
    nat\\sex    M    F
         BE  1.0  0.0
         FO  1.0  0.0

  Without passing an explicit order for labels like above (or an axis object), it should only be used on Python 3.6 or
  later because keyword arguments are NOT ordered on earlier Python versions.

    # use this only on Python 3.6 and later
    >>> stack(M=arr1, F=arr2, axis='sex')
    nat\\sex    M    F
         BE  1.0  0.0
         FO  1.0  0.0

* binary operations between session now ignore type errors. For example, if you are comparing two sessions with many
  arrays by computing the difference between them but a few arrays contain strings, the whole operation will not fail,
  the concerned arrays will be assigned a nan instead.


Fixes
-----

* fixed disambiguating an ambiguous key by adding the axis within the string, for example
  arr['axis_name[ambiguouslabel]'] (closes :issue:`331`).
