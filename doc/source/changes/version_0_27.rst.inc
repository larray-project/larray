Syntax changes
--------------

* renamed `Axis.translate` to `Axis.index` (closes :issue:`479`).


Backward incompatible changes
-----------------------------

* labels are checked during array subset assignment (closes :issue:`269`):

    >>> arr = ndtest(4)
    >>> arr
    a  a0  a1  a2  a3
        0   1   2   3
    >>> arr['a0,a1'] = arr['a2,a3']
    ValueError: incompatible axes:
    Axis(['a0', 'a1'], 'a')
    vs
    Axis(['a2', 'a3'], 'a')

  previous behavior can be recovered through `drop_labels` or by changing labels via `set_labels` or `set_axes`:

    >>> arr['a0,a1'] = arr['a2,a3'].drop_labels('a')
    >>> arr['a0,a1'] = arr['a2,a3'].set_labels('a', {'a2': 'a0', 'a3': 'a1'})

* from_frame `parse_header` argument defaults to `False` instead of `True`.


New features
------------

* implemented Axis.insert and LArray.insert to add values at a given position of an axis (closes :issue:`54`).

    >>> arr1 = ndtest((2, 3))
    >>> arr1
    a\\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr1.insert(42, before='b1', label='b0.5')
    a\\b  b0  b0.5  b1  b2
     a0   0    42   1   2
     a1   3    42   4   5

  insert an array

    >>> arr2 = ndtest(2)
    >>> arr2
    a  a0  a1
        0   1
    >>> arr1.insert(arr2, after='b0', label='b0.5')
    a\\b  b0  b0.5  b1  b2
     a0   0     0   1   2
     a1   3     1   4   5

  insert an array which already has the axis

    >>> arr3 = ndrange('a=a0,a1;b=b0.1,b0.2') + 42
    >>> arr3
    a\\b  b0.1  b0.2
     a0    42    43
     a1    44    45
    >>> arr1.insert(arr3, before='b1')
    a\\b  b0  b0.1  b0.2  b1  b2
     a0   0    42    43   1   2
     a1   3    44    45   4   5

* added new items in Help menu:
  - `Report Issue...`: to report an issue on the Github project website.
  - `Users Discussion...`: redirect to the LArray Users Google Group (you need to be registered to participate).
  - `New Releases And Announces Mailing List...`: redirect to the LArray Announce mailing list.
  - `About`: give information about the editor and the versions of packages currently installed on your computer.
  (closes :editor_issue:`88`)

* added possibility to show only rows with differences when comparing arrays or sessions through the `compare`
  function in the viewer (closes :editor_issue:`102`).

* added `reverse` argument to methods `indicesofsorted` and `labelsofsorted` to sort values in descending order
  instead of ascending (default behavior):

    >>> arr = LArray([[1, 5], [3, 2], [0, 4]], "nat=BE,FR,IT; sex=M,F")
    >>> arr
    nat\sex  M  F
         BE  1  5
         FR  3  2
         IT  0  4
    >>> arr.indicesofsorted("nat", reverse=True)
    nat\sex  M  F
          0  1  0
          1  0  2
          2  2  1
    >>> arr.labelsofsorted("nat", reverse=True)
    nat\sex   M   F
          0  FR  BE
          1  BE  IT
          2  IT  FR

  Closes :issue:`490`.


Miscellaneous improvements
--------------------------

* allowed to sort values of an array along an axis (closes :issue:`225`):
    >>> a = LArray([[10, 2, 4], [3, 7, 1]], "sex=M,F; nat=EU,FO,BE")
    >>> a
    sex\nat  EU  FO  BE
          M  10   2   4
          F   3   7   1
    >>> a.sort_values(axis='sex')
    sex*\nat  EU  FO  BE
           0   3   2   1
           1  10   7   4
    >>> a.sort_values(axis='nat')
    sex\nat*  0  1   2
           M  2  4  10
           F  1  3   7

* method `LArray.sort_values` can be called without argument (closes :issue:`478`):

    >>> arr = LArray([0, 1, 6, 3, -1], "a=a0..a4")
    >>> arr
    a  a0  a1  a2  a3  a4
        0   1   6   3  -1
    >>> arr.sort_values()
    a  a4  a0  a1  a3  a2
       -1   0   1   3   6

  If the array has more than one dimension, axes are combined together:

    >>> a = LArray([[10, 2, 4], [3, 7, 1]], "sex=M,F; nat=EU,FO,BE")
    >>> a
    sex\nat  EU  FO  BE
          M  10   2   4
          F   3   7   1
    >>> a.sort_values()
    sex_nat  F_BE  M_FO  F_EU  M_BE  F_FO  M_EU
                1     2     3     4     7    10

* made the editor more responsive when switching to or changing the filter of large arrays (closes :issue:`93`).

* added support for coloring numeric values for object arrays (e.g. arrays containing both strings and numbers).

* when appending/prepending/extending an array, both the original array and the added values will be converted to a
  data type which can hold both without loss of information. It used to convert the added values to the type of the
  original array. For example, given an array of integers like:

    >>> arr = ndtest(3)
    a  a0  a1  a2
        0   1   2

  Trying to add a floating point number to that array used to result in:

    >>> arr.append('a', 2.5, 'a3')
    a  a0  a1  a2  a3
        0   1   2   2

  Now it will result in:

    >>> arr.append('a', 2.5, 'a3')
    a   a0   a1   a2   a3
       0.0  1.0  2.0  2.5


Fixes
-----

* fixed array values being editable in view() (instead of only in edit()).
