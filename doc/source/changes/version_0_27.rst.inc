Syntax changes
--------------

* renamed `Axis.translate` to `Axis.index` (closes :issue:`479`).


Backward incompatible changes
-----------------------------

* labels are checked during array subset assignement (closes :issue:`269`):

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

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

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


Fixes
-----

* fixed array values being editable in view() (instead of only in edit()).
