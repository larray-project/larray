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

  previous behavior can be recovered through `drop_labels` or by changing labels via
  `set_labels` or `set_axes`:

    >>> arr['a0,a1'] = arr['a2,a3'].drop_labels('a')
    >>> arr['a0,a1'] = arr['a2,a3'].set_labels('a', {'a2': 'a0', 'a3': 'a1'})


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

* improved something.

Fixes
-----

* fixed something (closes :issue:`1`).
