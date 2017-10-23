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
