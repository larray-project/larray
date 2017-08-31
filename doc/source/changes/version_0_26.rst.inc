New features
------------

* added global_arrays() function which returns a Session containing all arrays defined in global variables. This
  complements the local_arrays() function. When used outside of a function, these two functions should have the same
  results, but inside a function local_arrays will return only arrays local to the function, while global_arrays will
  return only arrays defined globally.


Miscellaneous improvements
--------------------------

* view() and edit() without argument now display global arrays in addition to local ones (closes :editor_issue:`54`).

* allowed to pass an array of labels as `new_axis` argument to `reindex` method (closes :issue:`384`):

    >>> arr = ndrange('a=v0..v1;b=v0..v2')
    >>> arr
    a\b  v0  v1  v2
     v0   0   1   2
     v1   3   4   5
    >>> arr.reindex('a', arr.b.labels)
    a\b   v0   v1   v2
     v0    0    1    2
     v1    3    4    5
     v2  nan  nan  nan

* allowed to call the `reindex` method using a differently named axis for labels (closes :issue:`386`):

    >>> arr = ndrange('a=v0..v1;b=v0..v2')
    >>> arr
    a\b  v0  v1  v2
     v0   0   1   2
     v1   3   4   5
    >>> arr.reindex('a', arr.b)
    a\b   v0   v1   v2
     v0    0    1    2
     v1    3    4    5
     v2  nan  nan  nan

* `read_excel` displays an error message when arguments `na`, `sort_rows` or `sort_columns` are used
  with the default `xlwings` engine

* replaced `na` argument of `read_csv`, `read_excel`, `read_hdf` and `read_sas` functions by `fill_value`
  (closes :issue:`394`)

Fixes
-----

* fixed something (closes :issue:`1`).
