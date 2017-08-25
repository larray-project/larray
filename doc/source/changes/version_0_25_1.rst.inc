New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* some improvement.

Fixes
-----

* fixed error raised in viewer when switching between arrays when a filter was set.

* fixed displaying empty array when starting the viewer or a new session in it.

* fixed the `view()`, `edit()` and `compare()` functions not being available in the viewer console.

* fixed row and column resizing by double clicking on the edge of an header cell.

* setting values of a subset using another LArray is allowed only if axes and labels are compatible
  (closes :issue:`269`). For example:

    >>> arr = ndtest(4)
    >>> arr
    a  a0  a1  a2  a3
        0   1   2   3
    >>> arr['a2:'] = arr[':a1']
    ValueError: incompatible axes:
    Axis(['a2', 'a3'], 'a')
    vs
    Axis(['a0', 'a1'], 'a')

  is not allowed. To set values using a LArray with incompatible labels, use the method
  `set_labels` or `drop_labels`:

    >>> arr['a2:'] = arr[':a1'].set_labels('a', 'a2,a3')
    >>> arr['a2:'] = arr[':a1'].drop_labels('a')
