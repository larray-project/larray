New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* setting values of a subset using another LArray is allowed only if axes and labels are compatible.
  For example:

    >>> arr = ndtest(4)
    >>> arr
    a  a0  a1  a2  a3
        0   1   2   3
    >>> arr['a2:'] = arr[':a1']
    ...
    ValueError: incompatible axes:
    Axis(['a2', 'a3'], 'a')
    vs
    Axis(['a0', 'a1'], 'a')

  is not allowed and raises a ValueError (closes :issue:`269`).


Fixes
-----

* fixed error raised in viewer when switching between arrays.
