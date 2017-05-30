New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* improved auto-completion in ipython interactive consoles (e.g. the viewer console) for Axis, AxisCollection, Group
  and Workbook objects. These objects can now complete keys within [].

    >>> gender = Axis('gender=Male,Female')
    >>> gender
    Axis(['Male', 'Female'], 'gender')
    gender['Female
    >>> gender['Fe<tab>  # will be completed to `gender['Female`

    >>> arr = ndrange(gender)
    >>> arr.axes['gen<tab>  # will be completed to `arr.axes['gender`

    >>> wb = open_excel()
    >>> wb['Sh<tab>  # will be completed to `wb['Sheet1`


Fixes
-----

* fixed something (closes :issue:`1`).
