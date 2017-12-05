New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* added possibility to call the methods `diff` and `growth_rate` with a group (closes :issue:`532`):

    >>> data = [[2, 4, 5, 4, 6], [4, 6, 3, 6, 9]]
    >>> a = LArray(data, "sex=M,F; year=2016..2020")
    >>> a
    sex\year  2016  2017  2018  2019  2020
           M     2     4     5     4     6
           F     4     6     3     6     9
    >>> a.diff(a.year[2017:])
    sex\year  2018  2019  2020
           M     1    -1     2
           F    -3     3     3
    >>> a.growth_rate(a.year[2017:])
    sex\year  2018  2019  2020
           M  0.25  -0.2   0.5
           F  -0.5   1.0   0.5


Fixes
-----

* fixed something (closes :issue:`1`).