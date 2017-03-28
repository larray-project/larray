New features
------------

* a[bool_array_key] broadcasts missing/differently ordered dimensions and returns an LArray with combined axes

* a[bool_array_key] = value broadcasts missing/differently ordered dimensions on both key and value

* implemented argmin, argmax, argsort, posargmin, posargmax, posargsort.
   they do indirect operation along an axis. E.g. argmin gives the label of the minimum value, argsort gives the labels
   which would sort the array along that dimension.
   posargXXX gives the position/indexes instead of the labels.

* implemented Axis.__iter__ so that one can write:

    >>> for label in an_array.axes.an_axis:
    ...     <some code>

  instead of

    >>> for label in an_array.axes.an_axis.labels:
    ...     <some code>

* implemented the .info property on AxisCollection

* implement all/any top level functions, so that you can use them in with_total.


Miscellaneous improvements
--------------------------

* renamed ValueGroup to LabelGroup. We might want to rename it to LGroup to be consistent with LArray?

* allow a single int as argument to LArray creation functions (ndrange et al.)

   e.g. `ndrange(10)` is now allowed instead of `ndrange([10])`

* use display_name in .info (ie add * next to wildcard axes in .info).

* allow specifying a custom window title in view()

* viewer displays booleans as True/False instead of 1/0

* slightly better support for axes with no name (None). There is still a long way to go for full support though.

* improved a few docstrings

* nicer errors when tests results are different from expected

* removed debug prints from viewer

* misc cleanups

Fixes
-----

* fixed view() on all-negative arrays

* fixed view() on string arrays
