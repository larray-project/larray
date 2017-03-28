New features
------------

* implement AxisCollection.replace(old_axis, new_axis)
* implement positional indexing

Miscellaneous improvements
--------------------------

* more powerful AxisCollection.pop
  added support .pop(name) or .pop(Axis object)

* LArray.set_labels returns a new LArray by default
  use inplace=True to get previous behavior
* include ndrange and __version__ in __all__

Fixes
-----

* fixed shift with n <= 0
