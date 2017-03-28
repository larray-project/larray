Viewer
------

* implemented paste (ctrl-V) 

* implemented experimental array comparator:

    >>> compare(array1, array2)

  Known limitation: the arrays must have exactly the same axes and the background color is buggy when using filters

* when no title is specified in view(), it is determined automatically by inspecting the local variables of the function
  where view() is called and using the names of the ones matching the object passed. If several matches, up to 3 are
  displayed.

* added axes names to copy (ctrl-C)

* fixed copy (ctrl-C) of 0d array

Input/Output
------------

* added 'dialect' argument to to_csv. For example, dialect='classic' does not include the last (horizontal) axis name.

* fixed loading .csv files without \ (ie 'classic' .csv files), though one needs to specify nb_index in that case if
  ndim > 2

* strip spaces around axes names so that you can use "axis0<space>\<space>axis1" instead of "axis0\axis1" in .csv files

* fixed 1d arrays I/O

* more precise parsing of input headers: 1 and 0 come out as int, not bool

Misc
----

* nicer error message when using an invalid axes names

* changed LArray .df property to a to_frame() method so that we can pass options to it
