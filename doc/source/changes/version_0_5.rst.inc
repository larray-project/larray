New features
------------

* experimental support for indexing an LArray by another (integer) LArray

    >>> array[other_array]

* experimental support for LArray.drop_labels and the concept of wildcard axes

* added LArray.display_name and AxisCollection.display_names which add '*' next to wildcard axes

* implemented where(cond, array1, array2)

* implemented LArray.__iter__ so that this works:

    >>> for value in array:
    ...     <some code>

* implement keepaxes=label or keepaxes=True for aggregate functions on full axes
   
    array.sum(x.age, keepaxes='total')

* AxisCollection.replace can replace several axes in one call

* implemented .expand(out=) to expand into an existing array


Miscellaneous improvements
--------------------------

* removed Axis.sorted()

* removed LArray.axes_names & axes_labels. One should use .axes.names & .axes.labels instead.

* raise an error when trying to convert an array with more than one value to a Boolean. For example, this will fail:

    >>> arr = ndrange([sex])
    >>> if arr:
    ...     <some code>

* convert value to self.dtype in append/prepend

* faster .extend, .append, .prepend and .expand

* some code cleanup, better tests, …


Fixes
-----

* fixed .extend when other has longer axes than self

