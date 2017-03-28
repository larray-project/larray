Core
----

* implemented skipna argument for most aggregate functions. defaults to True.
* implemented LArray.sort_values(key)
* implemented percentile and median
* added isnan and isinf toplevel functions
* made axis argument optional for argsort & posargsort on 1D arrays

* fixed a[key] = value when key corresponds to a single cell of the array
* fixed keepaxes argument for aggregate functions
* fixed a[int_array] (when the axis needs to be guessed)
* fixed empty_like
* fixed aggregates on several axes given as integers e.g. arr.sum(axis=(0, 2))
* fixed "kind" argument in posargsort

Viewer
------

* added title argument to edit() (set automatically if not provided, like for view())
* fixed edit() on filtered arrays
* fixed view(expression).
  anything which was not stored in a variable was broken in 0.7.1
* reset background color when setting values if necessary (still buggy in some cases, but much less so ;-))
* background color for headers is always on
* view() => array cells are not editable, instead of being editable and ignoring entered values
* fixed compare() colors when arrays are entirely equal
* fixed error message for compare() when PyQt is not available

Misc
----

* bump numpy requirement to 1.10, implicitly dropping support for python 3.3
* renamed view module to editor to not collide with view function
* improved/added a few tests
