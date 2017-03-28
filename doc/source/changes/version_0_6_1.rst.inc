New features
------------

* added dtype argument to all array creation functions to override default data type

* aggregates can take an explicit "axis" keyword argument which can be used to target an axis by index

    >>> arr.sum(axis=0)

* implemented LGroup.__getitem__ & LGroup.__iter__, so that for list-based groups (ie not slices) you can write:

    >>> for v in my_group:
    ...     # some code

  or

    >>> my_group[0]


Miscellaneous improvements
--------------------------

* renamed LabelGroup to LGroup and PositionalKey to PGroup. We might want to rename the later to IGroup (to be
  consistent with axis.i[...]).

* slightly better support for axes without name
* better docstrings for a few functions
* misc cleanup


Fixes
-----

* fixed XXX_like(a) functions to use the same dtype than a instead of always float
* fixed to_XXX with 1d arrays (e.g. to_clipboard())
* fixed all() and any() toplevel functions without argument
* fixed LArray without axes in some cases
* fixed array creation functions with only shapes on python2

