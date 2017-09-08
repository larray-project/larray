New features
------------

* added global_arrays() function which returns a Session containing all arrays defined in global variables. This
  complements the local_arrays() function. When used outside of a function, these two functions should have the same
  results, but inside a function local_arrays will return only arrays local to the function, while global_arrays will
  return only arrays defined globally.


Miscellaneous improvements
--------------------------

* improved something.


Fixes
-----

* fixed something (closes :issue:`1`).
