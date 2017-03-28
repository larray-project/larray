A minor but backward incompatible version (hence the bump in version number)!

Core
----

* fixed int_array.mean() to return floats instead of int (regression in 0.8)
* larray_equal returns False when either value is not an LArray, instead of raising an exception

Session
-------

* changed `Session == Session` to return an array of booleans instead of a single boolean, so that we know which
  array(s) differ. Code like `session1 == session2`, should be changed to `all(session1 == session2)`.
* implemented Session != Session
* implemented Session.get(k, default) (returns default if k does not exist in Session)
* implemented len() for Session objects to know how many objects are in the Session

Viewer
------

* fixed view() (regression in 0.8.1)
* fixed edit() to actually apply changes on "OK"/accept_changes even when no filter change occurred after the last edit.
