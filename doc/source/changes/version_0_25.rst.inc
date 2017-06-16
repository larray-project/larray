New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* implemented Session.to_globals(inplace=True) which will update the content of existing arrays instead of creating new
  variables and overwriting them. This ensures the arrays have the same axes in the session than the existing variables.

Fixes
-----

* fixed something (closes :issue:`1`).
