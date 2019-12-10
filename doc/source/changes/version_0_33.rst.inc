.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed ``Array.old_method_name()`` to :py:obj:`Array.new_method_name()` (closes :issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`Array.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* other backward incompatible changes


New features
^^^^^^^^^^^^

* added official support for Python 3.9 (0.32.3 already supports it even though it was not mentioned).

* added :py:obj:`CheckedSession`, :py:obj:`CheckedParameters` and :py:obj:`CheckedArray` objects.

  `CheckedSession` is intended to be inherited by user defined classes in which the variables of a model
  are declared. By declaring variables, users will speed up the development of their models using the auto-completion
  (the feature in which development tools like PyCharm try to predict the variable or function a user intends
  to enter after only a few characters have been typed). All user defined classes inheriting from `CheckedSession`
  will have access to the same methods as `Session` objects.

  `CheckedParameters` is the same as `CheckedSession` but the declared variables cannot be
  modified after initialization.

  The special :py:funct:`CheckedArray` type represents an Array object with fixed axes and/or dtype.
  It is intended to be only used along with :py:class:`CheckedSession`.

  Closes :issue:`832`.


.. _misc:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved something.


Fixes
^^^^^

* fixed something (closes :issue:`1`).
