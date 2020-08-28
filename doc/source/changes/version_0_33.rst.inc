.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed ``Array.old_method_name()`` to :py:obj:`Array.new_method_name()` (closes :issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`Array.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dropped support for Python 2 (closes :issue:`567`).


New features
^^^^^^^^^^^^

* added support for Python 3.8 (closes :issue:`850`).

* added a feature (see the :ref:`miscellaneous section <misc>` for details). It works on :ref:`api-axis` and
  :ref:`api-group` objects.

  Here is an example of the new feature:

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5

  And it can also be used like this:

    >>> arr = ndtest("a=a0..a2")
    >>> arr
    a  a0  a1  a2
        0   1   2

* added another feature in the editor (closes :editor_issue:`1`).

  .. note::

     - It works for foo bar !
     - It does not work for foo baz !


.. _misc:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* scalar objects (i.e of type int, float, bool, string, date, time or datetime) belonging to a session
  are now also saved and loaded when using the HDF5 or pickle format (closes :issue:`842`).

* implemented :py:obj:`Axis.astype()` method (closes :issue:`880`).

Fixes
^^^^^

* fixed an edge case for group aggregates and labels in reverse order (closes :issue:`868`).

* fixed *_by functions returning the same value as if no axis was passed when specifying all axes
  (closes :issue:`913`)
