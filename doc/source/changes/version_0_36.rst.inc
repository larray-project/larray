.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* ``Axis.apply()`` and ``Axis.replace()`` are deprecated in favor of
  :py:obj:`Axis.set_labels()`.

* renamed ``Array.old_method_name()`` to :py:obj:`Array.new_method_name()` (closes :issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`Array.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* other backward incompatible changes


New features
^^^^^^^^^^^^

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

* :py:obj:`Array.set_labels()` and :py:obj:`Axis.set_labels()` (formerly
  ``Axis.replace()`` and ``Axis.apply()``) now accepts slices, Groups or
  selection strings as the labels to change, and callable and
  "creation strings" as the new labels. This makes it easier to change
  only a subset of labels or to change several labels in the same way
  (closes :issue:`906`).

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr.set_labels({'b1:': str.upper, 'a1': 'A-ONE'})
      a\b  b0  B1  B2
       a0   0   1   2
    A-ONE   3   4   5
    >>> arr.set_labels('b1:', 'B1..B2')
      a\b  b0  B1  B2
       a0   0   1   2
       a1   3   4   5


Fixes
^^^^^

* fixed something (closes :issue:`1`).
