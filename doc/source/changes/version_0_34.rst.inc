.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed ``Array.old_method_name()`` to :py:obj:`Array.new_method_name()` (closes :issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`Array.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dropped support for Python 3.6.

* deprecations dating to version 0.29 or earlier (released more than 3 years ago) now raise errors instead of
  printing a warning.


New features
^^^^^^^^^^^^

* implemented :py:obj:`Array.value_counts()`, which computes the number of occurrences of each unique value in an array.

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

* made all I/O functions/methods/constructors to accept either a string or a pathlib.Path object
  for all arguments representing a path (closes :issue:`896`).

* made ipfp faster (the smaller the array, the larger the improvement)
  For example, for small arrays it is several times faster than before, for 1000x1000 arrays it is about 30% faster.

* added type hints for all remaining functions and methods which improves autocompletion in editors (such as PyCharm).
  Closes :issue:`864`.

Fixes
^^^^^

* fixed Array.insert when no label is provided (closes :issue:`879`).

* fixed comparison between Array and None returning False instead of an array of boolean values
  (closes :issue:`988`)

* fixed binary operation between an Array and an Axis returning False.
