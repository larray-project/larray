.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed ``Array.old_method_name()`` to :py:obj:`Array.new_method_name()` (closes :issue:`1`).

* renamed ``stacked`` argument of :py:obj:`Array.plot()` to ``stack``. This
  also impacts all the relevant kind-specific sub-methods
  (:py:obj:`Array.plot.area()`, :py:obj:`Array.plot.bar()`,
  :py:obj:`Array.plot.barh()`, and :py:obj:`Array.plot.line()`).


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

* :py:obj:`Array.plot()` ``stack`` argument (previously called ``stacked``) can
  now take an axis (or several axes) to specify which axes to stack, instead of
  always stacking the last axis. For example, a plot with genders stacked
  could be specified as:

    >>> arr.plot.bar(stacked='gender')


Fixes
^^^^^

* fixed something (closes :issue:`1`).
