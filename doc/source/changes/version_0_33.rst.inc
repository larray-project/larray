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

* greatly improved :py:obj:`Array.plot()` method and "submethods" (:py:obj:`Array.plot.bar()`, etc.)

  - support `x`, `y` and `by` arguments in plot functions where it make sense
    When only some of them are specified, the other arguments pick from remaining
    available axes. This means a lot of plots can now be expressed more intuitively and concisely
    (you do not need to transpose your array to get the result you want, you just
    specify the axes you want to use in 'x' or 'y'.
  - `subplots` argument now accepts an axis (or tuple of them) in addition to a
    boolean to specify *which* axes to use as subplots.
  - support for *labels* (instead of axes) in x and y for line plot and scatter.
  - support passing a dict as legend to customize the legend.
  - many tweaks to make several plots look better out of the box.


Fixes
^^^^^

* fixed something (closes :issue:`1`).
