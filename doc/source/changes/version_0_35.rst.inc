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

* Plots made with Array.plot() in a Python script will be shown by default,
  unless either the filepath (see below) or ax arguments are used. Shown plots
  will open a window and pause the running script until the window is closed by
  the user. To revert to the previous behavior, use show=False.

* Using :py:obj:`CheckedSession`, :py:obj:`CheckedParameters` or :py:obj:`CheckedArray`
  now requires to install pydantic >= 2.12 (closes :issue:`1075`).

New features
^^^^^^^^^^^^

* Array.plot now has an ´animate´ argument to produce animated plots. The
  argument takes an axis (it also supports several axes but that is rarely
  useful) and will create an animation, with one image per label of that axis.
  For example,

    >>> arr.plot.bar(animate='year')

  will create an animated bar plot with one frame per year.

* implemented Array.plot `filepath` argument to save plots to a file directly,
  without having to use the matplotlib API.

* implemented Array.plot `show` argument to display plots directly, without
  having to use the matplotlib API. This is the new default behavior.

* implemented a new kind of plot: `heatmap`. It can be used like this:

    >>> arr.plot.heatmap()

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

* added support for Pandas Series in :py:obj:`asarray()`. This is considered a
  fix because it kind of worked but silently ignored the index and name of the
  series (closes :issue:`895`).

