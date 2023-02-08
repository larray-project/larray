.. py:currentmodule:: larray

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

  The special :py:obj:`CheckedArray` type represents an Array object with fixed axes and/or dtype.
  It is intended to be only used along with :py:class:`CheckedSession`.

  Closes :issue:`832`.


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

* :py:obj:`eye()` now supports an AxisCollection as argument, so you can use axes from another array by using
  ``eye(other_array.axes)``.

* added arguments `rtol`, `atol` and `nans_equal` to the :py:obj:`Session.element_equals()` and
  :py:obj:`Session.equals()` methods (closes :issue:`990`).


Fixes
^^^^^

* fixed Array.values(), zip_array_values and zip_array_items when axes=() (closes :issue:`883`).

* fixed several edge cases in :py:obj:`sequence()`.

* fixed set_labels(labels_dict) with several labels from the same axis (closes :issue:`851`).

* fixed loading arrays with anonymous axes and numeric labels from Excel using Pandas 1.3+ (closes :issue:`950`).

* fixed :py:obj:`read_hdf()` opening in RW mode instead of read mode (closes :issue:`980`).
