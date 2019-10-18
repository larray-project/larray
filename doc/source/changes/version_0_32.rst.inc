.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* renamed the ``LArray`` class to :py:obj:`Array` (closes :issue:`611`).


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Because it was broken, the possibility to dump and load Axis and Group objects
  contained in a session has been removed for the CSV and Excel formats.
  Fixing it would have taken too much time considering it is very rarely used
  (no one complains it was broken) so the decision to remove it was taken.
  However, this is still possible using the HDF format.
  Closes :issue:`815`.


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

* conda channel to install or update the larray, larray-editor, larray-eurostat and larrayenv packages
  switched from ``gdementen`` to ``larray-project`` (closes :issue:`560`).


Fixes
^^^^^

* fixed binary operations between a session and an array object (closes :issue:`807`).

* fixed the :py:obj:`Array.reindex()` printing a spurious warning message when the
  `axes_to_reindex` argument is the name of the axis to reindex (closes :issue:`812`).

* fixed :py:obj:`zip_array_values()` and :py:obj:`zip_array_items()` functions not available
  when importing the entire larray library as ``from larray import *`` (closes :issue:`816`).

* fixed wrong axes and groups names when loading a session from an HDF file
  (closes :issue:`803`).
