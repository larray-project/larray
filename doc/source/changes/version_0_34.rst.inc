.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* made :py:obj:`Array.append()` work for the cases previously covered by ``Array.extend()`` (when the appended value
  already has the axis being extended) and deprecated ``Array.extend()`` (closes :issue:`887`).

* renamed ``Array.sort_axes()`` to :py:obj:`Array.sort_labels()` (closes :issue:`861`).

* renamed :py:obj:`Array.percentile()` and :py:obj:`Array.percentile_by()` `interpolation` argument
  to `method` to follow numpy and thus support additional "interpolation" methods.

* deprecated the ability to target a label in an aggregated array using the group that created it.
  The aggregated array label should be used instead. This is a seldom used feature which is complex
  to keep working and has a significant performance cost in some cases, even when the feature is not used
  (closes :issue:`994`).

  In other words, the following code will now raise a warning:

    >>> arr = ndtest(4)
    >>> arr
    a  a0  a1  a2  a3
        0   1   2   3
    >>> group1 = arr.a['a0', 'a2'] >> 'a0_a2'
    >>> group2 = arr.a['a1', 'a3'] >> 'a1_a3'
    >>> agg_arr = arr.sum((group1, group2))
    >>> agg_arr
    a  a0_a2  a1_a3
           2      4
    >>> agg_arr[group1]
    FutureWarning: Using a Group object which was used to create an aggregate to target its aggregated label is deprecated.
    Please use the aggregated label directly instead. In this case, you should use 'a0_a2' instead of using
    a['a0', 'a2'] >> 'a0_a2'.
    2

  One should use the label on the aggregated array instead:

    >>> agg_arr['a0_a2']
    2

* deprecated passing individual session elements as non-keyword arguments to :py:obj:`Session()`.
  This means that, for example, ``Session(axis1, axis2, array1=array1)`` should be rewritten as
  ``Session(axis1name=axis1, axis2name=axis2, array1=array1)`` instead. Closes :issue:`1024`.

* deprecated :py:obj:`Session.add()`. Please use :py:obj:`Session.update()` instead (closes :issue:`999`).


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dropped support for Python 3.6.

* deprecations dating to version 0.29 or earlier (released more than 3 years ago) now raise errors instead of
  printing a warning.


New features
^^^^^^^^^^^^

* added support for Python 3.10.

* implemented :py:obj:`Array.value_counts()`, which computes the number of occurrences of each unique value in an array.

* added :py:obj:`Session.nbytes` and added :py:obj:`Session.memory_used` attributes.

* added ``display`` argument to :py:obj:`Array.compact()` to display a message if some axes were "compacted".


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* made all I/O functions/methods/constructors accept pathlib.Path objects in addition to strings
  for all arguments representing a path (closes :issue:`896`).

* added type hints for all remaining functions and methods which improves autocompletion in editors (such as PyCharm).
  Closes :issue:`864`.

* made several error messages more useful when trying to get an invalid subset of an array (closes :issue:`875`).

  - when a key is not valid on any axis, the error message includes the array axes
  - when a key is not valid for the axis specified by the user, the error message includes that axis labels
  - when a label is ambiguous (valid on several axes), the error message includes the axes labels in addition to the
    axes names
  - when several parts of a key seem to target the same axis, the error message includes the bad key in
    addition to the axis.

* made :py:obj:`ipfp()` faster (the smaller the array, the larger the improvement)
  For example, for small arrays it is several times faster than before, for 1000x1000 arrays it is about 30% faster.

* made arithmetic operations between two Arrays with the same axes much faster.

* made Array[] faster in the usual/simple cases.

* made Array.i[] much faster.


Fixes
^^^^^

* fixed displaying plots made via :py:obj:`Array.plot()` outside of the LArray editor (closes :issue:`1019`).

* fixed :py:obj:`Array.insert()` when no label is provided (closes :issue:`879`).

* fixed :py:obj:`Array.insert()` when (one of) the inserted label(s) is ambigous on the value.

* fixed comparison between :py:obj:`Array` and :py:obj:`None` returning False instead of an array of boolean values
  (closes :issue:`988`)

* fixed binary operations between an :py:obj:`Array` and an :py:obj:`Axis` returning False.

* fixed :py:obj:`AxisCollection.split_axes()` with anonymous axes.

* fixed the ``names`` argument in :py:obj:`Array.split_axes()` and :py:obj:`AxisCollection.split_axes()` not working
  in some cases.

* fixed taking a subset of an Excel range (e.g. myworkbook['my_sheet']['A2:C5'][1:])

* fixed setting the first sheet via position in a new workbook opened via open_excel(overwrite=True):

    >>> with open_excel(fpath, overwrite_file=True) as wb:
    ...     wb[0] = <something>

* fixed Array.ipoints[] when not all dimensions are given in the key.
