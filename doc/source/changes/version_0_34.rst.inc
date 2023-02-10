.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* deprecated :py:obj:`Session.add()`. Please use :py:obj:`Session.update()` instead (closes :issue:`999`).

* deprecated passing individual session elements as non-keyword arguments to :py:obj:`Session()`.
  This means that, for example, ``Session(axis1, axis2, array1=array1)`` should be rewritten as
  ``Session(axis1name=axis1, axis2name=axis2, array1=array1)`` instead. Closes :issue:`1024`.

* renamed ``Array.sort_axes()`` to :py:obj:`Array.sort_labels()` (closes :issue:`861`).


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dropped support for Python 3.6.

* deprecations dating to version 0.29 or earlier (released more than 3 years ago) now raise errors instead of
  printing a warning.


New features
^^^^^^^^^^^^

* implemented :py:obj:`Array.value_counts()`, which computes the number of occurrences of each unique value in an array.

* added :py:obj:`Session.nbytes` and added :py:obj:`Session.memory_used` attributes.

* added ``display`` argument to :py:obj:`Array.compact()` to display a message if some axes were "compacted".


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* made all I/O functions/methods/constructors to accept either a string or a pathlib.Path object
  for all arguments representing a path (closes :issue:`896`).

* added type hints for all remaining functions and methods which improves autocompletion in editors (such as PyCharm).
  Closes :issue:`864`.

* made :py:obj:`ipfp()` faster (the smaller the array, the larger the improvement)
  For example, for small arrays it is several times faster than before, for 1000x1000 arrays it is about 30% faster.

* made operations between two Arrays with the same axes much faster.

* made Array[] faster in the usual/simple cases.

* made Array.i[] much faster.


Fixes
^^^^^

* fixed displaying plots made via :py:obj:`Array.plot()` outside of LArray editor (closes :issue:`1019`).

* fixed :py:obj:`Array.insert()` when no label is provided (closes :issue:`879`).

* fixed comparison between :py:obj:`Array` and :py:obj:`None` returning False instead of an array of boolean values
  (closes :issue:`988`)

* fixed binary operation between an :py:obj:`Array` and an :py:obj:`Axis` returning False.

* fixed Array.ipoints[] when not all dimensions are given in the key.

* fixed :py:obj:`AxisCollection.split_axes()` with anonymous axes.

* fixed the ``names`` argument in :py:obj:`Array.split_axes()` and :py:obj:`AxisCollection.split_axes()` not working
  in some cases.

* fixed taking a subset of an Excel range (e.g. myworkbook['my_sheet']['A2:C5'][1:])

* fixed setting the first sheet via position in a new workbook opened via open_excel(overwrite=True)

    with open_excel(fpath, overwrite_file=True) as wb:
        wb[0] = <something>