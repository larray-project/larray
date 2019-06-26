.. py:currentmodule:: larray


Syntax changes
^^^^^^^^^^^^^^

* :py:obj:`stack()` ``axis`` argument was renamed to ``axes`` to reflect the fact that the function can now stack
  along multiple axes at once (see below).

* to accommodate for the "simpler pattern language" now supported for those functions, using a regular expression in
  :py:obj:`Axis.matching()` or :py:obj:`Group.matching()` now requires passing the pattern as an explicit ``regex``
  keyword argument instead of just the first argument of those methods. For example ``my_axis.matching('test.*')``
  becomes ``my_axis.matching(regex='test.*')``.

* ``LArray.as_table()`` is deprecated because it duplicated functionality found in :py:obj:`LArray.dump()`.
  Please only use :py:obj:`LArray.dump()` from now on.

* renamed ``a_min`` and ``a_max`` arguments of :py:obj:`LArray.clip()` to ``minval`` and ``maxval`` respectively
  and made them optional (closes :issue:`747`).


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* modified the behavior of the ``pattern`` argument of :py:obj:`Session.filter()` to actually support patterns instead
  of only checking if the object names start with the pattern. Special characters include ``?`` for matching any single
  character and ``*`` for matching any number of characters. Closes :issue:`703`.

  .. warning::

      If you were using Session.filter, you must add a ``*`` to your pattern to keep your code working.
      For example, ``my_session.filter('test')`` must be changed to ``my_session.filter('test*')``.

* :py:obj:`LArray.equals()` now returns True for arrays even when axes are in a different order or some axes are
  missing on either side (but the data is constant over that axis on the other side). Closes :issue:`237`.

  .. warning::

      If you were using :py:obj:`LArray.equals()` **and** want to keep the old, stricter, behavior, you must add
      ``check_axes=True``.


New features
^^^^^^^^^^^^

* added :py:obj:`set_options()` and :py:obj:`get_options()` functions to respectively set and get options for larray.
  Available options currently include ``display_precision`` for controlling the number of decimal digits used when
  showing floating point numbers, ``display_maxlines`` to control the maximum number of lines to use when displaying
  an array, etc. :py:obj:`set_options()` can used either like a normal function to set the options globally or within a
  ``with`` block to set them only temporarily. Closes :issue:`274`.

* implemented :py:obj:`read_stata()` and :py:obj:`LArray.to_stata()` to read arrays from and write arrays to Stata .dta
  files.

* implemented :py:obj:`LArray.isin()` method to check whether each value of an array is contained in a list (or array)
  of values.

* implemented :py:obj:`LArray.unique()` method to compute unique values (or sub-arrays) for an array,
  optionally along axes.

* implemented :py:obj:`LArray.apply()` method to apply a python function to all values of an array or to all sub-arrays
  along some axes of an array and return the result. This is an extremely versatile method as it can be used both with
  aggregating functions or element-wise functions.

* implemented :py:obj:`LArray.apply_map()` method to apply a transformation mapping to array elements. For example, this
  can be used to transform some numeric codes to labels.

* implemented :py:obj:`LArray.reverse()` method to reverse one or several axes of an array (closes :issue:`631`).

* implemented :py:obj:`LArray.roll()` method to roll the cells of an array n-times to the right along an axis. This is
  similar to :py:obj:`LArray.shift()`, except that cells which are pushed "outside of the axis" are reintroduced on the
  opposite side of the axis instead of being dropped.

* implemented :py:obj:`Axis.apply()` method to transform an axis labels by a function and return a new Axis.

* added :py:obj:`Session.update()` method to add and modify items from an existing session by passing
  either another session or a dict-like object or an iterable object with (key, value) pairs (closes :issue:`754`).

* implemented :py:obj:`AxisCollection.rename()` to rename axes of an AxisCollection, independently of any array.

* implemented :py:obj:`AxisCollection.set_labels()` (closes :issue:`782`).

* implemented :py:obj:`wrap_elementwise_array_func()` function to make a function defined in another library work with
  LArray arguments instead of with numpy arrays.

* implemented :py:obj:`LArray.keys()`, :py:obj:`LArray.values()` and :py:obj:`LArray.items()`
  methods to respectively loop on an array labels, values or (key, value) pairs.

* implemented :py:obj:`zip_array_values()` and :py:obj:`zip_array_items()` to loop respectively on several arrays values
  or (key, value) pairs.

* implemented :py:obj:`AxisCollection.iter_labels()` to iterate over all (possible combinations of) labels of the axes
  of the collection.


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved speed of :py:obj:`read_hdf()` function when reading a stored LArray object dumped with
  the current and future version of larray. To get benefit of the speedup of reading arrays dumped
  with older versions of larray, please read and re-dump them. Closes :issue:`563`.

* allowed to not specify the axes in :py:obj:`LArray.set_labels()` (closes :issue:`634`):

    >>> a = ndtest('nat=BE,FO;sex=M,F')
    >>> a
    nat\sex  M  F
         BE  0  1
         FO  2  3
    >>> a.set_labels({'M': 'Men', 'BE': 'Belgian'})
    nat\sex  Men  F
    Belgian    0  1
         FO    2  3

* :py:obj:`LArray.set_labels()` can now take functions to transform axes labels (closes :issue:`536`).

    >>> arr = ndtest((2, 2))
    >>> arr
    a\b  b0  b1
     a0   0   1
     a1   2   3
    >>> arr.set_labels('a', str.upper)
    a\b  b0  b1
     A0   0   1
     A1   2   3

* implemented the same "simpler pattern language" in :py:obj:`Axis.matching()` and :py:obj:`Group.matching()` than in
  :py:obj:`Session.filter()`, in addition to regular expressions (which now require using the ``regexp`` argument).

* py:obj:`stack()` can now stack along several axes at once (closes :issue:`56`).

    >>> country = Axis('country=BE,FR,DE')
    >>> gender = Axis('gender=M,F')
    >>> stack({('BE', 'M'): 0,
    ...        ('BE', 'F'): 1,
    ...        ('FR', 'M'): 2,
    ...        ('FR', 'F'): 3,
    ...        ('DE', 'M'): 4,
    ...        ('DE', 'F'): 5},
    ...       (country, gender))
    country\gender  M  F
                BE  0  1
                FR  2  3
                DE  4  5

* py:obj:`stack()` using a dictionary as elements can now use a simple axis name instead of requiring a full axis
  object. This will print a warning on Python < 3.7 though because the ordering of labels is not guaranteed in
  that case. Closes :issue:`755` and :issue:`581`.

* py:obj:`stack()` using keyword arguments can now use a simple axis name instead of requiring a full axis
  object, even on Python < 3.6. This will print a warning though because the ordering of labels is not guaranteed in
  that case.

* added password argument to :py:obj:`Workbook.save()` to allow protecting Excel files with a password.

* added option ``exact`` to ``join`` argument of :py:obj:`Axis.align()` and :py:obj:`LArray.align()` methods.
  Instead of aligning, passing ``join='exact'`` to the ``align`` method will raise an error when axes are not equal.
  Closes :issue:`338`.

* made :py:obj:`Axis.by()` and :py:obj:`Group.by()` return a list of named groups instead of anonymous groups.
  By default, group names are defined as ``<start>:<end>``. This can be changed via the new ``template`` argument:

    >>> age = Axis('age=0..6')
    >>> age
    Axis([0, 1, 2, 3, 4, 5, 6], 'age')
    >>> age.by(3)
    (age.i[0:3] >> '0:2', age.i[3:6] >> '3:5', age.i[6:7] >> '6')
    >>> age.by(3, step=2)
    (age.i[0:3] >> '0:2', age.i[2:5] >> '2:4', age.i[4:7] >> '4:6', age.i[6:7] >> '6')
    >>> age.by(3, template='{start}-{end}')
    (age.i[0:3] >> '0-2', age.i[3:6] >> '3-5', age.i[6:7] >> '6')

  Closes :issue:`669`.

* allowed to specify an axis by its position when selecting a subset of an array using the string notation:

    >>> pop_mouv = ndtest('geo_from=BE,FR,UK;geo_to=BE,FR,UK')
    >>> pop_mouv
    geo_from\geo_to  BE  FR  UK
                 BE   0   1   2
                 FR   3   4   5
                 UK   6   7   8
    >>> pop_mouv['0[BE, UK]']   # equivalent to pop_mouv[pop_mouv.geo_from['BE,UK']]
    geo_from\geo_to  BE  FR  UK
                 BE   0   1   2
                 UK   6   7   8
    >>> pop_mouv['1.i[0, 2]']   # equivalent to pop_mouv[pop_mouv.geo_to.i[0, 2]]
    geo_from\geo_to  BE  UK
                 BE   0   2
                 FR   3   5
                 UK   6   8

  Closes :issue:`671`.

* added documentation and examples for :py:obj:`where()`, :py:obj:`maximum()`
  and :py:obj:`minimum()` functions (closes :issue:`700`)

* updated the ``Working With Sessions`` section of the tutorial (closes :issue:`568`).

* added dtype argument to LArray to set the type of the array explicitly instead of relying on auto-detection.

* added dtype argument to stack to set the type of the resulting array explicitly instead of relying on auto-detection.

* allowed to pass a single axis or group as ``axes_to_reindex`` argument
  of the :py:obj:`LArray.reindex()` method (closes :issue:`712`).

* :py:obj:`LArray.dump()` gained a few extra arguments to further customize output :
  - axes_names : to specify whether or not the output should contain the axes names (and which)
  - maxlines and edgeitems : to dump only the start and end of large arrays
  - light : to output axes labels only when they change instead of repeating them on each line
  - na_repr : to specify how to represent N/A (NaN) values

* substantially improved performance of creating, iterating, and doing a few other operations over larray objects.
  This solves a few pathological cases of slow operations, especially those involving many small-ish arrays but sadly
  the overall performance improvement is negligible over most of the real-world models using larray that we tested these
  changes on.


Fixes
^^^^^

* fixed dumping to Excel arrays of "object" dtype containing NaN values using numpy float types (fixes the
  infamous 65535 bug).

* fixed :py:obj:`LArray.divnot0()` being slow when the divisor has many axes and many zeros (closes :issue:`705`).

* fixed maximum length of sheet names (31 characters instead of 30 characters) when adding a new sheet
  to an Excel Workbook (closes :issue:`713`).

* fixed missing documentation of many functions in :ref:`Utility Functions <api-ufuncs>` section
  of the API Reference (closes :issue:`698`).

* fixed arithmetic operations between two sessions returning a nan value for each axis and group (closes :issue:`725`).

* fixed dumping sessions with metadata in HDF format (closes :issue:`702`).

* fixed minimum version of pandas to install. The minimum version is now 0.20.0.

* fixed from_frame for dataframes with non string index names.

* fixed creating an LSet from an IGroup with a (single) scalar key

    >>> a = Axis('a=a0,a1,a2')
    >>> a.i[1].set()
    a['a1'].set()
