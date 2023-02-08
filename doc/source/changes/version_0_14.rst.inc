Important warning
-----------------

This version is not compatible with the new version of xlwings that just came
out. Consequently, upgrading to this version is different from the usual
"conda update larray". You should rather use:

    conda update larray --no-update-deps

To get the most of this release, you should also install the "qtconsole"
package via:

    conda install qtconsole

Viewer
------

* upgraded session viewer/editor to work like a super-calculator. The input box
  below the array view can be used to type any expression. eg
  array1.sum(x.age) / array2, which will be displayed in the viewer.
  One can also type assignment commands, like:
  array3 = array1.sum(x.age) / array2
  In which case, the new array will be displayed in the viewer AND added to the
  session (appear on the list on the left), so that you can use it in other
  expressions.

  If you have the "qtconsole" package installed (see above), that input box will be a full ipython console. This means:
   - history of typed commands,
   - tab-completion (for example, type "nd<tab>" and it will change to "ndrange"),
   - syntax highlighting,
   - calltips (show the documentation of functions when typing commands using them),
   - help on functions using "?". For example, type "ndrange?<enter>" to get the full documentation about ndrange.
     Use <ESC> or <q> to quit that screen !),
   - etc.

  When having the "qtconsole" package installed, you might get a warning when
  starting the viewer: ::

      WARNING:root:Message signing is disabled.  This is insecure and not recommended!

  This is totally harmless and can be safely ignored !

* made view() and edit() without argument equivalent to view(local_arrays())
  and edit(local_arrays()) respectively.

* made the viewer on large arrays start a lot faster by using a small
  subset of the array to guess the number of decimals to display and whether or
  not to use scientific notation.

* improved compare():
   - added support for comparing sessions. Arrays with differences between sessions are colored in red.
   - use a single array widget instead of 3. This is done by stacking arrays together to create a new dimension. This
     has the following advantages:

      * the filter and scrollbars are de-facto automatically synchronized.
      * any number of arrays can be compared, not just 2. All arrays are compared to the first one.
      * arrays with different sets of compatible axes can be compared (eg compare an array with its mean along an axis).

   - added label to show maximum absolute difference.

* implemented edit(session) in addition to view(session).


Excel support
-------------

* added support for copying sheets via: wb['x'] = wb['y']
  if 'x' sheet already existed, it is completely overwritten.


Core
----

* improved performance. My test models run about 10% faster than with 0.13.

* made cumsum and cumprod aggregate on the last axis by default so that the
  axis does not need to be specified when there is only one.

* implemented much better support for operations using arrays of different
  types. For example,

    - fixed create_sequential when mult, inc and initial are of different types
      eg create_sequential(..., initial=1, inc=0.1) had an unexpected integer
      result because it always used the type of the initial value for the output
    - when appending a string label to an integer axis (eg adding total to an
      age axis by using with_total()), the resulting axis should have a mixed
      type, and not be suddenly all string.
    - stack() now supports arrays with different types.

* made stack support arrays with different axes (the result has the union of all axes)


For completeness
----------------

Excel support
^^^^^^^^^^^^^

* use xlwings (ie live Excel instance) by default for all Excel input/output,
  including read_excel(), session.dump and session.load/Session(filename).
  This has the advantage of more coherent results among the different ways to
  load/save data to Excel and that simple sessions correctly survive a
  round-trip to an .xlsx workbook (ie (named) axes are detected properly).
  However, given the very different library involved, we loose most options
  that read_excel used to provide (courtesy of pandas.read_excel) and some
  bugs were probably introduced in the conversion.

* fixed creating a new file via open_excel()

* fixed loading 1D arrays (ranges with height 1 or width 1) via open_excel()

* fixed sheet['A1'] = array in some cases

* wb.close() only really close if the workbook was not already open in Excel
  when open_excel was called (so that we do not close a workbook a user is
  actually viewing).

* added support for wb.save(filename), or actually for using any relative
  path, instead of a full absolute path.

* when dumping a session to Excel, sort sheets alphabetically instead of
  dumping them in a "random" order.

* try to convert float to int in more situations

Core
^^^^

* added support for using stack() without providing an axis. It creates an
  anonymous wildcard axis of the correct length.

* added aslarray() top-level function to translate anything into an LArray if
  it is not already one

* made labels_array available via `from larray import *`

* fixed binary operations between an array and an axis where the array
  appeared first (eg array > axis). Confusingly, axis < array already worked.

* added check in "a[bool_larray_key]" to make sure key.axes are compatible
  with a.axes

* made create_sequential a lot faster when mult or inc are constants

* made axes without name compatible with any name
  (this is the equivalent of a wildcard name for labels)

* misc cleanup/docstring improvements/improved tests/improved error messages

