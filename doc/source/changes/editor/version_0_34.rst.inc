.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* added support for Python 3.10.

* the initial column width is now set depending on the content and resized automatically when changing the number of
  digits (closes :editor_issue:`145`).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* plot windows title now include the expression used to make the plot (the name of the array
  in most cases) (closes :editor_issue:`233`).

* when displaying an expression (computed array), the window title includes the actual expression
  instead of using '<expr>'.

* :py:obj:`compare()` can now take filepaths as argument (and will load them as a Session) to make
  comparing a in-memory Session with an earlier Session saved on the disk. Those filepaths
  can be given as both str or Path objects. Closes :editor_issue:`229`.

* added support for Path objects (in addition to str representing paths) in :py:obj:`view()` and :py:obj:`edit()`.
  See :issue:`896`.

* when the editor displays currently defined variables (via :py:obj:`debug()` :py:obj:`edit()` or
  :py:obj:`view()` without argument within user code or via an exception when run_editor_on_exception is active),
  LArray functions are not made available systematically in the console anymore (what is available in the console is
  really what was available in the users code). This closes :editor_issue:`199`.

* added support for incomplete slices in "save command history", like in Python slices.
  For example, one can save from line 10 onwards by using "10:" or "10..", i.e. without specifying the last line.
  See :editor_issue:`225`.


Fixes
^^^^^

* fixed :py:obj:`run_editor_on_exception()` so that the larray editor is not opened when trying to stop a program
  (via Ctrl-C or the IDE stop button). Closes :editor_issue:`231`.

* improved the situation when trying to stop a program (via `Ctrl-C` or the IDE stop button) with an LArray Editor
  window open. It used to ignore such events altogether, forcing the IDE to send a "kill" event when pressing the
  button a second time, which could leave some ressource open (e.g Excel instances). Now, the LArray Editor will
  close itself when its parent program is asked to stop *but* so far, it will only do so when the window is active
  again. This makes for an odd behavior but at least cleans up the program properly (closes :editor_issue:`231`).

* when save command history fails, do not do so silently. Closes :editor_issue:`225`.

* fixed saving command history to a path containing spaces. Closes :editor_issue:`244`.

* fixed compare() background color being red for two equal integer arrays instead of white
  (closes :editor_issue:`246`).
