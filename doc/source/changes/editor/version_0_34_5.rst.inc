.. py:currentmodule:: larray_editor

Fixes
^^^^^

* fixed console plots when xlwings 0.31.4+ is installed (closes :editor_issue:`278`).

* fixed some inplace modifications on arrays done via in the console not refreshing
  the displayed array automatically and/or not adding a `*` to the window title
  to inform the session is modified (closes :editor_issue:`22` and :editor_issue:`280`).
