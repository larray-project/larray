Miscellaneous improvements
--------------------------

* Excel Workbooks opened with open_excel(visible=False) will use the global Excel instance by default and those using
  visible=True will use a new Excel instance by default (closes :issue:`405`).


Fixes
-----

* fixed view() which did not show any array (closes :editor_issue:`57`).

* fixed exceptions in the viewer crashing it when a Qt app was created (e.g. from a plot) before the viewer was started
  (closes :editor_issue:`58`).

* fixed compare() arrays names not being determined correctly (closes :editor_issue:`61`).

* fixed filters and title not being updated when displaying array created via the console (closes :editor_issue:`55`).

* fixed array grid not being updated when selecting a variable when no variable was selected
  (closes :editor_issue:`56`).

* fixed digits not being automatically updated when changing filters.
