.. py:currentmodule:: larray_editor

Fixes
^^^^^

* changes made to arrays in the console using the "points" syntax (for example: `arr.points['a0,a1', 'b0,b1'] = 0`)
  and the other special `.something[]` syntaxes were not detected by the viewer and thus not displayed (closes
  :editor_issue:`269`).

* fixed copying to clipboard an array filtered on all dimensions (to a single value). Closes :editor_issue:`270`.
