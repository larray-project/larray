.. py:currentmodule:: larray


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved the tutorial and some examples to make them more intuitive (closes :issue:`829`).


Fixes
^^^^^

* fixed loading arrays with more than 2 dimensions but no axes names (even when specifying nb_axes explicitly). This
  case mostly occurs when trying to load a specific range of an Excel file (closes :issue:`830` and :issue:`831`).
