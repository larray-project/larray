New features
------------

* installing larray-editor (or larrayenv) from conda environment creates a new menu 'LArray' in the Windows start menu.
  It contains a link to open the documentation, a shortcut to launch the user interface in edition mode
  and a shortcut to update larrayenv. Closes issue:`281`.

Miscellaneous improvements
--------------------------

* implemented Session.to_globals(inplace=True) which will update the content of existing arrays instead of creating new
  variables and overwriting them. This ensures the arrays have the same axes in the session than the existing variables.

* added icon to display in Windows start menu and editor windows.

* added 'Getting Started' section in documentation.

Fixes
-----

* fixed disambiguating an ambiguous key by adding the axis within the string, for example
  arr['axis_name[ambiguouslabel]'] (closes :issue:`331`).
