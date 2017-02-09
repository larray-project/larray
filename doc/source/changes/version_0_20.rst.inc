IMPORTANT
---------

this release should installed using:

  conda install larrayenv

New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.

.. _misc:

Miscellaneous improvements
--------------------------

* viewer: view and edit functions accept path to file containing
  arrays to be loaded as argument.

* plot function from viewer updated: can plot several rows/columns.
  Draw a new curve for each row except if only one column has been
  selected.

* allowed to get axis from array using syntax a.axis_name
  (instead of a.axes.axis_name)

Fixes
-----

* viewer: filtering on anonymous axes is now possible
  (closes :issue:`33`).

* viewer: selection of entire rows/columns will
  load all data (closes :issue:`37`).

* 'by' aggregation functions accept both axes and groups
  (closes :issue:`59`).

* nb_index is no longer required in read_excel when
  using xlrd engine (closes :issue:`66`).


