.. _start_api:

#############
API Reference
#############

.. see larray/__init__.py
.. currentmodule:: larray

.. _api-axis:

Axis
====

.. autosummary::
   :toctree: _generated/

   Axis

Exploring
---------

=========================== ==============================================================
Axis.name                   Name of the axis. None in the case of an anonymous axis.
--------------------------- --------------------------------------------------------------
:attr:`Axis.labels`         Labels of the axis.
--------------------------- --------------------------------------------------------------
:attr:`Axis.labels_summary` Short representation of the labels.
=========================== ==============================================================

Copying
-------

.. autosummary::
   :toctree: _generated/

   Axis.copy

Searching
---------

.. autosummary::
   :toctree: _generated/

   Axis.translate
   Axis.matches
   Axis.startswith
   Axis.endswith

Modifying/Selecting/Searching
-----------------------------

.. autosummary::
   :toctree: _generated/

   Axis.__getitem__
   Axis.i
   Axis.by
   Axis.rename
   Axis.subaxis
   Axis.extend
   Axis.replace
   Axis.union
   Axis.intersection
   Axis.difference

Testing
-------

.. autosummary::
   :toctree: _generated/

   Axis.iscompatible
   Axis.equals

.. _api-group:

Group
=====

PGroup
------

.. autosummary::
   :toctree: _generated/

   PGroup

.. autosummary::
   :toctree: _generated/

   PGroup.named
   PGroup.with_axis
   PGroup.by
   PGroup.translate
   PGroup.union
   PGroup.intersection
   PGroup.difference

LGroup
------

.. autosummary::
   :toctree: _generated/

   LGroup

.. autosummary::
   :toctree: _generated/

   LGroup.named
   LGroup.with_axis
   LGroup.by
   LGroup.translate
   LGroup.union
   LGroup.intersection
   LGroup.difference

.. _api-set:

LSet
====

.. autosummary::
   :toctree: _generated/

   LSet

.. _api-axiscollection:

AxisCollection
==============

.. autosummary::
   :toctree: _generated/

   AxisCollection

.. autosummary::
   :toctree: _generated/

   AxisCollection.names
   AxisCollection.display_names
   AxisCollection.labels
   AxisCollection.shape
   AxisCollection.size
   AxisCollection.info
   AxisCollection.copy

Searching
---------

.. autosummary::
   :toctree: _generated/

   AxisCollection.keys
   AxisCollection.index
   AxisCollection.translate_full_key
   AxisCollection.axis_id
   AxisCollection.ids

Modifying/Selecting
-------------------

.. autosummary::
   :toctree: _generated/

   AxisCollection.get
   AxisCollection.get_by_pos
   AxisCollection.get_all
   AxisCollection.pop
   AxisCollection.append
   AxisCollection.extend
   AxisCollection.insert
   AxisCollection.replace
   AxisCollection.without
   AxisCollection.combine_axes
   AxisCollection.split_axis

Testing
-------

.. autosummary::
   :toctree: _generated/

   AxisCollection.isaxis
   AxisCollection.check_compatible

.. _api-larray:

LArray
======

* :ref:`la_overview`
* :ref:`la_creation_func`
* :ref:`la_copying`
* :ref:`la_inspecting`
* :ref:`la_selecting`
* :ref:`la_axes_labels`
* :ref:`la_agg`
* :ref:`la_sorting`
* :ref:`la_reshaping`
* :ref:`la_testing`
* :ref:`_la_op:`
* :ref:`la_misc`
* :ref:`la_to_pandas`
* :ref:`la_plotting`


.. _la_overview:

Overview
--------

.. autosummary::
   :toctree: _generated/

   LArray

.. _la_creation_func:

Array Creation Functions
------------------------

.. autosummary::
   :toctree: _generated/

   sequence
   ndrange
   ndtest
   zeros
   zeros_like
   ones
   ones_like
   empty
   empty_like
   full
   full_like

.. _la_copying:

Copying
-------

.. autosummary::
   :toctree: _generated/

   LArray.copy

.. _la_inspecting:

Inspecting
----------

=================== ==============================================================
LArray.data         Data of the array (Numpy ndarray)
------------------- --------------------------------------------------------------
LArray.axes         Axes of the array (AxisCollection)
------------------- --------------------------------------------------------------
LArray.title        Title of the array (str)
=================== ==============================================================

.. autosummary::
   :toctree: _generated/

   LArray.info
   LArray.shape
   LArray.ndim
   LArray.dtype
   LArray.size
   LArray.nbytes
   LArray.memory_used
   LArray.astype

.. _la_selecting:

Modifying/Selecting
-------------------

.. autosummary::
   :toctree: _generated/

   LArray.i
   LArray.points
   LArray.ipoints
   LArray.set
   LArray.drop_labels
   LArray.filter

.. _la_axes_labels:

Changing Axes or Labels
-----------------------

.. autosummary::
   :toctree: _generated/

   LArray.set_axes
   LArray.rename
   LArray.set_labels
   LArray.combine_axes
   LArray.split_axis

.. _la_agg:

Aggregation Functions
---------------------

.. autosummary::
   :toctree: _generated/

   LArray.sum
   LArray.sum_by
   LArray.prod
   LArray.prod_by
   LArray.cumsum
   LArray.cumprod
   LArray.mean
   LArray.mean_by
   LArray.median
   LArray.median_by
   LArray.var
   LArray.var_by
   LArray.std
   LArray.std_by
   LArray.percentile
   LArray.percentile_by
   LArray.ptp
   LArray.with_total
   LArray.percent
   LArray.growth_rate
   LArray.describe
   LArray.describe_by

.. _la_sorting:

Sorting
-------

.. autosummary::
   :toctree: _generated/

   LArray.sort_axis
   LArray.sort_values
   LArray.argsort
   LArray.posargsort

.. _la_reshaping:

Reshaping/Extending/Reordering
------------------------------

.. autosummary::
   :toctree: _generated/

   LArray.reshape
   LArray.reshape_like
   LArray.compact
   LArray.reindex
   LArray.transpose
   LArray.expand
   LArray.prepend
   LArray.append
   LArray.extend
   LArray.broadcast_with

.. _la_testing:

Testing/Searching
-----------------

.. autosummary::
   :toctree: _generated/

   LArray.nonzero
   LArray.all
   LArray.all_by
   LArray.any
   LArray.any_by
   LArray.min
   LArray.min_by
   LArray.max
   LArray.max_by
   LArray.argmin
   LArray.posargmin
   LArray.argmax
   LArray.posargmax

.. _la_op:

Operators
---------

=================================================== ==============================
:py:meth:`@ <LArray.__matmul__>`                    Matrix multiplication
=================================================== ==============================

.. _la_misc:

Miscellaneous
-------------

.. autosummary::
   :toctree: _generated/

   LArray.ratio
   LArray.rationot0
   LArray.divnot0
   LArray.clip
   LArray.shift
   LArray.diff
   LArray.to_clipboard
   round
   floor
   ceil
   trunc
   sqrt
   absolute
   fabs
   where
   isnan
   isinf
   nan_to_num

.. _la_to_pandas:

Converting to Pandas objects
----------------------------

.. autosummary::
   :toctree: _generated/

   LArray.to_series
   LArray.to_frame

.. _la_plotting:

Plotting
--------

.. autosummary::
   :toctree: _generated/

   LArray.plot

.. _api-IO:

Input/Output
============

Read
----

.. autosummary::
   :toctree: _generated/

   read_csv
   read_tsv
   read_excel
   read_hdf
   read_eurostat
   read_sas

Write
-----

.. autosummary::
   :toctree: _generated/

   LArray.to_csv
   LArray.to_excel
   LArray.to_hdf

Excel
=====

.. autosummary::
   :toctree: _generated/

   open_excel

.. _api-misc:

Miscellaneous
=============

.. autosummary::
   :toctree: _generated/

   aslarray
   labels_array
   larray_equal
   union
   stack
   identity
   diag
   eye
   ipfp
   load_example_data
   local_arrays

.. _api-session:

Session
=======

.. autosummary::
   :toctree: _generated/

   Session

Exploring
---------

.. autosummary::
   :toctree: _generated/

   Session.names
   Session.keys
   Session.values
   Session.items
   Session.summary

Copying
-------

.. autosummary::
   :toctree: _generated/

   Session.copy

Selecting
---------

.. autosummary::
   :toctree: _generated/

   Session.get

Modifying
---------

.. autosummary::
   :toctree: _generated/

   Session.add
   Session.get

Filtering/Cleaning
------------------

.. autosummary::
   :toctree: _generated/

   Session.filter
   Session.compact

Load/Save
---------

.. autosummary::
   :toctree: _generated/

   Session.load
   Session.save
   Session.to_csv
   Session.to_excel
   Session.to_hdf
   Session.to_pickle

.. _api-editor:

Editor
======

.. autosummary::
   :toctree: _generated/

   view
   edit
   compare
