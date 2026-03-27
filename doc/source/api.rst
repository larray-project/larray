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
--------------------------- --------------------------------------------------------------
:attr:`Axis.dtype`          Data type for the axis labels.
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

   Axis.index
   Axis.containing
   Axis.startingwith
   Axis.endingwith
   Axis.matching
   Axis.min
   Axis.max

Modifying/Selecting
-------------------

.. autosummary::
   :toctree: _generated/

   Axis.__getitem__
   Axis.i
   Axis.by
   Axis.rename
   Axis.extend
   Axis.insert
   Axis.replace
   Axis.apply
   Axis.union
   Axis.intersection
   Axis.difference
   Axis.align
   Axis.split
   Axis.ignore_labels
   Axis.astype

Testing
-------

.. autosummary::
   :toctree: _generated/

   Axis.iscompatible
   Axis.equals

Save
----

.. autosummary::
   :toctree: _generated/

   Axis.to_hdf

.. _api-group:

Group
=====

IGroup
------

.. autosummary::
   :toctree: _generated/

   IGroup

.. autosummary::
   :toctree: _generated/

   IGroup.named
   IGroup.with_axis
   IGroup.by
   IGroup.equals
   IGroup.translate
   IGroup.union
   IGroup.intersection
   IGroup.difference
   IGroup.containing
   IGroup.startingwith
   IGroup.endingwith
   IGroup.matching
   IGroup.to_hdf

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
   LGroup.equals
   LGroup.translate
   LGroup.union
   LGroup.intersection
   LGroup.difference
   LGroup.containing
   LGroup.startingwith
   LGroup.endingwith
   LGroup.matching
   LGroup.to_hdf

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
   AxisCollection.axis_id
   AxisCollection.ids
   AxisCollection.iter_labels

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
   AxisCollection.rename
   AxisCollection.replace
   AxisCollection.set_labels
   AxisCollection.without
   AxisCollection.combine_axes
   AxisCollection.split_axes
   AxisCollection.align

Testing
-------

.. autosummary::
   :toctree: _generated/

   AxisCollection.isaxis
   AxisCollection.check_compatible

.. _api-array:

Array
=====

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
* :ref:`la_iter`
* :ref:`la_op`
* :ref:`la_misc`
* :ref:`la_to_pandas`
* :ref:`la_plotting`


.. _la_overview:

Overview
--------

.. autosummary::
   :toctree: _generated/

   Array

.. _la_creation_func:

Array Creation Functions
------------------------

.. autosummary::
   :toctree: _generated/

   sequence
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

   Array.copy
   Array.astype

.. _la_inspecting:

Inspecting
----------

================== ==============================================================
Array.data         Data of the array (Numpy ndarray)
------------------ --------------------------------------------------------------
Array.axes         Axes of the array (AxisCollection)
------------------ --------------------------------------------------------------
Array.title        Title of the array (str)
================== ==============================================================

.. autosummary::
   :toctree: _generated/

   Array.info
   Array.shape
   Array.ndim
   Array.dtype
   Array.size
   Array.nbytes
   Array.memory_used

.. _la_selecting:

Modifying/Selecting
-------------------

.. autosummary::
   :toctree: _generated/

   Array.i
   Array.points
   Array.ipoints
   Array.iflat
   Array.set
   Array.drop
   Array.ignore_labels
   Array.filter
   Array.apply
   Array.apply_map

.. _la_axes_labels:

Changing Axes or Labels
-----------------------

.. autosummary::
   :toctree: _generated/

   Array.set_axes
   Array.rename
   Array.set_labels
   Array.combine_axes
   Array.split_axes
   Array.reverse

.. _la_agg:

Aggregation Functions
---------------------

.. autosummary::
   :toctree: _generated/

   Array.sum
   Array.sum_by
   Array.prod
   Array.prod_by
   Array.cumsum
   Array.cumprod
   Array.mean
   Array.mean_by
   Array.median
   Array.median_by
   Array.var
   Array.var_by
   Array.std
   Array.std_by
   Array.percentile
   Array.percentile_by
   Array.ptp
   Array.with_total
   Array.percent
   Array.ratio
   Array.rationot0
   Array.growth_rate
   Array.describe
   Array.describe_by
   Array.value_counts

.. _la_sorting:

Sorting
-------

.. autosummary::
   :toctree: _generated/

   Array.sort_labels
   Array.sort_values
   Array.labelsofsorted
   Array.indicesofsorted

.. _la_reshaping:

Reshaping/Extending/Reordering
------------------------------

.. autosummary::
   :toctree: _generated/

   Array.reshape
   Array.reshape_like
   Array.compact
   Array.reindex
   Array.transpose
   Array.expand
   Array.prepend
   Array.append
   Array.extend
   Array.insert
   Array.broadcast_with
   Array.align

.. _la_testing:

Testing/Searching
-----------------

.. autosummary::
   :toctree: _generated/

   Array.equals
   Array.allclose
   Array.eq
   Array.isin
   Array.nonzero
   Array.all
   Array.all_by
   Array.any
   Array.any_by
   Array.min
   Array.min_by
   Array.max
   Array.max_by
   Array.labelofmin
   Array.indexofmin
   Array.labelofmax
   Array.indexofmax

.. _la_iter:

Iterating
---------

.. autosummary::
   :toctree: _generated/

   Array.keys
   Array.values
   Array.items

.. _la_op:

Operators
---------

================================================== ==============================
:py:meth:`@ <Array.__matmul__>`                    Matrix multiplication
================================================== ==============================

.. _la_misc:

Miscellaneous
-------------

.. autosummary::
   :toctree: _generated/

   Array.divnot0
   Array.clip
   Array.shift
   Array.roll
   Array.diff
   Array.unique
   Array.to_clipboard

.. _la_to_pandas:

Converting to Pandas objects
----------------------------

.. autosummary::
   :toctree: _generated/

   Array.to_series
   Array.to_frame

.. _la_plotting:

Plotting
--------

.. autosummary::
   :toctree: _generated/

   Array.plot

.. _api-ufuncs:

Utility Functions
=================

* :ref:`ufuncs_misc`
* :ref:`ufuncs_rounding`
* :ref:`ufuncs_exp_log`
* :ref:`ufuncs_trigo`
* :ref:`ufuncs_hyper`
* :ref:`ufuncs_complex`
* :ref:`ufuncs_floating`

.. _ufuncs_misc:

Miscellaneous
-------------

.. autosummary::
   :toctree: _generated/

   where
   maximum
   minimum
   inverse
   interp
   convolve
   absolute
   fabs
   isscalar
   isnan
   isinf
   nan_to_num
   sqrt
   i0
   sinc

.. _ufuncs_rounding:

Rounding
--------

.. autosummary::
   :toctree: _generated/

   round
   floor
   ceil
   trunc
   rint
   fix

.. _ufuncs_exp_log:

Exponents And Logarithms
------------------------

.. autosummary::
   :toctree: _generated/

    exp
    expm1
    exp2
    log
    log10
    log2
    log1p
    logaddexp
    logaddexp2

.. _ufuncs_trigo:

Trigonometric functions
-----------------------

.. autosummary::
   :toctree: _generated/

    sin
    cos
    tan
    arcsin
    arccos
    arctan
    hypot
    arctan2
    degrees
    radians
    unwrap

.. _ufuncs_hyper:

Hyperbolic functions
--------------------

.. autosummary::
   :toctree: _generated/

   sinh
   cosh
   tanh
   arcsinh
   arccosh
   arctanh

.. _ufuncs_complex:

Complex Numbers
---------------

.. autosummary::
   :toctree: _generated/

   angle
   real
   imag
   conj

.. _ufuncs_floating:

Floating Point Routines
-----------------------

.. autosummary::
   :toctree: _generated/

   signbit
   copysign
   frexp
   ldexp

.. _api-metadata:

Metadata
========

.. autosummary::
   :toctree: _generated/

   Metadata

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
   read_stata

Write
-----

.. autosummary::
   :toctree: _generated/

   Array.to_csv
   Array.to_excel
   Array.to_hdf
   Array.to_stata
   Array.dump

Excel
=====

.. autosummary::
   :toctree: _generated/

   open_excel

.. autosummary::
   :toctree: _generated/

   Workbook
   Workbook.sheet_names
   Workbook.save
   Workbook.close
   Workbook.app

ExcelReport
===========

.. autosummary::
   :toctree: _generated/

   ExcelReport
   ExcelReport.template_dir
   ExcelReport.template
   ExcelReport.set_item_default_size
   ExcelReport.graphs_per_row
   ExcelReport.new_sheet
   ExcelReport.sheet_names
   ExcelReport.to_excel

ReportSheet
===========

.. autosummary::
   :toctree: _generated/

   ReportSheet
   ReportSheet.template_dir
   ReportSheet.template
   ReportSheet.set_item_default_size
   ReportSheet.graphs_per_row
   ReportSheet.add_title
   ReportSheet.add_graph
   ReportSheet.add_graphs
   ReportSheet.newline

.. _api-misc:

Miscellaneous
=============

.. autosummary::
   :toctree: _generated/

   asarray
   from_frame
   from_series
   get_example_filepath
   set_options
   get_options
   labels_array
   union
   stack
   identity
   diag
   eye
   ipfp
   wrap_elementwise_array_func
   zip_array_values
   zip_array_items

.. _api-session:

Session
=======

.. autosummary::
   :toctree: _generated/

   Session
   arrays
   local_arrays
   global_arrays
   load_example_data

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

Testing
-------

.. autosummary::
   :toctree: _generated/

   Session.element_equals
   Session.equals

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
   Session.update
   Session.apply
   Session.transpose

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

CheckedArray
============

.. autosummary::
   :toctree: _generated/

   CheckedArray

CheckedSession
==============

.. autosummary::
   :toctree: _generated/

   CheckedSession

CheckedParameters
=================

.. autosummary::
   :toctree: _generated/

   CheckedParameters

.. _api-editor:

Editor
======

.. autosummary::
   :toctree: _generated/

   view
   edit
   debug
   compare
   run_editor_on_exception

Random
======

.. autosummary::
   :toctree: _generated/

   random.randint
   random.normal
   random.uniform
   random.permutation
   random.choice

Constants
=========

.. currentmodule:: larray.core.constants
.. autosummary::
   :nosignatures:
   :toctree: _generated/

   nan
   inf
   pi
   e
   euler_gamma
