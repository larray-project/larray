#############
API Reference
#############

.. see larray/__init__.py
.. currentmodule:: larray

Axis
====

.. autosummary::
   :toctree: _generated/

   Axis
   Axis.name
   Axis.labels
   Axis.labels_summary
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

Testing
-------

.. autosummary::
   :toctree: _generated/

   Axis.iscompatible
   Axis.equals

Group
=====

PGroup
------

.. autosummary::
   :toctree: _generated/

   PGroup
   Group.named
   Group.with_axis
   Group.by
   PGroup.translate

LGroup
------

.. autosummary::
   :toctree: _generated/

   LGroup
   Group.named
   Group.with_axis
   Group.by
   LGroup.translate

LSet
====

.. autosummary::
   :toctree: _generated/

   LSet

AxisCollection
==============

.. autosummary::
   :toctree: _generated/

   AxisCollection
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


Array
=====

LArray
------

.. autoclass:: LArray
   :members:

Array Creation Functions
------------------------

.. autosummary::
   :toctree: _generated/

   create_sequential
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

Aggregation Functions
---------------------

.. autosummary::
   :toctree: _generated/

   all
   any
   min
   max
   sum
   prod
   cumsum
   cumprod
   mean
   median
   var
   std
   percentile
   ptp

Session
=======

.. autosummary::
   :toctree: _generated/

   Session
   Session.names
   Session.add
   Session.get
   Session.load
   Session.dump
   Session.dump_csv
   Session.dump_excel
   Session.dump_hdf
   Session.filter
   Session.compact
   Session.copy

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


Viewer
======

.. autosummary::
   :toctree: _generated/

   view
   edit
   compare

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
