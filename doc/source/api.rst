#############
API Reference
#############

.. see larray/__init__.py
.. currentmodule:: larray

Axis
====

.. autoclass:: Axis
   :members:

AxisCollection
--------------

.. autoclass:: AxisCollection
   :members:

Group & Set
===========

PGroup
------

.. autoclass:: PGroup
   :members:

LGroup
------

.. autoclass:: LGroup
   :members:

LSet
----

.. autoclass:: LSet
   :members:

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

.. autoclass:: Session
   :members:

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
