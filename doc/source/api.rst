#############
API Reference
#############

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
   :toctree: generated/

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


.. autofunction:: create_sequential

.. autofunction:: ndrange

.. autofunction:: ndtest

.. autofunction:: zeros

.. autofunction:: zeros_like

.. autofunction:: ones

.. autofunction:: ones_like

.. autofunction:: empty

.. autofunction:: empty_like

.. autofunction:: full

.. autofunction:: full_like

Aggregation Functions
---------------------

.. autofunction:: all

.. autofunction:: any

.. autofunction:: min

.. autofunction:: max

.. autofunction:: sum

.. autofunction:: prod

.. autofunction:: cumsum

.. autofunction:: cumprod

.. autofunction:: mean

.. autofunction:: median

.. autofunction:: var

.. autofunction:: std

.. autofunction:: percentile

.. autofunction:: ptp

Session
=======

.. autoclass:: Session
   :members:

Viewer
======

.. automodule:: viewer
   :members:

Input/Output
============

Excel
-----

.. automodule:: excel
   :members:

.. autoclass:: excel.Workbook
   :members:

.. autoclass:: excel.Sheet
   :members:

.. autoclass:: excel.Range
   :members:

Read Functions
--------------

.. autofunction:: read_csv

.. autofunction:: read_eurostat

.. autofunction:: read_excel

.. autofunction:: read_hdf

.. autofunction:: read_tsv

.. autofunction:: read_sas

Miscellaneous
=============

.. autofunction:: aslarray

.. autofunction:: labels_array

.. autofunction:: larray_equal

.. autofunction:: union

.. autofunction:: stack

.. autofunction:: identity

.. autofunction:: diag

.. autofunction:: eye

Apply Iterative Proportional Fitting Procedure
----------------------------------------------

.. automodule:: ipfp
   :members:
