.. currentmodule:: larray

Getting Started
===============

To use the LArray library, the first thing to do is to import it

.. ipython:: python

    from larray import *

Create an array
---------------

Working with the LArray library mainly consists of manipulating :ref:`LArray <api-larray>` data structures.
They represent N-dimensional labelled arrays and are composed of data (numpy ndarray), :ref:`axes <api-axis>`
and optionally some metadata. An axis contains a list of labels and may have a name (if not given, the axis is
anonymous).

You can create an array from scratch by supplying data, axes and optionally some metadata:

.. ipython:: python

    # define data
    data = [[20, 22],
            [33, 31],
            [79, 81],
            [28, 34]]

    # define axes
    age = Axis(["0-9", "10-17", "18-66", "67+"], "age")
    sex = Axis(["M", "F"], "sex")

    # create LArray object
    arr = LArray(data, [age, sex], meta=[("title", "population by age category and sex")])
    arr

Here are the key properties for an array:

.. ipython:: python

    # array summary : dimensions + description of axes
    arr.info

    # number of dimensions
    arr.ndim

    # array dimensions
    arr.shape

    # number of elements
    arr.size

    # size in memory
    arr.memory_used

    # type of the data of the array
    arr.dtype

Arrays can be generated through dedicated functions:

* :py:func:`zeros` : fills an array with 0
* :py:func:`ones` : fills an array with 1
* :py:func:`full` : fills an array with a given
* :py:func:`eye` : identity matrix
* :py:func:`ndtest` : creates a test array with increasing numbers as data
* :py:func:`sequence` : creates an array by sequentially applying modifications to the array along axis.

.. ipython:: python

   zeros([age, sex])

   ndtest((3, 3))

Save/Load an array
------------------

The LArray library offers many I/O functions to read and write arrays in various formats
(CSV, Excel, HDF5, pickle). For example, to save an array in a CSV file, call the method
:py:meth:`~LArray.to_csv`:

.. ipython:: python

    # let us first define one more axis
    year = Axis([2016, 2017, 2018], "year")

    # then create a test array with 3 axes
    arr = ndtest([age, sex, year])
    arr

    # now save that array to a .csv file
    arr.to_csv('test_array.csv')

The content of 'test_array.csv' file is ::

    age,sex\children,0,1,2+
    0-9,M,0,1,2
    0-9,F,3,4,5
    10-17,M,6,7,8
    10-17,F,9,10,11
    18-66,M,12,13,14
    18-66,F,15,16,17
    67+,M,18,19,20
    67+,F,21,22,23

.. note::
   In CSV or Excel files, the last dimension is horizontal and the names of the
   two last dimensions are separated by a ``\``.

To load a saved array, call the function :py:meth:`read_csv`:

.. ipython:: python

    arr = read_csv('test_array.csv')
    arr

Other input/output functions are described in the :ref:`corresponding section <api-IO>`
of the API documentation.

Selecting a subset
------------------

To select an element or a subset of an array, use brackets [ ]. In Python we usually use the term *indexing* for this
operation.

Let us start by selecting a single element:

.. ipython:: python

    arr['67+', 'F', 2017]

Labels can be given in arbitrary order

.. ipython:: python

    arr[2017, 'F', '67+']

When selecting a larger subset the result is an array

.. ipython:: python

    arr[2017]
    arr['M']

When selecting several labels for the same axis, they must be given as a list (enclosed by ``[ ]``)

.. ipython:: python
    arr['F', ['0-9', '10-17']]


.. warning::

    Selecting by labels as above only works as long as there is no ambiguity.
    When several axes have common labels and you do not specify explicitly
    on which axis to work, it fails with an error ending with something like
    ValueError: somelabel is ambiguous (valid in axis1, axis2).

For example, let us create a test array with an ambiguous label.
First create an axis (some kind of status code) with an "F" label (remember we already had an "F" label on the sex
axis).

.. ipython:: python

    status = Axis(["A", "C", "F"], "status")

Then create a test array using both axes

.. ipython:: python

    ambiguous_arr = ndtest([sex, status, year])
    ambiguous_arr

If we try to get to a subset of the array with "F"...

.. ipython:: python
    :verbatim:

    ambiguous_arr[2017, "F"]

... we receive back a volley of insults ::

    [some long error message ending with the line below]
    [...]
    ValueError: F is ambiguous (valid in sex, status)

In that case, we have to specify which axis the "F" we want belongs to:

.. ipython:: python

    ambiguous_arr[2017, sex["F"]]

You can also define slices (defined by 'start:stop' or 'start:stop:step').
A slice will select all labels between `start` and `stop` (stop included).
Specifying the start and stop bounds of a slice is optional: when not given,
start is the first label of an axis, stop the last one.

.. ipython:: python

    # "10-17":"67+" is a shortcut for ["10-17", "18-66", "67+"]
    arr["F", "10-17":"67+"]

    # :"18-66" will select all labels between the first one and "18-66"
    # 2017: will select all labels between 2017 and the last one
    arr[:"18-66", 2017:]


Aggregation
-----------

The LArray library includes many aggregations methods.
For example, to calculate the sum along an axis, write:

.. ipython:: python

    arr
    arr.sum("sex")
    arr.sum("age", "sex")

To aggregate along all axes except some, you simply have to append `_by`
to the aggregation method you want to use:

.. ipython:: python

    arr.sum_by("year")

See :ref:`here <la_agg>` to get the list of all available aggregation methods.


Groups
------

A :ref:`Group <api-group>` represents a subset of labels or positions of an axis:

.. ipython:: python

    arr

    children = age["0-9", "10-17"]
    children

It is often useful to attach them an explicit name using the ``>>`` operator:

.. ipython:: python

    working = age["10-17"] >> "working"
    working

    nonworking = age["0-9", "10-17", "67+"] >> "nonworking"
    nonworking

Groups can be used in selections:

.. ipython:: python

    arr[children]
    arr[nonworking]

or aggregations:

.. ipython:: python

    arr.sum(children)

When aggregating several groups, the names we set above using ``>>`` determines the label on the aggregated axis.
Since we did not give a name for the children group, the resulting label is generated automatically :

.. ipython:: python

    arr.sum((children, working, nonworking))


Grouping arrays in a Session
----------------------------

Arrays may be grouped in :ref:`Session <api-session>` objects.
A session is an ordered dict-like container of LArray objects with special I/O methods.
To create a session, you need to pass a list of pairs (array_name, array):

.. ipython:: python

    arr0 = ndtest((3, 3))
    arr1 = ndtest((2, 4))
    arr2 = ndtest((4, 2))

    arrays = [("arr0", arr0), ("arr1", arr1), ("arr2", arr2)]
    ses = Session(arrays)

    # displays names of arrays contained in the session
    ses.names
    # get an array
    ses["arr0"]
    # add/modify an array
    ses["arr3"] = ndtest((2, 2, 2))

.. warning::

    You can also pass a dictionary to the Session's constructor but since elements of a dict object are
    not ordered by default, you may lose the order. If you are using python 3.6 or later, using keyword
    arguments is a nice alternative which keeps ordering. For example, the session above can be defined
    using: `ses = Session(arr0=arr0, arr1=arr1, arr2=arr2)`.

One of the main interests of using sessions is to save and load many arrays at once:

.. ipython:: python
    :okwarning:

    ses.save("my_session.h5")
    ses = Session("my_session.h5")


Graphical User Interface (viewer)
---------------------------------

The LArray project provides an optional package called :ref:`larray-editor <start-dependencies-gui>`
allowing users to explore and edit arrays using a graphical interface.
This package is automatically installed with **larrayenv**.

To explore the content of arrays in read-only mode, import ``larray-editor`` and call :py:func:`view`

.. ipython:: python
    :verbatim:

    from larray_editor import *

    # shows the arrays of a given session in a graphical user interface
    view(ses)

    # the session may be directly loaded from a file
    view("my_session.h5")

    # creates a session with all existing arrays from the current namespace
    # and shows its content
    view()

To open the user interface in edit mode, call :py:func:`edit` instead.

.. image:: _static/editor.png
    :align: center

Once open, you can save and load any session using the `File` menu.

Finally, you can also visually compare two arrays or sessions using the :py:func:`compare` function.

.. ipython:: python
   :verbatim:

    arr0 = ndtest((3, 3))
    arr1 = ndtest((3, 3))
    arr1[["a1", "a2"]] = -arr1[["a1", "a2"]]
    compare(arr0, arr1)

.. image:: _static/compare.png
    :align: center

In case of two arrays, they must have compatible axes.

For Windows Users
^^^^^^^^^^^^^^^^^

Installing the ``larray-editor`` package on Windows will create a ``LArray`` menu in the
Windows Start Menu. This menu contains:

  * a shortcut to open the documentation of the last stable version of the library
  * a shortcut to open the graphical interface in edit mode.
  * a shortcut to update `larrayenv`.

.. image:: _static/menu_windows.png
    :align: center

.. image:: _static/editor_new.png
    :align: center

Once the graphical interface is open, all LArray objects and functions are directly accessible.
No need to start by `from larray import *`.

Compatibility with pandas
-------------------------

To convert a LArray object into a pandas DataFrame, the method :py:meth:`~LArray.to_frame` can be used:

.. ipython:: python

    df = arr.to_frame()
    df

Inversely, to convert a DataFrame into a LArray object, use the function :py:func:`aslarray`:

.. ipython:: python

    arr = aslarray(df)
    arr
