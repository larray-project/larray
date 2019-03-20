.. currentmodule:: larray

Getting Started
===============

The purpose of the present **Getting Started** section is to give a quick overview
of the main objects and features of the LArray library.
To get a more detailed presentation of all capabilities of LArray, read the
next sections of the tutorial.
The :ref:`API Reference <start_api>` section of the documentation give you the list
of all objects, methods and functions with their individual documentation and examples.

To use the LArray library, the first thing to do is to import it:

.. ipython:: python

    from larray import *

Create an array
---------------

Working with the LArray library mainly consists of manipulating :ref:`LArray <api-larray>` data structures.
They represent N-dimensional labelled arrays and are composed of raw data (NumPy ndarray), :ref:`axes <api-axis>`
and optionally some metadata.

An axis represents a dimension of an array. It contains a list of labels and has a name:

.. ipython:: python

    # define some axes to be used later
    age = Axis(['0-9', '10-17', '18-66', '67+'], 'age')
    sex = Axis(['F', 'M'], 'sex')
    year = Axis([2015, 2016, 2017], 'year')

The labels allow to select subsets and to manipulate the data without working with the positions
of array elements directly.

To create an array from scratch, you need to supply data and axes:

.. ipython:: python

    # define some data. This is the belgian population (in thousands). Source: eurostat.
    data = [[[633, 635, 634],
             [663, 665, 664]],
            [[484, 486, 491],
             [505, 511, 516]],
            [[3572, 3581, 3583],
             [3600, 3618, 3616]],
            [[1023, 1038, 1053],
             [756, 775, 793]]]

    # create an LArray object
    pop = LArray(data, axes=[age, sex, year])
    pop

You can optionally attach some metadata to an array:

.. ipython:: python

    # attach some metadata to the pop array
    pop.meta.title = 'population by age, sex and year'
    pop.meta.source = 'Eurostat'

    # display metadata
    pop.meta

To get a short summary of an array, type:

.. ipython:: python

    # Array summary: metadata + dimensions + description of axes
    pop.info


Create an array filled with predefined values
---------------------------------------------

Arrays filled with predefined values can be generated through dedicated functions:

* :py:func:`zeros` : creates an array filled with 0

.. ipython:: python

   zeros([age, sex])

* :py:func:`ones` : creates an array filled with 1

.. ipython:: python

   ones([age, sex])

* :py:func:`full` : creates an array filled with a given value

.. ipython:: python

   full([age, sex], fill_value=10.0)

* :py:func:`sequence` : creates an array by sequentially applying modifications to the array along axis.

.. ipython:: python

   sequence(age)

* :py:func:`ndtest` : creates a test array with increasing numbers as data

.. ipython:: python

   ndtest([age, sex])


Save/Load an array
------------------

The LArray library offers many I/O functions to read and write arrays in various formats
(CSV, Excel, HDF5). For example, to save an array in a CSV file, call the method
:py:meth:`~LArray.to_csv`:

.. ipython:: python

    # save our pop array to a CSV file
    pop.to_csv('belgium_pop.csv')

The content of the CSV file is then::

    age,sex\time,2015,2016,2017
    0-9,F,633,635,634
    0-9,M,663,665,664
    10-17,F,484,486,491
    10-17,M,505,511,516
    18-66,F,3572,3581,3583
    18-66,M,3600,3618,3616
    67+,F,1023,1038,1053
    67+,M,756,775,793

.. note::
   In CSV or Excel files, the last dimension is horizontal and the names of the
   last two dimensions are separated by a ``\``.

To load a saved array, call the function :py:meth:`read_csv`:

.. ipython:: python

    pop = read_csv('belgium_pop.csv')
    pop

Other input/output functions are described in the :ref:`Input/Output <api-IO>` section of the API documentation.

Selecting a subset
------------------

To select an element or a subset of an array, use brackets [ ].
In Python we usually use the term *indexing* for this operation.

Let us start by selecting a single element:

.. ipython:: python

    pop['67+', 'F', 2017]

Labels can be given in arbitrary order:

.. ipython:: python

    pop[2017, 'F', '67+']

When selecting a larger subset the result is an array:

.. ipython:: python

    pop[2017]
    pop['M']

When selecting several labels for the same axis, they must be given as a list (enclosed by ``[ ]``)

.. ipython:: python

    pop['F', ['0-9', '10-17']]

You can also select *slices*, which are all labels between two bounds (we usually call them the `start` and `stop`
bounds). Specifying the `start` and `stop` bounds of a slice is optional: when not given, `start` is the first label
of the corresponding axis, `stop` the last one:

.. ipython:: python

    # in this case '10-17':'67+' is equivalent to ['10-17', '18-66', '67+']
    pop['F', '10-17':'67+']

    # :'18-66' selects all labels between the first one and '18-66'
    # 2017: selects all labels between 2017 and the last one
    pop[:'18-66', 2017:]

.. note::
    Contrary to slices on normal Python lists, the ``stop`` bound **is** included in the selection.

.. warning::

    Selecting by labels as above only works as long as there is no ambiguity.
    When several axes have some labels in common and you do not specify explicitly
    on which axis to work, it fails with an error ending with something like
    ValueError: <somelabel> is ambiguous (valid in <axis1>, <axis2>).

For example, let us create a test array with an ambiguous label. We first create an axis (some kind of status code)
with an 'F' label (remember we already have an 'F' label on the sex axis).

.. ipython:: python

    status = Axis(['A', 'C', 'F'], 'status')

Then create a test array using both axes 'sex' and 'status':

.. ipython:: python

    ambiguous_arr = ndtest([sex, status, year])
    ambiguous_arr

If we try to get the subset of our array concerning women (represented by the 'F' label in our array), we might
try something like:

.. ipython:: python
    :verbatim:

    ambiguous_arr[2017, 'F']

... but we receive back a volley of insults ::

    [some long error message ending with the line below]
    [...]
    ValueError: F is ambiguous (valid in sex, status)

In that case, we have to specify explicitly which axis the 'F' label we want to select belongs to:

.. ipython:: python

    ambiguous_arr[2017, sex['F']]


Aggregation
-----------

The LArray library includes many :ref:`aggregations methods <la_agg>`: sum, mean, min, max, std, var, ...

For example, assuming we still have an array in the ``pop`` variable:

.. ipython:: python

    pop

We can sum along the 'sex' axis using:

.. ipython:: python

    pop.sum(sex)

Or sum along both 'age' and 'sex':

.. ipython:: python

    pop.sum(age, sex)

It is sometimes more convenient to aggregate along all axes **except** some. In that case, use the aggregation
methods ending with `_by`. For example:

.. ipython:: python

    pop.sum_by(year)


Groups
------

A :ref:`Group <api-group>` represents a subset of labels or positions of an axis:


.. ipython:: python

    children = age['0-9', '10-17']
    children

It is often useful to attach them an explicit name using the ``>>`` operator:

.. ipython:: python

    working = age['18-66'] >> 'working'
    working

    nonworking = age['0-9', '10-17', '67+'] >> 'nonworking'
    nonworking

Still using the same ``pop`` array:

.. ipython:: python

    pop

Groups can be used in selections:

.. ipython:: python

    pop[working]
    pop[nonworking]

or aggregations:

.. ipython:: python

    pop.sum(nonworking)

When aggregating several groups, the names we set above using ``>>`` determines the label on the aggregated axis.
Since we did not give a name for the children group, the resulting label is generated automatically :

.. ipython:: python

    pop.sum((children, working, nonworking))


Grouping arrays in a Session
----------------------------

Arrays may be grouped in :ref:`Session <api-session>` objects.
A session is an ordered dict-like container of LArray objects with special I/O methods.
To create a session, you need to pass a list of pairs (array_name, array):

.. ipython:: python

    pop = zeros([age, sex, year])
    births = zeros([age, sex, year])
    deaths = zeros([age, sex, year])

    # create a session containing the three arrays 'pop', 'births' and 'deaths'
    demo = Session(pop=pop, births=births, deaths=deaths)

    # displays names of arrays contained in the session
    demo.names
    # get an array
    demo['pop']
    # add/modify an array
    demo['foreigners'] = zeros([age, sex, year])

.. warning::

    If you are using a Python version prior to 3.6, you will have to pass a list of pairs
    to the Session constructor otherwise the arrays will be stored in an arbitrary order in
    the new session. For example, the session above must be created using the syntax:
    `demo=Session([('pop', pop), ('births', births), ('deaths', deaths)])`.

One of the main interests of using sessions is to save and load many arrays at once:

.. ipython:: python
    :okwarning:

    # dump all arrays contained in the session 'demo' in one HDF5 file
    demo.save('demo.h5')
    # load all arrays saved in the HDF5 file 'demo.h5' and store them in the session 'demo'
    demo = Session('demo.h5')


Graphical User Interface (viewer)
---------------------------------

The LArray project provides an optional package called :ref:`larray-editor <start-dependencies-gui>`
allowing users to explore and edit arrays through a graphical interface.
The larray-editor tool is automatically available when installing the **larrayenv** metapackage from conda.

To explore the content of arrays in read-only mode, import ``larray-editor`` and call :py:func:`view`

.. ipython:: python
    :verbatim:

    from larray_editor import *

    # shows the arrays of a given session in a graphical user interface
    view(ses)

    # the session may be directly loaded from a file
    view('my_session.h5')

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
    arr1[['a1', 'a2']] = -arr1[['a1', 'a2']]
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

