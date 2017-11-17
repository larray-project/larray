.. currentmodule:: larray

.. _start_tutorial:

Tutorial
========

This is an introduction to LArray. It is not intended to be a fully
comprehensive manual. It is mainly dedicated to help new users to
familiarize with it and others to remind essentials.

The first step to use the LArray library is to import it:

.. ipython:: python

    from larray import *

.. ipython:: python
   :suppress:

    import warnings
    warnings.filterwarnings('ignore')

Axis creation
-------------

An :py:class:`axis <Axis>` represents a dimension of an LArray object.
It consists of a name and a list of labels. They are several ways to create an axis:

.. ipython:: python

    # create a wildcard axis 
    age = Axis(3, 'age')
    # labels given as a list 
    time = Axis([2007, 2008, 2009], 'time')
    # create an axis using one string
    sex = Axis('sex=M,F')
    # labels generated using a special syntax  
    other = Axis('other=A01..C03')
    
    age, sex, time, other

Array creation
--------------

A :py:class:`LArray` object represents a multidimensional array with labeled axes.

From scratch
~~~~~~~~~~~~

To create an array from scratch, you need to provide the data and a list
of axes. Optionally, a title can be defined.

.. ipython:: python

    import numpy as np

    # list of the axes
    axes = [age, sex, time, other]
    # data (the shape of data array must match axes lengths)
    data = np.random.randint(100, size=[len(axis) for axis in axes])
    # title (optional)
    title = 'random data'
    
    arr = LArray(data, axes, title)
    arr

Array creation functions
~~~~~~~~~~~~~~~~~~~~~~~~

Arrays can also be generated in an easier way through creation functions:

-  :py:func:`ndrange` : fills an array with increasing numbers
-  :py:func:`ndtest` : same as ndrange but with axes generated automatically
   (for testing)
-  :py:func:`empty` : creates an array but leaves its allocated memory
   unchanged (i.e., it contains "garbage". Be careful !)
-  :py:func:`zeros` : fills an array with 0
-  :py:func:`ones` : fills an array with 1
-  :py:func:`full` : fills an array with a given value

Except for ndtest, a list of axes must be provided.
Axes can be passed in different ways:

-  as Axis objects
-  as integers defining the lengths of auto-generated wildcard axes
-  as a string : 'sex=M,F;time=2007,2008,2009' (name is optional)
-  as pairs (name, labels)

Optionally, the type of data stored by the array can be specified using argument dtype.

.. ipython:: python

    # start defines the starting value of data
    ndrange(['age=0..2', 'sex=M,F', 'time=2007..2009'], start=-1)

.. ipython:: python

    # start defines the starting value of data
    # label_start defines the starting index of labels
    ndtest((3, 3), start=-1, label_start=2)

.. ipython:: python

    # empty generates uninitialised array with correct axes (much faster but use with care!).
    # This not really random either, it just reuses a portion of memory that is available, with whatever content is there. 
    # Use it only if performance matters and make sure all data will be overridden. 
    empty(['age=0..2', 'sex=M,F', 'time=2007..2009'])

.. ipython:: python

    # example with anonymous axes
    zeros(['0..2', 'M,F', '2007..2009'])

.. ipython:: python

    # dtype=int forces to store int data instead of default float
    ones(['age=0..2', 'sex=M,F', 'time=2007..2009'], dtype=int)

.. ipython:: python

    full(['age=0..2', 'sex=M,F', 'time=2007..2009'], 1.23)

All the above functions exist in *{func}_like* variants which take
axes from another array

.. ipython:: python

    ones_like(arr)

Sequence
~~~~~~~~

The special :py:func:`sequence` function allows you to create an array from an
axis by iteratively applying a function to a given initial value. You
can choose between **inc** and **mult** functions or define your own.

.. ipython:: python

    # With initial=1.0 and inc=0.5, we generate the sequence 1.0, 1.5, 2.0, 2.5, 3.0, ... 
    sequence('sex=M,F', initial=1.0, inc=0.5)

.. ipython:: python

    # With initial=1.0 and mult=2.0, we generate the sequence 1.0, 2.0, 4.0, 8.0, ... 
    sequence('age=0..2', initial=1.0, mult=2.0) 

.. ipython:: python

    # Using your own function
    sequence('time=2007..2009', initial=2.0, func=lambda value: value**2)

You can also create N-dimensional array by passing (N-1)-dimensional
array to initial, inc or mult argument

.. ipython:: python

    birth = LArray([1.05, 1.15], 'sex=M,F')
    cumulate_newborns = sequence('time=2007..2009', initial=0.0, inc=birth)
    cumulate_newborns

.. ipython:: python

    initial = LArray([90, 100], 'sex=M,F') 
    survival = LArray([0.96, 0.98], 'sex=M,F')
    pop = sequence('age=80..83', initial=initial, mult=survival)
    pop

Load/Dump from files
--------------------

Load from files
~~~~~~~~~~~~~~~

.. ipython:: python

    example_dir = EXAMPLE_FILES_DIR

Arrays can be loaded from CSV files (see documentation of :py:func:`read_csv`
for more details)

.. ipython:: python

    # read_tsv is a shortcut when data are separated by tabs instead of commas (default separator of read_csv)
    # read_eurostat is a shortcut to read EUROSTAT TSV files  
    household = read_csv(example_dir + 'hh.csv')
    household.info

or Excel sheets (see documentation of :py:func:`read_excel` for more details)

.. ipython:: python

    # loads array from the first sheet if no sheetname is given
    @verbatim
    pop = read_excel(example_dir + 'demography.xlsx', 'pop')

    @suppress
    pop = read_csv(example_dir + 'pop.csv')

    pop.info

or HDF5 files (HDF5 is file format designed to store and organize large
amounts of data. An HDF5 file can contain multiple arrays. See
documentation of :py:func:`read_hdf` for more details)

.. ipython:: python

    mortality = read_hdf(example_dir + 'demography.h5','qx')
    mortality.info

Dump in files
~~~~~~~~~~~~~

Arrays can be dumped in CSV files (see documentation of :py:meth:`~LArray.to_csv` for
more details)

.. ipython:: python

    household.to_csv('hh2.csv')

or in Excel files (see documentation of :py:meth:`~LArray.to_excel` for more details)

.. ipython:: python
   :verbatim:

    # if the file does not already exist, it is created with a single sheet, 
    # otherwise a new sheet is added to it
    household.to_excel('demography_2.xlsx', overwrite_file=True)
    # it is usually better to specify the sheet explicitly (by name or position) though
    household.to_excel('demography_2.xlsx', 'hh')

or in HDF5 files (see documentation of :py:meth:`~LArray.to_hdf` for more details)

.. ipython:: python

    household.to_hdf('demography_2.h5', 'hh')

more Excel IO
~~~~~~~~~~~~~

.. ipython:: python

    # create a 3 x 2 x 3 array 
    age, sex, time = Axis('age=0..2'), Axis('sex=M,F'), Axis('time=2007..2009')
    arr = ndrange([age, sex, time])
    arr

Write Arrays
^^^^^^^^^^^^

Open an Excel file

.. ipython:: python
   :verbatim:

    wb = open_excel('test.xlsx', overwrite_file=True)

Put an array in an Excel Sheet, **excluding** headers (labels)

.. ipython:: python
   :verbatim:

    # put arr at A1 in Sheet1, excluding headers (labels)
    wb['Sheet1'] = arr
    # same but starting at A9
    # note that Sheet1 must exist
    wb['Sheet1']['A9'] = arr

Put an array in an Excel Sheet, **including** headers (labels)

.. ipython:: python
   :verbatim:

    # dump arr at A1 in Sheet2, including headers (labels)
    wb['Sheet2'] = arr.dump()
    # same but starting at A10
    wb['Sheet2']['A10'] = arr.dump()

Save file to disk

.. ipython:: python
   :verbatim:

    wb.save()

Close file

.. ipython:: python
   :verbatim:

    wb.close()

Read Arrays
^^^^^^^^^^^

Open an Excel file

.. ipython:: python
   :verbatim:

    wb = open_excel('test.xlsx')

Load an array from a sheet (assuming the presence of (correctly formatted)
headers and only one array in sheet)

.. ipython:: python

    # save one array in Sheet3 (including headers)
    @verbatim
    wb['Sheet3'] = arr.dump()
    
    # load array from the data starting at A1 in Sheet3
    @verbatim
    arr = wb['Sheet3'].load()

    arr

Load an array with its axes information from a range

.. ipython:: python

    # if you need to use the same sheet several times,
    # you can create a sheet variable
    @verbatim
    sheet2 = wb['Sheet2']
    
    # load array contained in the 4 x 4 table defined by cells A10 and D14
    @verbatim
    arr2 = sheet2['A10:D14'].load()

    @suppress
    arr2 = arr[[0, 1], ['M', 'F'], [2007, 2008]]

    arr2

Read Ranges (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Load an array (raw data) with no axis information from a range

.. ipython:: python

    @verbatim
    arr3 = wb['Sheet1']['A1:B4']

    @suppress
    arr3 = LArray([[3*i, 3*i+1] for i in range(4)])

    arr3

in fact, this is not really an LArray ...

.. ipython::

    @verbatim
    In [1]: type(arr3)
    larray.io.excel.Range

... but it can be used as such

.. ipython:: python

    arr3.sum(axis=0)

... and it can be used for other stuff, like setting the formula instead
of the value:

.. ipython:: python
   :verbatim:

    arr3.formula = '=D10+1'

In the future, we should also be able to set font name, size, style,
etc.

.. ipython:: python
   :verbatim:

    wb.close()

Inspecting
----------

.. ipython:: python

    # load population array
    pop = load_example_data('demography').pop

Get array summary : dimensions + description of axes

.. ipython:: python

    pop.info

Get axes

.. ipython:: python

    time, geo, age, sex, nat = pop.axes

Get array dimensions

.. ipython:: python

    pop.shape

Get number of elements

.. ipython:: python

    pop.size

Get size in memory

.. ipython:: python

    pop.nbytes

Start viewer (graphical user interface) in read-only mode.
This will open a new window and block execution of the rest
of code until the windows is closed! Required PyQt installed.

.. ipython::

    @verbatim
    In [1]: view(pop)

Load array in an Excel sheet

.. ipython::

    @verbatim
    In [1]: pop.to_excel()

Selection (Subsets)
-------------------

LArray allows to select a subset of an array either by labels or positions

Selection by Labels
~~~~~~~~~~~~~~~~~~~

To take a subset of an array using labels, use brackets [ ].
Let's start by selecting a single element:

.. ipython:: python

    # here we select the value associated with Belgian women of age 50 from Brussels region for the year 2015
    pop[2015, 'BruCap', 50, 'F', 'BE']

Continue with selecting a subset using slices and lists of labels

.. ipython:: python

    # here we select the subset associated with Belgian women of age 50, 51 and 52 
    # from Brussels region for the years 2010 to 2016
    pop[2010:2016, 'BruCap', 50:52, 'F', 'BE']

.. ipython:: python

    # slices bounds are optional: 
    # if not given start is assumed to be the first label and stop is the last one.
    # Here we select all years starting from 2010
    pop[2010:, 'BruCap', 50:52, 'F', 'BE']

.. ipython:: python

    # Slices can also have a step (defaults to 1), to take every Nth labels
    # Here we select all even years starting from 2010
    pop[2010::2, 'BruCap', 50:52, 'F', 'BE']

.. ipython:: python

    # one can also use list of labels to take non-contiguous labels.
    # Here we select years 2008, 2010, 2013 and 2015
    pop[[2008, 2010, 2013, 2015], 'BruCap', 50:52, 'F', 'BE']

The order of indexing does not matter either, so you usually do not
care/have to remember about axes positions during computation.
It only matters for output.

.. ipython:: python

    # order of index doesn't matter
    pop['F', 'BE', 'BruCap', [2008, 2010, 2013, 2015], 50:52]

.. warning::
   Selecting by labels as above works well as long as there is no ambiguity.
   When two or more axes have common labels, it may lead to a crash.
   The solution is then to precise to which axis belong the labels.

.. ipython:: python

    # let us now create an array with the same labels on several axes
    age, weight, size = Axis('age=0..80'), Axis('weight=0..120'), Axis('size=0..200')

    arr_ws = ndrange([age, weight, size])

.. ipython::

    # let's try to select teenagers with size between 1 m 60 and 1 m 65 and weight > 80 kg.
    # In this case the subset is ambiguous and this results in an error:
    @verbatim
    In [1]: arr_ws[10:18, :80, 160:165]
    <class 'ValueError'> slice(10, 18, None) is ambiguous (valid in age, weight, size)

.. ipython:: python

    # the solution is simple. You need to precise the axes on which you make a selection
    arr_ws[age[10:18], weight[:80], size[160:165]]

Special variable x
~~~~~~~~~~~~~~~~~~

When selecting, assiging or using aggregate functions, an axis can be
refered via the special variable ``x``:

-  pop[x.age[:20]]
-  pop.sum(x.age)

This gives you acces to axes of the array you are manipulating. The main
drawback of using **x** is that you lose the autocompletion available from
many editors. It only works with non-wildcard axes.

.. ipython:: python

    # the previous example could have been also written as  
    arr_ws[x.age[10:18], x.weight[:80], x.size[160:165]]

Selection by Positions
~~~~~~~~~~~~~~~~~~~~~~

Sometimes it is more practical to use positions along the axis, instead
of labels. You need to add the character ``i`` before the brackets:
``.i[positions]``. As for selection with labels, you can use single
position or slice or list of positions. Positions can be also negative
(-1 represent the last element of an axis).

.. note::
   Remember that positions (indices) are always **0-based** in Python.
   So the first element is at position 0, the second is at position 1, etc.

.. ipython:: python

    # here we select the subset associated with Belgian women of age 50, 51 and 52 
    # from Brussels region for the first 3 years
    pop[x.time.i[:3], 'BruCap', 50:52, 'F', 'BE']

.. ipython:: python

    # same but for the last 3 years
    pop[x.time.i[-3:], 'BruCap', 50:52, 'F', 'BE']

.. ipython:: python

    # using list of positions
    pop[x.time.i[-9,-7,-4,-2], 'BruCap', 50:52, 'F', 'BE']

.. warning::
   The end *indice* (position) is EXCLUSIVE while the end label is INCLUSIVE.

.. ipython:: python

    # with labels (3 is included)
    pop[2015, 'BruCap', x.age[:3], 'F', 'BE']

.. ipython:: python

    # with position (3 is out)
    pop[2015, 'BruCap', x.age.i[:3], 'F', 'BE']

You can use *.i[]* selection directly on array instead of axes. In this
context, if you want to select a subset of the first and third axes for
example, you must use a full slice ``:`` for the second one.

.. ipython:: python

    # here we select the last year and first 3 ages
    # equivalent to: pop.i[-1, :, :3, :, :]
    pop.i[-1, :, :3]

Assigning subsets
~~~~~~~~~~~~~~~~~

Assigning value
^^^^^^^^^^^^^^^

Assign a value to a subset

.. ipython:: python

    # let's take a smaller array
    pop = load_example_data('demography').pop[2016, 'BruCap', 100:105]
    pop2 = pop
    pop2

.. ipython:: python

    # set all data corresponding to age >= 102 to 0
    pop2[102:] = 0
    pop2

One very important gotcha though...

.. warning::
   Modifying a slice of an array in-place like we did above should be done with
   care otherwise you could have **unexpected effects**.
   The reason is that taking a **slice** subset of an array does not return a copy
   of that array, but rather a view on that array.
   To avoid such behavior, use ``.copy()`` method.

Remember:

-  taking a slice subset of an array is extremely fast (no data is
   copied)
-  if one modifies that subset in-place, one also **modifies the
   original array**
-  **.copy()** returns a copy of the subset (takes speed and memory) but
   allows you to change the subset without modifying the original array
   in the same time

.. ipython:: python

    # indeed, data from the original array have also changed
    pop

.. ipython:: python

    # the right way
    pop = load_example_data('demography').pop[2016, 'BruCap', 100:105]
    
    pop2 = pop.copy()
    pop2[102:] = 0
    pop2

.. ipython:: python

    # now, data from the original array have not changed this time
    pop

Assigning Arrays & Broadcasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of a value, we can also assign an array to a subset. In that
case, that array can have less axes than the target but those which are
present must be compatible with the subset being targeted.

.. ipython:: python

    sex, nat = Axis('sex=M,F'), Axis('nat=BE,FO')
    new_value = LArray([[1, -1], [2, -2]],[sex, nat])
    new_value

.. ipython:: python

    # this assigns 1, -1 to Belgian, Foreigner men 
    # and 2, -2 to Belgian, Foreigner women for all 
    # people older than 100
    pop[102:] = new_value
    pop

.. warning::
   The array being assigned must have compatible axes with the target subset.

.. ipython:: python

    # assume we define the following array with shape 3 x 2 x 2
    new_value = zeros(['age=0..2', sex, nat]) 
    new_value

.. ipython::

    # now let's try to assign the previous array in a subset with shape 7 x 2 x 2
    @verbatim
    In [1]: pop[102:] = new_value
    <class 'ValueError'> could not broadcast input array from shape (3,2,2) into shape (4,2,2)

.. ipython:: python

    # but this works
    pop[102:104] = new_value
    pop

Boolean filtering
~~~~~~~~~~~~~~~~~

Boolean filtering can be use to extract subsets.

.. ipython:: python

    #Let's focus on population living in Brussels during the year 2016
    pop = load_example_data('demography').pop[2016, 'BruCap']
    
    # here we select all males and females with age less than 5 and 10 respectively
    subset = pop[((x.sex == 'H') & (x.age <= 5)) | ((x.sex == 'F') & (x.age <= 10))]
    subset

.. note::
   Be aware that after boolean filtering, several axes may have merged.

.. ipython:: python

    # 'age' and 'sex' axes have been merged together
    subset.info

This may be not what you because previous selections on merged axes are
no longer valid

.. ipython::

    # now let's try to calculate the proportion of females with age less than 10
    @verbatim
    In [1]: subset['F'].sum() / pop['F'].sum()
    <class 'ValueError'> F is not a valid label for any axis

Therefore, it is sometimes more useful to not select, but rather set to 0
(or another value) non matching elements

.. ipython:: python

    subset = pop.copy()
    subset[((x.sex == 'F') & (x.age > 10))] = 0
    subset['F', :20]

.. ipython:: python

    # now we can calculate the proportion of females with age less than 10
    subset['F'].sum() / pop['F'].sum()

Boolean filtering can also mix axes and arrays. Example above could also
have been written as

.. ipython:: python

    age_limit = sequence('sex=M,F', initial=5, inc=5)
    age_limit

.. ipython:: python

    age = pop.axes['age']
    (age <= age_limit)[:20]

.. ipython:: python

    subset = pop.copy()
    subset[x.age > age_limit] = 0
    subset['F'].sum() / pop['F'].sum()

Finally, you can choose to filter on data instead of axes

.. ipython:: python

    # let's focus on females older than 90
    subset = pop['F', 90:110].copy()
    subset

.. ipython:: python

    # here we set to 0 all data < 10
    subset[subset < 10] = 0
    subset

Manipulates axes from arrays
----------------------------

.. ipython:: python

    # let's start with
    pop = load_example_data('demography').pop[2016, 'BruCap', 90:95]
    pop

Relabeling
~~~~~~~~~~

Replace all labels of one axis

.. ipython:: python

    # returns a copy by default
    pop_new_labels = pop.set_labels(x.sex, ['Men', 'Women'])
    pop_new_labels

.. ipython:: python

    # inplace flag avoids to create a copy
    pop.set_labels(x.sex, ['M', 'F'], inplace=True)

Renaming axes
~~~~~~~~~~~~~

Rename one axis

.. ipython:: python

    pop.info

.. ipython:: python

    # 'rename' returns a copy of the array
    pop2 = pop.rename(x.sex, 'gender')
    pop2

Rename several axes at once

.. ipython:: python

    # No x. here because sex and nat are keywords and not actual axes
    pop2 = pop.rename(sex='gender', nat='nationality')
    pop2

Reordering axes
~~~~~~~~~~~~~~~

Axes can be reordered using :py:meth:`~LArray.transpose` method. By default, *transpose*
reverse axes, otherwise it permutes the axes according to the list given as argument.
Axes not mentioned come after those which are mentioned(and keep their relative order).
Finally, *transpose* returns a copy of the array.

.. ipython:: python

    # starting order : age, sex, nat
    pop

.. ipython:: python

    # no argument --> reverse axes
    pop.transpose()
    
    # .T is a shortcut for .transpose()
    pop.T

.. ipython:: python

    # reorder according to list
    pop.transpose(x.age, x.nat, x.sex)

.. ipython:: python

    # axes not mentioned come after those which are mentioned (and keep their relative order)
    pop.transpose(x.sex)

Aggregates
----------

Calculate the sum along an axis

.. ipython:: python

    pop = load_example_data('demography').pop[2016, 'BruCap']
    pop.sum(x.age)

or along all axes except one by appending *_by* to the aggregation function

.. ipython:: python

    pop[90:95].sum_by(x.age)
    # is equivalent to 
    pop[90:95].sum(x.sex, x.nat)

There are many other :ref:`aggregate functions built-in <la_agg>`:

-  mean, min, max, median, percentile, var (variance), std (standard
   deviation)
-  labelofmin, labelofmax (label indirect minimum/maxium -- labels where the
   value is minimum/maximum)
-  indexofmin, indexofmax (positional indirect minimum/maxium -- position
   along axis where the value is minimum/maximum)
-  cumsum, cumprod (cumulative sum, cumulative product)

Groups
------

One can define groups of labels (or indices)

.. ipython:: python

    age = pop.axes['age']
    
    # using indices (remember: 20 will not be included)
    teens = age.i[10:20]
    # using labels
    pensioners = age[67:]
    strange = age[[30, 55, 52, 25, 99]]
    
    strange

or rename them

.. ipython:: python

    # method 'named' returns a new group with the given name
    teens = teens.named('children')
    
    # operator >> is a shortcut for 'named'
    pensioners = pensioners >> 'pensioners'
    
    pensioners

Then, use them in selections

.. ipython:: python

    pop[strange]

or aggregations

.. ipython:: python

    pop.sum(pensioners)

.. ipython:: python

    # several groups (here you see the interest of groups renaming)
    pop.sum((teens, pensioners, strange))

.. ipython:: python

    # combined with other axes
    pop.sum((teens, pensioners, strange), x.nat)

Arithmetic operations
---------------------

.. ipython:: python

    # go back to our 6 x 2 x 2 example array
    pop = load_example_data('demography').pop[2016, 'BruCap', 90:95]
    pop

Usual Operations
~~~~~~~~~~~~~~~~

One can do all usual arithmetic operations on an array, it will apply
the operation to all elements individually

.. ipython:: python

    # addition
    pop + 200

.. ipython:: python

    # multiplication
    pop * 2

.. ipython:: python

    # ** means raising to the power (squaring in this case)
    pop ** 2

.. ipython:: python

    # % means modulo (aka remainder of division)
    pop % 10

More interestingly, it also works between two arrays

.. ipython:: python

    # load mortality equivalent array
    mortality = load_example_data('demography').qx[2016, 'BruCap', 90:95] 
    
    # compute number of deaths
    death = pop * mortality
    death

.. note::
   Be careful when mixing different data types.
   See **type promotion** in programming.
   You can use the method :py:meth:`~LArray.astype` to change the data type of an array.

.. ipython:: python

    # to be sure to get number of deaths as integers
    # one can use .astype() method
    death = (pop * mortality).astype(int)
    death

But operations between two arrays only works when they have compatible axes (i.e. same labels)

.. ipython::

    @verbatim
    In [1]: pop[90:92] * mortality[93:95]
    <class 'ValueError'> incompatible axes:
    Axis([93, 94, 95], 'age')
    vs
    Axis([90, 91, 92], 'age')

You can override that but at your own risk. In that case only the
position on the axis is used and not the labels.

.. ipython:: python

    pop[90:92] * mortality[93:95].drop_labels(x.age)

Boolean Operations
~~~~~~~~~~~~~~~~~~

.. ipython:: python

    pop2 = pop.copy()
    pop2['F'] = -pop2['F']
    pop2

.. ipython:: python

    # testing for equality is done using == (a single = assigns the value)
    pop == pop2

.. ipython:: python

    # testing for inequality
    pop != pop2

.. ipython:: python

    # what was our original array like again?
    pop

.. ipython:: python

    # & means (boolean array) and
    (pop >= 500) & (pop <= 1000)

.. ipython:: python

    # | means (boolean array) or
    (pop < 500) | (pop > 1000)

Arithmetic operations with missing axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    pop.sum(x.age)

.. ipython:: python

    # arr has 3 dimensions
    pop.info

.. ipython:: python

    # and arr.sum(age) has two
    pop.sum(x.age).info

.. ipython:: python

    # you can do operation with missing axes so this works
    pop / pop.sum(x.age)

Axis order does not matter much (except for output)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can do operations between arrays having different axes order.
The axis order of the result is the same as the left array

.. ipython:: python

    pop

.. ipython:: python

    # let us change the order of axes
    pop_transposed = pop.T
    pop_transposed

.. ipython:: python

    # mind blowing
    pop_transposed + pop

Combining arrays
----------------

Append/Prepend
~~~~~~~~~~~~~~

Append/prepend one element to an axis of an array

.. ipython:: python

    pop = load_example_data('demography').pop[2016, 'BruCap', 90:95] 
    
    # imagine that you have now acces to the number of non-EU foreigners
    data = [[25, 54], [15, 33], [12, 28], [11, 37], [5, 21], [7, 19]]
    pop_non_eu = LArray(data, pop['FO'].axes)
    
    # you can do something like this
    pop = pop.append(nat, pop_non_eu, 'NEU')
    pop

.. ipython:: python

    # you can also add something at the start of an axis
    pop = pop.prepend(x.sex, pop.sum(x.sex), 'B')
    pop

The value being appended/prepended can have missing (or even extra) axes
as long as common axes are compatible

.. ipython:: python

    aliens = zeros(pop.axes['sex'])
    aliens

.. ipython:: python

    pop = pop.append(x.nat, aliens, 'AL')
    pop

Extend
~~~~~~

Extend an array along an axis with another array *with* that axis (but other labels)

.. ipython:: python

    _pop = load_example_data('demography').pop
    pop = _pop[2016, 'BruCap', 90:95] 
    pop_next = _pop[2016, 'BruCap', 96:100]
    
    # concatenate along age axis
    pop.extend(x.age, pop_next)

Stack
~~~~~

Stack several arrays together to create an entirely new dimension

.. ipython:: python

    # imagine you have loaded data for each nationality in different arrays (e.g. loaded from different Excel sheets)
    pop_be, pop_fo = pop['BE'], pop['FO']
    
    # first way to stack them
    nat = Axis('nat=BE,FO,NEU')
    pop = stack([pop_be, pop_fo, pop_non_eu], nat)
    
    # second way
    pop = stack([('BE', pop_be), ('FO', pop_fo), ('NEU', pop_non_eu)], 'nat')
    
    pop

Sorting
-------

Sort an axis (alphabetically if labels are strings)

.. ipython:: python

    pop_sorted = pop.sort_axes(x.nat)
    pop_sorted

Give labels which would sort the axis

.. ipython:: python

    pop_sorted.labelsofsorted(x.sex)

Sort according to values

.. ipython:: python

    pop_sorted.sort_values((90, 'F'))

Plotting
--------

Create a plot (last axis define the different curves to draw)

.. ipython:: python

    @savefig plot_tutorial_0.png height=10in
    pop.plot()

.. ipython:: python

    # plot total of both sex
    @savefig plot_tutorial_1.png height=10in
    pop.sum(x.sex).plot()

Interesting methods
-------------------

.. ipython:: python

    # starting array
    pop = load_example_data('demography').pop[2016, 'BruCap', 100:105]
    pop

with total
~~~~~~~~~~

Add totals to one axis

.. ipython:: python

    pop.with_total(x.sex, label='B')

Add totals to all axes at once

.. ipython:: python

    # by default label is 'total'
    pop.with_total()

where
~~~~~

where can be used to apply some computation depending on a condition

.. ipython:: python

    # where(condition, value if true, value if false)
    where(pop < 10, 0, -pop)

clip
~~~~

Set all data between a certain range

.. ipython:: python

    # clip(min, max)
    # values below 10 are set to 10 and values above 50 are set to 50
    pop.clip(10, 50)

divnot0
~~~~~~~

Replace division by 0 to 0

.. ipython:: python

    pop['BE'] / pop['FO']

.. ipython:: python

    # divnot0 replaces results of division by 0 by 0. 
    # Using it should be done with care though
    # because it can hide a real error in your data.
    pop['BE'].divnot0(pop['FO'])

diff
~~~~

:py:meth:`~LArray.diff` calculates the n-th order discrete difference along given axis.
The first order difference is given by out[n+1] = in[n + 1] - in[n]
along the given axis.

.. ipython:: python

    pop = load_example_data('demography').pop[2005:2015, 'BruCap', 50]
    pop

.. ipython:: python

    # calculates 'pop[year+1] - pop[year]'
    pop.diff(x.time)

.. ipython:: python

    # calculates 'pop[year+2] - pop[year]'
    pop.diff(x.time, d=2)

ratio
~~~~~

.. ipython:: python

    pop.ratio(x.nat)
    
    # which is equivalent to
    pop / pop.sum(x.nat)

percents
~~~~~~~~

.. ipython:: python

    # or, if you want the previous ratios in percents
    pop.percent(x.nat)

growth\_rate
~~~~~~~~~~~~

using the same principle than diff...

.. ipython:: python

    pop.growth_rate(x.time)

shift
~~~~~

The :py:meth:`~LArray.shift` method drops first label of an axis and shifts all
subsequent labels

.. ipython:: python

    pop.shift(x.time)

.. ipython:: python

    # when shift is applied on an (increasing) time axis, it effectively brings "past" data into the future
    pop.shift(x.time).drop_labels(x.time) == pop[2005:2014].drop_labels(x.time)

.. ipython:: python

    # this is mostly useful when you want to do operations between the past and now
    # as an example, here is an alternative implementation of the .diff method seen above:
    pop.i[1:] - pop.shift(x.time)

Misc other interesting functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a lot more :ref:`functions <la_misc>` available:

- round, floor, ceil, trunc,
- exp, log, log10,
- sqrt, absolute, nan_to_num, isnan, isinf, inverse,
- sin, cos, tan, arcsin, arccos, arctan
- and many many more...

Sessions
--------

You can group several arrays in a :py:class:`Session`

.. ipython:: python

    # load several arrays
    arr1, arr2, arr3 = ndtest((3, 3)), ndtest((4, 2)), ndtest((2, 4))
    
    # create and populate a 'session'
    s1 = Session()
    s1.arr1 = arr1
    s1.arr2 = arr2
    s1.arr3 = arr3
    
    s1

The advantage of sessions is that you can manipulate all of the arrays in them in one shot

.. ipython:: python

    # this saves all the arrays in a single excel file (each array on a different sheet)
    @verbatim
    s1.save('test.xlsx')

.. ipython:: python

    # this saves all the arrays in a single HDF5 file (which is a very fast format)
    s1.save('test.h5')

.. ipython:: python

    # this creates a session out of all arrays in the .h5 file
    s2 = Session('test.h5')
    s2

.. ipython:: python

    # this creates a session out of all arrays in the .xlsx file
    @verbatim
    s3 = Session('test.xlsx')

    @suppress
    s3 = Session('test.h5')

    s3

You can compare two sessions

.. ipython:: python

    s1 == s2

.. ipython:: python

    # let us introduce a difference (a variant, or a mistake perhaps)
    s2.arr1['a0', 'b1':] = 0

.. ipython:: python

    s1 == s2

.. ipython:: python

    s1_diff = s1[s1 != s2]
    s1_diff

.. ipython:: python

    s2_diff = s2[s1 != s2]
    s2_diff

This a bit experimental but can be useful nonetheless (Open a graphical interface)

.. ipython::

    @verbatim
    In [1]: compare(s1_diff.arr1, s2_diff.arr1)
