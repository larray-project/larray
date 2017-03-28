IMPORTANT
---------

To make sure all users have all optional dependencies installed and use the same version of packages, and to simplify
the update process, we created a new "larrayenv" package which will install larray itself AND all its dependencies
(*including* the optional ones). This means that this version needs to be installed using:

  conda install larrayenv

in the future, to update from one version to the next, it should always be enough to do:

  conda update larrayenv


New features
------------

* implemented from_lists() to create constant arrays (instead of using LArray directly as that is very error prone).
  We are not really happy with its name though, so it might change in the future. Any suggestion of a better name is
  very welcome (closes :issue:`30`).

    >>> from_lists([['sex\\year', 1991, 1992, 1993],
    ...             [ 'M',           0,    1,    2],
    ...             [ 'F',           3,    4,    5]])
    sex\year | 1991 | 1992 | 1993
           M |    0 |    1 |    2
           F |    3 |    4 |    5

* added support for loading sparse arrays via open_excel().

  For example, assuming you have a sheet like this: ::

    age | sex\year | 2015 | 2016
     10 |        F |  0.0 |  1.0
     10 |        M |  2.0 |  3.0
     20 |        M |  4.0 |  5.0

  loading it will yield:

    >>> wb = open_excel('test_sparse.xlsx')
    >>> arr = wb['Sheet1'].load()
    >>> arr
    age | sex\year | 2015 | 2016
     10 |        F |  0.0 |  1.0
     10 |        M |  2.0 |  3.0
     20 |        F |  nan |  nan
     20 |        M |  4.0 |  5.0


Miscellaneous improvements
--------------------------

* allowed to get an axis from an array by using array.axis_name in addition to array.axes.axis_name:

    >>> arr = ndtest((2, 3))
    >>> arr.axes
    AxisCollection([
        Axis('a', ['a0', 'a1']),
        Axis('b', ['b0', 'b1', 'b2'])
    ])
    >>> arr.a
    Axis('a', ['a0', 'a1'])

* viewer: several rows/columns can be plotted together. It draws a separate line for each row except if only one column
  has been selected.

* viewer: the array labels are used as "ticks" in plots.

* '_by' aggregation methods accept groups in addition to axes (closes :issue:`59`). It will keep only the mentioned
  groups and aggregate all other dimensions:

    >>> arr = ndtest((2, 3, 4))
    >>> arr
     a | b\c | c0 | c1 | c2 | c3
    a0 |  b0 |  0 |  1 |  2 |  3
    a0 |  b1 |  4 |  5 |  6 |  7
    a0 |  b2 |  8 |  9 | 10 | 11
    a1 |  b0 | 12 | 13 | 14 | 15
    a1 |  b1 | 16 | 17 | 18 | 19
    a1 |  b2 | 20 | 21 | 22 | 23

    >>> arr.sum_by('c0,c1;c1:c3')
    c | c0,c1 | c1:c3
      |   126 |   216

* viewer: view() and edit() now accept as argument a path to a file containing arrays.

  >>> view('myfile.h5')

  this is a shortcut for:

  >>> view(Session('myfile.h5'))

* AxisCollection.without now accepts a single integer position (to exclude an axis by position).

    >>> a = ndtest((2, 3))
    >>> a.axes
    AxisCollection([
        Axis('a', ['a0', 'a1']),
        Axis('b', ['b0', 'b1', 'b2'])
    ])
    >>> a.axes.without(0)
    AxisCollection([
        Axis('b', ['b0', 'b1', 'b2'])
    ])

* nicer display (repr) for LSet (closes :issue:`44`).

    >>> x.b['b0,b2'].set()
    x.b['b0', 'b2'].set()

* implemented sep argument for LArray & AxisCollection.combine_axes() to allow using a custom delimiter
  (closes :issue:`53`).

* added a check that ipfp target sums haves expected axes (closes :issue:`42`).

* when the nb_index argument is not provided explicitly in read_excel(engine='xlrd'), it is autodetected from the
  position of the first "\" (closes :issue:`66`).

* allow any special character except "." and whitespace when creating axes labels using ".." syntax
  (previously only _ was allowed).

* added many more I/O tests to hopefully lower our regression rate in the future (closes :issue:`70`).


Fixes
-----

* viewer: selection of entire rows/columns will load any remaining data, if any (closes :issue:`37`). Previously if you
  selected entire rows or columns of a large dataset (which is not loaded entirely from the start), it only selected
  (and thus copied/plotted) the part of the data which was already loaded.

* viewer: filtering on anonymous axes is now possible (closes :issue:`33`).

* fixed loading sparse files using read_excel() (fixes :issue:`29`).

* fixed nb_index argument for read_excel().

* fixed creating range axes with a negative start bound using string notation (e.g. `Axis('name', '-1..10')`)
  (fixes :issue:`51`).

* fixed ptp() function.

* fixed with_axes() to copy the title of the array.

* fixed Group >> 'name'.

* fixed workbook[sheet_position] when using open_excel().

* fixed plotting in the viewer when using Qt4.
