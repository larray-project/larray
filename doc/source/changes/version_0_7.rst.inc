Viewer
------

* implemented view() on Session objects
* added axes length in window title and add axes info even if title is provided manually (concatenate both)
* ndecimals are recomputed when toggling the scientific checkbox
* allow viewing (some) non-ndarray stuff (e.g. python lists)
* refactored viewer code so that the filter drop downs can be reused too
* Known regression: the viewer is slow on large arrays (this will be fixed in a later release, obviously)


Session
-------

* implemented local_arrays() to return all LArray in locals() as a Session
* implemented Session.__getitem__(int_position)
* implement Session(filename) to directly load all arrays from a file. Equivalent to:

    >>> s = Session()
    >>> s.load(filename)

* implemented Session.__eq__, so that you can compare two sessions and see if all arrays are equal.
  Suppose you want to refactor your code and make sure you get the same results.
    
    >>> # put results in a Session
    >>> res = Session({'array1': array1, 'array2': array2})
    >>> # before refactoring
    >>> res.dump('results.h5')
    >>> # after refactoring
    >>> assert Session('results.h5') == res

* you can load all sheets/arrays of a file (if you do not specify which ones you want, it takes all)
* loading several sheets from an excel file is now MUCH faster because the same file is kept open (apparently xlrd
  parses the whole file each time we open it).
* you can specify a subset of arrays to dump
* implemented rudimentary session I/O for .csv files, usage is a bit different from .h5 & excel files

    >>> # need to specify format manually
    >>> s.dump('directory_name', fmt='csv')
    >>> # need to specify format manually
    >>> s = Session()
    >>> s.load('directory_name', fmt='csv')

* pass `*args` and `**kwargs` to lower level functions in Session.load
* fail when trying to read an inexistant H5 file through Session, instead of creating it


Other new features
------------------

* added start argument in ndrange to specify starting value
* implemented Axis._rename. Not sure it's a good idea though…
* implemented identity function which takes an Axis and returns an LArray with the axis labels as values
* implemented size property on AxisCollection
* allow a single int in AxisCollection.without

Fixes
-----

* fixed broadcast_with when other_axes contains 0-len axes
* fixed a[bool_array] = value when the first axis of a is not in bool_array
* fixed view() on arrays with unnamed axes
* fixed view() on arrays of Python objects
* various other small bugs fixed
