LArray: N-dimensional labelled arrays
=====================================

|build-status| |docs|

.. _start-intro:

LArray is open source Python library that aims to provide tools for easy exploration and manipulation of
N-dimensional labelled data structures.

Library Highlights
------------------

* N-dimensional labelled array objects to store and manipulate multi-dimensional data

* I/O functions for reading and writing arrays in different formats:
  CSV, Microsoft Excel, HDF5, pickle

* Arrays can be grouped into Session objects and loaded/dumped at once

* User interface with an IPython console for rapid exploration of data

* Compatible with the pandas library: LArray objects can be converted into pandas DataFrame and vice versa.

.. _start-install:

Installation
============

Pre-built binaries
------------------

The easiest route to installing larray is through
`Conda <http://conda.pydata.org/miniconda.html>`_.
For all platforms installing larray can be done with::

    conda install -c gdementen larray

This will install a lightweight version of larray
depending only on Numpy and Pandas libraries only.
Additional libraries are required to use the included
graphical user interface, make plots or use special
I/O functions for easy dump/load from Excel or
HDF files. Optional dependencies are described
below.

Installing larray with all optional dependencies
can be done with ::

    conda install -c gdementen larrayenv

You can also first add the channel `gdementen` to
your channel list ::

    conda config --add channels gdementen

and then install larray (or larrayenv) as ::

    conda install larray


Building from source
--------------------

The latest release of LArray is available from
https://github.com/liam2/larray.git

Once you have satisfied the requirements detailed below, simply run::

    python setup.py install


Required Dependencies
---------------------

- Python 2.7, 3.4, 3.5, or 3.6
- `numpy <http://www.numpy.org/>`__ (1.10.0 or later)
- `pandas <http://pandas.pydata.org/>`__ (0.13.1 or later)


Optional Dependencies
---------------------

For IO (HDF, Excel)
~~~~~~~~~~~~~~~~~~~

- `pytables <http://www.pytables.org/>`__:
  for working with files in HDF5 format.
- `xlrd <http://www.python-excel.org/>`__:
  for reading data and formatting information from older Excel files (ie: .xls)
- `openpyxl <http://www.python-excel.org/>`__:
  recommended package for reading and writing
  Excel 2010 files (ie: .xlsx)
- `xlsxwriter <http://www.python-excel.org/>`__:
  alternative package for writing data, formatting
  information and, in particular, charts in the
  Excel 2010 format (ie: .xlsx)
- `larray_eurostat <https://github.com/larray-project/larray_eurostat>`__:
  provides functions to easily download EUROSTAT files as larray objects.
  Currently limited to TSV files.

.. _start-dependencies-gui:

For Graphical User Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LArray includes a graphical user interface to view, edit and compare arrays.

- `pyqt <https://riverbankcomputing.com/software/pyqt/intro>`__ (4 or 5):
  required by `larray-editor` (see below).
- `pyside <https://wiki.qt.io/PySide>`__:
  alternative to PyQt.
- `qtpy <https://github.com/spyder-ide/qtpy>`__:
  required by `larray-editor`.
  Provides support for PyQt5, PyQt4 and PySide using the PyQt5 layout.
- `larray-editor <https://github.com/larray-project/larray-editor>`__:
  required to use the graphical user interface associated with larray.
  It assumes that `qtpy` and `pyqt` or `pyside` are installed.
  On windows, creates also a menu ``LArray`` in the Windows Start Menu.

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__:
  required for plotting.

.. _start-documentation:

Documentation
=============

The official documentation is hosted on ReadTheDocs at http://larray.readthedocs.io/en/stable/

.. _start-get-in-touch:

Get in touch
============

- Report bugs, suggest features or view the source code `on GitHub`_.
- For questions, ideas or general discussion, use the `mailing list`_.

.. _on GitHub: http://github.com/liam2/larray
.. _mailing list: https://groups.google.com/forum/#!forum/larray

.. end-readme-file

.. |build-status| image:: https://travis-ci.org/liam2/larray.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/liam2/larray

.. |docs| image:: https://readthedocs.org/projects/larray/badge/?version=stable
    :alt: Documentation Status
    :scale: 100%
    :target: https://larray.readthedocs.io/en/latest/?badge=stable
