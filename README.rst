LArray: N-dimensional labelled arrays
=====================================

|ci-status| |docs|

.. _start-intro:

LArray is an open source Python library that aims to provide tools for easy exploration and manipulation of
N-dimensional labelled data structures.

Library Highlights
------------------

* N-dimensional labelled array objects to store and manipulate multi-dimensional data

* I/O functions for reading and writing arrays in different formats:
  CSV, Microsoft Excel, HDF5, pickle

* Arrays can be grouped into Session objects and loaded/dumped at once

* User interface with an IPython console for rapid exploration of data

* Compatible with the pandas library: Array objects can be converted into pandas DataFrame and vice versa.

.. _start-install:

Installation
============

Pre-built binaries
------------------

The easiest route to installing larray is through
`Conda <http://conda.pydata.org/miniconda.html>`_.
For all platforms installing larray can be done with::

    conda install -c larray-project larray

This will install a lightweight version of larray
depending only on Numpy and Pandas libraries only.
Additional libraries are required to use the included
graphical user interface, make plots or use special
I/O functions for easy dump/load from Excel or
HDF files. Optional dependencies are described
below.

Installing larray with all optional dependencies
can be done with ::

    conda install -c larray-project larrayenv

You can also first add the channel `larray-project` to
your channel list ::

    conda config --add channels larray-project

and then install larray (or larrayenv) as ::

    conda install larray


Building from source
--------------------

The latest release of LArray is available from
https://github.com/larray-project/larray.git

Once you have satisfied the requirements detailed below, simply run::

    python setup.py install


Required Dependencies
---------------------

- Python 3.9, 3.10, 3.11, 3.12, 3.13 or 3.14
- `numpy <http://www.numpy.org/>`__ (1.22 or later)
- `pandas <http://pandas.pydata.org/>`__ (0.20 or later)


Optional Dependencies
---------------------

For IO (HDF, Excel)
~~~~~~~~~~~~~~~~~~~

- `pytables <http://www.pytables.org/>`__:
  for working with files in HDF5 format.
- `xlwings <https://www.xlwings.org/>`__:
  recommended package to get benefit of all Excel features of LArray.
  Only available on Windows and Mac platforms.
- `openpyxl <http://www.python-excel.org/>`__:
  recommended package for reading and writing
  Excel 2010 files (ie: .xlsx)
- `xlsxwriter <http://www.python-excel.org/>`__:
  alternative package for writing data, formatting
  information and, in particular, charts in the
  Excel 2010 format (ie: .xlsx)
- `xlrd <http://www.python-excel.org/>`__:
  for reading data and formatting information from older Excel files (ie: .xls)
- `xlwt <http://www.python-excel.org/>`__:
   for writing data and formatting information to older Excel files (ie: .xls)
- `larray_eurostat <https://github.com/larray-project/larray_eurostat>`__:
  provides functions to easily download EUROSTAT files as larray objects.
  Currently limited to TSV files.

.. _start-dependencies-gui:

For Graphical User Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LArray includes a graphical user interface to view, edit and compare arrays.

- `pyqt <https://riverbankcomputing.com/software/pyqt/intro>`__ (version 5):
  required by `larray-editor` (see below).
- `pyside <https://wiki.qt.io/PySide>`__:
  alternative to PyQt.
- `qtpy <https://github.com/spyder-ide/qtpy>`__:
  required by `larray-editor`.
- `larray-editor <https://github.com/larray-project/larray-editor>`__:
  required to use the graphical user interface associated with larray.
  It assumes that `qtpy` and either `pyqt` or `pyside` are installed.
  On windows, creates also a menu ``LArray`` in the Windows Start Menu.

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__:
  required for plotting.

Miscellaneous
~~~~~~~~~~~~~

- `pydantic <https://github.com/samuelcolvin/pydantic>`__:
  required to use `CheckedSession`.

.. _start-documentation:

Documentation
=============

The official documentation is hosted on ReadTheDocs at http://larray.readthedocs.io/en/stable/

.. _start-get-in-touch:

Get in touch
============

- To be informed of each new release, please subscribe to the announce `mailing list`_.
- For questions, ideas or general discussion, please use the `Google Users Group`_.
- To report bugs, suggest features or view the source code, please go to our `GitHub website`_.

.. _mailing list: https://groups.google.com/d/forum/larray-announce
.. _Google Users Group: https://groups.google.com/d/forum/larray-users
.. _GitHub website: http://github.com/larray-project/larray

.. end-readme-file

.. |ci-status| image:: https://github.com/larray-project/larray/actions/workflows/ci.yml/badge.svg
    :alt: CI status
    :target: https://github.com/larray-project/larray/actions/workflows/ci.yml

.. |docs| image:: https://readthedocs.org/projects/larray/badge/?version=stable
    :alt: Documentation Status
    :target: https://larray.readthedocs.io/en/latest/?badge=stable
