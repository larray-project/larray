How to contribute
=================

Before to Start
---------------

Where to find the code
~~~~~~~~~~~~~~~~~~~~~~

The code is hosted on `GitHub <https://www.github.com/larray-project/larray>`_.

.. _contributing.tools:

Tools
~~~~~

To contribute you will need to sign up for a `free GitHub account <https://github.com/signup/free>`_.

We use `Git <http://git-scm.com/>`_ for version control to allow many people to work together on the project.

For managing the Python packages and developing the code we use `miniconda <https://conda.io/miniconda.html>`_
and the IDE `PyCharm <https://www.jetbrains.com/pycharm>`_ respectively.
Alternatively to `miniconda`, you may install `Anaconda <https://www.anaconda.com/download/>`_.

The documentation is written partly using reStructuredText and partly using Jupyter notebooks (for the tutorial).
It is built to various formats using `Sphinx <http://sphinx-doc.org/>`_
and `nbsphinx <https://nbsphinx.readthedocs.io>`_.

The unit tests are written using the `pytest library <https://docs.pytest.org>`_.

Licensing
~~~~~~~~~

LArray is licensed under the GPLv3. Before starting to work on any issue, make sure
you accept that your contributions will be released under that license.

Creating a development environment
----------------------------------

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions <http://help.github.com/set-up-git-redirect>`__ for installing git,
setting up your SSH key, and configuring git.

.. contributing.getting_code

Getting the code (for the first time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will need your own fork to work on the code. Go to the `larray project
page <https://github.com/larray-project/larray>`_ and hit the ``Fork`` button.

You will want to clone your fork to your machine.
To do it manually, follow these steps::

    git clone https://github.com/your-user-name/larray.git
    cd larray
    git remote add upstream https://github.com/larray-project/larray.git

Or do it with PyCharm following **VCS > Checkout from Version Control > GitHub** in the menu bar.

This creates the directory `larray` and connects your repository to
the upstream (main project) *larray* repository.

Creating a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting any development, you'll need to create an isolated larray
development environment:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_ as :ref:`suggest earlier <contributing.tools>`
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.getting_code>`
- ``cd`` to the *larray* source directory

We'll now kick off a two-step process:

1. Install the build dependencies

.. code-block:: none

   # Create and activate the build environment
   conda create -n larray_dev numpy pandas pytables pyqt qtpy matplotlib xlrd openpyxl xlsxwriter pytest
   conda activate larray_dev

2. Build and install larray

You could install LArray in the standard way:

.. code-block:: none

  python setup.py install

but in that case you need to "install" it again every time you change it. When developing, it is usually more
convenient to use:

.. code-block:: none

  python setup.py develop

This creates some kind of symlink between your python installation "modules" directory and your repository,
so that any change in your local copy is automatically usable by other modules.

At this point you should be able to import larray from your locally built version::

   $ python  # start an interpreter
   >>> import larray
   >>> larray.__version__
   '0.29'

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

To view your environments::

      conda info -e

To return to your root environment::

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`_.


Starting to contribute
----------------------

For developing the LArray library, we follow the `Forking Workflow
<https://gist.github.com/Chaser324/ce0505fbed06b947d962>`_.
In the :ref:`Getting code <contributing.getting_code>` section,
we have already explained how to get fork of the main larray repository.

To make a contribution, please follow the steps described bellow.

Step 1: Create a new branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch issue-to-fix
    git checkout issue-to-fix

The above can be simplified to::

    git checkout -b issue-to-fix

This changes your working directory to the issue-to-fix branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *larray*. You can have many "issue-to-fix"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the master branch::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest larray git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``stash`` them prior
to updating.  This will effectively store your changes and they can be reapplied
after updating.

For managing branches with PyCharm, please refer to
`this page <https://www.jetbrains.com/help/pycharm/manage-branches.html>`_.

Step 2: Write your code
~~~~~~~~~~~~~~~~~~~~~~~

When writing your code, please follow the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_
code conventions. Among others, this means:

- 120 characters lines
- 4 spaces indentation
- lowercase (with underscores if needed) variables, functions, methods and modules names
- CamelCase classes names
- all uppercase constants names
- whitespace around binary operators
- no whitespace before a comma, semicolon, colon or opening parenthesis
- whitespace after commas

This summary should not prevent you from reading the PEP!

LArray is currently compatible with both Python 2 and 3.
So make sure your code is compatible with both versions.

Step 3: Document your code
~~~~~~~~~~~~~~~~~~~~~~~~~~

We use Numpy conventions for docstrings. Here is a template: ::

  def funcname(arg1, arg2=default2, arg3=default3):
      """Summary line.

      Extended description of function.

      .. versionadded:: 0.2.0

      Parameters
      ----------
      arg1 : type1
          Description of arg1.
      arg2 : {value1, value2, value3}, optional
          Description of arg2.

          * value1 -- description of value1 (default2)
          * value2 -- description of value2
          * value3 -- description of value3
      arg3 : type3 or type3bis, optional
          Description of arg3. Default is default3.

          .. versionadded:: 0.3.0

      Returns
      -------
      type
          Description of return value.

      Notes
      -----
      Some interesting facts about this function.

      See Also
      --------
      LArray.otherfunc : How other function or method is related.

      Examples
      --------
      >>> funcname(arg)
      result
      """

For example: ::

  def check_number_string(number, string="1"):
      """Compares the string representation of a number to a string.

      Parameters
      ----------
      number : int
          The number to test.
      string : str, optional
          The string to test against. Default is "1".

      Returns
      -------
      bool
          Whether the string representation of the number is equal to the string.

      Examples
      --------
      >>> check_number_string(42, "42")
      True
      >>> check_number_string(25, "2")
      False
      >>> check_number_string(1)
      True
      """
      return str(number) == string


Step 4: Test your code
~~~~~~~~~~~~~~~~~~~~~~

Sometimes doctests are not enough and new features require to go a step further by writing unit tests.
Our unit tests modules are located in `/larray/tests/`.
See the :ref:`Tests <contributing.testing>` section bellow for more details.

Step 5: Add a change log
~~~~~~~~~~~~~~~~~~~~~~~~

Changes should be reflected in the release notes located in ``doc/source/changes/version_<next_release_version>.inc``.
This file contains an ongoing change log for the next release.
Add an entry to this file to document your fix, enhancement or (unavoidable) breaking change.
If you hesitate in which section to add your change log, feel free to ask.
Make sure to include the GitHub issue number when adding your entry (using `` closes :issue:`issue-number` ``
where `issue-number` is the number associated with the fixed issue).

Step 6: Commit your code
~~~~~~~~~~~~~~~~~~~~~~~~

When you think you have (finally) fixed the issue (after documenting your code, running all the tests
and adding a change log), make sure that one of your commit messages start with ``fix #issue-number :``
where `issue-number` is the number associated with the fixed issue before to start any pull request
(see `this github page <https://help.github.com/articles/closing-issues-using-keywords>`_ for more details).

Step 7: Push your changes
~~~~~~~~~~~~~~~~~~~~~~~~~

When you want your changes to appear publicly on your GitHub page, push your
forked feature branch's commits::

    git push origin issue-to-fix

Here ``origin`` is the default name given to your remote repository on GitHub.
You can see the remote repositories::

    git remote -v

If you added the upstream repository as described above you will see something
like::

    origin  git@github.com:yourname/larray.git (fetch)
    origin  git@github.com:yourname/larray.git (push)
    upstream        git://github.com/larray-project/larray.git (fetch)
    upstream        git://github.com/larray-project/larray.git (push)

Step 8: Start a pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If everything looks good, you are ready to make a pull request.
This pull request and its associated changes will eventually be committed to the master branch
and available in the next release.
To submit a pull request:

#. Navigate to your repository on GitHub
#. Click on the ``Pull Request`` button
#. You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
#. Write a description of your changes in the ``Preview Discussion`` tab
#. Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review
the code. If you need to make more changes, you can make them in
your branch, add them to a new commit, push them to GitHub, and the pull request
will be automatically updated. Pushing them to GitHub again is done by::

    git push origin shiny-new-feature

This will automatically update your pull request with the latest code and restart the
:ref:`Continuous Integration` tests.

The *larray* test suite will run automatically on `Travis-CI <https://travis-ci.org/>`__
continuous integration service. A pull-request will be considered for merging when you have
an all 'green' build. If any tests are failing, then you will get a red 'X', where you can click
through to see the individual failed tests.

``Warning``: Please do not rebase your local branch during the review process.

Documentation
-------------

The documentation is written using reStructuredText and built to various formats using
`Sphinx <http://sphinx-doc.org/>`_. See the `reStructuredText Primer <http://sphinx-doc.org/rest.html#rst-primer>`_
for a first introduction of the syntax.

Installing Requirements
~~~~~~~~~~~~~~~~~~~~~~~

Basic requirements (to generate an .html version of the documentation) can be installed using: ::

  > conda install sphinx numpydoc nbsphinx

To build the .pdf version, you need a LaTeX processor. We use `MiKTeX <http://miktex.org>`_.

To build the .chm version, you need `HTML Help Workshop
<http://www.microsoft.com/en-us/download/details.aspx?id=21138>`_.

Generating the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a command prompt and go to the documentation directory: ::

  > cd doc

If you just want to check that there is no syntax error in the documentation and that it formats properly, it is
usually enough to only generate the .html version, by using: ::

  > make html

Open the result in your favourite web browser. It is located in: ::

  build/html/index.html

If you want to also generate the .pdf and .chm (and you have the extra requirements to generate those), you could
use: ::

  > buildall


.. contributing.testing

Tests
-----

We use both unit tests and doctests. Unit tests are written using the `pytest library <https://docs.pytest.org>`_.
For example: ::

 from larray import to_ticks

 def test_split():
      assert to_ticks('M,F')  == ['M', 'F']
      assert to_ticks('M, F') == ['M', 'F']

To run all unit tests: ::

  > pytest larray\tests\test_la.py

Before writting any unit tests, please read the section `Conventions for Python test discovery
<https://docs.pytest.org/en/latest/goodpractices.html#test-discovery>`_ from the pytest documentation.


We also use doctests for some tests. Doctests is specially-formatted code within the docstring of a function which
embeds the result of calling said function with a particular set of arguments. This can be used both as documentation
and testing. We only use doctests for the cases where the test is simple enough to fit on one line and it can help
understand what the function does. For example: ::

  def slice_to_str(key):
      """Converts a slice to a string

      >>> slice_to_str(slice(None))
      ':'
      """
      # some clever code here
      return ':'

To run doc tests: ::

  > pytest larray\larray.py

To run all the tests, simply go to root directory and type: ::

  > pytest

`pytest` will automatically detect all existing unit tests and doctests and run them all.
