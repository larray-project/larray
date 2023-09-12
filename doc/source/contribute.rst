How to contribute
=================

Before Starting
---------------

Where to find the code
~~~~~~~~~~~~~~~~~~~~~~

The code is hosted on `GitHub <https://www.github.com/larray-project/larray>`_.

.. _contributing.tools:

Tools
~~~~~

To contribute you will need to sign up for a `free GitHub account <https://github.com/signup/free>`_.

We use `Git <http://git-scm.com/>`_ for version control to allow many people to work together on the project.

The documentation is written partly using reStructuredText and partly using Jupyter notebooks (for the tutorial).
It is built to various formats using `Sphinx <http://sphinx-doc.org/>`_
and `nbsphinx <https://nbsphinx.readthedocs.io>`_.

The unit tests are written using the `pytest library <https://docs.pytest.org>`_.
The compliance with the PEP8 conventions is tested using `ruff <https://github.com/astral-sh/ruff/>`_.

Many editors and IDE exist to edit Python code and provide integration with version control tools (like git).
A good IDE, such as PyCharm, can make many of the steps below much more efficient.

.. _contributing.licensing:

Licensing
~~~~~~~~~

LArray is licensed under the GPLv3. Before starting to work on any issue, make sure
you accept and are allowed to have your contributions released under that license.

Creating a development environment
----------------------------------

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions <http://help.github.com/set-up-git-redirect>`__
for installing and configuring git.

.. _contributing.getting_code:

Getting the code (for the first time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will need your own fork to work on the code. Go to the `larray project page
<https://github.com/larray-project/larray>`_ and hit the ``Fork`` button.

You will want to clone your fork to your machine.
To do it manually, follow these steps::

    git clone https://github.com/your-user-name/larray.git
    cd larray
    git remote add upstream https://github.com/larray-project/larray.git

This creates the directory `larray` and connects your repository to
the upstream (main project) *larray* repository.
You can see the remote repositories::

    git remote -v

If you added the upstream repository as described above you will see something
like::

    origin  git@github.com:yourname/larray.git (fetch)
    origin  git@github.com:yourname/larray.git (push)
    upstream        git://github.com/larray-project/larray.git (fetch)
    upstream        git://github.com/larray-project/larray.git (push)

Creating a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting any development, you will need a working Python installation.
It is recommended (but not required) to create an isolated larray development environment.
One of the easiest way to do it is via `Anaconda` or `Miniconda`:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_ as :ref:`suggest earlier <contributing.tools>`
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.getting_code>`
- ``cd`` to the *larray* source directory

We'll now kick off a two-step process:

1. Install the dependencies

.. code-block:: none

   # Create and activate the environment
   conda create -n larray_dev numpy pandas pytables pyqt qtpy matplotlib openpyxl xlsxwriter pytest
   conda activate larray_dev
   # Install ruff (as of September 2023, it is not available on Anaconda)
   pip install ruff

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

To view your environments::

      conda info -e

To return to your root environment::

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`_.

2. Install larray in "development mode"

Install larray using the following command:

.. code-block:: none

  python setup.py develop

This creates some kind of symbolic link between your python installation "modules"
directory and your repository, so that any change in your local copy is automatically
usable by other modules.

At this point you should be able to import larray from your local version::

   $ python  # start an interpreter
   >>> import larray
   >>> larray.__version__
   '0.29-dev'


Starting to contribute
----------------------

With your local version of larray, you are now ready to contribute to the project.
To make a contribution, please follow the steps described bellow.

Step 1: Create a new branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git checkout -b issue123

This changes your working directory to the issue123 branch.
Keep any changes in this branch specific to one bug or feature so it is clear
what the branch brings to the project. You can have many different branches
and switch between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the master branch::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest larray git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``stash`` them prior
to updating.  This will effectively store your changes and they can be reapplied
after updating.

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

You can check your code respects most of those conventions and some other style guidelines by running
the following command in the project directory: ::

  > ruff check .


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

.. _contributing.testing:

Step 4: Test your code
~~~~~~~~~~~~~~~~~~~~~~

Our unit tests are written using the `pytest library <https://docs.pytest.org>`_
and our tests modules are located in `/larray/tests/`.
The pytest library is able to automatically detect and run unit tests
as long as you respect some conventions:

  - pytest will search for ``test_*.py`` or ``*_test.py files``.
  - From those files, collect test items:

    - ``test_`` prefixed test functions or methods outside of class.
    - ``test_`` prefixed test functions or methods inside Test prefixed test classes
      (without an __init__ method).

For more details, please read the section `Conventions for Python test discovery
<https://docs.pytest.org/en/latest/goodpractices.html#test-discovery>`_
from the pytest documentation.

Here is an example of a unit test function using pytest: ::

  from larray.core.axis import _to_key

  def test_key_string_split():
      assert _to_key('M,F') == ['M', 'F']
      assert _to_key('M,') == ['M']

To run unit tests for a given test module: ::

  > pytest larray/tests/test_array.py

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

  > pytest larray/core/array.py

To run all the tests, simply go to root directory and type: ::

  > pytest

pytest will automatically detect all existing unit tests and doctests and run them all.

Step 5: Add a change log
~~~~~~~~~~~~~~~~~~~~~~~~

Changes should be reflected in the release notes located in ``doc/source/changes/version_<next_release_version>.inc``.
This file contains an ongoing change log for the next release.
Add an entry to this file to document your fix, enhancement or (unavoidable) breaking change.
If you hesitate in which section to add your change log, feel free to ask.
Make sure to include the GitHub issue number when adding your entry (using ``closes :issue:`123```
where 123 is the number associated with the fixed issue).

Step 6: Commit your changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all the above is done, commit your changes. Make sure that one of your commit messages starts with
``fix #123 :`` (where 123 is the issue number) before starting any pull request
(see `this github page <https://help.github.com/articles/closing-issues-using-keywords>`_ for more details).

Step 7: Push your changes
~~~~~~~~~~~~~~~~~~~~~~~~~

When you want your changes to appear publicly on the web page of your fork on GitHub,
push your forked feature branch's commits::

    git push origin issue123

Here ``origin`` is the default name given to your remote repository on GitHub.

Step 8: Start a pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You are ready to request your changes to be included in the master branch
(so that they will be available in the next release).
To submit a pull request:

#. Navigate to your repository on GitHub
#. Click on the ``Pull Request`` button
#. You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
#. Write a description of your changes in the ``Preview Discussion`` tab
#. If this is your first pull request, please state explicitly that you accept and are allowed
   to have your contribution (and any future contribution) licensed under the GPL license
   (See section :ref:`Licensing <contributing.licensing>` above).
#. Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review the code.
Your modifications will also be automatically tested by running the *larray* test suite via Github actions
continuous integration service. A pull request will only be
considered for merging when you have an all 'green' build.
If any tests are failing, then you will get a red 'X', where you can click through to see the individual failed tests.

If you need to make more changes to fix test failures or to take our comments into account, you can make them in
your branch, add them to a new commit and push them to GitHub using::

    git push origin issue123

This will automatically update your pull request with the latest code and trigger the automated tests again.


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
