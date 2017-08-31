How to contribute
=================

Getting the code (for the first time)
-------------------------------------

- install a Git client

  On Windows, TortoiseGit provides a nice graphical wrapper. You need to install both the console client from
  http://msysgit.github.io/ and `TortoiseGit <https://code.google.com/p/tortoisegit>`_ itself.

- create an account on `GitHub <https://github.com/>`_ (not necessary for readonly).

- clone the repository on your local machine ::

  > git clone https://github.com/liam2/larray.git


Installing the module
---------------------

You could install LArray in the standard way: ::

  > python setup.py install

but in that case you need to "install" it again every time you change it. When developing, it is usually more
convenient to use: ::

  > python setup.py develop

This creates some kind of symlink between your python installation "modules" directory and your repository, so that any
change in your local copy is automatically usable by other modules.


Updating your local copy with remote changes
--------------------------------------------

::

  > git pull  # or git fetch + git merge


Code conventions
----------------

`PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ is your friend. Among others, this means:

- 120 characters lines
- 4 spaces indentation
- lowercase (with underscores if needed) variables, functions, methods and modules names
- CamelCase classes names
- all uppercase constants names
- whitespace around binary operators
- no whitespace before a comma, semicolon, colon or opening parenthesis
- whitespace after commas

This summary should not prevent you from reading the PEP!


Docstring conventions
---------------------

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


Documentation
-------------

The documentation is written using reStructuredText and built to various formats using
`Sphinx <http://sphinx-doc.org/>`_. See the `reStructuredText Primer <http://sphinx-doc.org/rest.html#rst-primer>`_
for a first introduction of the syntax.

Installing Requirements
~~~~~~~~~~~~~~~~~~~~~~~

Basic requirements (to generate an .html version of the documentation) can be installed using: ::

  > conda install sphinx numpydoc

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


Tests
-----

We use both unit tests and doctests. Unit tests are written using Python's built-in
`unittest module <https://docs.python.org/3/library/unittest.html>`_.
For example: ::

  from unittest import TestCase

  class TestValueStrings(TestCase):
      def setUp(self):
          pass

      def tearDown(self):
          pass

      def test_split(self):
          self.assertEqual(to_ticks('M,F'), ['M', 'F'])
          self.assertEqual(to_ticks('M, F'), ['M', 'F'])

To run all unit tests: ::

  > python -m unittest -v larray\tests\test_la.py

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

  > python -m doctest -v larray\larray.py

To run both at the same time, one can use nosetests (install with `conda install nose`): ::

  > nosetests -v --with-doctest


Sending your changes
--------------------

::

  > git add       # tell git it should care about a file it previously ignored (only if needed)

  > git commit    # creates a new revision of the repository using its current state

  > git pull      # updates your local repository with "upstream" changes.
                  # this might create conflicts that you will need to resolve.
                  # this should also be done before you start making changes.

  > git push      # send all your committed changes "upstream".
