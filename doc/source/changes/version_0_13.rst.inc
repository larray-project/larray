New features
------------

* implemented a new way to do input/output from/to Excel

    >>> a = ndrange((2, 3))
    >>> wb = open_excel('c:/tmp/y.xlsx')
    # put a at A1 in Sheet1, excluding headers (labels)
    >>> wb['Sheet1'] = a
    # dump a at A1 in Sheet2, including headers (labels)
    >>> wb['Sheet2'] = a.dump()
    # save the file to disk
    >>> wb.save()
    # close it
    >>> wb.close()

    >>> wb = open_excel('c:/tmp/y.xlsx')
    # load a from the data starting at A1 in Sheet1, assuming the absence of headers.
    >>> a1 = wb['Sheet1']
    # load a from the data starting at A1 in Sheet1, assuming the presence of (correctly formatted) headers.
    >>> a2 = wb['Sheet2'].load()
    >>> wb.close()

    >>> wb = open_excel('c:/tmp/y.xlsx')
    # note that Sheet2 must exist
    >>> sheet2 = wb['Sheet2']
    # write a without labels starting at C5
    >>> sheet2['C5'] = a
    # write a with its labels starting at A10
    >>> sheet2['A10'] = a.dump()

  load an array with its axes information from a range. As you might have guessed, we could also use the sheet2
  variable here

    >>> b = wb['Sheet2']['A10:D12'].load()
    >>> b
    {0}*\{1}* | 0 | 1 | 2
            0 | 0 | 1 | 2
            1 | 3 | 4 | 5

  load an array (raw data) with no axis information from a range.

    >>> c = sheet['B11:D12']
    >>> # in fact, this is not really an LArray ...
    >>> c
    <larray.excel.Range at 0x1ff1bae22e8>
    >>> # but it can be used as such (this is currently very experimental)
    >>> c.sum(axis=0)
    {0}* |   0 |   1 |   2
         | 3.0 | 5.0 | 7.0
    >>> # ... and it can be used for other stuff, like setting the formula instead of the value:
    >>> c.formula = '=D10+1'
    >>> # in the future, we should also be able to set font name, size, style, etc.

* implemented LArray.rename({axis: new_name}) as well as using kwargs to rename several axes at once

    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> a = ndrange([nat, sex])
    >>> a
    nat\sex | M | F
         BE | 0 | 1
         FO | 2 | 3
    >>> a.rename(nat='nat2', sex='gender')
    nat2\gender | M | F
             BE | 0 | 1
             FO | 2 | 3
    >>> a.rename({'nat': 'nat2', 'sex': 'gender'})
    nat2\gender | M | F
             BE | 0 | 1
             FO | 2 | 3

* made tab-completion of axes names possible in an interactive console 

For completeness
----------------

* taking a subset of an array with wildcard axes now returns an array with wildcard axes

* fixed a case where wildcard axes were considered incompatible when they actually were compatible 

* better support for anonymous axes

* fix for obscure bugs, better doctests, cleaner implementation for a few functions, …
