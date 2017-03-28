New features
------------

* A single change in this release: a much more powerful to_excel function which (by default) use Excel itself to write
  files. Additional functionality include:

    - write in an existing file without overwriting existing data/sheet/…
    - write at a precise position
    - view an array in a live Excel instance (a new OR an existing workbook)

  See :meth:`~larray.LArray.to_excel` documentation for details.
