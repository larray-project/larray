Warning: this release needs to be installed using:

  conda update larray
  conda update xlwings


New features
------------

* implemented support for xlwings 0.9+. This allowed us to change the way we interact with Excel:

  - by default, the Excel instance we use is configured to be both hidden and
    silent (for example, it does not prompt to update/edit links).

  - by default, we now use a dedicated Excel instance for each call to
    open_excel, instead of reusing any existing instance if there was any open.
    In practice, it means input/output from/to Excel is more reliable and does
    not risk altering any workbook you had open (except if you ask for that
    explicitly). The cost of this is that it is slower by default. If you open
    many different workbooks, it is recommended that you create a single Excel
    instance and reuse it. This can be done with:

    >>> from larray import *
    >>> import xlwings as xw

    >>> app = xw.App(visible=False, add_book=False)
    >>> wb1 = open_excel('workbook1.xlsx', app=app)
    # use wb1 as before
    >>> wb1.close()
    >>> wb2 = open_excel('workbook2.xlsx', app=app)
    # use wb2 as before
    >>> wb2.close()
    >>> app.quit()    

* added ipfp function which does Iterative Proportional Fitting Procedure (also known as bi-proportional fitting in
  statistics or RAS algorithm in economics). Note that this new function is currently not in the core module,
  so it needs a specific import command:
  
    >>> from larray.ipfp import ipfp

    >>> a = Axis('a', 2)
    >>> b = Axis('b', 2)
    >>> initial = LArray([[2, 1],
    ...                   [1, 2]], [a, b])
    >>> initial
    a*\b* | 0 | 1
        0 | 2 | 1
        1 | 1 | 2
    >>> target_sum_along_a = LArray([2, 1], b)
    >>> target_sum_along_b = LArray([1, 2], a)
    >>> ipfp([target_sum_along_a, target_sum_along_b], initial, threshold=0.01)
    a*\b* |                  0 |                   1
        0 | 0.8450704225352113 | 0.15492957746478875
        1 | 1.1538461538461537 |  0.8461538461538463
                               
* made it possible to create arrays more succintly in some usual cases (especially for quick arrays for testing
  purposes). Previously, when one created an array from scratch, he had to provide Axis object(s) (or another array).
  Note that the following examples use zeros() but this change affects all array creation functions (ones, zeros,
  ndrange, full, empty):
  
    >>> nat = Axis('nat', ['BE', 'FO'])
    >>> sex = Axis('sex', ['M', 'F'])
    >>> zeros([nat, sex])
    nat\sex |   M |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0

  Now, when you have axe names and axes labels but do not have/want to reuse an
  existing axis, you can use this syntax:

    >>> zeros([('nat', ['BE', 'FO']),
    ...        ('sex', ['M', 'F'])])
    nat\sex |   M |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0

  If additionally all axe names and labels are strings (not integers or other types) which do not contain any special
  character ("=", "," or ";") you can use:

    >>> zeros('nat=BE,FO;sex=M,F')
    nat\sex |   M |   F
         BE | 0.0 | 0.0
         FO | 0.0 | 0.0

  See below (*) for some more alternate syntaxes and an explanation of how this works.

* added additional, less error-prone syntax for stack:

    >>> nat = Axis('nat', 'BE,FO')
    >>> arr1 = ones(nat)
    >>> arr1
    nat |  BE |  FO
        | 1.0 | 1.0
    >>> arr2 = zeros(nat)
    >>> arr2
    nat |  BE |  FO
        | 0.0 | 0.0
    >>> stack([('M', arr1), ('F', arr2)], 'sex')
    nat\sex |   H |   F
         BE | 1.0 | 0.0
         FO | 1.0 | 0.0

  in addition to the still supported but discouraged (because one has to remember the order of labels):

    >>> sex = Axis('sex', ['M', 'F'])
    >>> stack((arr1, arr2), sex)
    nat\sex |   H |   F
         BE | 1.0 | 0.0
         FO | 1.0 | 0.0

* added LArray.compact and Session.compact() to detect and remove "useless" axes
  (ie axes for which values are constant over the whole axis)

    >>> a = LArray([[1, 2], [1, 2]], [Axis('sex', 'M,F'), Axis('nat', 'BE,FO')])
    >>> a
    sex\nat | BE | FO
          M |  1 |  2
          F |  1 |  2
    >>> a.compact()
    nat | BE | FO
        |  1 |  2

* made Session keep the order in which arrays were added to it. The main goal was to make this work:

    >>> b, a = s['b', 'a']

  Previously, since sessions were always traversed alphabetically, this was a dangerous operation because if the keys
  (a and b) were not sorted alphabetically, the result would not be in the expected order:
  
  s['b', 'a'] previously returned a, b instead of b, a !!
  
  Session.names is still sorted alphabetically though (Session.keys() is not)

* added LArray.with_axes(axes) to return a new LArray with the same data but different axes

    >>> a = ndrange(2)
    >>> a
    {0}* | 0 | 1
         | 0 | 1
    >>> a.with_axes([Axis('sex', 'H,F')])
    sex | H | F
        | 0 | 1

* changed width from which an LArray is summarized (using "...") from 80 characters to 200.

* implemented memory_used property which displays nbytes in human-readable form

    >>> a = ndrange('sex=H,F;nat=BE,FO')
    >>> a.memory_used
    '16 bytes'
    >>> a = ndrange(100000)
    >>> a.memory_used
    '390.62 Kb'

* implemented Axis + AxisCollection

    >>> a = ndrange('sex=M,F;type=t1,t2')
    >>> Axis('nat', 'BE,FO') + a.axes
    AxisCollection([
        Axis('nat', ['BE', 'FO']),
        Axis('sex', ['M', 'F']),
        Axis('type', ['t1', 't2'])
    ])
  
(*) For the curious, there are also many syntaxes supported for array creation
functions. In fact, during array creation, at any place a list or tuple of
values is expected, you can specify it using a single string, which will be
split successively at the following characters if present: ";" then "=" then
",". If you apply that algorithm to 'nat=BE,FO;sex=M,F', you get:

1) 'nat=BE,FO;sex=M,F'
2) ('nat=BE,FO', 'sex=M,F')
3) (('nat', 'BE,FO'), ('sex', 'M,F'))
4) (('nat', ('BE', 'FO')), ('sex', ('M', 'F')))

Recognise this last syntax? This is the same as above, except above we replaced
some () with [] for clarity. In fact all the intermediate forms here above are
valid (and equivalent) in array creation functions.
