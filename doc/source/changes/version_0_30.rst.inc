Syntax changes
--------------

* new syntax


Backward incompatible changes
-----------------------------

* backward incompatible changes


New features
------------

* added a feature (see the :ref:`miscellaneous section <misc>` for details).

* added another feature.


.. _misc:

Miscellaneous improvements
--------------------------

* vastly improved indexing using arrays (of labels, indices or booleans). Many advanced cases did not work, including
  when combining several indexing arrays, or when (one of) the indexing array(s) had an axis present in the array.

    First let's create some test axes

    >>> a, b, c = ndtest((2, 3, 2)).axes

    Then create a test array.

    >>> arr = ndtest((a, b))
    >>> arr
    a\b b0 b1 b2
     a0  0  1  2
     a1  3  4  5

    If the key array has an axis not already present in arr (e.g. c), the target axis (a) is replaced by the extra
    axis (c). This already worked previously.

    >>> key = LArray(['a1', 'a0'], c)
    >>> key
    c  c0  c1
       a1  a0
    >>> arr[key]
    c\b  b0  b1  b2
     c0   3   4   5
     c1   0   1   2

    If the key array has the target axis, the axis stays the same, but the data is reordered (this also worked
    previously):

    >>> key = LArray(['b1', 'b0', 'b2'], b)
    >>> key
    b  b0  b1  b2
       b1  b0  b2
    >>> arr[key]
    a\b  b0  b1  b2
     a0   1   0   2
     a1   4   3   5

    From here on, the examples shown did not work previously...

    Now, if the key contains another axis present in the array (b) which is not the target axis (a), the target axis
    completely disappears (both axes are replaced by the key axis):

    >>> key = LArray(['a0', 'a1', 'a0'], b)
    >>> key
    b  b0  b1  b2
       a0  a1  a0
    >>> arr[key]
    b  b0  b1  b2
        0   4   2

    If the key has both the target axis (a) and another existing axis (b)

    >>> key
    a\b b0 b1 b2
     a0 a0 a1 a0
     a1 a1 a0 a1
    >>> arr[key]
    a\b  b0  b1  b2
     a0   0   4   2
     a1   3   1   5

    If the key has both another existing axis (a) and an extra axis (c)

    >>> key
    a\c  c0  c1
     a0  b0  b1
     a1  b2  b0
    >>> arr[key]
    a\c  c0  c1
     a0   0   1
     a1   5   3

    It also works if the key has the target axis (a), another existing axis (b) and an extra axis (c), but this is not
    shown for brevity.


Fixes
-----

* fixed something (closes :issue:`1`).
