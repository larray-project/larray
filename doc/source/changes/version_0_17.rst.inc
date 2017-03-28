Core
----

* added ndtest function to create n-dimensional test arrays (of given shape). Axes are named by single letters starting from 'a'. Axes labels are constructed using a '{axis_name}{label_pos}' pattern (e.g. 'a0'). 

    >>> ndtest(6)
    a | a0 | a1 | a2 | a3 | a4 | a5
      |  0 |  1 |  2 |  3 |  4 |  5
    >>> ndtest((2, 3))
    a\b | b0 | b1 | b2
    a0 |  0 |  1 |  2
    a1 |  3 |  4 |  5
    >>> ndtest((2, 3), label_start=1)
    a\b | b1 | b2 | b3
    a1 |  0 |  1 |  2
    a2 |  3 |  4 |  5

* allow naming "one-shot" groups in group aggregates.

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b | b0 | b1 | b2
    a0 |  0 |  1 |  2
    a1 |  3 |  4 |  5
    >>> arr.sum('g1=b0;g2=b1,b2;g3=b0:b2')
    a\b | 'g1' ('b0') | 'g2' (['b1' 'b2']) | 'g3' ('b0':'b2')
    a0 |           0 |                  3 |                3
    a1 |           3 |                  9 |               12

* implemented argmin, argmax, posargmin, posargmax without an axis argument (works on the full array).

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b | b0 | b1 | b2
    a0 |  0 |  1 |  2
    a1 |  3 |  4 |  5
    >>> arr.argmin()
    ('a0', 'b0')

* added preliminary code to add a title attribute to LArray.

  This needs a lot more work to be really useful though, as it can currently only be used
  in the LArray() function itself and is only used in Session.summary() (see below).
  There are many places where this should be used, but this is not done yet.

* added Session.summary() which displays a list of all arrays, their dimension names and title if any.
  
  This can be used in combination with local_arrays() to produce some kind of codebook with all
  the arrays of a function.

    >>> arr = LArray([[1, 2], [3, 4]], 'sex=M,F;nat=BE,FO', title='a test array')
    >>> arr
    sex\nat | BE | FO
          M |  1 |  2
          F |  3 |  4
    >>> s = Session({'arr': arr})
    >>> s
    Session(arr)
    >>> print(s.summary())
    arr: sex, nat
        a test array

* fixed using groups from other (compatible) axis
* fixed group aggregates using groups without axis
* fixed axis[another_label_group] when said group had a non-string Axis
* fixed axis.group(another_label_group, name='a_name') (name was not set correctly)
* fixed ipfp progress message when progress is negative


viewer
------

* when setting part of an array in the console (by using e.g. arr['M'] = 10), display that array
* when typing in the console the name of an existing array, select it in the list

* fixed missing tooltips for arrays added to the session from within the session viewer
* fixed window title (with axes info) not updating in many cases
* fixed the filters bar not being cleared when displaying a non-LArray object after an LArray object


misc
----

* improved messages in ipfp(display_progress=True)
* improved tests, docstrings, ...
