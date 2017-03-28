Major improvements
------------------

* the documentation (docstrings) of many functions was vastly improved (thanks to Alix)

* implemented a new optional syntax to generate sequences of labels for axes by using patterns

  integer strings generate integers

    >>> ndrange('age=0..10')
    age | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10

  you can combine letters and numbers. The number part is treated like increasing (or decreasing numbers)

    >>> ndrange('lipro=P01..P12')
    lipro | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12
          |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |  10 |  11

  letter patterns generate all combination of letters between the start and end:

    >>> ndrange('test=AA..CC')
    test | AA | AB | AC | BA | BB | BC | CA | CB | CC
         |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8

  other characters are left intact (and should be the same on the start and end patterns:

    >>> ndrange('test=A_1..C_2')
    test | A_1 | A_2 | B_1 | B_2 | C_1 | C_2
         |   0 |   1 |   2 |   3 |   4 |   5

  this also works within Axis()

    >>> Axis('age', '0..10')
    Axis('age', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

* implemented new syntax for defining groups using strings:

    >>> arr = ndtest((3, 4))
    >>> arr
    a\b | b0 | b1 | b2 | b3
     a0 |  0 |  1 |  2 |  3
     a1 |  4 |  5 |  6 |  7
     a2 |  8 |  9 | 10 | 11

  groups can be named using ">>" instead of "=" previously
    
    >>> arr.sum('b1,b3 >> b13;b0:b2 >> b012')
    a\b | b13 | b012
     a0 |   4 |    3
     a1 |  12 |   15
     a2 |  20 |   27

  if some labels are ambiguous, one can specify the axis by using "axis_name[labels]":

    >>> arr.sum('b[b1,b3] >> b13;b[b0:b2] >> b012')
    a\b | b13 | b012
     a0 |   4 |    3
     a1 |  12 |   15
     a2 |  20 |   27

  groups can also be defined by position using this syntax:

    >>> arr.sum('b.i[1,3] >> b13;b.i[0:3] >> b012')
    a\b | b13 | b012
     a0 |   4 |    3
     a1 |  12 |   15
     a2 |  20 |   27

  A few notes:

    - the goal was to have that syntax as close as the "normal" syntax as possible (just remove the "x." and all inner
      quotes).
    - in models, the normal syntax should be preferred, so that the groups can be stored in a variable and reused in
      several places
    - strings representing integers are evaluated as integers.
    - there is experimental support for evaluating expressions within string groups by using "{expr}", but this is
      fragile and might be removed in the future.

* implemented combine_axes & split_axis on arrays:

    >>> arr = ndtest((2, 3, 4))
    >>> arr
     a | b\c | c0 | c1 | c2 | c3
    a0 |  b0 |  0 |  1 |  2 |  3
    a0 |  b1 |  4 |  5 |  6 |  7
    a0 |  b2 |  8 |  9 | 10 | 11
    a1 |  b0 | 12 | 13 | 14 | 15
    a1 |  b1 | 16 | 17 | 18 | 19
    a1 |  b2 | 20 | 21 | 22 | 23

    >>> arr2 = arr.combine_axes((x.a, x.b))
    >>> arr2
    a_b\c | c0 | c1 | c2 | c3
    a0_b0 |  0 |  1 |  2 |  3
    a0_b1 |  4 |  5 |  6 |  7
    a0_b2 |  8 |  9 | 10 | 11
    a1_b0 | 12 | 13 | 14 | 15
    a1_b1 | 16 | 17 | 18 | 19
    a1_b2 | 20 | 21 | 22 | 23

    >>> arr2.split_axis(x.a_b)
     a | b\c | c0 | c1 | c2 | c3
    a0 |  b0 |  0 |  1 |  2 |  3
    a0 |  b1 |  4 |  5 |  6 |  7
    a0 |  b2 |  8 |  9 | 10 | 11
    a1 |  b0 | 12 | 13 | 14 | 15
    a1 |  b1 | 16 | 17 | 18 | 19
    a1 |  b2 | 20 | 21 | 22 | 23

* implemented .by() method on groups which splits them into subgroups of specified length

    >>> arr = ndtest((5, 2))
    >>> arr
    a\b | b0 | b1
     a0 |  0 |  1
     a1 |  2 |  3
     a2 |  4 |  5
     a3 |  6 |  7
     a4 |  8 |  9

    >>> arr.sum(a['a0':'a4'].by(2))
             a\b | b0 | b1
    a['a0' 'a1'] |  2 |  4
    a['a2' 'a3'] | 10 | 12
         a['a4'] |  8 |  9

  there is also an optional second argument to specify the "step" between groups

    >>> arr.sum(a['a0':'a4'].by(2, step=3))
             a\b | b0 | b1
    a['a0' 'a1'] |  2 |  4
    a['a3' 'a4'] | 14 | 16

  if the step is < the group size, you get overlapping groups:

    >>> arr.sum(a['a0':'a4'].by(2, step=1))
             a\b | b0 | b1
    a['a0' 'a1'] |  2 |  4
    a['a1' 'a2'] |  6 |  8
    a['a2' 'a3'] | 10 | 12
    a['a3' 'a4'] | 14 | 16
         a['a4'] |  8 |  9

* groups can be renamed using >> (in addition to the "named" method)

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b | b0 | b1 | b2
     a0 |  0 |  1 |  2
     a1 |  3 |  4 |  5
    >>> arr.sum((x.b['b0,b1'] >> 'b01', x.b['b1,b2'] >> 'b12'))
    a\b | b01 | b12
     a0 |   1 |   3
     a1 |   7 |   9

* implemented rationot0

    >>> a = Axis('a', 'a0,a1')
    >>> b = Axis('b', 'b0,b1,b2')
    >>> arr = LArray([[6, 0, 2],
    ...               [4, 0, 8]], [a, b])
    >>> arr
    a\b | b0 | b1 | b2
     a0 |  6 |  0 |  2
     a1 |  4 |  0 |  8
    >>> arr.sum()
    20
    >>> arr.rationot0()
    a\b |  b0 |  b1 |  b2
     a0 | 0.3 | 0.0 | 0.1
     a1 | 0.2 | 0.0 | 0.4
    >>> arr.rationot0(x.a)
    a\b |  b0 |  b1 |  b2
     a0 | 0.6 | 0.0 | 0.2
     a1 | 0.4 | 0.0 | 0.8

  for reference, the normal ratio method would return:

    >>> arr.ratio(x.a)
    a\b |  b0 |  b1 |  b2
     a0 | 0.6 | nan | 0.2
     a1 | 0.4 | nan | 0.8


Misc improvements
-----------------

* implemented [] on groups so that you can further subset them
* added a new "condensed" option for ipfp's display_progress argument to get back the old behavior
* changed how named groups are displayed (only the name is displayed)
* positional groups gained a few features and are almost on par with label groups now
* when iterating over an axis (for example when doing "for y in year_axis:" it yields groups (instead of raw labels) so
  that it works even in the presence of ambiguous labels.
* Axis.startswith, endswith, matches create groups which include the axis (so that those groups work even if the labels
  exist on several axes)


Bug fixes
---------

* fixed Session.summary() when arrays in the session have axes without name
* fixed full() and full_like() with an explicit dtype (the dtype was ignored)
