New features
------------

* Implemented a "by" variant to all aggregate methods (e.g. sum_by, mean_by, etc.). These methods aggregate all axes
  except those listed, which means the only axes remaining after the aggregate operation will be those listed.
  For example: arr.sum_by(x.a) is equivalent to arr.sum(arr.axes - x.a)

    >>> arr = ndtest((2, 3, 4))
    >>> arr
     a | b\c | c0 | c1 | c2 | c3
    a0 |  b0 |  0 |  1 |  2 |  3
    a0 |  b1 |  4 |  5 |  6 |  7
    a0 |  b2 |  8 |  9 | 10 | 11
    a1 |  b0 | 12 | 13 | 14 | 15
    a1 |  b1 | 16 | 17 | 18 | 19
    a1 |  b2 | 20 | 21 | 22 | 23
    >>> arr.sum_by(x.b)
    b | b0 | b1 |  b2
      | 60 | 92 | 124

* Added .extend() method to Axis class

    >>> a = Axis('a', 'a0..a2')
    >>> a
    Axis('a', ['a0', 'a1', 'a2'])
    >>> other = Axis('other', 'a3..a5')
    >>> a.extend(other)
    Axis('a', ['a0', 'a1', 'a2', 'a3', 'a4', 'a5'])

  or directly specify the extra labels as a list or as a "label string":

    >>> a.extend('a3..a5')
    Axis('a', ['a0', 'a1', 'a2', 'a3', 'a4', 'a5'])

* Added title argument to all array creation functions (ndrange, zeros, ones, ...) and display it in the .info of
  array objects.

    >>> a = ndrange(3, title='a simple test array')
    >>> a.info
    a simple test array
    3
     {0}* [3]: 0 1 2

* implemented creating an Axis using a group:

    >>> arr = ndtest((2, 3))
    >>> arr
    a\b | b0 | b1 | b2
    a0 |  0 |  1 |  2
    a1 |  3 |  4 |  5
    >>> a, b = arr.axes
    >>> zeros((a, b[:'b1']))
    a\b |  b0 |  b1
    a0 | 0.0 | 0.0
    a1 | 0.0 | 0.0

* made Axis.startswith, .endswith and .matches accept Group instances

    >>> a = Axis('a', 'a0..b2')
    >>> a
    Axis('a', ['a0', 'a1', 'a2', 'b0', 'b1', 'b2'])

    >>> prefix = Axis('prefix', 'a,b')
    >>> a.startswith(prefix['a'])
    a['a0', 'a1', 'a2']
    >>> a.startswith(prefix.i[1])
    a['b0', 'b1', 'b2']

* implemented all usual binary operations (+, -, `*`, /, ...) on Group

    >>> year = Axis('year', '2011..2016')
    >>> year[2013] + 1
    2014
    >>> year.i[2] + 1
    2014

* made the viewer is much more useful as a debugger in the middle of a function by generalizing SessionEditor to handle
  any mapping, instead of only Session objects but made it list and display only array objects. To view the value of
  non-array variable one should type their name in the console. Given those changes, view() will superficially behave
  as before, but behind the scene, *all* variables which were defined in the scope where view() was called will be
  available in the viewer console, even though they will not appear in the list on the left. This means that the viewer
  console will be able to use scalars defined at that point and call others functions of your code. In other words,
  there are more chances you can execute some code from the function calling view() by simply copy-pasting the code
  line.


Backward incompatible changes
-----------------------------

* LGroup lost set-like operations (intersection and union) to the profit of a specific subclass (LSet). In other words,
  this no longer works:

    >>> letters = Axis('letters', 'a..z')
    >>> letters[':c'] & letters['b:']

  To make it work, we need to convert the LGroup(s) to LSets explicitly:

    >>> letters[':c'].set() & letters['b:d'].set()
    letters.set[OrderedSet(['b', 'c'])]

    >>> letters[':c'].set() | letters['b:d'].set()
    letters.set[OrderedSet(['a', 'b', 'c', 'd'])]

    >>> letters[':c'].set() - 'b'
    letters.set[OrderedSet(['a', 'c'])]

* group aggregates produce simple string labels for the new aggregated axis instead of using the group themselves as
  labels. This means one can no longer know where a group comes from but this simplifies the code and fixes a few
  issues, most notably export of aggregated arrays to Excel, and some operations between two aggregated arrays.

    >>> arr = ndtest((3, 4))
    >>> arr
    a\b | b0 | b1 | b2 | b3
     a0 |  0 |  1 |  2 |  3
     a1 |  4 |  5 |  6 |  7
     a2 |  8 |  9 | 10 | 11
    >>> agg = arr.sum(':b2 >> tob2;b2,b3 >> other')
    >>> agg
    a\b | tob2 | other
     a0 |    3 |     5
     a1 |   15 |    13
     a2 |   27 |    21
    >>> agg.info
    3 x 2
     a [3]: 'a0' 'a1' 'a2'
     b [2]: 'tob2' 'other'
    >>> agg.axes.b.labels[0]
    'tob2'

  In previous versions this would have returned:

    >>> agg.axes.b.labels[0]
    LGroup(':b2', name='tob2', axis=Axis('b', ['b0', 'b1', 'b2', 'b3']))

* a string containing only a single "integer-like" is no longer transformed to an integer
  e.g. "10" will evaluate to (the string) "10" (like in version 0.17 and earlier) while "10,20" will evaluate to the
  list of integers: [10, 20]


Other changes
-------------

* changed how Group instances are displayed.

    >>> a = Axis('a', 'a0..a2')
    >>> a['a1,a2']
    a['a1', 'a2']


Fixes
-----

* fixed > and >= on Group using slices

* avoid a division by 0 warning when using divnot0

* viewer: fixed plots when Qt5 is installed. This also removes the matplotlib warning people got when running the
  viewer with Qt5 installed.

* viewer: display array when typing its name in the console even when no array was selected previously


Misc
----

* misc code cleanup, improved docstrings, ...
