Core
----

* added new methods on axes: matches, startswith, endswith

    >>> country = Axis('country', ['FR', 'BE', 'DE', 'BR'])
    >>> country.matches('BE|FR')
    LGroup(['FR', 'BE'])
    >>> country.matches('^..$') # labels 2 characters long
    LGroup(['FR', 'BE', 'DE', 'BR'])

    >>> country.startswith('B')
    LGroup(['BE', 'BR'])
    >>> country.endswith('R')
    LGroup(['FR', 'BR'])

* implemented set-like operations on LGroup: & (intersection), | (union),  - (difference).
  Slice groups do not work yet on axes references (x.) but that will come in the future…

    >>> alpha = Axis('alpha', 'a,b,c,d')
    >>> alpha['a', 'b'] | alpha['c', 'd']
    LGroup(['a', 'b', 'c', 'd'], axis=…)
    >>> alpha['a', 'b', 'c'] | alpha['c', 'd']
    LGroup(['a', 'b', 'c', 'd'], axis=…)

  a name is computed automatically when both operands are named

    >>> r = alpha['a', 'b'].named('ab') | alpha['c', 'd'].named('cd')
    >>> r.name
    'ab | cd'
    >>> r.key
    ['a', 'b', 'c', 'd']

  numeric axes work too

    >>> num = Axis('num', range(10))
    >>> num[:2] | num[8:]
    num[0, 1, 2, 8, 9]
    >>> num[:2] | num[5]
    num[0, 1, 2, 5])

  intersection

    >>> LGroup(['a', 'b', 'c']) & LGroup(['c', 'd'])
    LGroup(['c'])

  difference

    >>> LGroup(['a', 'b', 'c']) - LGroup(['c', 'd'])
    LGroup(['a', 'b'])
    >>> LGroup(['a', 'b', 'c']) - 'b'
    LGroup(['a', 'c'])

* fixed loading 1D arrays using open_excel 


Viewer
------

* added tooltip with the axes labels corresponding to each cell of the array viewer
* added name and dimensions of the current array to the window title bar in the session viewer
* added tooltip with each array .info() in the list of arrays of the session viewer

* fixed eval box throwing an exception when trying to set a new variable (if qtconsole is not present)
* fixed group aggregates using LGroups defined using axes references (x.), for example: 

    >>> arr.sum(x.age[:10])

* fixed group aggregates using anonymous axes
