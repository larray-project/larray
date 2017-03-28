Core
----

* better .info for arrays with groups in axes

    >>> # example using groups without a name
    >>> reg = la.sum((fla, wal, bru, belgium))
    >>> reg.info
    4 x 15
     geo [4]: ['A11' ... 'A73'] ['A25' ... 'A93'] 'A21' ['A11' ... 'A21']
     lipro [15]: 'P01' 'P02' 'P03' ... 'P13' 'P14' 'P15'

    >>> # example using groups with a name
    >>> fla = geo.group(fla_str, name='Flanders')
    >>> wal = geo.group(wal_str, name='Wallonia')
    >>> bru = geo.group(bru_str, name='Brussels')
    >>> reg = la.sum((fla, wal, bru))
    >>> reg.info
    3 x 15
     geo [3]: 'Flanders' (['A11' ... 'A73']) 'Wallonia' (['A25' ... 'A93']) 'Brussels' ('A21')
     lipro [15]: 'P01' 'P02' 'P03' ... 'P13' 'P14' 'P15'


Editor
------

* fixed edit() with non-string labels in axes
* fixed edit() with filters in some more cases
* fixed ArrayEditorWidget.reject_changes and accept_changes to update the model & view accordingly (in case the widget
  is kept open)
* avoid (harmless) error messages in some cases
