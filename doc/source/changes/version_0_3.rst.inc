New features
------------

* implemented LArray.with_total(): appends axes or group aggregates to the array.

  Without argument, it adds totals on all axes. It has optional keyword only arguments:

    * *label*: specify the label ("total" by default)
    * *op*: specify the aggregate function (sum by default, all other aggregates should work too)

  With multiple arguments, it adds totals sequentially. There are some tricky cases. For example when, for the same
  axis, you add group aggregates and axis aggregates:

    >>> # works but "wrong" for x.geo (double what is expected because the total also
    >>> # includes fla wal & bru)
    >>> la.with_total(x.sex, (fla, wal, bru), x.geo, x.lipro)

    >>> # correct total but the order is not very nice
    >>> la.with_total(x.sex, x.geo, (fla, wal, bru), x.lipro)

    >>> # the correct way to do it, but it is probably not entirely obvious
    >>> la.with_total(x.sex, (fla, wal, bru, x.geo.all()), x.lipro)

    >>> # we probably want to display a warning (or even an error?) in that case.
    >>> # If the user really wants that behavior, he can split the operation:
    >>> # .with_total((fla, wal, bru)).with_total(x.geo)

* implemented group aggregates without using keyword arguments. As a consequence of this, one can no longer use axis
  numbers in aggregates. Eg. `a.sum(0)` does not sum on the first axis anymore (but you can do `a.sum(a.axes[0])` if
  needed)

* implemented LArray.percent: equivalent to ratio * 100

* implemented Session.filter -> returns a new Session with only objects matching the filter

* implemented Session.dump -> dumps all LArray in the Session to a file

* implemented Session.load -> load several LArrays from a file to a Session
