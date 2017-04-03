New features
------------

* implemented a new syntax for stack(): `stack({label1: value1, label2: value2}, axis)`

    >>> nat = Axis('nat', 'BE, FO')
    >>> sex = Axis('sex', 'M, F')
    >>> males = ones(nat)
    >>> males
    nat |  BE |  FO
        | 1.0 | 1.0
    >>> females = zeros(nat)
    >>> females
    nat |  BE |  FO
        | 0.0 | 0.0

  In the case the axis has already been defined in a variable, this gives:

    >>> stack({'M': males, 'F': females}, sex)
    nat\\sex |   M |   F
         BE | 1.0 | 0.0
         FO | 1.0 | 0.0

  Additionally, axis can now be an axis string definition in addition to an Axis object, which means one can write this:

    >>> stack({'M': males, 'F': females}, 'sex=M,F')

  It is better than the simpler but *highly discouraged* alternative:

    >>> stack([males, females), sex)

  because it is all too easy to invert labels. It is very hard to spot the error in the following line, and larray
  cannot spot it for you either:

    >>> stack([females, males), sex)
    nat\\sex |   M |   F
         BE | 0.0 | 1.0
         FO | 0.0 | 1.0

  When creating an axis from scratch (it does not already exist in a variable), one might want to use this:

    >>> stack([males, females], 'sex=M,F')

  even if this could suffer, to a lesser extent, the same problem as above when stacking many arrays.

* added another feature (see the :ref:`miscellaneous section <misc>` for details).

.. _misc:

Miscellaneous improvements
--------------------------

* improved something.

Fixes
-----

* fixed group aggregates on integer arrays for median, percentile, var and std (closes :issue:`193`).

* fixed group sum over boolean arrays (closes :issue:`194`).
