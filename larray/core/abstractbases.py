# define abstract base classes to enable isinstance type checking on our objects
# idea taken from https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/generic.py
# we do not inherit from abc.ABC because it costs us a ~3% performance bump on
# our benchmarks for very little benefit
class ABCAxis:
    pass


class ABCAxisReference(ABCAxis):
    pass


class ABCArray:
    pass
