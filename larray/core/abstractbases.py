from abc import ABC


# define abstract base classes to enable isinstance type checking on our objects
# idea taken from https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/generic.py
class ABCAxis(ABC):
    pass


class ABCAxisReference(ABCAxis):
    pass


class ABCArray(ABC):
    pass
