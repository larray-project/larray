from __future__ import absolute_import

from abc import ABCMeta

# define abstract base classes to enable isinstance type checking on our objects
# idea taken from https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/generic.py
class ABCAxis(object):
    __metaclass__ = ABCMeta

class ABCAxisReference(ABCAxis):
    __metaclass__ = ABCMeta

class ABCLArray(object):
    __metaclass__ = ABCMeta