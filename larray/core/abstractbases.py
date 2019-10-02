from __future__ import absolute_import

from abc import ABCMeta


# define abstract base classes to enable isinstance type checking on our objects
# idea taken from https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/generic.py
# FIXME: __metaclass__ is ignored in Python 3
class ABCAxis(object):
    __metaclass__ = ABCMeta


class ABCAxisReference(ABCAxis):
    __metaclass__ = ABCMeta


class ABCArray(object):
    __metaclass__ = ABCMeta
