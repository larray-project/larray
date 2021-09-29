from typing import Union, TypeVar, List
from numpy import generic, ndarray

R = TypeVar('R')

Scalar = Union[bool, int, float, str, bytes, generic]
Key = Union[Scalar, 'Group', 'Array', ndarray, 'OrderedSet', List[Scalar], slice]   # noqa: F821
