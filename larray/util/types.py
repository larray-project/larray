from typing import Union, TypeVar
from numpy import generic

R = TypeVar('R')

Scalar = Union[bool, int, float, str, bytes, generic]
