from typing import Any

import numpy as np


def isscalar(element: Any) -> bool:
    r"""
    Return `True` if the type of element is a scalar type.

    Parameters
    ----------
    element: any
        Input argument, can be of any type and shape.

    Returns
    -------
    bool
        `True` if `element` is a scalar type, `False` if it is not.

    Examples
    --------
    >>> from larray import ndtest
    >>> isscalar(3.1)
    True
    >>> isscalar([3.1])
    False
    >>> isscalar(False)
    True
    >>> isscalar('larray')
    True

    >>> arr = ndtest((2, 2))
    >>> arr
    a\b  b0  b1
     a0   0   1
     a1   2   3
    >>> isscalar(arr)
    False
    """
    return np.isscalar(element)
