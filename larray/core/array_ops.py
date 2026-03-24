import builtins
import numpy as np
from typing import Union
from larray.util.types import Scalar

# Note: We use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from larray.core.array import Array


def all(values, axis=None) -> Union['Array', Scalar]:
    r"""
    Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    Array.all
    """
    from larray.core.array import Array
    if isinstance(values, Array):
        return values.all(axis)
    else:
        return builtins.all(values)


def any(values, axis=None) -> Union['Array', Scalar]:
    r"""
    Test whether any array elements along a given axis evaluate to True.

    See Also
    --------
    Array.any
    """
    from larray.core.array import Array
    if isinstance(values, Array):
        return values.any(axis)
    else:
        return builtins.any(values)


# commutative modulo float precision errors
def sum(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Sum of array elements.

    See Also
    --------
    Array.sum
    """
    from larray.core.array import Array
    # XXX: we might want to be more aggressive here (more types to convert),
    #      however, generators should still be computed via the builtin.
    if isinstance(array, (np.ndarray, list)):
        array = Array(array)
    if isinstance(array, Array):
        return array.sum(*args, **kwargs)
    else:
        return builtins.sum(array, *args, **kwargs)


def prod(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Product of array elements.

    See Also
    --------
    Array.prod
    """
    return array.prod(*args, **kwargs)


def cumsum(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Return the cumulative sum of array elements.

    See Also
    --------
    Array.cumsum
    """
    return array.cumsum(*args, **kwargs)


def cumprod(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Return the cumulative product of array elements.

    See Also
    --------
    Array.cumprod
    """
    return array.cumprod(*args, **kwargs)


def min(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Minimum of array elements.

    See Also
    --------
    Array.min
    """
    from larray.core.array import Array
    if isinstance(array, Array):
        return array.min(*args, **kwargs)
    else:
        return builtins.min(array, *args, **kwargs)


def max(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Maximum of array elements.

    See Also
    --------
    Array.max
    """
    from larray.core.array import Array
    if isinstance(array, Array):
        return array.max(*args, **kwargs)
    else:
        return builtins.max(array, *args, **kwargs)


def mean(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the arithmetic mean.

    See Also
    --------
    Array.mean
    """
    return array.mean(*args, **kwargs)


def median(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the median.

    See Also
    --------
    Array.median
    """
    return array.median(*args, **kwargs)


def percentile(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the qth percentile of the data along the specified axis.

    See Also
    --------
    Array.percentile
    """
    return array.percentile(*args, **kwargs)


# not commutative
def ptp(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Return the range of values (maximum - minimum).

    See Also
    --------
    Array.ptp
    """
    return array.ptp(*args, **kwargs)


def var(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the variance.

    See Also
    --------
    Array.var
    """
    return array.var(*args, **kwargs)


def std(array, *args, **kwargs) -> Union['Array', Scalar]:
    r"""
    Compute the standard deviation.

    See Also
    --------
    Array.std
    """
    return array.std(*args, **kwargs)


_np_op = {
    sum: np.sum,
    prod: np.prod,
    min: np.min,
    max: np.max,
    mean: np.mean,
    ptp: np.ptp,
    var: np.var,
    std: np.std,
    median: np.median,
    percentile: np.percentile,
}
