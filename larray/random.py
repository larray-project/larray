# Portions (part of the docstrings) of this file come from numpy/random/mtrand/mtrand.pyx
# that file (and thus those portions) are licensed under the terms below

# mtrand.pyx -- A Pyrex wrapper of Jean-Sebastien Roy's RandomKit
#
# Copyright 2005 Robert Kern (robert.kern@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np

from larray.core.axis import Axis, AxisCollection       # noqa: F401
from larray.core.array import Array, asarray
from larray.core.array import raw_broadcastable
import larray as la                                     # noqa: F401


__all__ = ['randint', 'normal', 'uniform', 'permutation', 'choice']


def generic_random(np_func, args, min_axes, meta) -> Array:
    args, res_axes = raw_broadcastable(args, min_axes=min_axes)
    res_data = np_func(*args, size=res_axes.shape)
    return Array(res_data, res_axes, meta=meta)


# We choose to place the axes argument in place of the numpy size argument, instead of having axes as the first
# argument, because that would make it ugly for scalars. As a consequence, it is slightly ugly when arguments
# before axes are not required.
def randint(low, high=None, axes=None, dtype='l', meta=None) -> Array:
    r"""Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval
    [`low`, `high`). If `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless ``high=None``, in which case this parameter
        is one above the *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior
        if ``high=None``).
    axes : int, tuple of int, str, Axis or tuple/list/AxisCollection of Axis, optional
        Axes (or shape) of the resulting array. If ``axes`` is None (the default), a single value is returned.
        Otherwise, if the resulting axes have a shape of, e.g., ``(m, n, k)``, then ``m * n * k`` samples are drawn.
    dtype : data-type, optional
        Desired dtype of the result. All dtypes are determined by their name, i.e., 'int64', 'int', etc, so byteorder
        is not available and a specific precision may have different C types depending on the platform.
        The default value is 'np.int'.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array

    Examples
    --------
    Generate a single int between 0 and 9, inclusive:

    >>> la.random.randint(10)                                   # doctest: +SKIP
    6

    Generate an array of 10 ints between 1 and 5, inclusive:

    >>> la.random.randint(1, 6, 10)                             # doctest: +SKIP
    {0}*  0  1  2  3  4  5  6  7  8  9
          1  1  5  1  5  4  3  4  2  1

    Generate a 2 x 3 array of ints between 0 and 4, inclusive:

    >>> la.random.randint(5, axes=(2, 3))                       # doctest: +SKIP
    {0}*\{1}*  0  1  2
            0  4  4  1
            1  1  2  2
    >>> la.random.randint(5, axes='a=a0,a1;b=b0..b2')           # doctest: +SKIP
    a\b  b0  b1  b2
     a0   0   3   1
     a1   4   0   1

    With varying low and high (each depending on a different axis)

    >>> low = la.sequence('a=a0,a1')
    >>> low
    a  a0  a1
        0   1
    >>> high = la.sequence('b=b0..b2', initial=3)
    >>> high
    b  b0  b1  b2
        3   4   5

    In other words, we want to generate values between low and high (high included) for each cell. Let's
    note that low..high:

    a\b    b0    b1    b2
     a0  0..2  0..3  0..4
     a1  1..2  1..3  1..4

    >>> la.random.randint(low, high)                            # doctest: +SKIP
    a\b  b0  b1  b2
     a0   0   2   2
     a1   2   3   4
    """
    return generic_random(np.random.randint, (low, high), axes, meta)


def normal(loc=0.0, scale=1.0, axes=None, meta=None) -> Array:
    r"""
    Draw random samples from a normal (Gaussian) distribution.

    Its probability density function is often called the bell curve because of its characteristic shape (see the
    example below)

    Parameters
    ----------
    loc : float or array_like of floats
        Mean ("centre") of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or "width") of the distribution.
    axes : int, tuple of int, str, Axis or tuple/list/AxisCollection of Axis, optional
        Minimum axes the resulting array must have. Defaults to None. The resulting array axes will be the union of
        those mentioned in ``axes`` and those of ``loc`` and ``scale``. If ``loc`` and ``scale`` are scalars and
        ``axes`` is None, a single value is returned. Otherwise, if the resulting axes have a shape of, e.g.,
        ``(m, n, k)``, then ``m * n * k`` samples are drawn.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array or scalar
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    The normal distributions occurs often in nature.  For example, it describes the commonly occurring distribution of
    samples influenced by a large number of tiny, random disturbances, each with its own unique distribution [2]_.

    The probability density function for the Gaussian distribution, first derived by De Moivre and 200 years later by
    both Gauss and Laplace independently [2]_, is

    .. math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                     e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },

    where :math:`\mu` is the mean and :math:`\sigma` the standard deviation. The square of the standard deviation,
    :math:`\sigma^2`, is called the variance.

    The function has its peak at the mean, and its "spread" increases with the standard deviation (the function reaches
    0.607 times its maximum at :math:`x + \sigma` and :math:`x - \sigma` [2]_).  This implies that
    `la.random.normal` is more likely to return samples lying close to the mean, rather than those far away.

    References
    ----------
    .. [1] Wikipedia, "Normal distribution",
           http://en.wikipedia.org/wiki/Normal_distribution
    .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001, pp. 51, 51, 125.

    Examples
    --------
    Generate a 2 x 3 array with numbers drawn from the distribution:

    >>> la.random.normal(0, 1, axes=(2, 3))                                         # doctest: +SKIP
    {0}*\{1}*                   0                     1                   2
            0  0.3564325741877542    0.8944149721039006  1.7206904920773107
            1  0.6904447654719367  -0.09395966570976753   0.185136309092257

    With named and labelled axes

    >>> la.random.normal(0, 1, axes='a=a0,a1;b=b0..b2')                             # doctest: +SKIP
    a\b                  b0                   b1                   b2
     a0  2.3096106652701827  -0.4269082412118316  -1.0862791566867225
     a1  0.8598817639620348   -2.386411240813283  0.10116503197279443

    With varying loc and scale (each depending on a different axis)

    >>> mu = la.sequence('a=a0,a1', initial=5, inc=5)
    >>> mu
    a  a0  a1
        5  10
    >>> sigma = la.sequence('b=b0..b2', initial=1)
    >>> sigma
    b  b0  b1  b2
        1   2   3
    >>> la.random.normal(mu, sigma)                                                 # doctest: +SKIP
    a\b                  b0                  b1                  b2
     a0   5.939369790854615  2.5043856460438403    8.33560126941519
     a1  10.759526714752091  10.093213549397403  11.705881778249683

    Draw 1000 samples from the distribution:

    >>> mu, sigma = 0, 0.1  # mean and standard deviation
    >>> sample = la.random.normal(mu, sigma, 1000)

    Verify the mean and the variance:

    >>> abs(mu - la.mean(sample)) < 0.01
    True
    >>> abs(sigma - la.std(sample, ddof=1)) < 0.01
    True

    Display the histogram of the samples, along with the probability density function:

    >>> import matplotlib.pyplot as plt                                         # doctest: +SKIP
    >>> count, bins, ignored = plt.hist(sample, 30, normed=True)                # doctest: +SKIP
    >>> pdf = 1 / (sigma * la.sqrt(2 * la.pi)) \
    ...       * la.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))                   # doctest: +SKIP
    >>> _ = plt.plot(bins, pdf, linewidth=2, color='r')                         # doctest: +SKIP
    >>> plt.show()                                                              # doctest: +SKIP
    """
    return generic_random(np.random.normal, (loc, scale), axes, meta)


def uniform(low=0.0, high=1.0, axes=None, meta=None) -> Array:
    r"""
    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval ``[low, high)`` (includes low, but excludes high).
    In other words, any value within the given interval is equally likely to be drawn by `uniform`.

    Parameters
    ----------
    low : float or array_like of floats, optional
        Lower boundary of the output interval.  All values generated will be greater than or equal to low.
        Defaults to 0.0.
    high : float or array_like of floats, optional
        Upper boundary of the output interval.  All values generated will be less than high.
        Defaults to 1.0.
    axes : int, tuple of int, str, Axis or tuple/list/AxisCollection of Axis, optional
        Minimum axes the resulting array must have. Defaults to None. The resulting array axes will be the union of
        those mentioned in ``axes`` and those of ``low`` and ``high``. If ``low`` and ``high`` are scalars and
        ``axes`` is None, a single value is returned. Otherwise, if the resulting axes have a shape of, e.g.,
        ``(m, n, k)``, then ``m * n * k`` samples are drawn.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array or scalar
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    randint : Discrete uniform distribution, yielding integers.

    Notes
    -----
    The probability density function of the uniform distribution is

    .. math:: p(x) = \frac{1}{b - a}

    anywhere within the interval ``[a, b)``, and zero elsewhere.

    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined and may eventually raise an error, i.e. do not rely on
    this function to behave when passed arguments satisfying that inequality condition.

    Examples
    --------
    Generate a single sample from the distribution:

    >>> la.random.uniform()                                                          # doctest: +SKIP
    0.4616049008844396

    Generate a 2 x 3 array with numbers drawn from the distribution:

    >>> la.random.uniform(0, 5, axes=(2, 3))                                         # doctest: +SKIP
    {0}*\{1}*                   0                  1                  2
            0  3.4951791043804192  3.888533056628081  4.347461073315136
            1   2.146211610940853  0.509146487437932  2.790852715735223

    With named and labelled axes

    >>> la.random.uniform(1, 2, axes='a=a0,a1;b=b0..b2')                             # doctest: +SKIP
    a\b                  b0                  b1                  b2
     a0  1.4167729850467825  1.6953091052066793  1.2321770607672526
     a1  1.4386221912579358  1.8480607144284926  1.1726213637670433

    With varying low and high (each depending on a different axis)

    >>> low = la.sequence('a=a0,a1')
    >>> low
    a  a0  a1
        0   1
    >>> high = la.sequence('b=b0..b2', initial=1, inc=0.5)
    >>> high
    b   b0   b1   b2
       1.0  1.5  2.0
    >>> la.random.uniform(low, high)                                                 # doctest: +SKIP
    a\b                   b0                  b1                  b2
     a0  0.44608671494167573   0.948315996350121    1.74189664009661
     a1                  1.0  1.1099944474264194  1.1362792569316835

    Draw 1000 samples from the distribution:

    >>> s = la.random.uniform(-1, 0, 1000)

    All values are within the given interval:

    >>> la.all(s >= -1)
    True
    >>> la.all(s < 0)
    True

    Display the histogram of the samples, along with the probability density function:

    >>> import matplotlib.pyplot as plt                                     # doctest: +SKIP
    >>> count, bins, ignored = plt.hist(s, 15, normed=True)                 # doctest: +SKIP
    >>> _ = plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')      # doctest: +SKIP
    >>> plt.show()                                                          # doctest: +SKIP
    """
    return generic_random(np.random.uniform, (low, high), axes, meta)


def permutation(x, axis=0) -> Array:
    r"""
    Randomly permute a sequence along an axis, or return a permuted range.

    Parameters
    ----------
    x : int or array_like
        If `x` is an integer, randomly permute ``sequence(x)``.
        If `x` is an array, returns a randomly shuffled copy.
    axis : int, str or Axis, optional
        Axis along which to permute. Defaults to the first axis.

    Returns
    -------
    Array
        Permuted sequence or array range.

    Examples
    --------
    >>> la.random.permutation(10)                               # doctest: +SKIP
    {0}*  0  1  2  3  4  5  6  7  8  9
          6  8  0  9  4  7  1  5  3  2
    >>> la.random.permutation([1, 4, 9, 12, 15])                # doctest: +SKIP
    {0}*  0   1   2  3  4
          1  15  12  9  4
    >>> la.random.permutation(la.ndtest(5))                     # doctest: +SKIP
    a  a3  a1  a2  a4  a0
        3   1   2   4   0
    >>> arr = la.ndtest((3, 3))                                 # doctest: +SKIP
    >>> la.random.permutation(arr)                              # doctest: +SKIP
    a\b  b0  b1  b2
     a1   3   4   5
     a2   6   7   8
     a0   0   1   2
    >>> la.random.permutation(arr, axis='b')                    # doctest: +SKIP
    a\b  b1  b2  b0
     a0   1   2   0
     a1   4   5   3
     a2   7   8   6
    """
    if isinstance(x, (int, np.integer)):
        return Array(np.random.permutation(x))
    else:
        x = asarray(x)
        axis = x.axes[axis]
        g = axis.i[np.random.permutation(len(axis))]
        return x[g]


def choice(choices=None, axes=None, replace=True, p=None, meta=None) -> Array:
    r"""
    Generate a random sample from given choices.

    Parameters
    ----------
    choices : 1-D array-like or int, optional
        Values to choose from.
        If an array, a random sample is generated from its elements.
        If an int n, the random sample is generated as if choices was la.sequence(n)
        If p is a 1-D Array, choices are taken from its axis.
    axes : int, tuple of int, str, Axis or tuple/list/AxisCollection of Axis, optional
        Axes (or shape) of the resulting array. If ``axes`` is None (the default), a single value is returned.
        Otherwise, if the resulting axes have a shape of, e.g., ``(m, n, k)``, then ``m * n * k`` samples are drawn.
    replace : boolean, optional
        Whether the sample is with or without replacement.
    p : array-like, optional
        The probabilities associated with each entry in choices.
        If p is a 1-D Array, choices are taken from its axis labels. If p is an N-D Array, each cell represents the
        probability that the combination of labels will occur.
        If not given the sample assumes a uniform distribution over all entries in choices.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Array or scalar
        The generated random samples with given ``axes`` (or shape).

    Raises
    ------
    ValueError
        If choices is an int and less than zero, if choices or p are not 1-dimensional,
        if choices is an array-like of size 0, if p is not a vector of probabilities,
        if choices and p have different lengths, or
        if replace=False and the sample size is greater than the population size.

    See Also
    --------
    randint, permutation

    Examples
    --------
    Generate one random value out of given choices (each choice has the same probability of occurring):

    >>> la.random.choice(['hello', 'world', '!'])                                       # doctest: +SKIP
    hello

    With given probabilities:

    >>> la.random.choice(['hello', 'world', '!'], p=[0.1, 0.8, 0.1])                    # doctest: +SKIP
    world

    Generate a 2 x 3 array with given axes and values drawn from the given choices using given probabilities:

    >>> la.random.choice([5, 10, 15], p=[0.3, 0.5, 0.2], axes='a=a0,a1;b=b0..b2')       # doctest: +SKIP
    a\b  b0  b1  b2
     a0  15  10  10
     a1  10   5  10

    Same as above with labels and probabilities given as a one dimensional Array

    >>> proba = Array([0.3, 0.5, 0.2], Axis([5, 10, 15], 'outcome'))                   # doctest: +SKIP
    >>> proba                                                                           # doctest: +SKIP
    outcome    5   10   15
             0.3  0.5  0.2
    >>> choice(p=proba, axes='a=a0,a1;b=b0..b2')                                        # doctest: +SKIP
    a\b  b0  b1  b2
     a0  10  15   5
     a1  10   5  10

    Generate a uniform random sample of size 3 from la.sequence(5):

    >>> la.random.choice(5, 3)                                                          # doctest: +SKIP
    {0}*  0  1  2
          3  2  0
    >>> # This is equivalent to la.random.randint(0, 5, 3)

    Generate a non-uniform random sample of size 3 from the given choices without replacement:

    >>> la.random.choice(['hello', 'world', '!'], 3, replace=False, p=[0.1, 0.6, 0.3])  # doctest: +SKIP
    {0}*      0  1      2
          world  !  hello

    Using an N-dimensional array as probabilities:

    >>> proba = Array([[0.15, 0.25, 0.10],
    ...                 [0.20, 0.10, 0.20]], 'a=a0,a1;b=b0..b2')                        # doctest: +SKIP
    >>> proba                                                                           # doctest: +SKIP
    a\b    b0    b1   b2
     a0  0.15  0.25  0.1
     a1   0.2   0.1  0.2
    >>> choice(p=proba, axes='draw=d0..d5')                                             # doctest: +SKIP
    draw\axis   a   b
           d0  a1  b2
           d1  a1  b1
           d2  a0  b1
           d3  a0  b0
           d4  a1  b2
           d5  a0  b1
    """
    axes = AxisCollection(axes)
    if isinstance(p, Array):
        if choices is not None:
            raise ValueError("choices argument cannot be used when p argument is an Array")

        if p.ndim > 1:
            flat_p = p.data.reshape(-1)
            flat_indices = choice(p.size, axes=axes, replace=replace, p=flat_p)
            return p.axes._flat_lookup(flat_indices)
        else:
            choices = p.axes[0].labels
            p = p.data
    if choices is None:
        raise ValueError("choices argument must be provided unless p is an Array")
    return Array(np.random.choice(choices, axes.shape, replace, p), axes, meta=meta)
