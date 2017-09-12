import math
from collections import deque

from larray.core.array import LArray, aslarray, ones, any
import numpy as np

__all__ = ['ipfp']


def badvalues(a, bad_filter):
    bad_values = a[bad_filter]
    assert bad_values.ndim == 1
    return '\n'.join('{}: {}'.format(k, v) for k, v in zip(bad_values.axes[0], bad_values))


def f2str(f, threshold=2):
    """Return string representation of floating point number f.
    Use scientific notation if f would have more than threshold decimal digits, otherwise use threshold as precision.

    Parameters
    ----------
    f : float
        Number to represent.
    threshold : int, optional
        Precision (number of decimal digits displayed). If the number needs more digits, scientific notation will be
        used.

    Examples
    --------
    >>> f2str(55.1)
    '55.10'
    >>> f2str(1.234)
    '1.23'
    >>> f2str(0.002)
    '2.00e-03'
    """
    kind = "e" if f and math.log10(1 / abs(f)) > threshold else "f"
    return "{:.{}{}}".format(f, threshold, kind)


def warn_or_raise(what, msg):
    if what == 'raise':
        raise ValueError(msg)
    else:
        print("WARNING: {}".format(msg))


def ipfp(target_sums, a=None, axes=None, maxiter=1000, threshold=0.5, stepstoabort=10, nzvzs='raise',
         no_convergence='raise', display_progress=False):
    """Apply Iterative Proportional Fitting Procedure (also known as bi-proportional fitting in statistics,
    RAS algorithm in economics) to array a, with target_sums as targets.

    Parameters
    ----------
    target_sums : tuple/list of array-like
        Target sums to achieve.
        First element must be the sum to achieve along axis 0, the second the sum along axis 1, ...
    a : array-like, optional
        Starting values to fit, if not given starts with an array filled with 1.
    axes : list/tuple of axes, optional
        Axes on which the fitting procedure should be applied. Defaults to all axes.
    maxiter : int, optional
        Maximum number of iteration, defaults to 1000.
    threshold : float, optional
        Threshold below which the result is deemed acceptable, defaults to 0.5.
    stepstoabort : int, optional
        Number of consecutive steps with no improvement after which to abort. Defaults to 10.
    nzvzs : 'fix', 'warn' or 'raise', optional
        Behavior when detecting non zero values where the sum is zero
        'fix': set to zero (silently)
        'warn': set to zero and print a warning
        'raise': raise an exception (default)
    no_convergence : 'ignore', 'warn' or 'raise, optional
        Behavior when the algorithm does not seem to converge. This condition is triggered both when the maximum number
        of iteration is reached or when the maximum absolute difference between the target and the current sums does
        not improve for `stepstoabort` iterations.
        'ignore': return values computed up to that point (silently)
        'warn': return values computed up to that point and print a warning
        'raise': raise an exception (default)
    display_progress : False, True or 'condensed', optional
        Whether or not to display progress. Defaults to False.
        If 'condensed' will display progress using a denser template (using one line per iteration).

    Returns
    -------
    LArray

    Examples
    --------
    >>> from larray import *
    >>> a = Axis('a=a0,a1')
    >>> b = Axis('b=b0,b1')
    >>> initial = LArray([[2, 1], [1, 2]], [a, b])
    >>> initial
    a\\b  b0  b1
     a0   2   1
     a1   1   2
    >>> target_sum_along_a = LArray([2, 1], b)
    >>> target_sum_along_a
    b  b0  b1
        2   1
    >>> target_sum_along_b = LArray([1, 2], a)
    >>> target_sum_along_b
    a  a0  a1
        1   2
    >>> result = ipfp([target_sum_along_a, target_sum_along_b], initial, threshold=0.01)
    >>> # round result so that its display is nicer
    ... round(result, 2)
    a\\b    b0    b1
     a0  0.85  0.15
     a1  1.15  0.85

    Now let us assume you have a 3D array like this:

    >>> year = Axis('year=2014..2016')
    >>> initial = ndrange([a, b, year])
    >>> initial
     a  b\year  2014  2015  2016
    a0      b0     0     1     2
    a0      b1     3     4     5
    a1      b0     6     7     8
    a1      b1     9    10    11

    and some targets for each year:

    >>> btargets = initial.sum(x.a) + 1
    >>> btargets
    b\year  2014  2015  2016
        b0     7     9    11
        b1    13    15    17
    >>> atargets = initial.sum(x.b) + 1
    >>> atargets
    a\year  2014  2015  2016
        a0     4     6     8
        a1    16    18    20

    You want to apply a 2D fitting procedure for each value of that year axis. You could call ipfp within a loop on
    the year axis, but you can also apply the procedure for all years at once by using the axes argument. This is
    *much* faster than an explicit loop.

    >>> result = ipfp([btargets, atargets], initial, axes=(x.a, x.b))
    """
    assert nzvzs in {'fix', 'warn', 'raise'}
    assert no_convergence in {'ignore', 'warn', 'raise'}
    assert isinstance(display_progress, bool) or display_progress == 'condensed'

    target_sums = [aslarray(ts) for ts in target_sums]

    n = len(target_sums)

    if axes is None:
        axes = list(range(n))

    def has_anonymous_axes(a):
        return any(axis.name is None for axis in a.axes)

    if any(has_anonymous_axes(ts) for ts in target_sums):
        if any(not isinstance(axis, int) for axis in axes):
            raise ValueError("ipfp does not support target sums with anonymous axes when using the axes argument with"
                             "non-integer (positional) axis references")

        names_for_missing_axes = ['axis{}'.format(i) for i in axes]
        new_target_sums = []
        for i, target_sum in zip(axes, target_sums):
            ts_axes_names = names_for_missing_axes[:i] + names_for_missing_axes[i + 1:]
            new_ts = target_sum.rename({axis: name
                                        for axis, name in zip(target_sum.axes, ts_axes_names)
                                        if axis.name is None})
            new_target_sums.append(new_ts)
        target_sums = new_target_sums

    if a is None:
        # reconstruct all axes from target_sums axes
        # assuming shape is the total shape of a
        # >>> shape = (3, 4, 5)
        # target_sums axes would be:
        # >>> shapes = [shape[:i] + shape[i+1:] for i in range(len(shape))]
        # >>> shapes
        # [(4, 5), (3, 5), (3, 4)]
        # >>> (shapes[1][0],) + shapes[0]
        # (3, 4, 5)
        # so, to reconstruct a.axes from target_sum axes, we need to take the first axis of the second target_sum and
        # all axes from the first target_sum:
        first_axis = target_sums[1].axes[0]
        other_axes = target_sums[0].axes
        all_axes = first_axis + other_axes
        a = ones(all_axes, dtype=np.float64)
    else:
        # TODO: only make a copy if there are actually any bad values, but I am unsure we should make a copy at all.
        # Either way, this should be documented.
        if nzvzs in {'warn', 'fix'} and isinstance(a, LArray):
            a = a.copy()
        else:
            a = aslarray(a)
        # TODO: this should be a builtin op
        a = a.rename({i: name if name is not None else 'axis{}'.format(i)
                      for i, name in enumerate(a.axes.names)})

    axes = a.axes[axes]

    # this test should only ever fail if the user passed larray for a and target sums
    for axis, axis_target_sum in zip(axes, target_sums):
        expected_axes = a.axes - axis
        if axis_target_sum.axes != expected_axes:
            raise ValueError("axes of target sum along {} (axis {}) do not match corresponding array axes: "
                             "got {} but expected {}. Are the target sums in the correct order?"
                             .format(axis.name, a.axes.index(axis), axis_target_sum.axes, expected_axes))

    axis0_total = target_sums[0].sum()
    for axis, axis_target_sum in zip(axes[1:], target_sums[1:]):
        axis_total = axis_target_sum.sum()
        if str(axis_total) != str(axis0_total):
            raise ValueError("target sum along {} (axis {}) is different than target sum along {} (axis {}): {} vs {}"
                             .format(axis, a.axes.index(axis), axes[0], a.axes.index(axes[0]), axis_total, axis0_total))

    negative = a < 0
    if any(negative):
        raise ValueError("negative value(s) found:\n{}".format(badvalues(a, negative)))

    for axis, axis_target_sum in zip(axes, target_sums):
        axis_idx = a.axes.index(axis)
        axis_sum = a.sum(axis)
        bad = (axis_sum == 0) & (axis_target_sum != 0)
        if any(bad):
            raise ValueError("found all zero values sum along {} (axis {}) but non zero target sum:\n{}"
                             .format(axis.name, axis_idx, badvalues(axis_target_sum, bad)))

        bad = (axis_sum != 0) & (axis_target_sum == 0)
        if any(bad):
            if nzvzs in {'warn', 'raise'}:
                msg = "found Non Zero Values but Zero target Sum (nzvzs) along {} (axis {})".format(axis.name, axis_idx)
                if nzvzs == 'raise':
                    raise ValueError("{}, use nzvzs='warn' or 'fix' to set them to zero automatically:\n{}"
                                     .format(msg, badvalues(axis_sum, bad)))
                else:
                    print("WARNING: {}, setting them to zero:\n{}".format(msg, badvalues(axis_sum, bad)))

            a[bad] = 0
            # verify we did fix the problem
            assert not any((a.sum(axis) != 0) & (axis_target_sum == 0))

    r = a
    lastdiffs = deque([float('nan')], maxlen=stepstoabort)

    # Here is the nice version of the algorithm

    # for i in range(maxiter):
    #     startr = r
    #     for axis, axis_target in zip(axes, target_sums):
    #         r = r * axis_target.divnot0(r.sum(axis))
    #     max_sum_diff = max(abs(r.sum(axis) - axis_target).max()
    #                        for axis, axis_target in zip(axes, target_sums))
    #     step_sum_improvement = ...

    # Here is the ugly optimized version which avoids computing the sum for the first axis twice per iteration
    # (saves ~10-15% running time).
    axis0_sum = r.sum(axes[0])
    for i in range(maxiter):
        startr = r
        r = r * target_sums[0].divnot0(axis0_sum)
        for axis, axis_target in zip(axes[1:], target_sums[1:]):
            r = r * axis_target.divnot0(r.sum(axis))

        axes_sum = [r.sum(axis) for axis in axes]
        max_sum_diff = max(abs(axis_sum - axis_target).max()
                           for axis_sum, axis_target in zip(axes_sum, target_sums))
        axis0_sum = axes_sum[0]

        step_sum_improvement = lastdiffs[-1] - max_sum_diff
        stepcelldiff = abs(r - startr).max()

        if display_progress:
            if display_progress == "condensed":
                template = "it {} max cell diff {} max diff to target {} ({})"
            else:
                template = """iteration {}
 * max(abs(prev_cell - cell)): {}
 * max(abs(sum - target_sum)): {}
   \- change since last iteration: {}
"""
            print(template.format(i, f2str(stepcelldiff), f2str(max_sum_diff), f2str(step_sum_improvement)))

        if np.all(np.array(lastdiffs) == max_sum_diff):
            if no_convergence in {'warn', 'raise'}:
                warn_or_raise(no_convergence, "does not seem to converge (no improvement for {} consecutive steps), "
                                              "stopping here.".format(stepstoabort))
            return r

        if max_sum_diff < threshold:
            if display_progress:
                print("acceptable max(abs(sum - target_sum)) found at iteration {}: {} < threshold ({})"
                      .format(i, f2str(max_sum_diff), threshold))
            return r

        lastdiffs.append(max_sum_diff)

    if no_convergence in {'warn', 'raise'}:
        warn_or_raise(no_convergence, "maximum iteration reached ({})".format(maxiter))
    return r
