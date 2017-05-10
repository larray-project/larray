import math
from collections import deque

import larray as la
import numpy as np


def badvalues(a, bad_filter):
    bad_values = a[bad_filter]
    assert bad_values.ndim == 1
    return '\n'.join('%s: %s' % (k, v) for k, v in zip(bad_values.axes[0], bad_values))


def f2str(f, threshold=2):
    """Return string representation of floating point number f. Use scientific
    notation if f would have more than threshold decimal digits, otherwise
    use threshold as precision.
    """
    kind = "e" if f and math.log10(1 / abs(f)) > threshold else "f"
    format_str = "%%.%d%s" % (threshold, kind)
    return format_str % f


def warn_or_raise(what, msg):
    if what == 'raise':
        raise ValueError(msg)
    else:
        print("WARNING: {}".format(msg))


def ipfp(target_sums, a=None, maxiter=1000, threshold=0.5, stepstoabort=10,
         nzvzs='raise', no_convergence='raise', display_progress=False):
    """Apply Iterative Proportional Fitting Procedure (also known as
    bi-proportional fitting in statistics, RAS algorithm in economics) to array
    a, with target_sums as targets.

    Parameters
    ----------
    target_sums : tuple/list of array-like
        target sums to achieve. First element must be the sum to achieve
        along axis 0, the second the sum along axis 1, ...
    a : array-like, optional
        starting values to fit, if not given starts with an array filled
        with 1.
    maxiter : int, optional
        maximum number of iteration, defaults to 1000.
    threshold : float, optional
        threshold below which the result is deemed acceptable, defaults to 0.5.
    stepstoabort : int, optional
        number of consecutive steps with no improvement after which to abort.
        Defaults to 10.
    nzvzs : 'fix', 'warn' or 'raise', optional
        behavior when detecting non zero values where the sum is zero
        'fix': set to zero (silently)
        'warn': set to zero and print a warning
        'raise': raise an exception (default)
    no_convergence : 'ignore', 'warn' or 'raise, optional
        behavior when the algorithm does not seem to converge. This
        condition is triggered both when the maximum number of iteration is
        reached or when the maximum absolute difference between the target and
        the current sums does not improve for `stepstoabort` iterations.
        'ignore': return values computed up to that point (silently)
        'warn': return values computed up to that point and print a warning
        'raise': raise an exception (default)
    display_progress : False, True or 'condensed', optional
        whether or not to display progress. Defaults to False.
        if 'condensed' will display progress using a denser template (using one
        line per iteration).

    Returns
    -------
    LArray

    Examples
    --------
    >>> from larray import *
    >>> from larray.ipfp import ipfp
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
    """
    assert nzvzs in {'fix', 'warn', 'raise'}
    assert no_convergence in {'ignore', 'warn', 'raise'}
    assert isinstance(display_progress, bool) or display_progress == 'condensed'

    target_sums = [la.aslarray(ts) for ts in target_sums]

    n = len(target_sums)
    axes_names = ['axis%d' % i for i in range(n)]
    new_target_sums = []
    for i, ts in enumerate(target_sums):
        ts_axes_names = axes_names[:i] + axes_names[i+1:]
        new_ts = ts.rename({axis: axis.name if axis.name is not None else name
                            for axis, name in zip(ts.axes, ts_axes_names)})
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
        # so, to reconstruct a.axes from target_sum axes, we need to take the
        # first axis of the second target_sum and all axes from the first
        # target_sum:
        first_axis = target_sums[1].axes[0]
        other_axes = target_sums[0].axes
        all_axes = first_axis + other_axes
        a = la.ones(all_axes, dtype=np.float64)
    else:
        # TODO: this should be a builtin op
        if isinstance(a, la.LArray):
            a = a.copy()
        else:
            a = la.aslarray(a)
        # TODO: this should be a builtin op
        a = a.rename({i: name if name is not None else 'axis%d' % i
                      for i, name in enumerate(a.axes.names)})

    # this test should only ever fail if the user passed larray for a and target sums
    for i, axis_target in enumerate(target_sums):
        expected_axes = a.axes - i
        if axis_target.axes != expected_axes:
            raise ValueError("axes of target sum along axis {} ({}) do not match corresponding array "
                             "axes: got {} but expected {}. Are the target sums in the correct order?"
                             .format(i, a.axes[i].name, axis_target.axes, expected_axes))

    axis0_total = target_sums[0].sum()
    for i, axis_target in enumerate(target_sums[1:], start=1):
        axis_total = axis_target.sum()
        if str(axis_total) != str(axis0_total):
            raise ValueError("target sum along %s (axis %d) is different "
                             "than target sum along %s (axis %d): %s vs %s"
                             % (a.axes[i], i,
                                a.axes[0], 0,
                                axis_total, axis0_total))

    negative = a < 0
    if la.any(negative):
        raise ValueError("negative value(s) found:\n%s"
                         % badvalues(a, negative))

    for dim, axis_target in enumerate(target_sums):
        axis_sum = a.sum(axis=dim)
        bad = (axis_sum == 0) & (axis_target != 0)
        if la.any(bad):
            raise ValueError("found all zero values sum along %s (%d) but non "
                             "zero target sum:\n%s"
                             % (a.axes[dim].name, dim,
                                badvalues(axis_target, bad)))

        bad = (axis_sum != 0) & (axis_target == 0)
        if la.any(bad):
            if nzvzs in {'warn', 'raise'}:
                msg = "found non zero values sum along {} ({}) but zero " \
                      "target sum".format(a.axes[dim].name, dim)
                if nzvzs == 'raise':
                    raise ValueError("{}:\n{}"
                                     .format(msg, badvalues(axis_sum, bad)))
                else:
                    print("WARNING: {}, setting them to zero:\n{}"
                          .format(msg, badvalues(axis_sum, bad)))
            a[bad] = 0
            # verify we did fix the problem
            assert not np.any((a.sum(axis=dim) != 0) & (axis_target == 0))

    r = a
    lastdiffs = deque([float('nan')], maxlen=stepstoabort)
    for i in range(maxiter):
        startr = r.copy()
        for dim, axis_target in enumerate(target_sums):
            axis_sum = r.sum(axis=dim)
            factor = axis_target.divnot0(axis_sum)
            r *= factor
        stepcelldiff = abs(r - startr).max()
        max_sum_diff = max(abs(r.sum(axis=dim) - axis_target).max()
                         for dim, axis_target in enumerate(target_sums))
        step_sum_improvement = lastdiffs[-1] - max_sum_diff

        if display_progress:
            if display_progress == "condensed":
                template = "it %d max cell diff %s max diff to target %s (%s)"
            else:
                template = """iteration %d
 * max(abs(prev_cell - cell)): %s
 * max(abs(sum - target_sum)): %s
   \- change since last iteration: %s
"""
            print(template % (i, f2str(stepcelldiff), f2str(max_sum_diff),
                              f2str(step_sum_improvement)))

        if np.all(np.array(lastdiffs) == max_sum_diff):
            if no_convergence in {'warn', 'raise'}:
                warn_or_raise(no_convergence,
                              "does not seem to converge (no improvement "
                              "for %d consecutive steps), stopping here."
                              % stepstoabort)
            return r

        if max_sum_diff < threshold:
            if display_progress:
                print("acceptable max(abs(sum - target_sum)) found at "
                      "iteration {}: {} < threshold ({})"
                      .format(i, f2str(max_sum_diff), threshold))
            return r

        lastdiffs.append(max_sum_diff)

    if no_convergence in {'warn', 'raise'}:
        warn_or_raise(no_convergence,
                      "maximum iteration reached ({})".format(maxiter))
    return r
