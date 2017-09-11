from __future__ import absolute_import, division, print_function

import os
import numpy as np
from larray import LArray


TESTDATADIR = os.path.dirname(__file__)


def abspath(relpath):
    """
    :param relpath: path relative to current module
    :return: absolute path
    """
    return os.path.join(TESTDATADIR, relpath)

# XXX: maybe we should force value groups to use tuple and families (group of groups to use lists, or vice versa, so
# that we know which is which) or use a class, just for that? group(a, b, c) vs family(group(a), b, c)


def assert_equal_factory(test_func, check_shape=True, check_axes=True):
    def assert_equal(a, b):
        if isinstance(a, LArray) and isinstance(b, LArray) and a.axes != b.axes:
            raise AssertionError("axes differ:\n%s\n\nvs\n\n%s" % (a.axes.info, b.axes.info))
        if not isinstance(a, (np.ndarray, LArray)):
            a = np.asarray(a)
        if not isinstance(b, (np.ndarray, LArray)):
            b = np.asarray(b)
        if a.shape != b.shape:
            raise AssertionError("shapes differ: %s != %s" % (a.shape, b.shape))
        equal = test_func(a, b)
        if not equal.all():
            # XXX: for some reason ndarray[bool_larray] does not work as we would like, so we cannot do b[~equal]
            #      directly. I should at least understand why this happens and fix this if possible.
            notequal = np.asarray(~equal)
            raise AssertionError("\ngot:\n\n%s\n\nexpected:\n\n%s" % (a[notequal], b[notequal]))
    return assert_equal


def equal(a, b):
    return a == b


def nan_equal(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


# numpy.testing.assert_array_equal/assert_equal would work too but it does not (as of numpy 1.10) display specifically
# the non equal items
assert_array_equal = assert_equal_factory(equal)
assert_array_nan_equal = assert_equal_factory(nan_equal)
