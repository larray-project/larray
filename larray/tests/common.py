from __future__ import absolute_import, division, print_function

import os
import numpy as np
from larray import LArray, isnan, aslarray


TESTDATADIR = os.path.dirname(__file__)


def inputpath(relpath):
    """
    Parameters
    ----------
    relpath: str
        path relative to current module

    Returns
    -------
    absolute path to input data file
    """
    return os.path.join(TESTDATADIR, 'data', relpath)


# XXX: maybe we should force value groups to use tuple and families (group of groups to use lists, or vice versa, so
# that we know which is which) or use a class, just for that? group(a, b, c) vs family(group(a), b, c)


def assert_equal_factory(test_func):
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


def assert_larray_equal_factory(test_func, convert=True, check_axes=False):
    def assert_equal(a, b):
        if convert:
            a = aslarray(a)
            b = aslarray(b)
        if check_axes and a.axes != b.axes:
            raise AssertionError("axes differ:\n%s\n\nvs\n\n%s" % (a.axes.info, b.axes.info))
        equal = test_func(a, b)
        if not equal.all():
            notequal = ~equal
            raise AssertionError("\ngot:\n\n%s\n\nexpected:\n\n%s" % (a[notequal], b[notequal]))
    return assert_equal


def assert_nparray_equal_factory(test_func, convert=True, check_shape=False):
    def assert_equal(a, b):
        if convert:
            a = np.asarray(a)
            b = np.asarray(b)
        if check_shape and a.shape != b.shape:
            raise AssertionError("shapes differ: %s != %s" % (a.shape, b.shape))
        equal = test_func(a, b)
        if not equal.all():
            notequal = ~equal
            raise AssertionError("\ngot:\n\n%s\n\nexpected:\n\n%s" % (a[notequal], b[notequal]))
    return assert_equal


def equal(a, b):
    return a == b


def nan_equal(a, b):
    return (a == b) | (isnan(a) & isnan(b))


# numpy.testing.assert_array_equal/assert_equal would work too but it does not (as of numpy 1.10) display specifically
# the non equal items
# TODO: this is defined for backward compatibility only (until we update all tests to use either assert_larray* or
#       assert_nparray*)
assert_array_equal = assert_equal_factory(equal)
assert_array_nan_equal = assert_equal_factory(nan_equal)

assert_larray_equal = assert_larray_equal_factory(equal, check_axes=True)
assert_larray_nan_equal = assert_larray_equal_factory(nan_equal, check_axes=True)

assert_larray_equiv = assert_larray_equal_factory(equal)
assert_larray_nan_equiv = assert_larray_equal_factory(nan_equal)

assert_nparray_equal = assert_nparray_equal_factory(equal, check_shape=True)
assert_nparray_nan_equal = assert_nparray_equal_factory(nan_equal, check_shape=True)

assert_nparray_equiv = assert_nparray_equal_factory(equal)
assert_nparray_nan_equiv = assert_nparray_equal_factory(nan_equal)


def assert_axis_eq(axis1, axis2):
    assert axis1.equals(axis2)
