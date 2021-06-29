import os
import re
import sys
import inspect
from contextlib import contextmanager

import pytest
import datetime as dt
import numpy as np
import pandas as pd
try:
    import xlwings as xw
except ImportError:
    xw = None
try:
    import tables
except ImportError:
    tables = None
try:
    import openpyxl
except ImportError:
    openpyxl = None
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None

from larray import Array, isnan, asarray, Metadata


SKIP_EXCEL_TESTS = False
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
        if isinstance(a, Array) and isinstance(b, Array) and a.axes != b.axes:
            raise AssertionError(f"axes differ:\n{a.axes.info}\n\nvs\n\n{b.axes.info}")
        if not isinstance(a, (np.ndarray, Array)):
            a = np.asarray(a)
        if not isinstance(b, (np.ndarray, Array)):
            b = np.asarray(b)
        if a.shape != b.shape:
            raise AssertionError(f"shapes differ: {a.shape} != {b.shape}")
        equal = test_func(a, b)
        if not equal.all():
            # XXX: for some reason ndarray[bool_larray] does not work as we would like, so we cannot do b[~equal]
            #      directly. I should at least understand why this happens and fix this if possible.
            notequal = np.asarray(~equal)
            raise AssertionError(f"\ngot:\n\n{a[notequal]}\n\nexpected:\n\n{b[notequal]}")
    return assert_equal


def assert_larray_equal_factory(test_func, convert=True, check_axes=False):
    def assert_equal(a, b):
        if convert:
            a = asarray(a)
            b = asarray(b)
        if check_axes and a.axes != b.axes:
            raise AssertionError(f"axes differ:\n{a.axes.info}\n\nvs\n\n{b.axes.info}")
        equal = test_func(a, b)
        if not equal.all():
            notequal = ~equal
            raise AssertionError(f"\ngot:\n\n{a[notequal]}\n\nexpected:\n\n{b[notequal]}")
    return assert_equal


def assert_nparray_equal_factory(test_func, convert=True, check_shape=False):
    def assert_equal(a, b):
        if convert:
            a = np.asarray(a)
            b = np.asarray(b)
        if check_shape and a.shape != b.shape:
            raise AssertionError(f"shapes differ: {a.shape} != {b.shape}")
        equal = test_func(a, b)
        if not equal.all():
            notequal = ~equal
            raise AssertionError(f"\ngot:\n\n{a[notequal]}\n\nexpected:\n\n{b[notequal]}")
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


def tmp_path(tmpdir, fname):
    return os.path.join(tmpdir.strpath, fname)


@pytest.fixture
def meta():
    title = 'test array'
    description = 'Array used for testing'
    author = 'John Cleese'
    location = 'Ministry of Silly Walks'
    office_number = 42
    score = 9.70
    date = pd.Timestamp(dt.datetime(1970, 3, 21))
    return Metadata([('title', title), ('description', description), ('author', author),
                     ('location', location), ('office_number', office_number),
                     ('score', score), ('date', date)])


needs_pytables = pytest.mark.skipif(tables is None, reason="pytables is required for this test")
needs_xlwings = pytest.mark.skipif(SKIP_EXCEL_TESTS or xw is None, reason="xlwings is required for this test")
needs_openpyxl = pytest.mark.skipif(SKIP_EXCEL_TESTS or openpyxl is None, reason="openpyxl is required for this test")
needs_xlsxwriter = pytest.mark.skipif(SKIP_EXCEL_TESTS or xlsxwriter is None,
                                      reason="xlsxwriter is required for this test")

needs_python37 = pytest.mark.skipif(sys.version_info < (3, 7), reason="Python 3.7 is required for this test")


@contextmanager
def must_warn(warn_cls=None, msg=None, match=None, check_file=True, check_num=True):
    if msg is not None and match is not None:
        raise ValueError("bad test: can't use both msg and match arguments")
    elif msg is not None:
        match = re.escape(msg)

    try:
        with pytest.warns(warn_cls, match=match) as caught_warnings:
            yield caught_warnings
    finally:
        if check_num:
            assert len(caught_warnings) == 1
        if check_file:
            caller_path = inspect.stack()[2].filename
            warning_path = caught_warnings[0].filename
            assert warning_path == caller_path, \
                f"{warning_path} != {caller_path}"
