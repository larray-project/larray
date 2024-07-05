import re
import inspect
from contextlib import contextmanager
from pathlib import Path
import datetime as dt

import pytest
import numpy as np
from numpy.lib import NumpyVersion
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


NUMPY2 = NumpyVersion(np.__version__) >= '2.0.0'
SKIP_EXCEL_TESTS = False
TESTDATADIR = Path(__file__).parent


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
    return (TESTDATADIR / 'data' / relpath).absolute()


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
        else:
            assert isinstance(a, Array) and isinstance(b, Array)
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
        else:
            assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
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

assert_larray_equal = assert_larray_equal_factory(equal, convert=False, check_axes=True)
assert_larray_nan_equal = assert_larray_equal_factory(nan_equal, convert=False, check_axes=True)

assert_larray_equiv = assert_larray_equal_factory(equal)
assert_larray_nan_equiv = assert_larray_equal_factory(nan_equal)

assert_nparray_equal = assert_nparray_equal_factory(equal, convert=False, check_shape=True)
assert_nparray_nan_equal = assert_nparray_equal_factory(nan_equal, convert=False, check_shape=True)

assert_nparray_equiv = assert_nparray_equal_factory(equal)
assert_nparray_nan_equiv = assert_nparray_equal_factory(nan_equal)


def assert_axis_eq(axis1, axis2):
    assert axis1.equals(axis2)


@pytest.fixture
def meta():
    return Metadata(title='test array', description='Array used for testing', author='John Cleese',
                    location='Ministry of Silly Walks', office_number=42,
                    score=9.70, date=pd.Timestamp(dt.datetime(1970, 3, 21)))


needs_pytables = pytest.mark.skipif(tables is None, reason="pytables is required for this test")
needs_xlwings = pytest.mark.skipif(SKIP_EXCEL_TESTS or xw is None, reason="xlwings is required for this test")
needs_openpyxl = pytest.mark.skipif(SKIP_EXCEL_TESTS or openpyxl is None, reason="openpyxl is required for this test")
needs_xlsxwriter = pytest.mark.skipif(SKIP_EXCEL_TESTS or xlsxwriter is None,
                                      reason="xlsxwriter is required for this test")


@contextmanager
def must_warn(warn_cls=None, msg=None, match=None, check_file=True, num_expected=1):
    if num_expected == 0:
        yield []
    else:
        if msg is not None and match is not None:
            raise ValueError("BAD TEST: can't use both msg and match arguments")
        elif msg is None and match is None:
            raise ValueError("BAD TEST: not checking the warning message")
        elif isinstance(msg, (tuple, list)):
            if num_expected != 1:
                raise ValueError("BAD TEST: cannot use a tuple/list msg and num_expected")
            match = None
            num_expected = len(msg)
        elif msg is not None:
            match = '^' + re.escape(msg) + '$'

        try:
            with pytest.warns(warn_cls, match=match) as caught_warnings:
                # yield to the tested code
                yield caught_warnings
        # this code executes after the tested code has finished (whether or not it raised an exception)
        finally:
            if isinstance(msg, (tuple, list)):
                caught_messages = [str(caught_w.message) for caught_w in caught_warnings]
                expected_messages = list(msg)
                assert caught_messages == expected_messages, (f"Caught messages:\n{caught_messages}\n"
                                                              f"different from expected:\n{msg}")

            elif num_expected is not None:
                # pytest.warns only checks there is at least *one* warning with the correct
                # class and message
                num_caught = len(caught_warnings)
                caught_messages = [str(caught_w.message) for caught_w in caught_warnings]
                pattern = re.compile(match)
                messages_matching_pattern = [msg for msg in caught_messages if pattern.match(msg)]
                messages_not_matching_pattern = [msg for msg in caught_messages if not pattern.match(msg)]
                num_matching_msgs = len(messages_matching_pattern)
                num_unexpected_msgs = len(messages_not_matching_pattern)
                assert num_matching_msgs + num_unexpected_msgs == num_caught, "problem in testing framework"
                if num_unexpected_msgs and num_matching_msgs == num_expected:
                    assert_msg = (f"caught {num_expected} matching warning(s) *as expected* but also "
                                  f"{num_unexpected_msgs} unexpected warning(s): {messages_not_matching_pattern}")
                elif num_unexpected_msgs:
                    assert_msg = (f"caught {num_matching_msgs} matching warning(s) but expected {num_expected} "
                                  f"instead and also caught {num_unexpected_msgs} unexpected warning(s): "
                                  f"{messages_not_matching_pattern}")
                elif num_matching_msgs != num_expected:
                    assert_msg = (f"caught {num_matching_msgs} matching warning(s) but expected {num_expected} "
                                  f"instead")
                else:
                    assert_msg = None
                if assert_msg is not None:
                    raise AssertionError(assert_msg)

            if check_file:
                caller_path = inspect.stack()[2].filename
                warning_path = caught_warnings[0].filename
                assert warning_path == caller_path, f"{warning_path} != {caller_path}"


def must_raise(exception_cls=None, msg=None, match=None):
    from _pytest.python_api import RaisesContext

    if msg is not None and match is not None:
        raise ValueError("BAD TEST: can't use both msg and match arguments")
    elif msg is None and match is None:
        raise ValueError("BAD TEST: not checking the error message")
    elif msg is not None:
        match = f'^{re.escape(msg)}$'

    # This version starts the traceback at the right level. Unfortunately, it uses
    # pytest private API, so it might break in the future. Given that our end-users should
    # not use this function, I think it is worth it.
    return RaisesContext(exception_cls, f"DID NOT RAISE {exception_cls}", match)

