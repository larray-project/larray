import pickle
import warnings

import pytest
try:
    import pydantic         # noqa: F401
except ImportError:
    pytest.skip("pydantic is required for testing Checked* classes", allow_module_level=True)
import numpy as np

from larray import (CheckedSession, CheckedArray, Axis, AxisCollection, Group, Array,
                    ndtest, full, full_like, zeros_like, ones, ones_like, isnan)
from larray.tests.common import (inputpath, assert_array_nan_equal, meta,                           # noqa: F401
                                 needs_pytables, needs_openpyxl, needs_xlwings,
                                 must_warn, must_raise)
from larray.tests.test_session import (a, a2, a3, anonymous, a01, ano01, b, b2, b024,               # noqa: F401
                                       c, d, e, f, g, h,
                                       assert_seq_equal, session, test_getitem, test_getattr,
                                       test_add, test_element_equals, test_eq, test_ne)
from larray.core.checked import NotLoaded


# avoid flake8 errors
meta = meta


class CheckedSessionExample(CheckedSession):
    b = b
    b024 = b024
    a: Axis
    a2: Axis
    anonymous = anonymous
    a01: Group
    ano01 = ano01
    c: str = c
    d = d
    e: Array
    g: Array
    f: CheckedArray((Axis(3), Axis(2)))
    h: CheckedArray((a3, b2), dtype=int)


@pytest.fixture()
def checkedsession():
    return CheckedSessionExample(a=a, a2=a2, a01=a01, e=e, g=g, f=f, h=h)


def test_create_checkedsession_instance(meta):
    # As of v1.0 of pydantic all fields with annotations (whether annotation-only or with a default value)
    # will precede all fields without an annotation. Within their respective groups, fields remain in the
    # order they were defined.
    # See https://pydantic-docs.helpmanual.io/usage/models/#field-ordering
    declared_variable_keys = ['a', 'a2', 'a01', 'c', 'e', 'g', 'f', 'h', 'b', 'b024', 'anonymous', 'ano01', 'd']

    # setting variables without default values
    cs = CheckedSessionExample(a=a, a01=a01, a2=a2, e=e, f=f, g=g, h=h)
    assert list(cs.keys()) == declared_variable_keys
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.a.equals(a)
    assert cs.a2.equals(a2)
    assert cs.anonymous.equals(anonymous)
    assert cs.a01.equals(a01)
    assert cs.ano01.equals(ano01)
    assert cs.c == c
    assert cs.d == d
    assert cs.e.equals(e)
    assert cs.g.equals(g)
    assert cs.f.equals(f)
    assert cs.h.equals(h)

    # metadata
    cs = CheckedSessionExample(a=a, a01=a01, a2=a2, e=e, f=f, g=g, h=h, meta=meta)
    assert cs.meta == meta

    # override default value
    b_alt = Axis('b=b0..b4')
    cs = CheckedSessionExample(a=a, a01=a01, b=b_alt, a2=a2, e=e, f=f, g=g, h=h)
    assert cs.b is b_alt

    # test for "NOT_LOADED" variables
    with must_warn(UserWarning, msg="No value passed for the declared variable 'a'"):
        CheckedSessionExample(a01=a01, a2=a2, e=e, f=f, g=g, h=h)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=7):
        cs = CheckedSessionExample()
    assert list(cs.keys()) == declared_variable_keys
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.c == c
    assert cs.d == d
    # --- variables without default values ---
    assert isinstance(cs.a, NotLoaded)
    assert isinstance(cs.a2, NotLoaded)
    assert isinstance(cs.a01, NotLoaded)
    assert isinstance(cs.e, NotLoaded)
    assert isinstance(cs.g, NotLoaded)
    assert isinstance(cs.f, NotLoaded)
    assert isinstance(cs.h, NotLoaded)

    # passing a scalar to set all elements a CheckedArray
    cs = CheckedSessionExample(a=a, a01=a01, a2=a2, e=e, f=f, g=g, h=5)
    assert cs.h.axes == AxisCollection((a3, b2))
    assert cs.h.equals(full(axes=(a3, b2), fill_value=5))

    # add the undeclared variable 'i'
    with must_warn(UserWarning, f"'i' is not declared in '{cs.__class__.__name__}'"):
        cs = CheckedSessionExample(a=a, a01=a01, a2=a2, i=5, e=e, f=f, g=g, h=h)
    assert list(cs.keys()) == declared_variable_keys + ['i']

    # test inheritance between checked sessions
    class TestInheritance(CheckedSessionExample):
        # override variables
        b = b2
        c: int = 5
        f: CheckedArray((a3, b2), dtype=int)
        h: CheckedArray((Axis(3), Axis(2)))
        # new variables
        n0 = 'first new var'
        n1: str

    declared_variable_keys += ['n1', 'n0']
    cs = TestInheritance(a=a, a01=a01, a2=a2, e=e, f=h, g=g, h=f, n1='second new var')
    assert list(cs.keys()) == declared_variable_keys
    # --- overridden variables ---
    assert cs.b.equals(b2)
    assert cs.c == 5
    assert cs.f.equals(h)
    assert cs.h.equals(f)
    # --- new variables ---
    assert cs.n0 == 'first new var'
    assert cs.n1 == 'second new var'
    # --- variables declared in the base class ---
    assert cs.b024.equals(b024)
    assert cs.a.equals(a)
    assert cs.a2.equals(a2)
    assert cs.anonymous.equals(anonymous)
    assert cs.a01.equals(a01)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    assert cs.e.equals(e)
    assert cs.g.equals(g)

    # test deprecated *args to Session
    # no checkfile because it points to session.py instead of test_checked_session.py
    with must_warn(FutureWarning, "Session(obj1, ...) is deprecated, please use Session(obj1name=obj1, ...) instead",
                   check_file=False):
        _ = CheckedSessionExample(a, a01, a2=a2, e=e, f=f, g=g, h=h)


@needs_pytables
def test_init_checkedsession_hdf():
    cs = CheckedSessionExample(inputpath('test_session.h5'))
    assert set(cs.keys()) == {'b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01', 'c', 'd', 'e', 'g', 'f', 'h'}


def test_getitem_cs(checkedsession):
    test_getitem(checkedsession)


def test_setitem_cs(checkedsession):
    cs = checkedsession

    # only change values of an array -> OK
    cs['h'] = zeros_like(h)

    # trying to add an undeclared variable -> prints a warning message
    with must_warn(UserWarning, msg=f"'i' is not declared in '{cs.__class__.__name__}'"):
        cs['i'] = ndtest((3, 3))

    # trying to set a variable with an object of different type -> should fail
    # a) type given explicitly
    # -> Axis
    with must_raise(TypeError, msg="instance of Axis expected"):
        cs['a'] = 0
    # -> CheckedArray
    with must_raise(TypeError, msg="Expected object of type 'Array' or a scalar for the variable 'h' but got "
                                   "object of type 'ndarray'"):
        cs['h'] = h.data
    # b) type deduced from the given default value
    with must_raise(TypeError, msg="instance of Axis expected"):
        cs['b'] = ndtest((3, 3))

    # trying to set a CheckedArray variable using a scalar -> OK
    cs['h'] = 5

    # trying to set a CheckedArray variable using an array with axes in different order -> OK
    cs['h'] = h.transpose()
    assert cs.h.axes.names == h.axes.names

    # broadcasting (missing axis) is allowed
    cs['h'] = ndtest(a3)
    assert_array_nan_equal(cs['h']['b0'], cs['h']['b1'])

    # trying to set a CheckedArray variable using an array with wrong axes -> should fail
    # a) extra axis
    with must_raise(ValueError, msg="Array 'h' was declared with axes {a, b} but got array with axes {a, b, c} "
                                    "(unexpected {c} axis)"):
        cs['h'] = ndtest((a3, b2, 'c=c0..c2'))
    # b) incompatible axis
    msg = """\
Incompatible axis for array 'h':
Axis(['a0', 'a1', 'a2', 'a3', 'a4'], 'a')
vs
Axis(['a0', 'a1', 'a2', 'a3'], 'a')"""
    with must_raise(ValueError, msg=msg):
        cs['h'] = h.append('a', 0, 'a4')


def test_getattr_cs(checkedsession):
    test_getattr(checkedsession)


def test_setattr_cs(checkedsession):
    cs = checkedsession

    # only change values of an array -> OK
    cs.h = zeros_like(h)

    # trying to add an undeclared variable -> prints a warning message
    with must_warn(UserWarning, msg=f"'i' is not declared in '{cs.__class__.__name__}'"):
        cs.i = ndtest((3, 3))

    # trying to set a variable with an object of different type -> should fail
    # a) type given explicitly
    # -> Axis
    with must_raise(TypeError, msg="instance of Axis expected"):
        cs.a = 0
    # -> CheckedArray
    with must_raise(TypeError, msg="Expected object of type 'Array' or a scalar for the variable 'h' but got "
                                   "object of type 'ndarray'"):
        cs.h = h.data
    # b) type deduced from the given default value
    with must_raise(TypeError, msg="instance of Axis expected"):
        cs.b = ndtest((3, 3))

    # trying to set a CheckedArray variable using a scalar -> OK
    cs.h = 5

    # trying to set a CheckedArray variable using an array with axes in different order -> OK
    cs.h = h.transpose()
    assert cs.h.axes.names == h.axes.names

    # broadcasting (missing axis) is allowed
    cs.h = ndtest(a3)
    assert_array_nan_equal(cs.h['b0'], cs.h['b1'])

    # trying to set a CheckedArray variable using an array with wrong axes -> should fail
    # a) extra axis
    with must_raise(ValueError, msg="Array 'h' was declared with axes {a, b} but got array with axes {a, b, c} "
                                    "(unexpected {c} axis)"):
        cs.h = ndtest((a3, b2, 'c=c0..c2'))
    # b) incompatible axis
    msg = """\
Incompatible axis for array 'h':
Axis(['a0', 'a1', 'a2', 'a3', 'a4'], 'a')
vs
Axis(['a0', 'a1', 'a2', 'a3'], 'a')"""
    with must_raise(ValueError, msg=msg):
        cs.h = h.append('a', 0, 'a4')


def test_add_cs(checkedsession):
    cs = checkedsession
    test_add(cs)

    u = Axis('u=u0..u2')
    with must_warn(UserWarning, msg=("Session.add() is deprecated. Please use Session.update() instead.",
                                     f"'u' is not declared in '{cs.__class__.__name__}'")):
        cs.add(u)


def test_iter_cs(checkedsession):
    # As of v1.0 of pydantic all fields with annotations (whether annotation-only or with a default value)
    # will precede all fields without an annotation. Within their respective groups, fields remain in the
    # order they were defined.
    # See https://pydantic-docs.helpmanual.io/usage/models/#field-ordering
    expected = [a, a2, a01, c, e, g, f, h, b, b024, anonymous, ano01, d]
    assert_seq_equal(checkedsession, expected)


def test_filter_cs(checkedsession):
    # see comment in test_iter_cs() about fields ordering
    cs = checkedsession
    with must_warn(UserWarning, msg="'ax' is not declared in 'CheckedSessionExample'"):
        cs.ax = 'ax'
    assert_seq_equal(cs.filter(), [a, a2, a01, c, e, g, f, h, b, b024, anonymous, ano01, d, 'ax'])
    assert_seq_equal(cs.filter('a*'), [a, a2, a01, anonymous, ano01, 'ax'])
    assert list(cs.filter('a*', dict)) == []
    assert list(cs.filter('a*', str)) == ['ax']
    assert list(cs.filter('a*', Axis)) == [a, a2, anonymous]
    assert list(cs.filter(kind=Axis)) == [a, a2, b, anonymous]
    assert list(cs.filter('a01', Group)) == [a01]
    assert list(cs.filter(kind=Group)) == [a01, b024, ano01]
    assert_seq_equal(cs.filter(kind=Array), [e, g, f, h])
    assert list(cs.filter(kind=dict)) == [{}]
    assert list(cs.filter(kind=(Axis, Group))) == [a, a2, a01, b, b024, anonymous, ano01]


def test_names_cs(checkedsession):
    assert checkedsession.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                                    'c', 'd', 'e', 'f', 'g', 'h']


def _test_io_cs(tmp_path, meta, engine, ext):
    filename = f"test_{engine}.{ext}" if 'csv' not in engine else f"test_{engine}{ext}"
    fpath = tmp_path / filename

    is_excel_or_csv = 'excel' in engine or 'csv' in engine

    # Save and load
    # -------------

    # a) - all typed variables have a defined value
    #    - no extra variables are added
    csession = CheckedSessionExample(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=7):
        cs = CheckedSessionExample()
    cs.load(fpath, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g.equals(g)
    assert cs.f.equals(f)
    assert cs.h.equals(h)
    # Axis and Group are not supported by the Excel and CSV formats
    if is_excel_or_csv:
        assert isinstance(cs.a, NotLoaded)
        assert isinstance(cs.a2, NotLoaded)
        assert isinstance(cs.a01, NotLoaded)
    else:
        assert cs.a.equals(a)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a01)
    # --- dtype of Axis variables ---
    if not is_excel_or_csv:
        for key in cs.filter(kind=Axis).keys():
            assert cs[key].dtype == csession[key].dtype
    # --- metadata ---
    if engine != 'pandas_excel':
        assert cs.meta == meta

    # b) - not all typed variables have a defined value
    #    - no extra variables are added
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=4):
        csession = CheckedSessionExample(a=a, d=d, e=e, h=h, meta=meta)
    if 'csv' in engine:
        import shutil
        shutil.rmtree(fpath)
    csession.save(fpath, engine=engine)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=7):
        cs = CheckedSessionExample()
    cs.load(fpath, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e)
    assert isinstance(cs.g, NotLoaded)
    assert isinstance(cs.f, NotLoaded)
    assert cs.h.equals(h)
    # Axis and Group are not supported by the Excel and CSV formats
    if is_excel_or_csv:
        assert isinstance(cs.a, NotLoaded)
        assert isinstance(cs.a2, NotLoaded)
        assert isinstance(cs.a01, NotLoaded)
    else:
        assert cs.a.equals(a)
        assert isinstance(cs.a2, NotLoaded)
        assert isinstance(cs.a01, NotLoaded)

    # c) - all typed variables have a defined value
    #    - extra variables are added
    i = ndtest(6)
    j = ndtest((3, 3))
    k = ndtest((2, 2))
    with must_warn(UserWarning, match=r"'\w' is not declared in 'CheckedSessionExample'", num_expected=3):
        csession = CheckedSessionExample(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, k=k, j=j, i=i, meta=meta)
    csession.save(fpath, engine=engine)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=7):
        cs = CheckedSessionExample()

    with must_warn(UserWarning, match=r"'\w' is not declared in 'CheckedSessionExample'", num_expected=3):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    module='openpyxl',
                                    message=r"datetime.datetime.utcnow\(\) is deprecated.*")

            cs.load(fpath, engine=engine)

    # --- names ---
    # we do not use keys() since order of undeclared variables
    # may not be preserved (at least for the HDF format)
    assert cs.names == csession.names
    # --- extra variable ---
    assert cs.i.equals(i)
    assert cs.j.equals(j)
    assert cs.k.equals(k)

    # Update a Group + an Axis + an array (overwrite=False)
    # -----------------------------------------------------
    csession = CheckedSessionExample(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    a4 = Axis('a=0..3')
    a4_01 = a3['0,1'] >> 'a01'
    e2 = ndtest((a4, 'b=b0..b2'))
    h2 = full_like(h, fill_value=10)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=3):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    module=r'openpyxl|xlsxwriter',
                                    message=r"datetime.datetime.utcnow\(\) is deprecated.*")

            CheckedSessionExample(a=a4, a01=a4_01, e=e2, h=h2).save(fpath, overwrite=False, engine=engine)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=7):
        cs = CheckedSessionExample()

    # number of expected warnings is different depending on engine
    expected_warnings = {
        'pandas_excel': 0,
        'xlwings_excel': 0,
        'pandas_hdf': 0,
        'pandas_csv': 3,
        'pickle': 0,
    }
    num_expected = expected_warnings[engine]
    with must_warn(UserWarning, match=r"'\w' is not declared in 'CheckedSessionExample'",
                   num_expected=num_expected):
        cs.load(fpath, engine=engine)
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e2)
    assert cs.h.equals(h2)
    if engine == 'pandas_excel':
        # Session.save() via engine='pandas_excel' always overwrite the output Excel files
        # arrays 'g' and 'f' have been dropped
        assert isinstance(cs.g, NotLoaded)
        assert isinstance(cs.f, NotLoaded)
        # Axis and Group are not supported by the Excel and CSV formats
        assert isinstance(cs.a, NotLoaded)
        assert isinstance(cs.a2, NotLoaded)
        assert isinstance(cs.a01, NotLoaded)
    elif is_excel_or_csv:
        assert cs.g.equals(g)
        assert cs.f.equals(f)
        # Axis and Group are not supported by the Excel and CSV formats
        assert isinstance(cs.a, NotLoaded)
        assert isinstance(cs.a2, NotLoaded)
        assert isinstance(cs.a01, NotLoaded)
    else:
        assert list(cs.keys()) == list(csession.keys())
        assert cs.a.equals(a4)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a4_01)
        assert cs.g.equals(g)
        assert cs.f.equals(f)
    if engine != 'pandas_excel':
        assert cs.meta == meta

    # Load only some objects
    # ----------------------
    csession = CheckedSessionExample(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    with must_warn(UserWarning, match=r"No value passed for the declared variable '\w+'", num_expected=7):
        cs = CheckedSessionExample()
    names_to_load = ['e', 'h'] if is_excel_or_csv else ['a', 'a01', 'a2', 'e', 'h']
    cs.load(fpath, names=names_to_load, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e)
    assert isinstance(cs.g, NotLoaded)
    assert isinstance(cs.f, NotLoaded)
    assert cs.h.equals(h)
    # Axis and Group are not supported by the Excel and CSV formats
    if is_excel_or_csv:
        assert isinstance(cs.a, NotLoaded)
        assert isinstance(cs.a2, NotLoaded)
        assert isinstance(cs.a01, NotLoaded)
    else:
        assert cs.a.equals(a)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a01)

    return fpath


@needs_pytables
def test_h5_io_cs(tmp_path, meta):
    _test_io_cs(tmp_path, meta, engine='pandas_hdf', ext='h5')


@needs_openpyxl
def test_xlsx_pandas_io_cs(tmp_path, meta):
    _test_io_cs(tmp_path, meta, engine='pandas_excel', ext='xlsx')


@needs_xlwings
def test_xlsx_xlwings_io_cs(tmp_path, meta):
    _test_io_cs(tmp_path, meta, engine='xlwings_excel', ext='xlsx')


def test_csv_io_cs(tmp_path, meta):
    _test_io_cs(tmp_path, meta, engine='pandas_csv', ext='csv')


def test_pickle_io_cs(tmp_path, meta):
    _test_io_cs(tmp_path, meta, engine='pickle', ext='pkl')


def test_pickle_roundtrip_cs(checkedsession, meta):
    cs = checkedsession
    cs.meta = meta
    s = pickle.dumps(cs)
    res = pickle.loads(s)
    assert res.equals(cs)
    assert res.meta == meta


def test_element_equals_cs(checkedsession):
    test_element_equals(checkedsession)


def test_eq_cs(checkedsession):
    test_eq(checkedsession)


def test_ne_cs(checkedsession):
    test_ne(checkedsession)


def test_sub_cs(checkedsession):
    cs = checkedsession
    session_cls = cs.__class__

    # session - session
    other = session_cls(a=a, a2=a2, a01=a01, e=e - 1, g=zeros_like(g), f=zeros_like(f), h=ones_like(h))
    diff = cs - other
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- array variables ---
    assert_array_nan_equal(diff.e, np.full((2, 3), 1, dtype=np.int32))
    assert_array_nan_equal(diff.g, g)
    assert_array_nan_equal(diff.f, f)
    assert_array_nan_equal(diff.h, h - ones_like(h))

    # session - scalar
    diff = cs - 2
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, e - 2)
    assert_array_nan_equal(diff.g, g - 2)
    assert_array_nan_equal(diff.f, f - 2)
    assert_array_nan_equal(diff.h, h - 2)

    # session - dict(Array and scalar)
    other = {'e': ones_like(e), 'h': 1}
    diff = cs - other
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, e - ones_like(e))
    assert isnan(diff.g).all()
    assert isnan(diff.f).all()
    assert_array_nan_equal(diff.h, h - 1)

    # session - array
    axes = cs.h.axes
    cs.e = ndtest(axes)
    cs.g = ones_like(cs.h)
    diff = cs - ones(axes)
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, cs.e - ones(axes))
    assert_array_nan_equal(diff.g, cs.g - ones(axes))
    assert isnan(diff.f).all()
    assert_array_nan_equal(diff.h, cs.h - ones(axes))


def test_rsub_cs(checkedsession):
    cs = checkedsession
    session_cls = cs.__class__

    # scalar - session
    diff = 2 - cs
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, 2 - e)
    assert_array_nan_equal(diff.g, 2 - g)
    assert_array_nan_equal(diff.f, 2 - f)
    assert_array_nan_equal(diff.h, 2 - h)

    # dict(Array and scalar) - session
    other = {'e': ones_like(e), 'h': 1}
    diff = other - cs
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, ones_like(e) - e)
    assert isnan(diff.g).all()
    assert isnan(diff.f).all()
    assert_array_nan_equal(diff.h, 1 - h)


def test_neg_cs(checkedsession):
    cs = checkedsession
    neg_cs = -cs
    # --- non-array variables ---
    assert isnan(neg_cs.b)
    assert isnan(neg_cs.b024)
    assert isnan(neg_cs.a)
    assert isnan(neg_cs.a2)
    assert isnan(neg_cs.anonymous)
    assert isnan(neg_cs.a01)
    assert isnan(neg_cs.ano01)
    assert isnan(neg_cs.c)
    assert isnan(neg_cs.d)
    # --- non constant arrays ---
    assert_array_nan_equal(neg_cs.e, -e)
    assert_array_nan_equal(neg_cs.g, -g)
    assert_array_nan_equal(neg_cs.f, -f)
    assert_array_nan_equal(neg_cs.h, -h)


if __name__ == "__main__":
    pytest.main()
