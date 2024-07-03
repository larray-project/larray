import re
import shutil
import pickle
from datetime import date, time, datetime

import numpy as np
import pandas as pd
import pytest

from larray.tests.common import meta
from larray.tests.common import (assert_larray_equal, assert_array_nan_equal, inputpath,
                                 needs_xlwings, needs_pytables, needs_openpyxl, must_warn, must_raise)
from larray.inout.common import _supported_scalars_types
from larray import (Session, Axis, Array, Group, isnan, zeros_like, ndtest, ones_like,
                    ones, full, full_like, stack, local_arrays, global_arrays, arrays, CheckedSession)


# avoid flake8 errors
meta = meta


def equal(o1, o2):
    if isinstance(o1, Array) or isinstance(o2, Array):
        return o1.equals(o2)
    elif isinstance(o1, Axis) or isinstance(o2, Axis):
        return o1.equals(o2)
    else:
        return o1 == o2


def assert_seq_equal(got, expected):
    assert len(got) == len(expected)
    for e1, e2 in zip(got, expected):
        assert equal(e1, e2), f"{e1} != {e2}"


a = Axis('a=a0..a2')
a2 = Axis('a=a0..a4')
a3 = Axis('a=a0..a3')
anonymous = Axis(4)
a01 = a['a0,a1'] >> 'a01'
ano01 = a['a0,a1']
b = Axis('b=0..4')
b2 = Axis('b=b0..b4')
b024 = b[[0, 2, 4]] >> 'b024'
c = 'c'
d = {}
e = ndtest([(2, 'a'), (3, 'b')])
_e = ndtest((3, 3))
f = ndtest((Axis(3), Axis(2)), dtype=float)
g = ndtest([(2, 'a'), (4, 'b')])
h = ndtest((a3, b2))
k = ndtest((3, 3))

# ########################### #
#           SESSION           #
# ########################### #


@pytest.fixture()
def session():
    return Session({'b': b, 'b024': b024, 'a': a, 'a2': a2, 'anonymous': anonymous,
                    'a01': a01, 'ano01': ano01, 'c': c, 'd': d, 'e': e, 'g': g, 'f': f, 'h': h})


def test_init_session(meta):
    s = Session(b=b, b024=b024, a=a, a01=a01, a2=a2, anonymous=anonymous, ano01=ano01, c=c, d=d, e=e, g=g, f=f, h=h)
    assert list(s.keys()) == ['b', 'b024', 'a', 'a01', 'a2', 'anonymous', 'ano01', 'c', 'd', 'e', 'g', 'f', 'h']

    # TODO: format auto-detection does not work in this case
    # s = Session('test_session_csv')
    # assert list(s.keys()) == ['e', 'f', 'g']

    # metadata
    s = Session(b=b, b024=b024, a=a, a01=a01, a2=a2, anonymous=anonymous, ano01=ano01, c=c, d=d, e=e, f=f, g=g, h=h,
                meta=meta)
    assert s.meta == meta


@needs_xlwings
def test_init_session_xlsx():
    s = Session(inputpath('demography_eurostat.xlsx'))
    assert list(s.keys()) == ['population', 'population_benelux', 'population_5_countries',
                              'births', 'deaths', 'immigration']


@needs_pytables
def test_init_session_hdf():
    s = Session(inputpath('test_session.h5'))
    assert list(s.keys()) == ['e', 'f', 'g', 'h', 'a', 'a2', 'anonymous', 'b', 'a01', 'ano01', 'b024']


def test_getitem(session):
    assert session['a'] is a
    assert session['a2'] is a2
    assert session['anonymous'] is anonymous
    assert session['b'] is b
    assert session['a01'] is a01
    assert session['ano01'] is ano01
    assert session['b024'] is b024
    assert session['c'] == 'c'
    assert session['d'] == {}
    assert equal(session['e'], e)
    assert equal(session['h'], h)


def test_getitem_list(session):
    assert list(session[[]]) == []
    assert list(session[['b', 'a']]) == [b, a]
    assert list(session[['a', 'b']]) == [a, b]
    assert list(session[['a', 'a2']]) == [a, a2]
    assert list(session[['anonymous', 'ano01']]) == [anonymous, ano01]
    assert list(session[['b024', 'a']]) == [b024, a]
    assert list(session[['e', 'a01']]) == [e, a01]
    assert list(session[['a', 'e', 'g']]) == [a, e, g]
    assert list(session[['g', 'a', 'e']]) == [g, a, e]


def test_getitem_larray(session):
    s1 = session.filter(kind=Array)
    s2 = Session({'e': e + 1, 'f': f})
    res_eq = s1[s1.element_equals(s2)]
    res_neq = s1[~(s1.element_equals(s2))]
    assert list(res_eq) == [f]
    assert list(res_neq) == [e, g, h]


def test_setitem(session):
    s = session.copy()
    s['g'] = 'g'
    assert s['g'] == 'g'


def test_getattr(session):
    assert session.a is a
    assert session.a2 is a2
    assert session.anonymous is anonymous
    assert session.b is b
    assert session.a01 is a01
    assert session.ano01 is ano01
    assert session.b024 is b024
    assert session.c == 'c'
    assert session.d == {}


def test_setattr(session):
    s = session.copy()
    s.i = 'i'
    assert s.i == 'i'


def test_add(session):
    i = Axis('i=i0..i2')
    i01 = i['i0,i1'] >> 'i01'
    with must_warn((UserWarning, FutureWarning), match='.*', num_expected=None) as caught_warnings:
        session.add(i, i01, j='j')

    future_warnings = [w for w in caught_warnings if w.category is FutureWarning]
    assert len(future_warnings) == 1
    # .message is the w object itself, .message.args[0] is the actual message
    assert future_warnings[0].message.args[0] == "Session.add() is deprecated. Please use Session.update() instead."

    if isinstance(session, CheckedSession):
        user_warnings = [w for w in caught_warnings if w.category is UserWarning]
        assert len(user_warnings) == 3
        assert all(re.match(r"'\w+' is not declared in 'CheckedSessionExample'", w.message.args[0])
                   for w in user_warnings)

    assert i.equals(session.i)
    assert i01 == session.i01
    assert session.j == 'j'


def test_iter(session):
    expected = [b, b024, a, a2, anonymous, a01, ano01, c, d, e, g, f, h]
    assert_seq_equal(session, expected)


def test_filter(session):
    session.ax = 'ax'
    assert_seq_equal(session.filter(), [b, b024, a, a2, anonymous, a01, ano01, 'c', {}, e, g, f, h, 'ax'])
    assert_seq_equal(session.filter('a*'), [a, a2, anonymous, a01, ano01, 'ax'])
    assert list(session.filter('a*', dict)) == []
    assert list(session.filter('a*', str)) == ['ax']
    assert list(session.filter('a*', Axis)) == [a, a2, anonymous]
    assert list(session.filter(kind=Axis)) == [b, a, a2, anonymous]
    assert list(session.filter('a01', Group)) == [a01]
    assert list(session.filter(kind=Group)) == [b024, a01, ano01]
    assert_seq_equal(session.filter(kind=Array), [e, g, f, h])
    assert list(session.filter(kind=dict)) == [{}]
    assert list(session.filter(kind=(Axis, Group))) == [b, b024, a, a2, anonymous, a01, ano01]


def test_names(session):
    assert session.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                             'c', 'd', 'e', 'f', 'g', 'h']
    # add them in the "wrong" order
    session['j'] = 'j'
    session['i'] = 'i'
    assert session.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                             'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def _test_io(tmp_path, session, meta, engine, ext):
    filename = f"test_{engine}.{ext}" if 'csv' not in engine else f"test_{engine}{ext}"
    fpath = tmp_path / filename

    is_excel_or_csv = 'excel' in engine or 'csv' in engine

    kind = Array if is_excel_or_csv else (Axis, Group, Array) + _supported_scalars_types
    session = session.filter(kind=kind)

    session.meta = meta

    # save and load
    session.save(fpath, engine=engine)
    s = Session()
    s.load(fpath, engine=engine)
    # use Session.names instead of Session.keys because CSV, Excel and HDF do *not* keep ordering
    assert s.names == session.names
    assert s.equals(session)
    if not is_excel_or_csv:
        for key in s.filter(kind=Axis).keys():
            assert s[key].dtype == session[key].dtype
    if engine != 'pandas_excel':
        assert s.meta == meta

    # update a Group + an Axis + an array (overwrite=False)
    a4 = Axis('a=0..3')
    a4_01 = a3['0,1'] >> 'a01'
    e2 = ndtest((a4, 'b=b0..b2'))
    h2 = full_like(h, fill_value=10)
    Session(a=a4, a01=a4_01, e=e2, h=h2).save(fpath, overwrite=False, engine=engine)
    s = Session()
    s.load(fpath, engine=engine)
    if engine == 'pandas_excel':
        # Session.save() via engine='pandas_excel' always overwrite the output Excel files
        assert s.names == ['e', 'h']
    elif is_excel_or_csv:
        assert s.names == ['e', 'f', 'g', 'h']
    else:
        assert s.names == session.names
        assert s['a'].equals(a4)
        assert s['a01'].equals(a4_01)
    assert_array_nan_equal(s['e'], e2)
    if engine != 'pandas_excel':
        assert s.meta == meta

    # load only some objects
    session.save(fpath, engine=engine)
    s = Session()
    names_to_load = ['e', 'f'] if is_excel_or_csv else ['a', 'a01', 'a2', 'anonymous', 'e', 'f', 's_bool', 's_int']
    s.load(fpath, names=names_to_load, engine=engine)
    assert s.names == names_to_load
    if engine != 'pandas_excel':
        assert s.meta == meta

    return fpath


def _add_scalars_to_session(s):
    # 's' for scalar
    s['s_int'] = 5
    s['s_float'] = 5.5
    s['s_bool'] = True
    s['s_str'] = 'string'
    s['s_date'] = date(2020, 1, 10)
    s['s_time'] = time(11, 23, 54)
    s['s_datetime'] = datetime(2020, 1, 10, 11, 23, 54)
    return s


@needs_pytables
def test_h5_io(tmp_path, session, meta):
    session = _add_scalars_to_session(session)

    # for some reason the PerformanceWarning is not detected as such, so this does not work:
    # with must_warn(tables.PerformanceWarning, ...):
    # cannot use check_file=True because of the extra level from _test_io
    with must_warn(Warning, match="^\nyour performance may suffer as PyTables will pickle object types that .*",
                   num_expected=4, check_file=False):
        _test_io(tmp_path, session, meta, engine='pandas_hdf', ext='h5')


@needs_openpyxl
def test_xlsx_pandas_io(tmp_path, session, meta):
    _test_io(tmp_path, session, meta, engine='pandas_excel', ext='xlsx')


@needs_xlwings
def test_xlsx_xlwings_io(tmp_path, session, meta):
    _test_io(tmp_path, session, meta, engine='xlwings_excel', ext='xlsx')


def test_csv_io(tmp_path, session, meta):
    fpath = _test_io(tmp_path, session, meta, engine='pandas_csv', ext='csv')
    try:
        names = Session({k: v for k, v in session.items() if isinstance(v, Array)}).names

        # test loading with a pattern
        pattern = fpath / '*.csv'
        s = Session(pattern)
        assert s.names == names
        assert s.meta == meta

        # create an invalid .csv file
        invalid_fpath = fpath / 'invalid.csv'
        with open(invalid_fpath, 'w') as f:
            f.write(',",')

        # try loading the directory with the invalid file
        # we do not try to validate the exact Pandas error message
        with must_raise(pd.errors.ParserError, match='.*'):
            s = Session(pattern)

        # test loading a pattern, ignoring invalid/unsupported files
        s = Session()
        s.load(pattern, ignore_exceptions=True)
        assert s.names == names
        assert s.meta == meta
    finally:
        shutil.rmtree(fpath)


def test_pickle_io(tmp_path, session, meta):
    session = _add_scalars_to_session(session)
    _test_io(tmp_path, session, meta, engine='pickle', ext='pkl')


def test_pickle_roundtrip(session, meta):
    original = session.filter(kind=Array)
    original.meta = meta
    s = pickle.dumps(original)
    res = pickle.loads(s)
    assert res.equals(original)
    assert res.meta == meta


def test_to_globals(session):
    msg = "Session.to_globals should usually only be used in interactive consoles and not in scripts. " \
          "Use warn=False to deactivate this warning."
    with must_warn(RuntimeWarning, msg=msg):
        session.to_globals()

    assert a is session.a
    assert b is session.b
    assert c is session.c
    assert d is session.d
    assert e is session.e
    assert f is session.f
    assert g is session.g

    # test inplace
    backup_dest = e
    backup_value = session.e.copy()
    session.e = zeros_like(e)
    session.to_globals(inplace=True, warn=False)
    # check the variable is correct (the same as before)
    assert e is backup_dest
    assert e is not session.e
    # check the content has changed
    assert_array_nan_equal(e, session.e)
    assert not e.equals(backup_value)
    # reset e to its original value
    e[:] = backup_value


def test_element_equals(session):
    session_cls = session.__class__
    other_session = session_cls(session)

    keys = [key for key, value in session.items() if isinstance(value, (Axis, Group, Array))]
    expected_res = full(Axis(keys, 'name'), fill_value=True, dtype=bool)

    # ====== same sessions ======
    res = session.element_equals(other_session)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with missing/extra items ======
    # delete some items
    for deleted_key in ['b', 'b024', 'g']:
        del other_session[deleted_key]
        expected_res[deleted_key] = False
    # add one item
    expected_warnings = 1 if isinstance(other_session, CheckedSession) else 0
    with must_warn(UserWarning, msg="'k' is not declared in 'CheckedSessionExample'",
                   num_expected=expected_warnings):
        other_session['k'] = k
    expected_res = expected_res.append('name', False, label='k')

    res = session.element_equals(other_session)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with a modified array ======
    h2 = h.copy()
    h2['a1', 'b1'] = 42
    other_session['h'] = h2
    expected_res['h'] = False

    res = session.element_equals(other_session)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)


def to_boolean_array_eq(res):
    return stack([(key, item.all() if isinstance(item, Array) else item)
                  for key, item in res.items()], 'name')


def test_eq(session):
    session_cls = session.__class__
    other_session = session_cls(session)
    expected_res = full(Axis(list(session.keys()), 'name'), fill_value=True, dtype=bool)

    # ====== same sessions ======
    res = session == other_session
    res = to_boolean_array_eq(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with missing/extra items ======
    del other_session['g']
    expected_res['g'] = False
    expected_warnings = 1 if isinstance(other_session, CheckedSession) else 0
    with must_warn(UserWarning, msg="'k' is not declared in 'CheckedSessionExample'",
                   num_expected=expected_warnings):
        other_session['k'] = k
    expected_res = expected_res.append('name', False, label='k')

    res = session == other_session
    res = to_boolean_array_eq(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with a modified array ======
    h2 = h.copy()
    h2['a1', 'b1'] = 42
    other_session['h'] = h2
    expected_res['h'] = False

    res = session == other_session
    assert res['h'].equals(session['h'] == other_session['h'])
    res = to_boolean_array_eq(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)


def to_boolean_array_ne(res):
    return stack([(key, item.any() if isinstance(item, Array) else item)
                  for key, item in res.items()], 'name')


def test_ne(session):
    session_cls = session.__class__
    other_session = session_cls([(key, value) for key, value in session.items()])
    expected_res = full(Axis(list(session.keys()), 'name'), fill_value=False, dtype=bool)

    # ====== same sessions ======
    res = session != other_session
    res = to_boolean_array_ne(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with missing/extra items ======
    del other_session['g']
    expected_res['g'] = True
    expected_warnings = 1 if isinstance(other_session, CheckedSession) else 0
    with must_warn(UserWarning, msg="'k' is not declared in 'CheckedSessionExample'",
                   num_expected=expected_warnings):
        other_session['k'] = k
    expected_res = expected_res.append('name', True, label='k')

    res = session != other_session
    res = to_boolean_array_ne(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with a modified array ======
    h2 = h.copy()
    h2['a1', 'b1'] = 42
    other_session['h'] = h2
    expected_res['h'] = True

    res = session != other_session
    assert res['h'].equals(session['h'] != other_session['h'])
    res = to_boolean_array_ne(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)


def test_sub(session):
    sess = session

    # session - session
    other = Session({'e': e, 'f': f})
    other['e'] = e - 1
    other['f'] = ones_like(f)
    diff = sess - other
    assert_array_nan_equal(diff['e'], np.full((2, 3), 1, dtype=np.int32))
    assert_array_nan_equal(diff['f'], f - ones_like(f))
    assert isnan(diff['g']).all()
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # session - scalar
    diff = sess - 2
    assert_array_nan_equal(diff['e'], e - 2)
    assert_array_nan_equal(diff['f'], f - 2)
    assert_array_nan_equal(diff['g'], g - 2)
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # session - dict(Array and scalar)
    other = {'e': ones_like(e), 'f': 1}
    diff = sess - other
    assert_array_nan_equal(diff['e'], e - ones_like(e))
    assert_array_nan_equal(diff['f'], f - 1)
    assert isnan(diff['g']).all()
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # session - array
    axes = [a, b]
    other = Session({'a': a, 'a01': a01, 'c': c, 'e': ndtest((a, b)),
                     'f': full((a, b), fill_value=3), 'g': ndtest('c=c0..c2')})
    diff = other - ones(axes)
    assert_array_nan_equal(diff['e'], other['e'] - ones(axes))
    assert_array_nan_equal(diff['f'], other['f'] - ones(axes))
    assert_array_nan_equal(diff['g'], other['g'] - ones(axes))
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c


def test_rsub(session):
    sess = session

    # scalar - session
    diff = 2 - sess
    assert_array_nan_equal(diff['e'], 2 - e)
    assert_array_nan_equal(diff['f'], 2 - f)
    assert_array_nan_equal(diff['g'], 2 - g)
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # dict(Array and scalar) - session
    other = {'e': ones_like(e), 'f': 1}
    diff = other - sess
    assert_array_nan_equal(diff['e'], ones_like(e) - e)
    assert_array_nan_equal(diff['f'], 1 - f)
    assert isnan(diff['g']).all()
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c


def test_div(session):
    sess = session
    session_cls = session.__class__

    other = session_cls({'e': e, 'f': f})
    other['e'] = e - 1
    other['f'] = f + 1

    with must_warn(RuntimeWarning, msg="divide by zero encountered during operation"):
        res = sess / other

    with np.errstate(divide='ignore', invalid='ignore'):
        flat_e = np.arange(6) / np.arange(-1, 5)
    assert_array_nan_equal(res['e'], flat_e.reshape(2, 3))

    flat_f = np.arange(6) / np.arange(1, 7)
    assert_array_nan_equal(res['f'], flat_f.reshape(3, 2))
    assert isnan(res['g']).all()


def test_rdiv(session):
    sess = session

    # scalar / session
    with must_warn(RuntimeWarning, msg="divide by zero encountered during operation", num_expected=4):
        res = 2 / sess
    with np.errstate(divide='ignore'):
        expected_e = 2 / e
        expected_f = 2 / f
        expected_g = 2 / g
    assert_array_nan_equal(res['e'], expected_e)
    assert_array_nan_equal(res['f'], expected_f)
    assert_array_nan_equal(res['g'], expected_g)
    assert res.a is a
    assert res.a01 is a01
    assert res.c is c

    # dict(Array and scalar) - session
    other = {'e': e, 'f': f}
    with must_warn(RuntimeWarning,
                   msg="invalid value (NaN) encountered during operation (this is typically caused by a 0 / 0)",
                   num_expected=2):
        res = other / sess
    with np.errstate(invalid='ignore'):
        expected_e = e / e
        expected_f = f / f
    assert_array_nan_equal(res['e'], expected_e)
    assert_array_nan_equal(res['f'], expected_f)
    assert res.a is a
    assert res.a01 is a01
    assert res.c is c


def test_local_arrays():
    h = ndtest(2)
    _h = ndtest(3)

    # exclude private local arrays
    s = local_arrays()
    s_expected = Session({'h': h})
    assert s.equals(s_expected)

    # all local arrays
    s = local_arrays(include_private=True)
    s_expected = Session({'h': h, '_h': _h})
    assert s.equals(s_expected)


def test_global_arrays():
    # exclude private global arrays
    s = global_arrays()
    s_expected = Session({'e': e, 'f': f, 'g': g, 'h': h, 'k': k})
    assert s.equals(s_expected)

    # all global arrays
    s = global_arrays(include_private=True)
    s_expected = Session({'e': e, '_e': _e, 'f': f, 'g': g, 'h': h, 'k': k})
    assert s.equals(s_expected)


def test_arrays():
    i = ndtest(2)
    _i = ndtest(3)

    # exclude private arrays
    s = arrays()
    s_expected = Session({'e': e, 'f': f, 'g': g, 'h': h, 'i': i, 'k': k})
    assert s.equals(s_expected)

    # all arrays
    s = arrays(include_private=True)
    s_expected = Session({'_e': _e, '_i': _i, 'e': e, 'f': f, 'g': g, 'h': h, 'i': i, 'k': k})
    assert s.equals(s_expected)


def test_stack():
    # stacking all arrays of a single session
    # =======================================
    # a) using explicit axis
    s = Session(arr1=ndtest(3), arr2=ndtest(3) + 10)
    axis = Axis("array=arr2,arr1")
    res = stack(s, axis)
    expected = stack(arr1=s.arr1, arr2=s.arr2, axes=axis)
    assert_larray_equal(res, expected)

    # b) using explicit axis name
    s = Session(arr1=ndtest(3), arr2=ndtest(3) + 10)
    res = stack(s, "array")
    expected = stack(arr1=s.arr1, arr2=s.arr2, axes="array")
    assert_larray_equal(res, expected)

    # c) using not axis information
    s = Session(arr1=ndtest(3), arr2=ndtest(3) + 10)
    res = stack(s)
    expected = stack(arr1=s.arr1, arr2=s.arr2)
    assert_larray_equal(res, expected)

    # stacking two sessions (it will stack all arrays with corresponding names
    s1 = Session(arr1=ndtest(3), arr2=ndtest(3) + 10)
    s2 = Session(arr1=ndtest(3), arr2=ndtest(3) + 30)
    expected_arr1 = stack(s1=s1.arr1, s2=s2.arr1, axes="session")
    expected_arr2 = stack(s1=s1.arr2, s2=s2.arr2, axes="session")
    res = stack(s1=s1, s2=s2, axes="session")
    assert isinstance(res, Session)
    assert_larray_equal(res.arr1, expected_arr1)
    assert_larray_equal(res.arr2, expected_arr2)


if __name__ == "__main__":
    pytest.main()
