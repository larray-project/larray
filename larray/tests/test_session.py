import os
import shutil
import pickle
from datetime import date, time, datetime

import numpy as np
import pandas as pd
import pytest

from larray.tests.common import meta            # noqa: F401
from larray.tests.common import (assert_array_nan_equal, inputpath, tmp_path,
                                 needs_xlwings, needs_pytables, needs_openpyxl)
from larray.inout.common import _supported_scalars_types
from larray import (Session, Axis, Array, Group, isnan, zeros_like, ndtest, ones_like, ones, full,
                    local_arrays, global_arrays, arrays)


def equal(o1, o2):
    if isinstance(o1, Array) or isinstance(o2, Array):
        return o1.equals(o2)
    elif isinstance(o1, Axis) or isinstance(o2, Axis):
        return o1.equals(o2)
    else:
        return o1 == o2


def assertObjListEqual(got, expected):
    assert len(got) == len(expected)
    for e1, e2 in zip(got, expected):
        assert equal(e1, e2), f"{e1} != {e2}"


a = Axis('a=a0..a2')
a2 = Axis('a=a0..a4')
anonymous = Axis(4)
a01 = a['a0,a1'] >> 'a01'
ano01 = a['a0,a1']
b = Axis('b=0..4')
b024 = b[[0, 2, 4]] >> 'b024'
c = 'c'
d = {}
e = ndtest([(2, 'a'), (3, 'b')])
_e = ndtest((3, 3))
f = ndtest((Axis(3), Axis(2)))
g = ndtest([(2, 'a'), (4, 'b')])
h = ndtest(('a=a0..a2', 'b=b0..b4'))


@pytest.fixture()
def session():
    return Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                    ('a01', a01), ('ano01', ano01), ('c', c), ('d', d), ('e', e), ('g', g), ('f', f), ('h', h)])


def test_init_session(meta):
    s = Session(b, b024, a, a01, a2=a2, anonymous=anonymous, ano01=ano01, c=c, d=d, e=e, f=f, g=g, h=h)
    assert s.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024', 'c', 'd', 'e', 'f', 'g', 'h']

    # TODO: format autodetection does not work in this case
    # s = Session('test_session_csv')
    # assert s.names == ['e', 'f', 'g']

    # metadata
    s = Session(b, b024, a, a01, a2=a2, anonymous=anonymous, ano01=ano01, c=c, d=d, e=e, f=f, g=g, h=h, meta=meta)
    assert s.meta == meta


@needs_xlwings
def test_init_session_xlsx():
    s = Session(inputpath('demography_eurostat.xlsx'))
    assert s.names == ['births', 'deaths', 'immigration', 'population',
                       'population_5_countries', 'population_benelux']


@needs_pytables
def test_init_session_hdf():
    s = Session(inputpath('test_session.h5'))
    assert s.names == ['e', 'f', 'g']


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
    session.add(i, i01, j='j')
    assert i.equals(session.i)
    assert i01 == session.i01
    assert session.j == 'j'


def test_iter(session):
    expected = [b, b024, a, a2, anonymous, a01, ano01, c, d, e, g, f, h]
    assertObjListEqual(session, expected)


def test_filter(session):
    session.ax = 'ax'
    assertObjListEqual(session.filter(), [b, b024, a, a2, anonymous, a01, ano01, 'c', {}, e, g, f, h, 'ax'])
    assertObjListEqual(session.filter('a*'), [a, a2, anonymous, a01, ano01, 'ax'])
    assert list(session.filter('a*', dict)) == []
    assert list(session.filter('a*', str)) == ['ax']
    assert list(session.filter('a*', Axis)) == [a, a2, anonymous]
    assert list(session.filter(kind=Axis)) == [b, a, a2, anonymous]
    assert list(session.filter('a01', Group)) == [a01]
    assert list(session.filter(kind=Group)) == [b024, a01, ano01]
    assertObjListEqual(session.filter(kind=Array), [e, g, f, h])
    assert list(session.filter(kind=dict)) == [{}]
    assert list(session.filter(kind=(Axis, Group))) == [b, b024, a, a2, anonymous, a01, ano01]


def test_names(session):
    assert session.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                             'c', 'd', 'e', 'f', 'g', 'h']
    # add them in the "wrong" order
    session.add(i='i')
    session.add(j='j')
    assert session.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                             'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def _test_io(fpath, session, meta, engine):
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
    a3 = Axis('a=0..3')
    a3_01 = a3['0,1'] >> 'a01'
    e2 = ndtest((a3, 'b=b0..b2'))
    Session(a=a3, a01=a3_01, e=e2).save(fpath, overwrite=False, engine=engine)
    s = Session()
    s.load(fpath, engine=engine)
    if engine == 'pandas_excel':
        # Session.save() via engine='pandas_excel' always overwrite the output Excel files
        assert s.names == ['e']
    elif is_excel_or_csv:
        assert s.names == ['e', 'f', 'g', 'h']
    else:
        assert s.names == session.names
        assert s['a'].equals(a3)
        assert s['a01'].equals(a3_01)
    assert_array_nan_equal(s['e'], e2)
    if engine != 'pandas_excel':
        assert s.meta == meta

    # load only some objects
    session.save(fpath, engine=engine)
    s = Session()
    names_to_load = ['e', 'f'] if is_excel_or_csv else ['a', 'a01', 'a2', 'anonymous', 'e', 'f']
    s.load(fpath, names=names_to_load, engine=engine)
    assert s.names == names_to_load
    if engine != 'pandas_excel':
        assert s.meta == meta


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
def test_h5_io(tmpdir, session, meta):
    session = _add_scalars_to_session(session)
    fpath = tmp_path(tmpdir, 'test_session.h5')
    _test_io(fpath, session, meta, engine='pandas_hdf')


@needs_openpyxl
def test_xlsx_pandas_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session.xlsx')
    _test_io(fpath, session, meta, engine='pandas_excel')


@needs_xlwings
def test_xlsx_xlwings_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session.xlsx')
    _test_io(fpath, session, meta, engine='xlwings_excel')


def test_csv_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session_csv')
    try:
        _test_io(fpath, session, meta, engine='pandas_csv')

        names = session.filter(kind=Array).names

        # test loading with a pattern
        pattern = os.path.join(fpath, '*.csv')
        s = Session(pattern)
        assert s.names == names
        assert s.meta == meta

        # create an invalid .csv file
        invalid_fpath = os.path.join(fpath, 'invalid.csv')
        with open(invalid_fpath, 'w') as f:
            f.write(',",')

        # try loading the directory with the invalid file
        with pytest.raises(pd.errors.ParserError):
            s = Session(pattern)

        # test loading a pattern, ignoring invalid/unsupported files
        s = Session()
        s.load(pattern, ignore_exceptions=True)
        assert s.names == names
        assert s.meta == meta
    finally:
        shutil.rmtree(fpath)


def test_pickle_io(tmpdir, session, meta):
    session = _add_scalars_to_session(session)
    fpath = tmp_path(tmpdir, 'test_session.pkl')
    _test_io(fpath, session, meta, engine='pickle')


def test_to_globals(session):
    with pytest.warns(RuntimeWarning) as caught_warnings:
        session.to_globals()
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == "Session.to_globals should usually only be used in interactive " \
                                                 "consoles and not in scripts. Use warn=False to deactivate this " \
                                                 "warning."
    assert caught_warnings[0].filename == __file__

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
    sess = session.filter(kind=(Axis, Group, Array))
    expected = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                        ('a01', a01), ('ano01', ano01), ('e', e), ('g', g), ('f', f), ('h', h)])
    assert all(sess.element_equals(expected))

    other = Session([('a', a), ('a2', a2), ('anonymous', anonymous),
                     ('a01', a01), ('ano01', ano01), ('e', e), ('f', f), ('h', h)])
    res = sess.element_equals(other)
    assert res.ndim == 1
    assert res.axes.names == ['name']
    assert np.array_equal(res.axes.labels[0], ['b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01',
                                               'e', 'g', 'f', 'h'])
    assert list(res) == [False, False, True, True, True, True, True, True, False, True, True]

    e2 = e.copy()
    e2.i[1, 1] = 42
    other = Session([('a', a), ('a2', a2), ('anonymous', anonymous),
                     ('a01', a01), ('ano01', ano01), ('e', e2), ('f', f), ('h', h)])
    res = sess.element_equals(other)
    assert res.axes.names == ['name']
    assert np.array_equal(res.axes.labels[0], ['b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01',
                                               'e', 'g', 'f', 'h'])
    assert list(res) == [False, False, True, True, True, True, True, False, False, True, True]


def test_eq(session):
    sess = session.filter(kind=(Axis, Group, Array))
    expected = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                        ('a01', a01), ('ano01', ano01), ('e', e), ('g', g), ('f', f), ('h', h)])
    assert all([item.all() if isinstance(item, Array) else item
                for item in (sess == expected).values()])

    other = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                     ('a01', a01), ('ano01', ano01), ('e', e), ('f', f), ('h', h)])
    res = sess == other
    assert list(res.keys()) == ['b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01',
                                'e', 'g', 'f', 'h']
    assert [item.all() if isinstance(item, Array) else item
            for item in res.values()] == [True, True, True, True, True, True, True, True, False, True, True]

    e2 = e.copy()
    e2.i[1, 1] = 42
    other = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                     ('a01', a01), ('ano01', ano01), ('e', e2), ('f', f), ('h', h)])
    res = sess == other
    assert [item.all() if isinstance(item, Array) else item
            for item in res.values()] == [True, True, True, True, True, True, True, False, False, True, True]


def test_ne(session):
    sess = session.filter(kind=(Axis, Group, Array))
    expected = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                        ('a01', a01), ('ano01', ano01), ('e', e), ('g', g), ('f', f), ('h', h)])
    assert ([(~item).all() if isinstance(item, Array) else not item
             for item in (sess != expected).values()])

    other = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                     ('a01', a01), ('ano01', ano01), ('e', e), ('f', f), ('h', h)])
    res = sess != other
    assert list(res.keys()) == ['b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01',
                                'e', 'g', 'f', 'h']
    assert [(~item).all() if isinstance(item, Array) else not item
            for item in res.values()] == [True, True, True, True, True, True, True, True, False, True, True]

    e2 = e.copy()
    e2.i[1, 1] = 42
    other = Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                     ('a01', a01), ('ano01', ano01), ('e', e2), ('f', f), ('h', h)])
    res = sess != other
    assert [(~item).all() if isinstance(item, Array) else not item
            for item in res.values()] == [True, True, True, True, True, True, True, False, False, True, True]


def test_sub(session):
    sess = session

    # session - session
    other = Session({'e': e - 1, 'f': ones_like(f)})
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
    sess = Session([('a', a), ('a01', a01), ('c', c), ('e', ndtest(axes)),
                    ('f', full(axes, fill_value=3)), ('g', ndtest('c=c0..c2'))])
    diff = sess - ones(axes)
    assert_array_nan_equal(diff['e'], sess['e'] - ones(axes))
    assert_array_nan_equal(diff['f'], sess['f'] - ones(axes))
    assert_array_nan_equal(diff['g'], sess['g'] - ones(axes))
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
    other = Session({'e': e - 1, 'f': f + 1})

    with pytest.warns(RuntimeWarning) as caught_warnings:
        res = sess / other
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == "divide by zero encountered during operation"
    assert caught_warnings[0].filename == __file__

    with np.errstate(divide='ignore', invalid='ignore'):
        flat_e = np.arange(6) / np.arange(-1, 5)
    assert_array_nan_equal(res['e'], flat_e.reshape(2, 3))

    flat_f = np.arange(6) / np.arange(1, 7)
    assert_array_nan_equal(res['f'], flat_f.reshape(3, 2))
    assert isnan(res['g']).all()


def test_rdiv(session):
    sess = session

    # scalar / session
    res = 2 / sess
    assert_array_nan_equal(res['e'], 2 / e)
    assert_array_nan_equal(res['f'], 2 / f)
    assert_array_nan_equal(res['g'], 2 / g)
    assert res.a is a
    assert res.a01 is a01
    assert res.c is c

    # dict(Array and scalar) - session
    other = {'e': e, 'f': f}
    res = other / sess
    assert_array_nan_equal(res['e'], e / e)
    assert_array_nan_equal(res['f'], f / f)
    assert res.a is a
    assert res.a01 is a01
    assert res.c is c


def test_pickle_roundtrip(session, meta):
    original = session.filter(kind=Array)
    original.meta = meta
    s = pickle.dumps(original)
    res = pickle.loads(s)
    assert res.equals(original)
    assert res.meta == meta


def test_local_arrays():
    h = ndtest(2)
    _h = ndtest(3)

    # exclude private local arrays
    s = local_arrays()
    s_expected = Session([('h', h)])
    assert s.equals(s_expected)

    # all local arrays
    s = local_arrays(include_private=True)
    s_expected = Session([('h', h), ('_h', _h)])
    assert s.equals(s_expected)


def test_global_arrays():
    # exclude private global arrays
    s = global_arrays()
    s_expected = Session([('e', e), ('f', f), ('g', g), ('h', h)])
    assert s.equals(s_expected)

    # all global arrays
    s = global_arrays(include_private=True)
    s_expected = Session([('e', e), ('_e', _e), ('f', f), ('g', g), ('h', h)])
    assert s.equals(s_expected)


def test_arrays():
    i = ndtest(2)
    _i = ndtest(3)

    # exclude private arrays
    s = arrays()
    s_expected = Session([('e', e), ('f', f), ('g', g), ('h', h), ('i', i)])
    assert s.equals(s_expected)

    # all arrays
    s = arrays(include_private=True)
    s_expected = Session([('_e', _e), ('_i', _i), ('e', e), ('f', f), ('g', g), ('h', h), ('i', i)])
    assert s.equals(s_expected)


if __name__ == "__main__":
    pytest.main()
