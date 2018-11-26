from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from larray.tests.common import assert_array_nan_equal, inputpath, tmp_path, meta
from larray import (Session, Axis, LArray, Group, isnan, zeros_like, ndtest, ones_like,
                    local_arrays, global_arrays, arrays)
from larray.util.misc import pickle

try:
    import xlwings as xw
except ImportError:
    xw = None


def equal(o1, o2):
    if isinstance(o1, LArray) or isinstance(o2, LArray):
        return o1.equals(o2)
    elif isinstance(o1, Axis) or isinstance(o2, Axis):
        return o1.equals(o2)
    else:
        return o1 == o2


def assertObjListEqual(got, expected):
    assert len(got) == len(expected)
    for e1, e2 in zip(got, expected):
        assert equal(e1, e2), "{} != {}".format(e1, e2)


a = Axis('a=a0..a2')
a01 = a['a0,a1'] >> 'a01'
b = Axis('b=b0..b2')
b12 = b['b1,b2'] >> 'b12'
c = 'c'
d = {}
e = ndtest([(2, 'a0'), (3, 'a1')])
_e = ndtest((3, 3))
e2 = ndtest(('a=a0..a2', 'b=b0..b2'))
f = ndtest([(3, 'a0'), (2, 'a1')])
g = ndtest([(2, 'a0'), (4, 'a1')])


@pytest.fixture()
def session():
    return Session([('b', b), ('b12', b12), ('a', a), ('a01', a01),
                    ('c', c), ('d', d), ('e', e), ('g', g), ('f', f)])


def test_init_session(meta):
    s = Session(b, b12, a, a01, c=c, d=d, e=e, f=f, g=g)
    assert s.names == ['a', 'a01', 'b', 'b12', 'c', 'd', 'e', 'f', 'g']

    s = Session(inputpath('test_session.h5'))
    assert s.names == ['e', 'f', 'g']

    # this needs xlwings installed
    # s = Session('test_session_ef.xlsx')
    # assertEqual(s.names, ['e', 'f'])

    # TODO: format autodetection does not work in this case
    # s = Session('test_session_csv')
    # assertEqual(s.names, ['e', 'f', 'g'])

    # metadata
    s = Session(b, b12, a, a01, c=c, d=d, e=e, f=f, g=g, meta=meta)
    assert s.meta == meta


def test_getitem(session):
    assert session['a'] is a
    assert session['b'] is b
    assert session['a01'] is a01
    assert session['b12'] is b12
    assert session['c'] == 'c'
    assert session['d'] == {}


def test_getitem_list(session):
    assert list(session[[]]) == []
    assert list(session[['b', 'a']]) == [b, a]
    assert list(session[['a', 'b']]) == [a, b]
    assert list(session[['b12', 'a']]) == [b12, a]
    assert list(session[['e', 'a01']]) == [e, a01]
    assert list(session[['a', 'e', 'g']]) == [a, e, g]
    assert list(session[['g', 'a', 'e']]) == [g, a, e]


def test_getitem_larray(session):
    s1 = session.filter(kind=LArray)
    s2 = Session({'e': e + 1, 'f': f})
    res_eq = s1[s1.element_equals(s2)]
    res_neq = s1[~(s1.element_equals(s2))]
    assert list(res_eq) == [f]
    assert list(res_neq) == [e, g]


def test_setitem(session):
    s = session.copy()
    s['g'] = 'g'
    assert s['g'] == 'g'


def test_getattr(session):
    assert session.a is a
    assert session.b is b
    assert session.a01 is a01
    assert session.b12 is b12
    assert session.c == 'c'
    assert session.d == {}


def test_setattr(session):
    s = session.copy()
    s.h = 'h'
    assert s.h == 'h'


def test_add(session):
    h = Axis('h=h0..h2')
    h01 = h['h0,h1'] >> 'h01'
    session.add(h, h01, i='i')
    assert h.equals(session.h)
    assert h01 == session.h01
    assert session.i == 'i'


def test_iter(session):
    expected = [b, b12, a, a01, c, d, e, g, f]
    assertObjListEqual(session, expected)


def test_filter(session):
    session.ax = 'ax'
    assertObjListEqual(session.filter(), [b, b12, a, a01, 'c', {}, e, g, f, 'ax'])
    assertObjListEqual(session.filter('a*'), [a, a01, 'ax'])
    assert list(session.filter('a*', dict)) == []
    assert list(session.filter('a*', str)) == ['ax']
    assert list(session.filter('a*', Axis)) == [a]
    assert list(session.filter(kind=Axis)) == [b, a]
    assert list(session.filter('a01', Group)) == [a01]
    assert list(session.filter(kind=Group)) == [b12, a01]
    assertObjListEqual(session.filter(kind=LArray), [e, g, f])
    assert list(session.filter(kind=dict)) == [{}]
    assert list(session.filter(kind=(Axis, Group))) == [b, b12, a, a01]


def test_names(session):
    assert session.names == ['a', 'a01', 'b', 'b12', 'c', 'd', 'e', 'f', 'g']
    # add them in the "wrong" order
    session.add(i='i')
    session.add(h='h')
    assert session.names == ['a', 'a01', 'b', 'b12', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


def test_h5_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session.h5')
    session.meta = meta
    session.save(fpath)

    s = Session()
    s.load(fpath)
    # HDF does *not* keep ordering (ie, keys are always sorted +
    # read Axis objects, then Groups objects and finally LArray objects)
    assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']
    assert s.meta == meta

    # update a Group + an Axis + an array (overwrite=False)
    a2 = Axis('a=0..2')
    a2_01 = a2['0,1'] >> 'a01'
    e2 = ndtest((a2, 'b=b0..b2'))
    Session(a=a2, a01=a2_01, e=e2).save(fpath, overwrite=False)
    s = Session()
    s.load(fpath)
    assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']
    assert s['a'].equals(a2)
    assert all(s['a01'] == a2_01)
    assert_array_nan_equal(s['e'], e2)
    assert s.meta == meta

    # load only some objects
    s = Session()
    s.load(fpath, names=['a', 'a01', 'e', 'f'])
    assert list(s.keys()) == ['a', 'a01', 'e', 'f']
    assert s.meta == meta


def test_xlsx_pandas_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session.xlsx')
    session.meta = meta
    session.save(fpath, engine='pandas_excel')

    s = Session()
    s.load(fpath, engine='pandas_excel')
    assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'g', 'f']
    assert s.meta == meta

    # update a Group + an Axis + an array
    # XXX: overwrite is not taken into account by the pandas_excel engine
    a2 = Axis('a=0..2')
    a2_01 = a2['0,1'] >> 'a01'
    e2 = ndtest((a2, 'b=b0..b2'))
    Session(a=a2, a01=a2_01, e=e2, meta=meta).save(fpath, engine='pandas_excel')
    s = Session()
    s.load(fpath, engine='pandas_excel')
    assert list(s.keys()) == ['a', 'a01', 'e']
    assert s['a'].equals(a2)
    assert all(s['a01'] == a2_01)
    assert_array_nan_equal(s['e'], e2)
    assert s.meta == meta

    # load only some objects
    session.save(fpath, engine='pandas_excel')
    s = Session()
    s.load(fpath, names=['a', 'a01', 'e', 'f'], engine='pandas_excel')
    assert list(s.keys()) == ['a', 'a01', 'e', 'f']
    assert s.meta == meta


@pytest.mark.skipif(xw is None, reason="xlwings is not available")
def test_xlsx_xlwings_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session_xw.xlsx')
    session.meta = meta
    # test save when Excel file does not exist
    session.save(fpath, engine='xlwings_excel')

    s = Session()
    s.load(fpath, engine='xlwings_excel')
    # ordering is only kept if the file did not exist previously (otherwise the ordering is left intact)
    assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'g', 'f']
    assert s.meta == meta

    # update a Group + an Axis + an array (overwrite=False)
    a2 = Axis('a=0..2')
    a2_01 = a2['0,1'] >> 'a01'
    e2 = ndtest((a2, 'b=b0..b2'))
    Session(a=a2, a01=a2_01, e=e2).save(fpath, engine='xlwings_excel', overwrite=False)
    s = Session()
    s.load(fpath, engine='xlwings_excel')
    assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'g', 'f']
    assert s['a'].equals(a2)
    assert all(s['a01'] == a2_01)
    assert_array_nan_equal(s['e'], e2)
    assert s.meta == meta

    # load only some objects
    s = Session()
    s.load(fpath, names=['a', 'a01', 'e', 'f'], engine='xlwings_excel')
    assert list(s.keys()) == ['a', 'a01', 'e', 'f']
    assert s.meta == meta


def test_csv_io(tmpdir, session, meta):
    try:
        fpath = tmp_path(tmpdir, 'test_session_csv')
        session.meta = meta
        session.to_csv(fpath)

        # test loading a directory
        s = Session()
        s.load(fpath, engine='pandas_csv')
        # CSV cannot keep ordering (so we always sort keys)
        # Also, Axis objects are read first, then Groups objects and finally LArray objects
        assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']
        assert s.meta == meta

        # test loading with a pattern
        pattern = os.path.join(fpath, '*.csv')
        s = Session(pattern)
        # s = Session()
        # s.load(pattern)
        assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']
        assert s.meta == meta

        # create an invalid .csv file
        invalid_fpath = os.path.join(fpath, 'invalid.csv')
        with open(invalid_fpath, 'w') as f:
            f.write(',",')

        # try loading the directory with the invalid file
        with pytest.raises(pd.errors.ParserError) as e_info:
            s = Session(pattern)

        # test loading a pattern, ignoring invalid/unsupported files
        s = Session()
        s.load(pattern, ignore_exceptions=True)
        assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']
        assert s.meta == meta

        # load only some objects
        s = Session()
        s.load(fpath, names=['a', 'a01', 'e', 'f'])
        assert list(s.keys()) == ['a', 'a01', 'e', 'f']
        assert s.meta == meta
    finally:
        shutil.rmtree(fpath)


def test_pickle_io(tmpdir, session, meta):
    fpath = tmp_path(tmpdir, 'test_session.pkl')
    session.meta = meta
    session.save(fpath)

    s = Session()
    s.load(fpath, engine='pickle')
    assert list(s.keys()) == ['b', 'a', 'b12', 'a01', 'e', 'g', 'f']
    assert s.meta == meta

    # update a Group + an Axis + an array (overwrite=False)
    a2 = Axis('a=0..2')
    a2_01 = a2['0,1'] >> 'a01'
    e2 = ndtest((a2, 'b=b0..b2'))
    Session(a=a2, a01=a2_01, e=e2).save(fpath, overwrite=False)
    s = Session()
    s.load(fpath, engine='pickle')
    assert list(s.keys()) == ['b', 'a', 'b12', 'a01', 'e', 'g', 'f']
    assert s['a'].equals(a2)
    assert isinstance(a2_01, Group)
    assert isinstance(s['a01'], Group)
    assert s['a01'].eval() == a2_01.eval()
    assert_array_nan_equal(s['e'], e2)
    assert s.meta == meta

    # load only some objects
    s = Session()
    s.load(fpath, names=['a', 'a01', 'e', 'f'], engine='pickle')
    assert list(s.keys()) == ['a', 'a01', 'e', 'f']
    assert s.meta == meta


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


def test_array_equals(session):
    sess = session.filter(kind=LArray)
    expected = Session([('e', e), ('f', f), ('g', g)])
    assert all(sess.element_equals(expected))

    other = Session({'e': e, 'f': f})
    res = sess.element_equals(other)
    assert res.ndim == 1
    assert res.axes.names == ['name']
    assert np.array_equal(res.axes.labels[0], ['e', 'g', 'f'])
    assert list(res) == [True, False, True]

    e2 = e.copy()
    e2.i[1, 1] = 42
    other = Session({'e': e2, 'f': f})
    res = sess.element_equals(other)
    assert res.axes.names == ['name']
    assert np.array_equal(res.axes.labels[0], ['e', 'g', 'f'])
    assert list(res) == [False, False, True]


def test_eq(session):
    sess = session.filter(kind=LArray)
    expected = Session([('e', e), ('f', f), ('g', g)])
    assert all([array.all() for array in (sess == expected).values()])

    other = Session([('e', e), ('f', f)])
    res = sess == other
    assert list(res.keys()) == ['e', 'g', 'f']
    assert [arr.all() for arr in res.values()] == [True, False, True]

    e2 = e.copy()
    e2.i[1, 1] = 42
    other = Session([('e', e2), ('f', f)])
    res = sess == other
    assert [arr.all() for arr in res.values()] == [False, False, True]


def test_ne(session):
    sess = session.filter(kind=LArray)
    expected = Session([('e', e), ('f', f), ('g', g)])
    assert ([(~array).all() for array in (sess != expected).values()])

    other = Session([('e', e), ('f', f)])
    res = sess != other
    assert [(~arr).all() for arr in res.values()] == [True, False, True]

    e2 = e.copy()
    e2.i[1, 1] = 42
    other = Session([('e', e2), ('f', f)])
    res = sess != other
    assert [(~arr).all() for arr in res.values()] == [False, False, True]


def test_sub(session):
    sess = session.filter(kind=LArray)

    # session - session
    other = Session({'e': e - 1, 'f': ones_like(f)})
    diff = sess - other
    assert_array_nan_equal(diff['e'], np.full((2, 3), 1, dtype=np.int32))
    assert_array_nan_equal(diff['f'], f - ones_like(f))
    assert isnan(diff['g']).all()

    # session - scalar
    diff = sess - 2
    assert_array_nan_equal(diff['e'], e - 2)
    assert_array_nan_equal(diff['f'], f - 2)
    assert_array_nan_equal(diff['g'], g - 2)

    # session - dict(LArray and scalar)
    other = {'e': ones_like(e), 'f': 1}
    diff = sess - other
    assert_array_nan_equal(diff['e'], e - ones_like(e))
    assert_array_nan_equal(diff['f'], f - 1)
    assert isnan(diff['g']).all()


def test_rsub(session):
    sess = session.filter(kind=LArray)

    # scalar - session
    diff = 2 - sess
    assert_array_nan_equal(diff['e'], 2 - e)
    assert_array_nan_equal(diff['f'], 2 - f)
    assert_array_nan_equal(diff['g'], 2 - g)

    # dict(LArray and scalar) - session
    other = {'e': ones_like(e), 'f': 1}
    diff = other - sess
    assert_array_nan_equal(diff['e'], ones_like(e) - e)
    assert_array_nan_equal(diff['f'], 1 - f)
    assert isnan(diff['g']).all()


def test_div(session):
    sess = session.filter(kind=LArray)
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
    sess = session.filter(kind=LArray)

    # scalar / session
    res = 2 / sess
    assert_array_nan_equal(res['e'], 2 / e)
    assert_array_nan_equal(res['f'], 2 / f)
    assert_array_nan_equal(res['g'], 2 / g)

    # dict(LArray and scalar) - session
    other = {'e': e, 'f': f}
    res = other / sess
    assert_array_nan_equal(res['e'], e / e)
    assert_array_nan_equal(res['f'], f / f)


def test_pickle_roundtrip(session, meta):
    original = session.filter(kind=LArray)
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
    s_expected = Session([('e', e), ('e2', e2), ('f', f), ('g', g)])
    assert s.equals(s_expected)

    # all global arrays
    s = global_arrays(include_private=True)
    s_expected = Session([('e', e), ('_e', _e), ('e2', e2), ('f', f), ('g', g)])
    assert s.equals(s_expected)


def test_arrays():
    h = ndtest(2)
    _h = ndtest(3)

    # exclude private arrays
    s = arrays()
    s_expected = Session([('e', e), ('e2', e2), ('f', f), ('g', g), ('h', h)])
    assert s.equals(s_expected)

    # all arrays
    s = arrays(include_private=True)
    s_expected = Session([('_e', _e), ('_h', _h), ('e', e), ('e2', e2), ('f', f), ('g', g), ('h', h)])
    assert s.equals(s_expected)


if __name__ == "__main__":
    pytest.main()
