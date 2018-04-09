from __future__ import absolute_import, division, print_function

import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from larray.tests.common import assert_array_nan_equal, inputpath
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


global_arr1 = ndtest((2, 2))
_global_arr2 = ndtest((3, 3))


class TestSession(TestCase):
    def setUp(self):
        self.a = Axis('a=a0..a2')
        self.a01 = self.a['a0,a1'] >> 'a01'
        self.b = Axis('b=b0..b2')
        self.b12 = self.b['b1,b2'] >> 'b12'
        self.c = 'c'
        self.d = {}
        self.e = ndtest([(2, 'a0'), (3, 'a1')])
        self.e2 = ndtest(('a=a0..a2', 'b=b0..b2'))
        self.f = ndtest([(3, 'a0'), (2, 'a1')])
        self.g = ndtest([(2, 'a0'), (4, 'a1')])
        self.session = Session([
            ('b', self.b), ('b12', self.b12), ('a', self.a), ('a01', self.a01),
            ('c', self.c), ('d', self.d), ('e', self.e), ('g', self.g), ('f', self.f),
        ])

    @pytest.fixture(autouse=True)
    def output_dir(self, tmpdir_factory):
        self.tmpdir = tmpdir_factory.mktemp('tmp_session').strpath

    def get_path(self, fname):
        return os.path.join(str(self.tmpdir), fname)

    def assertObjListEqual(self, got, expected):
        assert len(got) == len(expected)
        for e1, e2 in zip(got, expected):
            assert equal(e1, e2), "{} != {}".format(e1, e2)

    def test_init(self):
        s = Session(self.b, self.b12, self.a, self.a01, c=self.c, d=self.d, e=self.e, f=self.f, g=self.g)
        assert s.names == ['a', 'a01', 'b', 'b12', 'c', 'd', 'e', 'f', 'g']

        s = Session(inputpath('test_session.h5'))
        assert s.names == ['e', 'f', 'g']

        # this needs xlwings installed
        # s = Session('test_session_ef.xlsx')
        # self.assertEqual(s.names, ['e', 'f'])

        # TODO: format autodetection does not work in this case
        # s = Session('test_session_csv')
        # self.assertEqual(s.names, ['e', 'f', 'g'])

    def test_getitem(self):
        s = self.session
        assert s['a'] is self.a
        assert s['b'] is self.b
        assert s['a01'] is self.a01
        assert s['b12'] is self.b12
        assert s['c'] == 'c'
        assert s['d'] == {}

    def test_getitem_list(self):
        s = self.session
        assert list(s[[]]) == []
        assert list(s[['b', 'a']]) == [self.b, self.a]
        assert list(s[['a', 'b']]) == [self.a, self.b]
        assert list(s[['b12', 'a']]) == [self.b12, self.a]
        assert list(s[['e', 'a01']]) == [self.e, self.a01]
        assert list(s[['a', 'e', 'g']]) == [self.a, self.e, self.g]
        assert list(s[['g', 'a', 'e']]) == [self.g, self.a, self.e]

    def test_getitem_larray(self):
        s1 = self.session.filter(kind=LArray)
        s2 = Session({'e': self.e + 1, 'f': self.f})
        res_eq = s1[s1.array_equals(s2)]
        res_neq = s1[~(s1.array_equals(s2))]
        assert list(res_eq) == [self.f]
        assert list(res_neq) == [self.e, self.g]

    def test_setitem(self):
        s = self.session
        s['g'] = 'g'
        assert s['g'] == 'g'

    def test_getattr(self):
        s = self.session
        assert s.a is self.a
        assert s.b is self.b
        assert s.a01 is self.a01
        assert s.b12 is self.b12
        assert s.c == 'c'
        assert s.d == {}

    def test_setattr(self):
        s = self.session
        s.h = 'h'
        assert s.h == 'h'

    def test_add(self):
        s = self.session
        h = Axis('h=h0..h2')
        h01 = h['h0,h1'] >> 'h01'
        s.add(h, h01, i='i')
        assert h.equals(s.h)
        assert h01 == s.h01
        assert s.i == 'i'

    def test_iter(self):
        expected = [self.b, self.b12, self.a, self.a01, self.c, self.d, self.e, self.g, self.f]
        self.assertObjListEqual(self.session, expected)

    def test_filter(self):
        s = self.session
        s.ax = 'ax'
        self.assertObjListEqual(s.filter(), [self.b, self.b12, self.a, self.a01, 'c', {},
                                             self.e, self.g, self.f, 'ax'])
        self.assertObjListEqual(s.filter('a'), [self.a, self.a01, 'ax'])
        assert list(s.filter('a', dict)) == []
        assert list(s.filter('a', str)) == ['ax']
        assert list(s.filter('a', Axis)) == [self.a]
        assert list(s.filter(kind=Axis)) == [self.b, self.a]
        assert list(s.filter('a01', Group)) == [self.a01]
        assert list(s.filter(kind=Group)) == [self.b12, self.a01]
        self.assertObjListEqual(s.filter(kind=LArray), [self.e, self.g, self.f])
        assert list(s.filter(kind=dict)) == [{}]
        assert list(s.filter(kind=(Axis, Group))) == [self.b, self.b12, self.a, self.a01]

    def test_names(self):
        s = self.session
        assert s.names == ['a', 'a01', 'b', 'b12', 'c', 'd', 'e', 'f', 'g']
        # add them in the "wrong" order
        s.add(i='i')
        s.add(h='h')
        assert s.names == ['a', 'a01', 'b', 'b12', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    def test_h5_io(self):
        fpath = self.get_path('test_session.h5')
        self.session.save(fpath)

        s = Session()
        s.load(fpath)
        # HDF does *not* keep ordering (ie, keys are always sorted +
        # read Axis objects, then Groups objects and finally LArray objects)
        assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']

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

        # load only some objects
        s = Session()
        s.load(fpath, names=['a', 'a01', 'e', 'f'])
        assert list(s.keys()) == ['a', 'a01', 'e', 'f']

    def test_xlsx_pandas_io(self):
        fpath = self.get_path('test_session.xlsx')
        self.session.save(fpath, engine='pandas_excel')

        s = Session()
        s.load(fpath, engine='pandas_excel')
        assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'g', 'f']

        # update a Group + an Axis + an array (overwrite=False)
        a2 = Axis('a=0..2')
        a2_01 = a2['0,1'] >> 'a01'
        e2 = ndtest((a2, 'b=b0..b2'))
        Session(a=a2, a01=a2_01, e=e2).save(fpath, engine='pandas_excel')
        s = Session()
        s.load(fpath, engine='pandas_excel')
        assert list(s.keys()) == ['a', 'a01', 'e']
        assert s['a'].equals(a2)
        assert all(s['a01'] == a2_01)
        assert_array_nan_equal(s['e'], e2)

        # load only some objects
        self.session.save(fpath, engine='pandas_excel')
        s = Session()
        s.load(fpath, names=['a', 'a01', 'e', 'f'], engine='pandas_excel')
        assert list(s.keys()) == ['a', 'a01', 'e', 'f']

    @pytest.mark.skipif(xw is None, reason="xlwings is not available")
    def test_xlsx_xlwings_io(self):
        fpath = self.get_path('test_session_xw.xlsx')
        # test save when Excel file does not exist
        self.session.save(fpath, engine='xlwings_excel')

        s = Session()
        s.load(fpath, engine='xlwings_excel')
        # ordering is only kept if the file did not exist previously (otherwise the ordering is left intact)
        assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'g', 'f']

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

        # load only some objects
        s = Session()
        s.load(fpath, names=['a', 'a01', 'e', 'f'], engine='xlwings_excel')
        assert list(s.keys()) == ['a', 'a01', 'e', 'f']

    def test_csv_io(self):
        try:
            fpath = self.get_path('test_session_csv')
            self.session.to_csv(fpath)

            # test loading a directory
            s = Session()
            s.load(fpath, engine='pandas_csv')
            # CSV cannot keep ordering (so we always sort keys)
            # Also, Axis objects are read first, then Groups objects and finally LArray objects
            assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']

            # test loading with a pattern
            pattern = os.path.join(fpath, '*.csv')
            s = Session(pattern)
            # s = Session()
            # s.load(pattern)
            assert list(s.keys()) == ['a', 'b', 'a01', 'b12', 'e', 'f', 'g']

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

            # load only some objects
            s = Session()
            s.load(fpath, names=['a', 'a01', 'e', 'f'])
            assert list(s.keys()) == ['a', 'a01', 'e', 'f']
        finally:
            shutil.rmtree(fpath)

    def test_pickle_io(self):
        fpath = self.get_path('test_session.pkl')
        self.session.save(fpath)

        s = Session()
        s.load(fpath, engine='pickle')
        assert list(s.keys()) == ['b', 'a', 'b12', 'a01', 'e', 'g', 'f']

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

        # load only some objects
        s = Session()
        s.load(fpath, names=['a', 'a01', 'e', 'f'], engine='pickle')
        assert list(s.keys()) == ['a', 'a01', 'e', 'f']

    def test_to_globals(self):
        with pytest.warns(RuntimeWarning) as caught_warnings:
            self.session.to_globals()
        assert len(caught_warnings) == 1
        assert caught_warnings[0].message.args[0] == "Session.to_globals should usually only be used in interactive " \
                                                     "consoles and not in scripts. Use warn=False to deactivate this " \
                                                     "warning."
        assert caught_warnings[0].filename == __file__

        self.assertIs(a, self.a)
        self.assertIs(b, self.b)
        self.assertIs(c, self.c)
        self.assertIs(d, self.d)
        self.assertIs(e, self.e)
        self.assertIs(f, self.f)
        self.assertIs(g, self.g)

        # test inplace
        backup_dest = e
        backup_value = self.session.e.copy()
        self.session.e = zeros_like(e)
        self.session.to_globals(inplace=True, warn=False)
        # check the variable is correct (the same as before)
        self.assertIs(e, backup_dest)
        self.assertIsNot(e, self.session.e)
        # check the content has changed
        assert_array_nan_equal(e, self.session.e)
        self.assertFalse(e.equals(backup_value))

    def test_array_equals(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        assert all(sess.array_equals(expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess.array_equals(other)
        assert res.ndim == 1
        assert res.axes.names == ['name']
        assert np.array_equal(res.axes.labels[0], ['e', 'g', 'f'])
        assert list(res) == [True, False, True]

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess.array_equals(other)
        assert res.axes.names == ['name']
        assert np.array_equal(res.axes.labels[0], ['e', 'g', 'f'])
        assert list(res) == [False, False, True]

    def test_eq(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        assert all([array.all() for array in (sess == expected).values()])

        other = Session([('e', self.e), ('f', self.f)])
        res = sess == other
        assert list(res.keys()) == ['e', 'g', 'f']
        assert [arr.all() for arr in res.values()] == [True, False, True]

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session([('e', e2), ('f', self.f)])
        res = sess == other
        assert [arr.all() for arr in res.values()] == [False, False, True]

    def test_ne(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        assert ([(~array).all() for array in (sess != expected).values()])

        other = Session([('e', self.e), ('f', self.f)])
        res = sess != other
        assert [(~arr).all() for arr in res.values()] == [True, False, True]

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session([('e', e2), ('f', self.f)])
        res = sess != other
        assert [(~arr).all() for arr in res.values()] == [False, False, True]

    def test_sub(self):
        sess = self.session.filter(kind=LArray)

        # session - session
        other = Session({'e': self.e - 1, 'f': ones_like(self.f)})
        diff = sess - other
        assert_array_nan_equal(diff['e'], np.full((2, 3), 1, dtype=np.int32))
        assert_array_nan_equal(diff['f'], self.f - ones_like(self.f))
        assert isnan(diff['g']).all()

        # session - scalar
        diff = sess - 2
        assert_array_nan_equal(diff['e'], self.e - 2)
        assert_array_nan_equal(diff['f'], self.f - 2)
        assert_array_nan_equal(diff['g'], self.g - 2)

        # session - dict(LArray and scalar)
        other = {'e': ones_like(self.e), 'f': 1}
        diff = sess - other
        assert_array_nan_equal(diff['e'], self.e - ones_like(self.e))
        assert_array_nan_equal(diff['f'], self.f - 1)
        assert isnan(diff['g']).all()

    def test_rsub(self):
        sess = self.session.filter(kind=LArray)

        # scalar - session
        diff = 2 - sess
        assert_array_nan_equal(diff['e'], 2 - self.e)
        assert_array_nan_equal(diff['f'], 2 - self.f)
        assert_array_nan_equal(diff['g'], 2 - self.g)

        # dict(LArray and scalar) - session
        other = {'e': ones_like(self.e), 'f': 1}
        diff = other - sess
        assert_array_nan_equal(diff['e'], ones_like(self.e) - self.e)
        assert_array_nan_equal(diff['f'], 1 - self.f)
        assert isnan(diff['g']).all()

    def test_div(self):
        sess = self.session.filter(kind=LArray)
        other = Session({'e': self.e - 1, 'f': self.f + 1})

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
        self.assertTrue(isnan(res['g']).all())

    def test_rdiv(self):
        sess = self.session.filter(kind=LArray)

        # scalar / session
        res = 2 / sess
        assert_array_nan_equal(res['e'], 2 / self.e)
        assert_array_nan_equal(res['f'], 2 / self.f)
        assert_array_nan_equal(res['g'], 2 / self.g)

        # dict(LArray and scalar) - session
        other = {'e': self.e, 'f': self.f}
        res = other / sess
        assert_array_nan_equal(res['e'], self.e / self.e)
        assert_array_nan_equal(res['f'], self.f / self.f)

    def test_summary(self):
        # only arrays
        sess = self.session.filter(kind=LArray)
        assert sess.summary() == "e: a0*, a1*\n    \n\ng: a0*, a1*\n    \n\nf: a0*, a1*\n    \n"
        # all objects
        sess = self.session
        assert sess.summary() == "e: a0*, a1*\n    \n\ng: a0*, a1*\n    \n\nf: a0*, a1*\n    \n"

    def test_pickle_roundtrip(self):
        original = self.session.filter(kind=LArray)
        s = pickle.dumps(original)
        res = pickle.loads(s)
        assert res.equals(original)

    def test_local_arrays(self):
        local_arr1 = ndtest(2)
        _local_arr2 = ndtest(3)

        # exclude private local arrays
        s = local_arrays()
        s_expected = Session([('local_arr1', local_arr1)])
        assert s.equals(s_expected)

        # all local arrays
        s = local_arrays(include_private=True)
        s_expected = Session([('local_arr1', local_arr1), ('_local_arr2', _local_arr2)])
        assert s.equals(s_expected)

    def test_global_arrays(self):
        # exclude private global arrays
        s = global_arrays()
        s_expected = Session([('global_arr1', global_arr1)])
        assert s.equals(s_expected)

        # all global arrays
        s = global_arrays(include_private=True)
        s_expected = Session([('global_arr1', global_arr1), ('_global_arr2', _global_arr2)])
        assert s.equals(s_expected)

    def test_arrays(self):
        local_arr1 = ndtest(2)
        _local_arr2 = ndtest(3)

        # exclude private arrays
        s = arrays()
        s_expected = Session([('local_arr1', local_arr1), ('global_arr1', global_arr1)])
        assert s.equals(s_expected)

        # all arrays
        s = arrays(include_private=True)
        s_expected = Session([('local_arr1', local_arr1), ('_local_arr2', _local_arr2),
                              ('global_arr1', global_arr1), ('_global_arr2', _global_arr2)])
        assert s.equals(s_expected)


if __name__ == "__main__":
    pytest.main()
