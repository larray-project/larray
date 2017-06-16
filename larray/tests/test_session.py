from __future__ import absolute_import, division, print_function

import os
from unittest import TestCase

import numpy as np
import pytest

from larray.tests.common import assert_array_nan_equal, abspath
from larray import Session, Axis, LArray, ndrange, isnan, larray_equal, zeros_like
from larray.util.misc import pickle

try:
    import xlwings as xw
except ImportError:
    xw = None


def equal(o1, o2):
    if isinstance(o1, LArray) or isinstance(o2, LArray):
        return larray_equal(o1, o2)
    elif isinstance(o1, Axis) or isinstance(o2, Axis):
        return o1.equals(o2)
    else:
        return o1 == o2


class TestSession(TestCase):
    def setUp(self):
        self.a = Axis([], 'a')
        self.b = Axis([], 'b')
        self.c = 'c'
        self.d = {}
        self.e = ndrange([(2, 'a0'), (3, 'a1')])
        self.e2 = ndrange(('a=a0..a2', 'b=b0..b2'))
        self.f = ndrange([(3, 'a0'), (2, 'a1')])
        self.g = ndrange([(2, 'a0'), (4, 'a1')])
        self.session = Session([
            ('b', self.b), ('a', self.a),
            ('c', self.c), ('d', self.d),
            ('e', self.e), ('g', self.g), ('f', self.f),
        ])

    def assertObjListEqual(self, got, expected):
        self.assertEqual(len(got), len(expected))
        for e1, e2 in zip(got, expected):
            self.assertTrue(equal(e1, e2), "{} != {}".format(e1, e2))

    def test_init(self):
        s = Session(self.b, self.a, c=self.c, d=self.d,
                    e=self.e, f=self.f, g=self.g)
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        s = Session(abspath('test_session.h5'))
        self.assertEqual(s.names, ['e', 'f', 'g'])

        # this needs xlwings installed
        # s = Session('test_session_ef.xlsx')
        # self.assertEqual(s.names, ['e', 'f'])

        # TODO: format autodetection does not work in this case
        # s = Session('test_session_csv')
        # self.assertEqual(s.names, ['e', 'f', 'g'])

    def test_getitem(self):
        s = self.session
        self.assertIs(s['a'], self.a)
        self.assertIs(s['b'], self.b)
        self.assertEqual(s['c'], 'c')
        self.assertEqual(s['d'], {})

    def test_getitem_list(self):
        s = self.session
        self.assertEqual(list(s[[]]), [])
        self.assertEqual(list(s[['b', 'a']]), [self.b, self.a])
        self.assertEqual(list(s[['a', 'b']]), [self.a, self.b])
        self.assertEqual(list(s[['a', 'e', 'g']]), [self.a, self.e, self.g])
        self.assertEqual(list(s[['g', 'a', 'e']]), [self.g, self.a, self.e])

    def test_getitem_larray(self):
        s1 = self.session.filter(kind=LArray)
        s2 = Session({'e': self.e + 1, 'f': self.f})
        res_eq = s1[s1 == s2]
        res_neq = s1[s1 != s2]
        self.assertEqual(list(res_eq), [self.f])
        self.assertEqual(list(res_neq), [self.e, self.g])

    def test_setitem(self):
        s = self.session
        s['g'] = 'g'
        self.assertEqual(s['g'], 'g')

    def test_getattr(self):
        s = self.session
        self.assertIs(s.a, self.a)
        self.assertIs(s.b, self.b)
        self.assertEqual(s.c, 'c')
        self.assertEqual(s.d, {})

    def test_setattr(self):
        s = self.session
        s.h = 'h'
        self.assertEqual(s.h, 'h')

    def test_add(self):
        s = self.session
        h = Axis([], 'h')
        s.add(h, i='i')
        self.assertTrue(h.equals(s.h))
        self.assertEqual(s.i, 'i')

    def test_iter(self):
        expected = [self.b, self.a, self.c, self.d, self.e, self.g, self.f]
        self.assertObjListEqual(self.session, expected)

    def test_filter(self):
        s = self.session
        s.ax = 'ax'
        self.assertObjListEqual(s.filter(), [self.b, self.a, 'c', {},
                                             self.e, self.g, self.f, 'ax'])
        self.assertEqual(list(s.filter('a')), [self.a, 'ax'])
        self.assertEqual(list(s.filter('a', dict)), [])
        self.assertEqual(list(s.filter('a', str)), ['ax'])
        self.assertEqual(list(s.filter('a', Axis)), [self.a])
        self.assertEqual(list(s.filter(kind=Axis)), [self.b, self.a])
        self.assertObjListEqual(s.filter(kind=LArray), [self.e, self.g, self.f])
        self.assertEqual(list(s.filter(kind=dict)), [{}])

    def test_names(self):
        s = self.session
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        # add them in the "wrong" order
        s.add(i='i')
        s.add(h='h')
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

    def test_h5_io(self):
        fpath = abspath('test_session.h5')
        self.session.save(fpath)

        s = Session()
        s.load(fpath)
        # HDF does *not* keep ordering (ie, keys are always sorted)
        self.assertEqual(list(s.keys()), ['e', 'f', 'g'])

        # update an array (overwrite=False)
        Session(e=self.e2).save(fpath, overwrite=False)
        s.load(fpath)
        self.assertEqual(list(s.keys()), ['e', 'f', 'g'])
        assert_array_nan_equal(s['e'], self.e2)

        s = Session()
        s.load(fpath, ['e', 'f'])
        self.assertEqual(list(s.keys()), ['e', 'f'])

    def test_xlsx_pandas_io(self):
        fpath = abspath('test_session.xlsx')
        self.session.save(fpath, engine='pandas_excel')

        s = Session()
        s.load(fpath, engine='pandas_excel')
        self.assertEqual(list(s.keys()), ['e', 'g', 'f'])

        # update an array (overwrite=False)
        Session(e=self.e2).save(fpath, engine='pandas_excel', overwrite=False)
        s.load(fpath, engine='pandas_excel')
        self.assertEqual(list(s.keys()), ['e', 'g', 'f'])
        assert_array_nan_equal(s['e'], self.e2)

        fpath = abspath('test_session_ef.xlsx')
        self.session.save(fpath, ['e', 'f'], engine='pandas_excel')
        s = Session()
        s.load(fpath, engine='pandas_excel')
        self.assertEqual(list(s.keys()), ['e', 'f'])

    @pytest.mark.skipif(xw is None, reason="xlwings is not available")
    def test_xlsx_xlwings_io(self):
        fpath = abspath('test_session_xw.xlsx')
        # test save when Excel file does not exist
        self.session.save(fpath, engine='xlwings_excel')

        s = Session()
        s.load(fpath, engine='xlwings_excel')
        # ordering is only kept if the file did not exist previously (otherwise the ordering is left intact)
        self.assertEqual(list(s.keys()), ['e', 'g', 'f'])

        # update an array (overwrite=False)
        Session(e=self.e2).save(fpath, engine='xlwings_excel', overwrite=False)
        s.load(fpath, engine='xlwings_excel')
        self.assertEqual(list(s.keys()), ['e', 'g', 'f'])
        assert_array_nan_equal(s['e'], self.e2)

        fpath = abspath('test_session_ef_xw.xlsx')
        self.session.save(fpath, ['e', 'f'], engine='xlwings_excel')
        s = Session()
        s.load(fpath, engine='xlwings_excel')
        self.assertEqual(list(s.keys()), ['e', 'f'])

    def test_csv_io(self):
        fpath = abspath('test_session_csv')
        self.session.to_csv(fpath)

        s = Session()
        s.load(fpath, engine='pandas_csv')
        # CSV cannot keep ordering (so we always sort keys)
        self.assertEqual(list(s.keys()), ['e', 'f', 'g'])

    def test_pickle_io(self):
        fpath = abspath('test_session.pkl')
        self.session.save(fpath)

        s = Session()
        s.load(fpath, engine='pickle')
        self.assertEqual(list(s.keys()), ['e', 'g', 'f'])

        # update an array (overwrite=False)
        Session(e=self.e2).save(fpath, overwrite=False)
        s.load(fpath, engine='pickle')
        self.assertEqual(list(s.keys()), ['e', 'g', 'f'])
        assert_array_nan_equal(s['e'], self.e2)

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
        self.assertFalse(larray_equal(e, backup_value))

    def test_eq(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        self.assertTrue(all(sess == expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess == other
        self.assertEqual(res.ndim, 1)
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'g', 'f']))
        self.assertEqual(list(res), [True, False, True])

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess == other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'g', 'f']))
        self.assertEqual(list(res), [False, False, True])

    def test_ne(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        self.assertFalse(any(sess != expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess != other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'g', 'f']))
        self.assertEqual(list(res), [False, True, False])

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess != other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'g', 'f']))
        self.assertEqual(list(res), [True, True, False])

    def test_sub(self):
        sess = self.session.filter(kind=LArray)
        other = Session({'e': self.e - 1, 'f': 1})
        diff = sess - other
        assert_array_nan_equal(diff['e'], np.full((2, 3), 1, dtype=np.int32))
        assert_array_nan_equal(diff['f'], np.arange(-1, 5).reshape(3, 2))
        self.assertTrue(isnan(diff['g']).all())

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

    def test_summary(self):
        sess = self.session.filter(kind=LArray)
        self.assertEqual(sess.summary(),
                         "e: a0*, a1*\n    \n\n"
                         "g: a0*, a1*\n    \n\n"
                         "f: a0*, a1*\n    \n")

    def test_pickle_roundtrip(self):
        original = self.session
        s = pickle.dumps(original)
        res = pickle.loads(s)
        self.assertTrue(all(res == original))


if __name__ == "__main__":
    pytest.main()
