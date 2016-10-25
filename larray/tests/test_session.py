from __future__ import absolute_import, division, print_function

from unittest import TestCase
import unittest

import numpy as np
from larray import Session, Axis, LArray, ndrange, isnan, larray_equal
from larray.tests.test_la import assert_array_nan_equal


def equal(o1, o2):
    if isinstance(o1, LArray) or isinstance(o2, LArray):
        return larray_equal(o1, o2)
    elif isinstance(o1, Axis) or isinstance(o2, Axis):
        return o1.equals(o2)
    else:
        return o1 == o2


class TestSession(TestCase):
    def setUp(self):
        self.a = Axis('a', [])
        self.b = Axis('b', [])
        self.c = 'c'
        self.d = {}
        self.e = ndrange((2, 3)).rename(0, 'a0').rename(1, 'a1')
        self.f = ndrange((3, 2)).rename(0, 'a0').rename(1, 'a1')
        self.g = ndrange((2, 4)).rename(0, 'a0').rename(1, 'a1')
        self.session = Session([('b', self.b), ('a', self.a),
                                ('c', self.c), ('d', self.d),
                                ('e', self.e), ('f', self.f),
                                ('g', self.g)])

    def assertObjListEqual(self, got, expected):
        self.assertEqual(len(got), len(expected))
        for e1, e2 in zip(got, expected):
            self.assertTrue(equal(e1, e2), "{} != {}".format(e1, e2))

    def test_init(self):
        s = Session(self.b, self.a, c=self.c, d=self.d,
                    e=self.e, f=self.f, g=self.g)
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        s = Session('test_session.h5')
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
        h = Axis('h', [])
        s.add(h, i='i')
        self.assertTrue(h.equals(s.h))
        self.assertEqual(s.i, 'i')

    def test_iter(self):
        expected = [self.b, self.a, self.c, self.d, self.e, self.f, self.g]
        self.assertObjListEqual(self.session, expected)

    def test_filter(self):
        s = self.session
        s.ax = 'ax'
        self.assertObjListEqual(s.filter(), [self.b, self.a, 'c', {},
                                             self.e, self.f, self.g, 'ax'])
        self.assertEqual(list(s.filter('a')), [self.a, 'ax'])
        self.assertEqual(list(s.filter('a', dict)), [])
        self.assertEqual(list(s.filter('a', str)), ['ax'])
        self.assertEqual(list(s.filter('a', Axis)), [self.a])
        self.assertEqual(list(s.filter(kind=Axis)), [self.b, self.a])
        self.assertObjListEqual(s.filter(kind=LArray), [self.e, self.f, self.g])
        self.assertEqual(list(s.filter(kind=dict)), [{}])

    def test_names(self):
        s = self.session
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        # add them in the "wrong" order
        s.add(i='i')
        s.add(h='h')
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

    def test_h5_io(self):
        self.session.dump('test_session.h5')
        s = Session()
        s.load('test_session.h5')
        self.assertEqual(s.names, ['e', 'f', 'g'])

        s = Session()
        s.load('test_session.h5', ['e', 'f'])
        self.assertEqual(s.names, ['e', 'f'])

    def test_xlsx_io(self):
        self.session.dump('test_session.xlsx', engine='pandas_excel')
        self.session.dump('test_session_ef.xlsx', ['e', 'f'],
                          engine='pandas_excel')
        # dump_excel uses default engine (xlwings) which is not available on
        # travis
        # self.session.dump_excel('test_session2.xlsx')

        s = Session()
        s.load('test_session_ef.xlsx', engine='pandas_excel')
        self.assertEqual(s.names, ['e', 'f'])

    def test_csv_io(self):
        self.session.dump_csv('test_session_csv')

        s = Session()
        s.load('test_session_csv', engine='pandas_csv')
        self.assertEqual(s.names, ['e', 'f', 'g'])

    def test_eq(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        self.assertTrue(all(sess == expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess == other
        self.assertEqual(res.ndim, 1)
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [True, True, False])

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess == other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [False, True, False])

    def test_ne(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        self.assertFalse(any(sess != expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess != other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [False, False, True])

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess != other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [True, False, True])

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
        res = sess / other

        flat_e = np.arange(6) / np.arange(-1, 5)
        assert_array_nan_equal(res['e'], flat_e.reshape(2, 3))

        flat_f = np.arange(6) / np.arange(1, 7)
        assert_array_nan_equal(res['f'], flat_f.reshape(3, 2))
        self.assertTrue(isnan(res['g']).all())

if __name__ == "__main__":
    # import doctest
    # doctest.testmod(larray.core)
    unittest.main()
