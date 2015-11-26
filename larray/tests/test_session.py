from __future__ import absolute_import, division, print_function

from unittest import TestCase
import unittest

from larray import Session, Axis, LArray, ndrange


class TestSession(TestCase):
    def setUp(self):
        self.a = Axis('a', [])
        self.b = Axis('b', [])
        self.e = ndrange((2, 3))
        self.f = ndrange((3, 2))
        self.session = Session(self.a, self.b, c='c', d={},
                               e=self.e, f=self.f)

    def test_getitem(self):
        s = self.session
        self.assertEqual(s['a'], self.a)
        self.assertEqual(s['b'], self.b)
        self.assertEqual(s['c'], 'c')
        self.assertEqual(s['d'], {})

    def test_setitem(self):
        s = self.session
        s['g'] = 'g'
        self.assertEqual(s['g'], 'g')

    def test_getattr(self):
        s = self.session
        self.assertEqual(s.a, self.a)
        self.assertEqual(s.b, self.b)
        self.assertEqual(s.c, 'c')
        self.assertEqual(s.d, {})

    def test_setattr(self):
        s = self.session
        s.g = 'g'
        self.assertEqual(s.g, 'g')

    def test_add(self):
        s = self.session
        g = Axis('g', [])
        s.add(g, h='h')
        self.assertEqual(s.g, g)
        self.assertEqual(s.h, 'h')

    def test_iter(self):
        self.assertEqual(list(self.session), [self.a, self.b, 'c', {},
                                              self.e, self.f])

    def test_filter(self):
        s = self.session
        s.ax = 'ax'
        self.assertEqual(list(s.filter()), [self.a, 'ax', self.b, 'c', {},
                                            self.e, self.f])
        self.assertEqual(list(s.filter('a')), [self.a, 'ax'])
        self.assertEqual(list(s.filter('a', dict)), [])
        self.assertEqual(list(s.filter('a', str)), ['ax'])
        self.assertEqual(list(s.filter('a', Axis)), [self.a])
        self.assertEqual(list(s.filter(kind=Axis)), [self.a, self.b])
        self.assertEqual(list(s.filter(kind=LArray)), [self.e, self.f])
        self.assertEqual(list(s.filter(kind=dict)), [{}])

    def test_names(self):
        s = self.session
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f'])
        # add them in the "wrong" order
        s.add(h='h')
        s.add(g='g')
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    def test_dump(self):
        self.session.dump('test.h5')
        self.session.dump('test.xlsx')
        self.session.dump_excel('test2.xlsx')

    def test_load(self):
        s = Session()
        s.load('test.h5', ['e', 'f'])
        self.assertEqual(s.names, ['e', 'f'])

if __name__ == "__main__":
    # import doctest
    # doctest.testmod(larray.core)
    unittest.main()
