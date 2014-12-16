from __future__ import absolute_import, division, print_function

from unittest import TestCase
import unittest

from larray import Session, Axis


class TestSession(TestCase):
    def setUp(self):
        self.a = Axis('a', [])
        self.b = Axis('b', [])
        self.session = Session(self.a, self.b, c='c', d={})

    def test_getitem(self):
        s = self.session
        self.assertEqual(s['a'], self.a)
        self.assertEqual(s['b'], self.b)
        self.assertEqual(s['c'], 'c')
        self.assertEqual(s['d'], {})

    def test_setitem(self):
        s = self.session
        s['e'] = 'e'
        self.assertEqual(s['e'], 'e')

    def test_getattr(self):
        s = self.session
        self.assertEqual(s.a, self.a)
        self.assertEqual(s.b, self.b)
        self.assertEqual(s.c, 'c')
        self.assertEqual(s.d, {})

    def test_setattr(self):
        s = self.session
        s.f = 'f'
        self.assertEqual(s.f, 'f')

    def test_add(self):
        s = self.session
        g = Axis('g', [])
        s.add(g, h='h')
        self.assertEqual(s.g, g)
        self.assertEqual(s.h, 'h')

    def test_objects(self):
        s = self.session
        s.ax = 'ax'
        self.assertEqual(s.objects(), [self.a, 'ax', self.b, 'c', {}])
        self.assertEqual(s.objects('a'), [self.a, 'ax'])
        self.assertEqual(s.objects('a', dict), [])
        self.assertEqual(s.objects('a', str), ['ax'])
        self.assertEqual(s.objects('a', Axis), [self.a])
        self.assertEqual(s.objects(kind=Axis), [self.a, self.b])
        self.assertEqual(s.objects(kind=dict), [{}])

if __name__ == "__main__":
    # import doctest
    # doctest.testmod(larray.core)
    unittest.main()
