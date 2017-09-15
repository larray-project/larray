from __future__ import absolute_import, division, print_function

from unittest import TestCase

import pytest
import numpy as np

from larray.tests.common import abspath, assert_array_equal, assert_array_nan_equal
from larray import Axis, LGroup, LSet
from larray.core.group import Group


class TestLGroup(TestCase):
    def setUp(self):
        self.age = Axis('age=0..10')
        self.lipro = Axis('lipro=P01..P05')
        self.anonymous = Axis(range(3))

        self.slice_both_named_wh_named_axis = LGroup('1:5', "full", self.age)
        self.slice_both_named = LGroup('1:5', "named")
        self.slice_both = LGroup('1:5')
        self.slice_start = LGroup('1:')
        self.slice_stop = LGroup(':5')
        self.slice_none_no_axis = LGroup(':')
        self.slice_none_wh_named_axis = LGroup(':', axis=self.lipro)
        self.slice_none_wh_anonymous_axis = LGroup(':', axis=self.anonymous)

        self.single_value = LGroup('P03')
        self.list = LGroup('P01,P03,P04')
        self.list_named = LGroup('P01,P03,P04', "P134")

    def test_init(self):
        self.assertEqual(self.slice_both_named_wh_named_axis.name, "full")
        self.assertEqual(self.slice_both_named_wh_named_axis.key, slice(1, 5, None))
        self.assertIs(self.slice_both_named_wh_named_axis.axis, self.age)

        self.assertEqual(self.slice_both_named.name, "named")
        self.assertEqual(self.slice_both_named.key, slice(1, 5, None))

        self.assertEqual(self.slice_both.key, slice(1, 5, None))
        self.assertEqual(self.slice_start.key, slice(1, None, None))
        self.assertEqual(self.slice_stop.key, slice(None, 5, None))
        self.assertEqual(self.slice_none_no_axis.key, slice(None, None, None))
        self.assertIs(self.slice_none_wh_named_axis.axis, self.lipro)
        self.assertIs(self.slice_none_wh_anonymous_axis.axis, self.anonymous)

        self.assertEqual(self.single_value.key, 'P03')
        self.assertEqual(self.list.key, ['P01', 'P03', 'P04'])

        # passing an axis as name
        group = Group('1:5', self.age, self.age)
        assert group.name == self.age.name
        group = self.age['1:5'] >> self.age
        assert group.name == self.age.name
        # passing an group as name
        group2 = Group('1:5', group, self.age)
        assert group2.name == group.name
        group2 = self.age['1:5'] >> group
        assert group2.name == group.name

    def test_eq(self):
        # with axis vs no axis do not compare equal
        # self.assertEqual(self.slice_both, self.slice_both_named_wh_named_axis)
        self.assertEqual(self.slice_both, self.slice_both_named)

        res = self.slice_both_named_wh_named_axis == self.age[1:5]
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (5,))
        self.assertTrue(res.all())

        self.assertEqual(self.slice_both, LGroup(slice(1, 5)))
        self.assertEqual(self.slice_start, LGroup(slice(1, None)))
        self.assertEqual(self.slice_stop, LGroup(slice(5)))
        self.assertEqual(self.slice_none_no_axis, LGroup(slice(None)))

        self.assertEqual(self.single_value, LGroup('P03'))
        self.assertEqual(self.list, LGroup(['P01', 'P03', 'P04']))
        self.assertEqual(self.list_named, LGroup(['P01', 'P03', 'P04']))

        # test with raw objects
        self.assertEqual(self.slice_both, slice(1, 5))
        self.assertEqual(self.slice_start, slice(1, None))
        self.assertEqual(self.slice_stop, slice(5))
        self.assertEqual(self.slice_none_no_axis, slice(None))

        self.assertEqual(self.single_value, 'P03')
        self.assertEqual(self.list, ['P01', 'P03', 'P04'])
        self.assertEqual(self.list_named, ['P01', 'P03', 'P04'])

    def test_sorted(self):
        self.assertEqual(sorted(LGroup(['c', 'd', 'a', 'b'])),
                         [LGroup('a'), LGroup('b'), LGroup('c'), LGroup('d')])

    def test_asarray(self):
        assert_array_equal(np.asarray(self.slice_both_named_wh_named_axis), np.array([1, 2, 3, 4, 5]))
        assert_array_equal(np.asarray(self.slice_none_wh_named_axis), np.array(['P01', 'P02', 'P03', 'P04', 'P05']))

    def test_hash(self):
        # this test is a lot less important than what it used to, because we cannot have Group ticks on an axis anymore
        d = {self.slice_both: 1,
             self.single_value: 2,
             self.list_named: 3}
        # target a LGroup with an equivalent LGroup
        self.assertEqual(d.get(self.slice_both), 1)
        self.assertEqual(d.get(self.single_value), 2)
        self.assertEqual(d.get(self.list), 3)
        self.assertEqual(d.get(self.list_named), 3)

    def test_repr(self):
        self.assertEqual(repr(self.slice_both_named_wh_named_axis), "age[1:5] >> 'full'")
        self.assertEqual(repr(self.slice_both_named), "LGroup(slice(1, 5, None)) >> 'named'")
        self.assertEqual(repr(self.slice_both), "LGroup(slice(1, 5, None))")
        self.assertEqual(repr(self.list), "LGroup(['P01', 'P03', 'P04'])")
        self.assertEqual(repr(self.slice_none_no_axis), "LGroup(slice(None, None, None))")
        self.assertEqual(repr(self.slice_none_wh_named_axis), "lipro[:]")
        self.assertEqual(repr(self.slice_none_wh_anonymous_axis),
                         "LGroup(slice(None, None, None), axis=Axis([0, 1, 2], None))")

    def test_to_int(self):
        a = Axis(['42'], 'a')
        self.assertEqual(int(a['42']), 42)

    def test_to_float(self):
        a = Axis(['42'], 'a')
        self.assertEqual(float(a['42']), 42.0)


class TestLSet(TestCase):
    def test_or(self):
        # without axis
        self.assertEqual(LSet(['a', 'b']) | LSet(['c', 'd']),
                         LSet(['a', 'b', 'c', 'd']))
        self.assertEqual(LSet(['a', 'b', 'c']) | LSet(['c', 'd']),
                         LSet(['a', 'b', 'c', 'd']))
        # with axis (pure)
        alpha = Axis('alpha=a,b,c,d')
        res = alpha['a', 'b'].set() | alpha['c', 'd'].set()
        self.assertIs(res.axis, alpha)
        self.assertEqual(res, alpha['a', 'b', 'c', 'd'].set())
        self.assertEqual(alpha['a', 'b', 'c'].set() | alpha['c', 'd'].set(),
                         alpha['a', 'b', 'c', 'd'].set())

        # with axis (mixed)
        alpha = Axis('alpha=a,b,c,d')
        res = alpha['a', 'b'].set() | alpha['c', 'd']
        self.assertIs(res.axis, alpha)
        self.assertEqual(res, alpha['a', 'b', 'c', 'd'].set())
        self.assertEqual(alpha['a', 'b', 'c'].set() | alpha['c', 'd'],
                         alpha['a', 'b', 'c', 'd'].set())

        # with axis & name
        alpha = Axis('alpha=a,b,c,d')
        res = alpha['a', 'b'].set().named('ab') | alpha['c', 'd'].set().named('cd')
        self.assertIs(res.axis, alpha)
        self.assertEqual(res.name, 'ab | cd')
        self.assertEqual(res, alpha['a', 'b', 'c', 'd'].set())
        self.assertEqual(alpha['a', 'b', 'c'].set() | alpha['c', 'd'],
                         alpha['a', 'b', 'c', 'd'].set())

        # numeric axis
        num = Axis(range(10), 'num')
        # single int
        self.assertEqual(num[1, 5, 3].set() | 4, num[1, 5, 3, 4].set())
        self.assertEqual(num[1, 5, 3].set() | num[4], num[1, 5, 3, 4].set())
        self.assertEqual(num[4].set() | num[1, 5, 3], num[4, 1, 5, 3].set())
        # slices
        self.assertEqual(num[:2].set() | num[8:], num[0, 1, 2, 8, 9].set())
        self.assertEqual(num[:2].set() | num[5], num[0, 1, 2, 5].set())

    def test_and(self):
        # without axis
        self.assertEqual(LSet(['a', 'b', 'c']) & LSet(['c', 'd']), LSet(['c']))
        # with axis & name
        alpha = Axis('alpha=a,b,c,d')
        res = alpha['a', 'b', 'c'].named('abc').set() & alpha['c', 'd'].named('cd')
        self.assertIs(res.axis, alpha)
        self.assertEqual(res.name, 'abc & cd')
        self.assertEqual(res, alpha[['c']].set())

    def test_sub(self):
        self.assertEqual(LSet(['a', 'b', 'c']) - LSet(['c', 'd']), LSet(['a', 'b']))
        self.assertEqual(LSet(['a', 'b', 'c']) - ['c', 'd'], LSet(['a', 'b']))
        self.assertEqual(LSet(['a', 'b', 'c']) - 'b', LSet(['a', 'c']))
        self.assertEqual(LSet([1, 2, 3]) - 4, LSet([1, 2, 3]))
        self.assertEqual(LSet([1, 2, 3]) - 2, LSet([1, 3]))


class TestPGroup(TestCase):
    def _assert_array_equal_is_true_array(self, a, b):
        res = a == b
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, np.asarray(b).shape)
        self.assertTrue(res.all())

    def setUp(self):
        self.code_axis = Axis('code=a0..a4')

        self.slice_both_named = self.code_axis.i[1:4] >> 'a123'
        self.slice_both = self.code_axis.i[1:4]
        self.slice_start = self.code_axis.i[1:]
        self.slice_stop = self.code_axis.i[:4]
        self.slice_none = self.code_axis.i[:]

        self.first_value = self.code_axis.i[0]
        self.last_value = self.code_axis.i[-1]
        self.list = self.code_axis.i[[0, 1, -2, -1]]
        self.tuple = self.code_axis.i[0, 1, -2, -1]

    def test_asarray(self):
        assert_array_equal(np.asarray(self.slice_both), np.array(['a1', 'a2', 'a3']))

    def test_eq(self):
        self._assert_array_equal_is_true_array(self.slice_both, ['a1', 'a2', 'a3'])
        self._assert_array_equal_is_true_array(self.slice_both_named, ['a1', 'a2', 'a3'])
        self._assert_array_equal_is_true_array(self.slice_both, self.slice_both_named)
        self._assert_array_equal_is_true_array(self.slice_both_named, self.slice_both)
        self._assert_array_equal_is_true_array(self.slice_start, ['a1', 'a2', 'a3', 'a4'])
        self._assert_array_equal_is_true_array(self.slice_stop, ['a0', 'a1', 'a2', 'a3'])
        self._assert_array_equal_is_true_array(self.slice_none, ['a0', 'a1', 'a2', 'a3', 'a4'])

        self.assertEqual(self.first_value, 'a0')
        self.assertEqual(self.last_value, 'a4')

        self._assert_array_equal_is_true_array(self.list, ['a0', 'a1', 'a3', 'a4'])
        self._assert_array_equal_is_true_array(self.tuple, ['a0', 'a1', 'a3', 'a4'])

    def test_getattr(self):
        agg = Axis(['a1:a2', ':a2', 'a1:'], 'agg')
        self.assertEqual(agg.i[0].split(':'), ['a1', 'a2'])
        self.assertEqual(agg.i[1].split(':'), ['', 'a2'])
        self.assertEqual(agg.i[2].split(':'), ['a1', ''])

    def test_dir(self):
        agg = Axis(['a', 1], 'agg')
        self.assertTrue('split' in dir(agg.i[0]))
        self.assertTrue('strip' in dir(agg.i[0]))
        self.assertTrue('strip' in dir(agg.i[0]))

    def test_repr(self):
        self.assertEqual(repr(self.slice_both_named), "code.i[1:4] >> 'a123'")
        self.assertEqual(repr(self.slice_both), "code.i[1:4]")
        self.assertEqual(repr(self.slice_start), "code.i[1:]")
        self.assertEqual(repr(self.slice_stop), "code.i[:4]")
        self.assertEqual(repr(self.slice_none), "code.i[:]")
        self.assertEqual(repr(self.first_value), "code.i[0]")
        self.assertEqual(repr(self.last_value), "code.i[-1]")
        self.assertEqual(repr(self.list), "code.i[0, 1, -2, -1]")
        self.assertEqual(repr(self.tuple), "code.i[0, 1, -2, -1]")

    def test_to_int(self):
        a = Axis(['42'], 'a')
        self.assertEqual(int(a.i[0]), 42)

    def test_to_float(self):
        a = Axis(['42'], 'a')
        self.assertEqual(float(a.i[0]), 42.0)


if __name__ == "__main__":
    pytest.main()
