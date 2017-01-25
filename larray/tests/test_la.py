from __future__ import absolute_import, division, print_function

import os.path
import sys
from unittest import TestCase
import unittest

import numpy as np
import pandas as pd

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray import (LArray, Axis, AxisCollection, LGroup, LSet, PGroup, union,
                    read_csv, read_excel, zeros, zeros_like, ndrange, ndtest,
                    ones, eye, diag, clip, exp, where, x, mean, isnan, round)
from larray.core import _to_ticks, _to_key, df_aslarray


TESTDATADIR = os.path.dirname(__file__)


def abspath(relpath):
    """
    :param relpath: path relative to current module
    :return: absolute path
    """
    return os.path.join(TESTDATADIR, relpath)

# XXX: maybe we should force value groups to use tuple and families (group of
# groups to use lists, or vice versa, so that we know which is which)
# or use a class, just for that?
# group(a, b, c)
# family(group(a), b, c)


def assert_equal_factory(test_func, check_shape=True, check_axes=True):
    def assert_equal(a, b):
        if isinstance(a, LArray) and isinstance(b, LArray) and a.axes != b.axes:
            raise AssertionError("axes differ:\n%s\n\nvs\n\n%s"
                                 % (a.axes.info, b.axes.info))
        if not isinstance(a, (np.ndarray, LArray)):
            a = np.asarray(a)
        if not isinstance(b, (np.ndarray, LArray)):
            b = np.asarray(b)
        if a.shape != b.shape:
            raise AssertionError("shapes differ: %s != %s" % (a.shape, b.shape))
        equal = test_func(a, b)
        if not equal.all():
            # XXX: for some reason ndarray[bool_larray] does not work as we
            #      would like, so we cannot do b[~equal] directly. I should
            #      at least understand why this happens and fix this if
            #      possible.
            notequal = np.asarray(~equal)
        assert equal.all(), "\ngot:\n\n%s\n\nexpected:\n\n%s" % (a[notequal],
                                                                 b[notequal])
    return assert_equal


def equal(a, b):
    return a == b


def nan_equal(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


# numpy.testing.assert_array_equal/assert_equal would work too but it does not
# (as of numpy 1.10) display specifically the non equal items
assert_array_equal = assert_equal_factory(equal)
assert_array_nan_equal = assert_equal_factory(nan_equal)


class TestValueStrings(TestCase):
    def test_split(self):
        self.assertEqual(_to_ticks('H,F'), ['H', 'F'])
        self.assertEqual(_to_ticks('H, F'), ['H', 'F'])

    def test_union(self):
        self.assertEqual(union('A11,A22', 'A12,A22'), ['A11', 'A22', 'A12'])

    def test_range(self):
        self.assertEqual(_to_ticks('0..115'), range(116))
        self.assertEqual(_to_ticks('..115'), range(116))
        with self.assertRaises(ValueError):
            _to_ticks('10..')
        with self.assertRaises(ValueError):
            _to_ticks('..')


class TestKeyStrings(TestCase):
    def test_nonstring(self):
        self.assertEqual(_to_key(('H', 'F')), ['H', 'F'])
        self.assertEqual(_to_key(['H', 'F']), ['H', 'F'])

    def test_split(self):
        self.assertEqual(_to_key('H,F'), ['H', 'F'])
        self.assertEqual(_to_key('H, F'), ['H', 'F'])
        self.assertEqual(_to_key('H,'), ['H'])
        self.assertEqual(_to_key('H'), 'H')

    def test_slice_strings(self):
        # XXX: these two examples return different things, do we want that?
        self.assertEqual(_to_key('0:115'), slice(0, 115))
        self.assertEqual(_to_key(':115'), slice(115))
        self.assertEqual(_to_key('10:'), slice(10, None))
        self.assertEqual(_to_key(':'), slice(None))


class TestAxis(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        sex_tuple = ('H', 'F')
        sex_list = ['H', 'F']
        sex_array = np.array(sex_list)

        # tuple of strings
        assert_array_equal(Axis('sex', sex_tuple).labels, sex_array)
        # list of strings
        assert_array_equal(Axis('sex', sex_list).labels, sex_array)
        # array of strings
        assert_array_equal(Axis('sex', sex_array).labels, sex_array)
        # single string
        assert_array_equal(Axis('sex', 'H,F').labels, sex_array)
        # list of ints
        assert_array_equal(Axis('age', range(116)).labels, np.arange(116))
        # range-string
        assert_array_equal(Axis('age', '..115').labels, np.arange(116))

    def test_equals(self):
        self.assertTrue(Axis('sex', 'H,F').equals(Axis('sex', 'H,F')))
        self.assertTrue(Axis('sex', 'H,F').equals(Axis('sex', ['H', 'F'])))
        self.assertFalse(Axis('sex', 'M,F').equals(Axis('sex', 'H,F')))
        self.assertFalse(Axis('sex1', 'H,F').equals(Axis('sex2', 'H,F')))
        self.assertFalse(Axis('sex1', 'M,F').equals(Axis('sex2', 'H,F')))

    def test_group(self):
        age = Axis('age', '..115')
        ages_list = [1, 5, 9]
        self.assertEqual(age.group(ages_list), LGroup(ages_list, axis=age))
        self.assertEqual(age.group(ages_list), LGroup(ages_list))
        self.assertEqual(age.group('1,5,9'), LGroup(ages_list))
        self.assertEqual(age.group(1, 5, 9), LGroup(ages_list))

        # with a slice string
        self.assertEqual(age.group('10:20'), LGroup(slice(10, 20)))

        # with name
        group = age.group(ages_list, name='teens')
        self.assertEqual(group.key, ages_list)
        self.assertEqual(group.name, 'teens')
        self.assertIs(group.axis, age)

    def test_group_using_lgroup(self):
        def group_equal(g1, g2):
            return (g1.key == g2.key and g1.name == g2.name and
                    g1.axis is g2.axis)

        age = Axis('age', range(100))
        ages = [1, 5, 9]

        val_only = LGroup(ages)
        self.assertTrue(group_equal(age.group(val_only),
                                    LGroup(ages, axis=age)))
        self.assertTrue(group_equal(age.group(val_only, name='a_name'),
                                    LGroup(ages, 'a_name', axis=age)))

        val_name = LGroup(ages, 'val_name')
        self.assertTrue(group_equal(age.group(val_name),
                                    LGroup(ages, 'val_name', age)))
        self.assertTrue(group_equal(age.group(val_name, name='a_name'),
                                    LGroup(ages, 'a_name', age)))

        val_axis = LGroup(ages, axis=age)
        self.assertTrue(group_equal(age.group(val_axis),
                                    LGroup(ages, axis=age)))
        self.assertTrue(group_equal(age.group(val_axis, name='a_name'),
                                    LGroup(ages, 'a_name', axis=age)))

        val_axis_name = LGroup(ages, 'val_axis_name', age)
        self.assertTrue(group_equal(age.group(val_axis_name),
                                    LGroup(ages, 'val_axis_name', age)))
        self.assertTrue(group_equal(age.group(val_axis_name, name='a_name'),
                                    LGroup(ages, 'a_name', age)))

    def test_init_from_group(self):
        code = Axis('code', 'C01..C03')
        code_group = code[:'C02']
        subset_axis = Axis('code_subset', code_group)
        assert_array_equal(subset_axis.labels, ['C01', 'C02'])

    def test_match(self):
        sutcode = Axis('sutcode', ['A23', 'A2301', 'A25', 'A2501'])
        self.assertEqual(sutcode.matches('^...$'), LGroup(['A23', 'A25']))
        self.assertEqual(sutcode.startswith('A23'), LGroup(['A23', 'A2301']))
        self.assertEqual(sutcode.endswith('01'), LGroup(['A2301', 'A2501']))

    def test_getitem(self):
        age = Axis('age', '0..10')
        lg = age.group(':3')
        # these are equivalent
        # self.assertEqual(age[:'17'], lg)
        group = age[':3']
        self.assertEqual(group.key, slice(None, 3, None))
        self.assertTrue(group.axis.equals(age))
        self.assertEqual(group, lg, "%r != %r" % (group, lg))
        # self.assertEqual(age[':7'], lg)

        group = age[:]
        self.assertEqual(group.key, slice(None))
        self.assertIs(group.axis, age)

    def test_iter(self):
        sex = Axis('sex', 'H,F')
        self.assertEqual(list(sex), [PGroup(0, axis=sex), PGroup(1, axis=sex)])

    def test_positional(self):
        age = Axis('age', '..115')

        # these are NOT equivalent (not translated until used in an LArray
        # self.assertEqual(age.i[:17], age[':17'])
        key = age.i[:-1]
        self.assertEqual(key.key, slice(None, -1))
        self.assertIs(key.axis, age)

    def test_all(self):
        age = Axis('age', '..115')
        group = age.all()
        self.assertEqual(group.key, slice(None))
        self.assertIs(group.axis, age)

    def test_contains(self):
        # normal Axis
        age = Axis('age', '..10')

        age2 = age.group(2)
        age2bis = age.group((2,))
        age2ter = age.group([2])
        age2qua = '2,'

        age20 = LGroup('20')
        age20bis = LGroup('20,')
        age20ter = LGroup(['20'])
        age20qua = '20,'

        # TODO: move assert to another test
        # self.assertEqual(age2bis, age2ter)

        age247 = age.group('2,4,7')
        age247bis = age.group(['2', '4', '7'])
        age359 = age.group(['3', '5', '9'])
        age468 = age.group('4,6,8', name='even')

        self.assertTrue(5 in age)
        self.assertFalse('5' in age)

        self.assertTrue(age2 in age)
        # only single ticks are "contained" in the axis, not "collections"
        self.assertFalse(age2bis in age)
        self.assertFalse(age2ter in age)
        self.assertFalse(age2qua in age)

        self.assertFalse(age20 in age)
        self.assertFalse(age20bis in age)
        self.assertFalse(age20ter in age)
        self.assertFalse(age20qua in age)
        self.assertFalse(['3', '5', '9'] in age)
        self.assertFalse('3,5,9' in age)
        self.assertFalse('3:9' in age)
        self.assertFalse(age247 in age)
        self.assertFalse(age247bis in age)
        self.assertFalse(age359 in age)
        self.assertFalse(age468 in age)

        # aggregated Axis
        # FIXME: _to_tick(age2) == 2, but then np.asarray([2, '2,4,7', ...]) returns np.array(['2', '2,4,7'])
        # instead of returning an object array
        agg = Axis("agg", (age2, age247, age359, age468,
                           '2,6', ['3', '5', '7'], ('6', '7', '9')))
        # fails because of above FIXME
        # self.assertTrue(age2 in agg)
        self.assertFalse(age2bis in agg)
        self.assertFalse(age2ter in agg)
        self.assertFalse(age2qua in age)

        self.assertTrue(age247 in agg)
        self.assertTrue(age247bis in agg)
        self.assertTrue('2,4,7' in agg)
        self.assertTrue(['2', '4', '7'] in agg)

        self.assertTrue(age359 in agg)
        self.assertTrue('3,5,9' in agg)
        self.assertTrue(['3', '5', '9'] in agg)

        self.assertTrue(age468 in agg)
        # no longer the case
        # self.assertTrue('4,6,8' in agg)
        # self.assertTrue(['4', '6', '8'] in agg)
        self.assertTrue('even' in agg)

        self.assertTrue('2,6' in agg)
        self.assertTrue(['2', '6'] in agg)
        self.assertTrue(age.group('2,6') in agg)
        self.assertTrue(age.group(['2', '6']) in agg)

        self.assertTrue('3,5,7' in agg)
        self.assertTrue(['3', '5', '7'] in agg)
        self.assertTrue(age.group('3,5,7') in agg)
        self.assertTrue(age.group(['3', '5', '7']) in agg)

        self.assertTrue('6,7,9' in agg)
        self.assertTrue(['6', '7', '9'] in agg)
        self.assertTrue(age.group('6,7,9') in agg)
        self.assertTrue(age.group(['6', '7', '9']) in agg)

        self.assertFalse(5 in agg)
        self.assertFalse('5' in agg)
        self.assertFalse(age20 in agg)
        self.assertFalse(age20bis in agg)
        self.assertFalse(age20ter in agg)
        self.assertFalse(age20qua in agg)
        self.assertFalse('2,7' in agg)
        self.assertFalse(['2', '7'] in agg)
        self.assertFalse(age.group('2,7') in agg)
        self.assertFalse(age.group(['2', '7']) in agg)


class TestLGroup(TestCase):
    def setUp(self):
        self.age = Axis('age', '..115')
        self.lipro = Axis('lipro', ['P%02d' % i for i in range(1, 10)])
        self.anonymous = Axis(None, range(3))

        self.slice_both_named_wh_named_axis = LGroup('1:5', "full", self.age)
        self.slice_both_named = LGroup('1:5', "named")
        self.slice_both = LGroup('1:5')
        self.slice_start = LGroup('1:')
        self.slice_stop = LGroup(':5')
        self.slice_none_no_axis = LGroup(':')
        self.slice_none_wh_named_axis = LGroup(':', axis=self.lipro)
        self.slice_none_wh_anonymous_axis = LGroup(':', axis=self.anonymous)

        self.single_value = LGroup('P03')
        self.list = LGroup('P01,P03,P07')
        self.list_named = LGroup('P01,P03,P07', "P137")

    def test_init(self):
        self.assertEqual(self.slice_both_named_wh_named_axis.name, "full")
        self.assertEqual(self.slice_both_named_wh_named_axis.key, slice(1, 5, None))
        self.assertEqual(self.slice_both_named.name, "named")
        self.assertEqual(self.slice_both_named.key, slice(1, 5, None))
        self.assertEqual(self.slice_both.key, slice(1, 5, None))
        self.assertEqual(self.slice_start.key, slice(1, None, None))
        self.assertEqual(self.slice_stop.key, slice(None, 5, None))
        self.assertEqual(self.slice_none_no_axis.key, slice(None, None, None))
        self.assertIs(self.slice_none_wh_named_axis.axis, self.lipro)
        self.assertIs(self.slice_none_wh_anonymous_axis.axis, self.anonymous)

        self.assertEqual(self.single_value.key, 'P03')
        self.assertEqual(self.list.key, ['P01', 'P03', 'P07'])

    def test_eq(self):
        # with axis vs no axis do not compare equal
        # self.assertEqual(self.slice_both, self.slice_both_named_wh_named_axis)
        self.assertEqual(self.slice_both, self.slice_both_named)
        self.assertEqual(self.slice_both, LGroup(slice('1', '5')))
        self.assertEqual(self.slice_start, LGroup(slice('1', None)))
        self.assertEqual(self.slice_stop, LGroup(slice('5')))
        self.assertEqual(self.slice_none_no_axis, LGroup(slice(None)))
        self.assertEqual(self.list, LGroup(['P01', 'P03', 'P07']))
        # test with raw objects
        self.assertEqual(self.slice_both, '1:5')
        self.assertEqual(self.slice_start, '1:')
        self.assertEqual(self.slice_stop, ':5')
        self.assertEqual(self.slice_none_no_axis, ':')
        self.assertEqual(self.slice_both, slice('1', '5'))
        self.assertEqual(self.slice_start, slice('1', None))
        self.assertEqual(self.slice_stop, slice('5'))
        self.assertEqual(self.slice_none_no_axis, slice(None))
        self.assertEqual(self.list, 'P01,P03,P07')
        self.assertEqual(self.list, ' P01 , P03 , P07 ')
        self.assertEqual(self.list, ['P01', 'P03', 'P07'])
        self.assertEqual(self.list, ('P01', 'P03', 'P07'))

    def test_sorted(self):
        self.assertEqual(sorted(LGroup(['c', 'd', 'a', 'b'])),
                         [LGroup('a'), LGroup('b'), LGroup('c'), LGroup('d')])

    def test_hash(self):
        d = {self.slice_both: 1,
             self.single_value: 2,
             self.list_named: 3}
        # target a LGroup with an equivalent LGroup
        self.assertEqual(d.get(self.slice_both), 1)
        self.assertEqual(d.get(self.single_value), 2)
        self.assertEqual(d.get(self.list), 3)
        self.assertEqual(d.get(self.list_named), 3)
        # this cannot and will never work, because we cannot have the LGroup
        # hash both like its key and like its name!
        # we could make it work with a special dict class, but do we WANT to
        # make it work?
        # yes, probably
        # self.assertEqual(d.get("P137"), 3)

        # target a LGroup with an equivalent key
        self.assertEqual(d.get('1:5'), 1)
        self.assertEqual(d.get('P03'), 2)
        self.assertEqual(d.get('P01,P03,P07'), 3)

        # this cannot and will never work!
        # hash(str) and hash(tuple) are not special, so ' P01 ,...' and
        # ('P01', ...) do not hash to the same value than 'P01,P03...", which is
        # our "canonical hash"
        # self.assertEqual(d.get(' P01 , P03 , P07 '), 3)
        # self.assertEqual(d.get(('P01', 'P03', 'P07')), 3)

        # target a simple key with an equivalent LGroup
        d = {'1:5': 1,
             'P03': 2,
             'P01,P03,P07': 3}
        self.assertEqual(d.get(self.slice_both), 1)
        self.assertEqual(d.get(self.single_value), 2)
        self.assertEqual(d.get(self.list), 3)
        self.assertEqual(d.get(LGroup(' P01 , P03 , P07 ')), 3)
        self.assertEqual(d.get(LGroup(('P01', 'P03', 'P07'))), 3)

    def test_repr(self):
        self.assertEqual(repr(self.slice_both_named),
                         "LGroup(slice(1, 5, None)) >> 'named'")
        self.assertEqual(repr(self.slice_both), "LGroup(slice(1, 5, None))")
        self.assertEqual(repr(self.list), "LGroup(['P01', 'P03', 'P07'])")
        self.assertEqual(repr(self.slice_none_no_axis), "LGroup(slice(None, None, None))")
        self.assertEqual(repr(self.slice_none_wh_named_axis), "lipro[:]")
        self.assertEqual(repr(self.slice_none_wh_anonymous_axis),
                         "LGroup(slice(None, None, None), axis=Axis(None, [0, 1, 2]))")


class TestLSet(TestCase):
    def test_or(self):
        # without axis
        self.assertEqual(LSet(['a', 'b']) | LSet(['c', 'd']),
                         LSet(['a', 'b', 'c', 'd']))
        self.assertEqual(LSet(['a', 'b', 'c']) | LSet(['c', 'd']),
                         LSet(['a', 'b', 'c', 'd']))
        # with axis (pure)
        alpha = Axis('alpha', 'a,b,c,d')
        res = alpha['a', 'b'].set() | alpha['c', 'd'].set()
        self.assertIs(res.axis, alpha)
        self.assertEqual(res, alpha['a', 'b', 'c', 'd'].set())
        self.assertEqual(alpha['a', 'b', 'c'].set() | alpha['c', 'd'].set(),
                         alpha['a', 'b', 'c', 'd'].set())

        # with axis (mixed)
        alpha = Axis('alpha', 'a,b,c,d')
        res = alpha['a', 'b'].set() | alpha['c', 'd']
        self.assertIs(res.axis, alpha)
        self.assertEqual(res, alpha['a', 'b', 'c', 'd'].set())
        self.assertEqual(alpha['a', 'b', 'c'].set() | alpha['c', 'd'],
                         alpha['a', 'b', 'c', 'd'].set())

        # with axis & name
        alpha = Axis('alpha', 'a,b,c,d')
        res = alpha['a', 'b'].set().named('ab') | alpha['c', 'd'].set().named('cd')
        self.assertIs(res.axis, alpha)
        self.assertEqual(res.name, 'ab | cd')
        self.assertEqual(res, alpha['a', 'b', 'c', 'd'].set())
        self.assertEqual(alpha['a', 'b', 'c'].set() | alpha['c', 'd'],
                         alpha['a', 'b', 'c', 'd'].set())

        # numeric axis
        num = Axis('num', range(10))
        # single int
        self.assertEqual(num[1, 5, 3].set() | 4, num[1, 5, 3, 4].set())
        self.assertEqual(num[1, 5, 3].set() | num[4], num[1, 5, 3, 4].set())
        self.assertEqual(num[4].set() | num[1, 5, 3], num[4, 1, 5, 3].set())
        # slices
        self.assertEqual(num[:2].set() | num[8:], num[0, 1, 2, 8, 9].set())
        self.assertEqual(num[:2].set() | num[5], num[0, 1, 2, 5].set())

    def test_and(self):
        # without axis
        self.assertEqual(LSet(['a', 'b', 'c']) & LSet(['c', 'd']),
                         LSet(['c']))
        # with axis & name
        alpha = Axis('alpha', 'a,b,c,d')
        res = alpha['a', 'b', 'c'].named('abc').set() & alpha['c', 'd'].named('cd')
        self.assertIs(res.axis, alpha)
        self.assertEqual(res.name, 'abc & cd')
        self.assertEqual(res, alpha[['c']].set())

    def test_sub(self):
        self.assertEqual(LSet(['a', 'b', 'c']) - LSet(['c', 'd']),
                         LSet(['a', 'b']))
        self.assertEqual(LSet(['a', 'b', 'c']) - ['c', 'd'],
                         LSet(['a', 'b']))
        self.assertEqual(LSet(['a', 'b', 'c']) - 'b',
                         LSet(['a', 'c']))
        self.assertEqual(LSet([1, 2, 3]) - 4, LSet([1, 2, 3]))
        self.assertEqual(LSet([1, 2, 3]) - 2, LSet([1, 3]))


class TestAxisCollection(TestCase):
    def setUp(self):
        self.lipro = Axis('lipro', 'P01..P04')
        self.sex = Axis('sex', 'H,F')
        self.sex2 = Axis('sex', 'F,H')
        self.age = Axis('age', '..7')
        self.geo = Axis('geo', 'A11,A12,A13')
        self.value = Axis('value', '..10')
        self.collection = AxisCollection((self.lipro, self.sex, self.age))

    def test_init_from_group(self):
        lipro_subset = self.lipro[:'P03']
        col2 = AxisCollection((lipro_subset, self.sex))
        self.assertEqual(col2.names, ['lipro', 'sex'])
        assert_array_equal(col2.lipro.labels, ['P01', 'P02', 'P03'])
        assert_array_equal(col2.sex.labels, ['H', 'F'])

    def test_eq(self):
        col = self.collection
        self.assertEqual(col, col)
        self.assertEqual(col, AxisCollection((self.lipro, self.sex, self.age)))
        self.assertEqual(col, (self.lipro, self.sex, self.age))
        self.assertNotEqual(col, (self.lipro, self.age, self.sex))

    def test_getitem_name(self):
        col = self.collection
        self.assert_axis_eq(col['lipro'], self.lipro)
        self.assert_axis_eq(col['sex'], self.sex)
        self.assert_axis_eq(col['age'], self.age)

    def test_getitem_int(self):
        col = self.collection
        self.assert_axis_eq(col[0], self.lipro)
        self.assert_axis_eq(col[-3], self.lipro)
        self.assert_axis_eq(col[1], self.sex)
        self.assert_axis_eq(col[-2], self.sex)
        self.assert_axis_eq(col[2], self.age)
        self.assert_axis_eq(col[-1], self.age)

    def test_getitem_slice(self):
        col = self.collection[:2]
        self.assertEqual(len(col), 2)
        self.assert_axis_eq(col[0], self.lipro)
        self.assert_axis_eq(col[1], self.sex)

    def test_setitem_name(self):
        col = self.collection[:]
        # replace an axis with one with another name
        col['lipro'] = self.geo
        self.assertEqual(len(col), 3)
        self.assertEqual(col, [self.geo, self.sex, self.age])
        # replace an axis with one with the same name
        col['sex'] = self.sex2
        self.assertEqual(col, [self.geo, self.sex2, self.age])
        col['geo'] = self.lipro
        self.assertEqual(col, [self.lipro, self.sex2, self.age])
        col['age'] = self.geo
        self.assertEqual(col, [self.lipro, self.sex2, self.geo])
        col['sex'] = self.sex
        col['geo'] = self.age
        self.assertEqual(col, self.collection)

    def test_setitem_int(self):
        col = self.collection[:]
        col[1] = self.geo
        self.assertEqual(len(col), 3)
        self.assertEqual(col, [self.lipro, self.geo, self.age])
        col[2] = self.sex
        self.assertEqual(col, [self.lipro, self.geo, self.sex])
        col[-1] = self.age
        self.assertEqual(col, [self.lipro, self.geo, self.age])

    def test_setitem_list_replace(self):
        col = self.collection[:]
        col[['lipro', 'age']] = [self.geo, self.lipro]
        self.assertEqual(col, [self.geo, self.sex, self.lipro])

    def test_setitem_slice_replace(self):
        col = self.collection[:]
        # replace by list
        col[1:] = [self.geo, self.sex]
        self.assertEqual(col, [self.lipro, self.geo, self.sex])
        # replace by collection
        col[1:] = self.collection[1:]
        self.assertEqual(col, self.collection)

    def test_setitem_slice_insert(self):
        col = self.collection[:]
        col[1:1] = [self.geo]
        self.assertEqual(col, [self.lipro, self.geo, self.sex, self.age])

    def test_setitem_slice_delete(self):
        col = self.collection[:]
        col[1:2] = []
        self.assertEqual(col, [self.lipro, self.age])
        col[0:1] = []
        self.assertEqual(col, [self.age])

    def assert_axis_eq(self, axis1, axis2):
        self.assertTrue(axis1.equals(axis2))

    def test_delitem(self):
        col = self.collection[:]
        self.assertEqual(len(col), 3)
        del col[0]
        self.assertEqual(len(col), 2)
        self.assert_axis_eq(col[0], self.sex)
        self.assert_axis_eq(col[1], self.age)
        del col['age']
        self.assertEqual(len(col), 1)
        self.assert_axis_eq(col[0], self.sex)
        del col[self.sex]
        self.assertEqual(len(col), 0)

    def test_delitem_slice(self):
        col = self.collection[:]
        self.assertEqual(len(col), 3)
        del col[0:2]
        self.assertEqual(len(col), 1)
        self.assertEqual(col, [self.age])
        del col[:]
        self.assertEqual(len(col), 0)

    def test_pop(self):
        col = self.collection[:]
        lipro, sex, age = col
        self.assertEqual(len(col), 3)
        self.assertIs(col.pop(), age)
        self.assertEqual(len(col), 2)
        self.assertIs(col[0], lipro)
        self.assertIs(col[1], sex)
        self.assertIs(col.pop(), sex)
        self.assertEqual(len(col), 1)
        self.assertIs(col[0], lipro)
        self.assertIs(col.pop(), lipro)
        self.assertEqual(len(col), 0)

    def test_replace(self):
        col = self.collection[:]
        newcol = col.replace('sex', self.geo)
        # original collection is not modified
        self.assertEqual(col, self.collection)
        self.assertEqual(len(newcol), 3)
        self.assertEqual(newcol.names, ['lipro', 'geo', 'age'])
        newcol = newcol.replace(self.geo, self.sex)
        self.assertEqual(len(newcol), 3)
        self.assertEqual(newcol.names, ['lipro', 'sex', 'age'])

    def test_replace_multiple(self):
        col = self.collection.replace(['lipro', 'age'], [self.geo, self.lipro])
        self.assertEqual(col, [self.geo, self.sex, self.lipro])

    # TODO: add contains_test (using both axis name and axis objects)
    def test_get(self):
        col = self.collection
        self.assert_axis_eq(col.get('lipro'), self.lipro)
        self.assertIsNone(col.get('nonexisting'))
        self.assertIs(col.get('nonexisting', self.value), self.value)

    def test_keys(self):
        self.assertEqual(self.collection.keys(), ['lipro', 'sex', 'age'])

    def test_getattr(self):
        col = self.collection
        self.assert_axis_eq(col.lipro, self.lipro)
        self.assert_axis_eq(col.sex, self.sex)
        self.assert_axis_eq(col.age, self.age)

    def test_append(self):
        col = self.collection
        geo = Axis('geo', 'A11,A12,A13')
        col.append(geo)
        self.assertEqual(col, [self.lipro, self.sex, self.age, geo])

    def test_extend(self):
        col = self.collection
        col.extend([self.geo, self.value])
        self.assertEqual(col,
                         [self.lipro, self.sex, self.age, self.geo, self.value])

    def test_insert(self):
        col = self.collection
        col.insert(1, self.geo)
        self.assertEqual(col, [self.lipro, self.geo, self.sex, self.age])

    def test_add(self):
        col = self.collection.copy()
        lipro, sex, age = self.lipro, self.sex, self.age
        geo, value = self.geo, self.value

        # 1) list
        # a) no dupe
        new = col + [self.geo, value]
        self.assertEqual(new, [lipro, sex, age, geo, value])
        # check the original has not been modified
        self.assertEqual(col, self.collection)

        # b) with compatible dupe
        # the "new" age axis is ignored (because it is compatible)
        new = col + [Axis('geo', 'A11,A12,A13'), Axis('age', '..7')]
        self.assertEqual(new, [lipro, sex, age, geo])

        # c) with incompatible dupe
        # XXX: the "new" age axis is ignored. We might want to ignore it if it
        #  is the same but raise an exception if it is different
        with self.assertRaises(ValueError):
            col + [Axis('geo', 'A11,A12,A13'), Axis('age', '..6')]

        # 2) other AxisCollection
        new = col + AxisCollection([geo, value])
        self.assertEqual(new, [lipro, sex, age, geo, value])

    def test_combine(self):
        col = self.collection.copy()
        lipro, sex, age = self.lipro, self.sex, self.age
        res = col.combine_axes((lipro, sex))
        self.assertEqual(res.names, ['lipro_sex', 'age'])
        self.assertEqual(res.size, col.size)
        self.assertEqual(res.shape, (4 * 2, 8))
        print(res.info)
        assert_array_equal(res.lipro_sex.labels[0], 'P01_H')
        res = col.combine_axes((lipro, age))
        self.assertEqual(res.names, ['lipro_age', 'sex'])
        self.assertEqual(res.size, col.size)
        self.assertEqual(res.shape, (4 * 8, 2))
        assert_array_equal(res.lipro_age.labels[0], 'P01_0')
        res = col.combine_axes((sex, age))
        self.assertEqual(res.names, ['lipro', 'sex_age'])
        self.assertEqual(res.size, col.size)
        self.assertEqual(res.shape, (4, 2 * 8))
        assert_array_equal(res.sex_age.labels[0], 'H_0')

    def test_info(self):
        expected = """\
4 x 2 x 8
 lipro [4]: 'P01' 'P02' 'P03' 'P04'
 sex [2]: 'H' 'F'
 age [8]: 0 1 2 ... 5 6 7"""
        self.assertEqual(self.collection.info, expected)

    def test_str(self):
        self.assertEqual(str(self.collection), "{lipro, sex, age}")

    def test_repr(self):
        self.assertEqual(repr(self.collection), """AxisCollection([
    Axis('lipro', ['P01', 'P02', 'P03', 'P04']),
    Axis('sex', ['H', 'F']),
    Axis('age', [0, 1, 2, 3, 4, 5, 6, 7])
])""")


class TestLArray(TestCase):
    def setUp(self):
        self.title = 'test array'
        self.lipro = Axis('lipro', ['P%02d' % i for i in range(1, 16)])
        self.age = Axis('age', range(116))
        self.sex = Axis('sex', 'H,F')

        vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,' \
              'A43,A44,A45,A46,A71,A72,A73'
        wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,' \
              'A83,A84,A85,A91,A92,A93'
        bru = 'A21'
        self.vla_str = vla
        self.wal_str = wal
        # string without commas
        self.bru_str = bru
        # list of strings
        self.belgium = union(vla, wal, bru)

        # belgium = vla + wal + bru # equivalent
        # wal_bru = belgium - vla
        # wal_bru = wal + bru # equivalent

        self.geo = Axis('geo', self.belgium)

        self.array = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
                                                 .astype(float)
        self.larray = LArray(self.array,
                             axes=(self.age, self.geo, self.sex, self.lipro),
                             title=self.title)

        self.small_title = 'small test array'
        self.small_data = np.arange(30).reshape(2, 15)
        self.small = LArray(self.small_data, axes=(self.sex, self.lipro),
                            title=self.small_title)

    def test_zeros(self):
        la = zeros((self.geo, self.age))
        self.assertEqual(la.shape, (44, 116))
        assert_array_equal(la, np.zeros((44, 116)))

    def test_zeros_like(self):
        la = zeros_like(self.larray)
        self.assertEqual(la.shape, (116, 44, 2, 15))
        assert_array_equal(la, np.zeros((116, 44, 2, 15)))

    def test_bool(self):
        a = ones([2])
        # ValueError: The truth value of an array with more than one element
        #             is ambiguous. Use a.any() or a.all()
        self.assertRaises(ValueError, bool, a)

        a = ones([1])
        self.assertTrue(bool(a))

        a = zeros([1])
        self.assertFalse(bool(a))

        a = LArray(np.array(2), [])
        self.assertTrue(bool(a))

        a = LArray(np.array(0), [])
        self.assertFalse(bool(a))

    def test_iter(self):
        array = self.small
        l = list(array)
        assert_array_equal(l[0], array['H'])
        assert_array_equal(l[1], array['F'])

    def test_rename(self):
        la = self.larray
        new = la.rename('sex', 'gender')
        # old array axes names not modified
        self.assertEqual(la.axes.names, ['age', 'geo', 'sex', 'lipro'])
        self.assertEqual(new.axes.names, ['age', 'geo', 'gender', 'lipro'])

        new = la.rename(self.sex, 'gender')
        # old array axes names not modified
        self.assertEqual(la.axes.names, ['age', 'geo', 'sex', 'lipro'])
        self.assertEqual(new.axes.names, ['age', 'geo', 'gender', 'lipro'])

    def test_info(self):
        expected = """\
test array
116 x 44 x 2 x 15
 age [116]: 0 1 2 ... 113 114 115
 geo [44]: 'A11' 'A12' 'A13' ... 'A92' 'A93' 'A21'
 sex [2]: 'H' 'F'
 lipro [15]: 'P01' 'P02' 'P03' ... 'P13' 'P14' 'P15'"""
        self.assertEqual(self.larray.info, expected)

    def test_str(self):
        lipro = self.lipro
        lipro3 = lipro['P01:P03']
        sex = self.sex

        # zero dimension / scalar
        self.assertEqual(str(self.small[lipro['P01'], sex['F']]), "15")

        # empty / len 0 first dimension
        self.assertEqual(str(self.small[sex[[]]]), "LArray([])")

        # one dimension
        self.assertEqual(str(self.small[lipro3, sex['H']]), """\
lipro | P01 | P02 | P03
      |   0 |   1 |   2""")
        # two dimensions
        self.assertEqual(str(self.small.filter(lipro=lipro3)), """\
sex\lipro | P01 | P02 | P03
        H |   0 |   1 |   2
        F |  15 |  16 |  17""")
        # four dimensions (too many rows)
        self.assertEqual(str(self.larray.filter(lipro=lipro3)), """\
age | geo | sex\lipro |      P01 |      P02 |      P03
  0 | A11 |         H |      0.0 |      1.0 |      2.0
  0 | A11 |         F |     15.0 |     16.0 |     17.0
  0 | A12 |         H |     30.0 |     31.0 |     32.0
  0 | A12 |         F |     45.0 |     46.0 |     47.0
  0 | A13 |         H |     60.0 |     61.0 |     62.0
... | ... |       ... |      ... |      ... |      ...
115 | A92 |         F | 153045.0 | 153046.0 | 153047.0
115 | A93 |         H | 153060.0 | 153061.0 | 153062.0
115 | A93 |         F | 153075.0 | 153076.0 | 153077.0
115 | A21 |         H | 153090.0 | 153091.0 | 153092.0
115 | A21 |         F | 153105.0 | 153106.0 | 153107.0""")
        # too many columns
        self.assertEqual(str(self.larray['P01', 'A11', 'H']), """\
age |   0 |      1 |      2 |      3 |      4 |      5 |      6 |      7 | ... \
|      107 |      108 |      109 |      110 |      111 |      112 |      113 |\
      114 |      115
    | 0.0 | 1320.0 | 2640.0 | 3960.0 | 5280.0 | 6600.0 | 7920.0 | 9240.0 | ... \
| 141240.0 | 142560.0 | 143880.0 | 145200.0 | 146520.0 | 147840.0 \
| 149160.0 | 150480.0 | 151800.0""")

    def test_getitem(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = age[[1, 5, 9]]
        lipro159 = lipro['P01,P05,P09']

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis('age', [1, 5, 9])))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # LGroup at "incorrect" place
        assert_array_equal(la[lipro159], raw[..., [0, 4, 8]])

        # multiple LGroup key (in "incorrect" order)
        res = la[lipro159, age159]
        self.assertEqual(res.axes.names, ['age', 'geo', 'sex', 'lipro'])
        assert_array_equal(res,
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # LGroup key and scalar
        res = la[lipro159, 5]
        self.assertEqual(res.axes.names, ['geo', 'sex', 'lipro'])
        assert_array_equal(res, raw[..., [0, 4, 8]][5])

        # mixed LGroup/positional key
        assert_array_equal(la[[1, 5, 9], lipro159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # single None slice
        assert_array_equal(la[:], raw)

        # only Ellipsis
        assert_array_equal(la[...], raw)

        # Ellipsis and VG
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaises(ValueError):
            la[age[1, 2], age[3, 4]]

        # key with invalid axis
        bad = Axis('bad', 3)
        with self.assertRaises(KeyError):
            la[bad[1, 2], age[3, 4]]

    def test_getitem_abstract_axes(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = x.age[1, 5, 9]
        lipro159 = x.lipro['P01,P05,P09']

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis('age', [1, 5, 9])))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # LGroup at "incorrect" place
        assert_array_equal(la[lipro159], raw[..., [0, 4, 8]])

        # multiple LGroup key (in "incorrect" order)
        assert_array_equal(la[lipro159, age159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # mixed LGroup/positional key
        assert_array_equal(la[[1, 5, 9], lipro159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # single None slice
        assert_array_equal(la[:], raw)

        # only Ellipsis
        assert_array_equal(la[...], raw)

        # Ellipsis and VG
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaises(ValueError):
            la[x.age[1, 2], x.age[3]]

        # key with invalid axis
        with self.assertRaises(KeyError):
            la[x.bad[1, 2], x.age[3, 4]]

    def test_getitem_anonymous_axes(self):
        la = ndrange((3, 4))
        raw = la.data
        assert_array_equal(la[x[0][1:]], raw[1:])
        assert_array_equal(la[x[1][2:]], raw[:, 2:])
        assert_array_equal(la[x[0][2:], x[1][1:]], raw[2:, 1:])
        assert_array_equal(la.i[2:, 1:], raw[2:, 1:])

    def test_getitem_guess_axis(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes

        # key at "correct" place
        assert_array_equal(la[[1, 5, 9]], raw[[1, 5, 9]])
        subset = la[[1, 5, 9]]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis('age', [1, 5, 9])))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # key at "incorrect" place
        assert_array_equal(la['P01,P05,P09'], raw[..., [0, 4, 8]])
        assert_array_equal(la[['P01', 'P05', 'P09']], raw[..., [0, 4, 8]])

        # multiple keys (in "incorrect" order)
        assert_array_equal(la['P01,P05,P09', [1, 5, 9]],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # mixed LGroup/key
        assert_array_equal(la[lipro['P01,P05,P09'], [1, 5, 9]],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # single None slice
        assert_array_equal(la[:], raw)

        # only Ellipsis
        assert_array_equal(la[...], raw)

        # Ellipsis and VG
        assert_array_equal(la[..., 'P01,P05,P09'], raw[..., [0, 4, 8]])
        assert_array_equal(la[..., ['P01', 'P05', 'P09']],
                           raw[..., [0, 4, 8]])

        # LGroup without axis (which also needs to be guessed)
        g = LGroup(['P01', 'P05', 'P09'])
        assert_array_equal(la[g], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaisesRegexp(ValueError, "key has several values for axis: age"):
            la[[1, 2], [3, 4]]

        # key with invalid label (ie label not found on any axis)
        with self.assertRaisesRegexp(ValueError, "999 is not a valid label for any axis"):
            la[[1, 2], 999]

        # key with invalid label list (ie list of labels not found on any axis)
        with self.assertRaisesRegexp(ValueError, "\[998, 999\] is not a valid label for any axis"):
            la[[1, 2], [998, 999]]

        # key with partial invalid list (ie list containing a label not found
        # on any axis)
        # FIXME: the message should be the same as for 999, 4 (ie it should NOT mention age).
        with self.assertRaisesRegexp(ValueError, "age\[3, 999\] is not a valid label for any axis"):
            la[[1, 2], [3, 999]]

        with self.assertRaisesRegexp(ValueError, "\[999, 4\] is not a valid label for any axis"):
            la[[1, 2], [999, 4]]

        # ambiguous key
        a = ndrange((sex, sex.rename('sex2')))
        with self.assertRaisesRegexp(ValueError, "F is ambiguous \(valid in sex, sex2\)"):
            a['F']

    def test_getitem_positional_group(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = age.i[1, 5, 9]
        lipro159 = lipro.i[0, 4, 8]

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis('age', [1, 5, 9])))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # LGroup at "incorrect" place
        assert_array_equal(la[lipro159], raw[..., [0, 4, 8]])

        # multiple LGroup key (in "incorrect" order)
        assert_array_equal(la[lipro159, age159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # mixed LGroup/positional key
        assert_array_equal(la[[1, 5, 9], lipro159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # single None slice
        assert_array_equal(la[:], raw)

        # only Ellipsis
        assert_array_equal(la[...], raw)

        # Ellipsis and VG
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaisesRegexp(ValueError,
                                     "key has several values for axis: age"):
            la[age.i[1, 2], age.i[3, 4]]

    def test_getitem_abstract_positional(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = x.age.i[1, 5, 9]
        lipro159 = x.lipro.i[0, 4, 8]

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis('age', [1, 5, 9])))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # LGroup at "incorrect" place
        assert_array_equal(la[lipro159], raw[..., [0, 4, 8]])

        # multiple LGroup key (in "incorrect" order)
        assert_array_equal(la[lipro159, age159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # mixed LGroup/positional key
        assert_array_equal(la[[1, 5, 9], lipro159],
                           raw[[1, 5, 9]][..., [0, 4, 8]])

        # single None slice
        assert_array_equal(la[:], raw)

        # only Ellipsis
        assert_array_equal(la[...], raw)

        # Ellipsis and VG
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaisesRegexp(ValueError,
                                     "key has several values for axis: age"):
            la[x.age.i[2, 3], x.age.i[1, 5]]

    def test_getitem_bool_larray_key(self):
        raw = self.array
        la = self.larray

        # all dimensions
        res = la[la < 5]
        self.assertTrue(isinstance(res, LArray))
        self.assertEqual(res.ndim, 1)
        assert_array_equal(res, raw[raw < 5])

        # missing dimension
        res = la[la['H'] % 5 == 0]
        self.assertTrue(isinstance(res, LArray))
        self.assertEqual(res.ndim, 2)
        self.assertEqual(res.shape, (116 * 44 * 15 / 5, 2))
        raw_key = raw[:, :, 0, :] % 5 == 0
        raw_d1, raw_d2, raw_d4 = raw_key.nonzero()
        assert_array_equal(res, raw[raw_d1, raw_d2, :, raw_d4])

    def test_getitem_bool_ndarray_key(self):
        raw = self.array
        la = self.larray

        res = la[raw < 5]
        self.assertTrue(isinstance(res, LArray))
        self.assertEqual(res.ndim, 1)
        assert_array_equal(res, raw[raw < 5])

    def test_getitem_bool_anonymous_axes(self):
        a = ndrange((2, 3, 4, 5))
        mask = ones(a.axes[1, 3], dtype=bool)
        res = a[mask]
        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape, (15, 2, 4))

        # XXX: we might want to transpose the result to always move
        # combined axes to the front
        a = ndrange((2, 3, 4, 5))
        mask = ones(a.axes[1, 2], dtype=bool)
        res = a[mask]
        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape, (2, 12, 5))

    def test_getitem_int_larray_lgroup_key(self):
        # e axis go from 0 to 3
        arr = ndrange((2, 2, 4)).rename(0, 'c').rename(1, 'd').rename(2, 'e')
        # key values go from 0 to 3
        key = ndrange((2, 2)).rename(0, 'a').rename(1, 'b')
        # this replaces 'e' axis by 'a' and 'b' axes
        res = arr[x.e[key]]
        self.assertEqual(res.shape, (2, 2, 2, 2))
        self.assertEqual(res.axes.names, ['c', 'd', 'a', 'b'])

    def test_getitem_single_larray_key_guess(self):
        a = Axis('a', ['a1', 'a2'])
        b = Axis('b', ['b1', 'b2', 'b3'])
        c = Axis('c', ['c1', 'c2', 'c3', 'c4'])

        # 1) key with extra axis
        arr = ndrange([a, b])
        # replace the values_axis by the extra axis
        key = LArray(['a1', 'a2', 'a2', 'a1'], [c])
        self.assertEqual(arr[key].axes, [c, b])

        # 2) key with the values axis (the one being replaced)
        arr = ndrange([a, b])
        key = LArray(['b2', 'b1', 'b3'], [b])
        # axis stays the same but data should be flipped/shuffled
        self.assertEqual(arr[key].axes, [a, b])

        # 2bis) key with part of the values axis (the one being replaced)
        arr = ndrange([a, b])
        b_bis = Axis('b', ['b1', 'b2'])
        key = LArray(['b3', 'b2'], [b_bis])
        self.assertEqual(arr[key].axes, [a, b_bis])

        # 3) key with another existing axis (not the values axis)
    #     arr = ndrange([a, b])
    #     key = LArray(['a1', 'a2', 'a1'], [b])
    #     # we need points indexing
    #     # equivalent to
    #     # tmp = arr[x.a['a1', 'a2', 'a1'] ^ x.b['b1', 'b2', 'b3']]
    #     # res = tmp.set_axes([b])
    #     # both the values axis and the other existing axis
    #     self.assertEqual(arr[key].axes, [b])
    #
    #     # 3bis) key with part of another existing axis (not the values axis)
    #     arr = ndrange([a, b])
    #     b_bis = Axis('b', ['b1', 'b2'])
    #     key = LArray(['a2', 'a1'], [b_bis])
    #     # we need points indexing
    #     # equivalent to
    #     # tmp = arr[x.a['a2', 'a1'] ^ x.b['b1', 'b2']]
    #     # res = tmp.set_axes([b_bis])
    #     self.assertEqual(arr[key].axes, [b_bis])
    #
    #     # 4) key has both the values axis and another existing axis
    #     # a\b b1 b2 b3
    #     #  a1  0  1  2
    #     #  a2  3  4  5
    #     arr = ndrange([a, b])
    #     # a\b b1 b2 b3
    #     #  a1 a1 a2 a1
    #     #  a2 a2 a1 a2
    #     key = LArray([['a1', 'a2', 'a1'],
    #                   ['a2', 'a1', 'a2']],
    #                  [a, b])
    #     # a\b b1 b2 b3
    #     #  a1  0  4  2
    #     #  a2  3  1  5
    #     # we need to produce the following keys for numpy:
    #     # k0:
    #     # [[0, 1, 0],
    #     #  [1, 0, 1]]
    #     # TODO: [0, 1, 2] is enough in this case (thanks to broadcasting)
    #     #       because in numpy missing dimensions are filled by adding
    #     #       length 1 dimensions to the left. Ie it works because b is the
    #     #       last dimension.
    #     # k1:
    #     # [[0, 1, 2],
    #     #  [0, 1, 2]]
    #
    #     # we need points indexing
    #     # equivalent to
    #     # tmp = arr[x.a[['a1', 'a2', 'a1'],
    #     #               ['a2', 'a1', 'a2']] ^ x.b['b1', 'b2', 'b3']]
    #     # res = tmp.set_axes([a, b])
    #     # this is kinda ugly because a ND x.a has implicit (positional dimension
    #     res = arr[key]
    #     self.assertEqual(res.axes, [a, b])
    #     assert_array_equal(res, [[0, 4, 2],
    #                              [3, 1, 5]])
    #
    #     # 5) key has both the values axis and an extra axis
    #     arr = ndrange([a, b])
    #     key = LArray([['a1', 'a2', 'a2', 'a1'], ['a2', 'a1', 'a1', 'a2']],
    #                  [a, c])
    #     self.assertEqual(arr[key].axes, [a, c])
    #
    #     # 6) key has both another existing axis (not values) and an extra axis
    #     arr = ndrange([a, b])
    #     key = LArray([['b1', 'b2', 'b1', 'b2'], ['b3', 'b4', 'b3', 'b4']],
    #                  [a, c])
    #     self.assertEqual(arr[key].axes, [a, c])
    #
    #     # 7) key has the values axis, another existing axis and an extra axis
    #     arr = ndrange([a, b])
    #     key = LArray([[['a1', 'a2', 'a1', 'a2'],
    #                    ['a2', 'a1', 'a2', 'a1'],
    #                    ['a1', 'a2', 'a1', 'a2']],
    #
    #                   [['a1', 'a2', 'a2', 'a1'],
    #                    ['a2', 'a2', 'a2', 'a2'],
    #                    ['a1', 'a2', 'a2', 'a1']]],
    #                  [a, b, c])
    #     self.assertEqual(arr[key].axes, [a, c])
    #
    # def test_getitem_multiple_larray_key_guess(self):
    #     a = Axis('a', ['a1', 'a2'])
    #     b = Axis('b', ['b1', 'b2', 'b3'])
    #     c = Axis('c', ['c1', 'c2', 'c3', 'c4'])
    #     d = Axis('d', ['d1', 'd2', 'd3', 'd4', 'd5'])
    #     e = Axis('e', ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])
    #
    #     # 1) key with extra disjoint axes
    #     arr = ndrange([a, b])
    #     k1 = LArray(['a1', 'a2', 'a2', 'a1'], [c])
    #     k2 = LArray(['b1', 'b2', 'b3', 'b1'], [d])
    #     self.assertEqual(arr[k1, k2].axes, [c, d])
    #
    #     # 2) key with common extra axes
    #     arr = ndrange([a, b])
    #     k1 = LArray(['a1', 'a2', 'a2', 'a1'], [c, d])
    #     k2 = LArray(['b1', 'b2', 'b3', 'b1'], [c, e])
    #     # TODO: not sure what *should* happen in this case!
    #     self.assertEqual(arr[k1, k2].axes, [c, d, e])

    def test_getitem_ndarray_key_guess(self):
        raw = self.array
        la = self.larray
        keys = ['P04', 'P01', 'P03', 'P02']
        key = np.array(keys)
        res = la[key]
        self.assertTrue(isinstance(res, LArray))
        self.assertEqual(res.axes,
                         la.axes.replace(x.lipro, Axis('lipro', keys)))
        assert_array_equal(res, raw[:, :, :, [3, 0, 2, 1]])

    def test_getitem_int_larray_key_guess(self):
        a = Axis('a', [0, 1])
        b = Axis('b', [2, 3])
        c = Axis('c', [4, 5])
        d = Axis('d', [6, 7])
        e = Axis('e', [8, 9, 10, 11])

        arr = ndrange([c, d, e])
        key = LArray([[8, 9], [10, 11]], [a, b])
        self.assertEqual(arr[key].axes, [c, d, a, b])

    def test_getitem_int_ndarray_key_guess(self):
        c = Axis('c', [4, 5])
        d = Axis('d', [6, 7])
        e = Axis('e', [8, 9, 10, 11])

        arr = ndrange([c, d, e])
        # ND keys do not work yet
        # key = np.array([[8, 11], [10, 9]])
        key = np.array([8, 11, 10])
        res = arr[key]
        self.assertEqual(res.axes, [c, d, Axis('e', [8, 11, 10])])

    def test_positional_indexer_getitem(self):
        raw = self.array
        la = self.larray
        for key in [0, (0, 5, 1, 2), (slice(None), 5, 1), (0, 5), [1, 0],
                    ([1, 0], 5)]:
            assert_array_equal(la.i[key], raw[key])
        assert_array_equal(la.i[[1, 0], [5, 4]], raw[np.ix_([1, 0], [5, 4])])
        with self.assertRaises(IndexError):
            la.i[0, 0, 0, 0, 0]

    def test_positional_indexer_setitem(self):
        for key in [0, (0, 2, 1, 2), (slice(None), 2, 1), (0, 2), [1, 0],
                    ([1, 0], 2)]:
            la = self.larray.copy()
            raw = self.array.copy()
            la.i[key] = 42
            raw[key] = 42
            assert_array_equal(la, raw)

        la = self.larray.copy()
        raw = self.array.copy()
        la.i[[1, 0], [5, 4]] = 42
        raw[np.ix_([1, 0], [5, 4])] = 42
        assert_array_equal(la, raw)

    def test_setitem_larray(self):
        """
        tests LArray.__setitem__(key, value) where value is an LArray
        """
        age, geo, sex, lipro = self.larray.axes

        # 1) using a LGroup key
        ages1_5_9 = age[[1, 5, 9]]

        # a) value has exactly the same shape as the target slice
        la = self.larray.copy()
        raw = self.array.copy()

        la[ages1_5_9] = la[ages1_5_9] + 25.0
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
        assert_array_equal(la, raw)

        # b) value has exactly the same shape but VG at a "wrong" positions
        la = self.larray.copy()
        la[geo[:], ages1_5_9] = la[ages1_5_9] + 25.0
        # same raw as previous test
        assert_array_equal(la, raw)

        # c) value has an extra length-1 axis
        la = self.larray.copy()
        raw = self.array.copy()

        raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
        fake_axis = Axis('fake', ['label'])
        age_axis = la[ages1_5_9].axes.age
        value = LArray(raw_value, axes=(age_axis, fake_axis, self.geo, self.sex,
                                        self.lipro))
        la[ages1_5_9] = value
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 26.0
        assert_array_equal(la, raw)

        # d) value has the same axes than target but one has length 1
        # la = self.larray.copy()
        # raw = self.array.copy()
        # raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
        # la[ages1_5_9] = la[ages1_5_9].sum(geo=(geo.all(),))
        # assert_array_equal(la, raw)

        # e) value has a missing dimension
        la = self.larray.copy()
        raw = self.array.copy()
        la[ages1_5_9] = la[ages1_5_9].sum(geo)
        raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
        assert_array_equal(la, raw)

        # 2) using a LGroup and scalar key (triggers advanced indexing/cross)

        # a) value has exactly the same shape as the target slice
        la = self.larray.copy()
        raw = self.array.copy()

        # using 1, 5, 8 and not 9 so that the list is not collapsed to slice
        value = la[age[1, 5, 8], sex['H']] + 25.0
        la[age[1, 5, 8], sex['H']] = value
        raw[[1, 5, 8], :, 0] = raw[[1, 5, 8], :, 0] + 25.0
        assert_array_equal(la, raw)

        # 3) using a string key
        la = self.larray.copy()
        raw = self.array.copy()
        la[[1, 5, 9]] = la[[2, 7, 3]] + 27.0
        raw[[1, 5, 9]] = raw[[2, 7, 3]] + 27.0
        assert_array_equal(la, raw)

        # 4) using ellipsis keys
        # only Ellipsis
        la = self.larray.copy()
        la[...] = 0
        assert_array_equal(la, np.zeros_like(raw))

        # Ellipsis and VG
        la = self.larray.copy()
        raw = self.array.copy()
        la[..., lipro['P01,P05,P09']] = 0
        raw[..., [0, 4, 8]] = 0
        assert_array_equal(la, raw)

        # 5) using a single slice(None) key
        la = self.larray.copy()
        la[:] = 0
        assert_array_equal(la, np.zeros_like(raw))

    def test_setitem_ndarray(self):
        """
        tests LArray.__setitem__(key, value) where value is a raw ndarray.
        In that case, value.shape is more restricted as we rely on
        numpy broadcasting.
        """
        # a) value has exactly the same shape as the target slice
        la = self.larray.copy()
        raw = self.array.copy()
        value = raw[[1, 5, 9]] + 25.0
        la[[1, 5, 9]] = value
        raw[[1, 5, 9]] = value
        assert_array_equal(la, raw)

        # b) value has the same axes than target but one has length 1
        la = self.larray.copy()
        raw = self.array.copy()
        value = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
        la[[1, 5, 9]] = value
        raw[[1, 5, 9]] = value
        assert_array_equal(la, raw)

    def test_setitem_scalar(self):
        """
        tests LArray.__setitem__(key, value) where value is a scalar
        """
        # a) list key (one dimension)
        la = self.larray.copy()
        raw = self.array.copy()
        la[[1, 5, 9]] = 42
        raw[[1, 5, 9]] = 42
        assert_array_equal(la, raw)

        # b) full scalar key (ie set one cell)
        la = self.larray.copy()
        raw = self.array.copy()
        la[0, 'P02', 'A12', 'H'] = 42
        raw[0, 1, 0, 1] = 42
        assert_array_equal(la, raw)

    def test_setitem_bool_array_key(self):
        # XXX: this test is awfully slow (more than 1s)
        age, geo, sex, lipro = self.larray.axes

        # LArray key
        # a1) same shape, same order
        la = self.larray.copy()
        raw = self.array.copy()
        la[la < 5] = 0
        raw[raw < 5] = 0
        assert_array_equal(la, raw)

        # a2) same shape, different order
        la = self.larray.copy()
        raw = self.array.copy()
        key = (la < 5).T
        la[key] = 0
        raw[raw < 5] = 0
        assert_array_equal(la, raw)

        # b) numpy-broadcastable shape
        # la = self.larray.copy()
        # raw = self.array.copy()
        # key = la[sex['F,']] < 5
        # self.assertEqual(key.ndim, 4)
        # la[key] = 0
        # raw[raw[:, :, [1]] < 5] = 0
        # assert_array_equal(la, raw)

        # c) LArray-broadcastable shape (missing axis)
        la = self.larray.copy()
        raw = self.array.copy()
        key = la[sex['H']] < 5
        self.assertEqual(key.ndim, 3)
        la[key] = 0

        raw_key = raw[:, :, 0, :] < 5
        raw_d1, raw_d2, raw_d4 = raw_key.nonzero()
        raw[raw_d1, raw_d2, :, raw_d4] = 0
        assert_array_equal(la, raw)

        # ndarray key
        la = self.larray.copy()
        raw = self.array.copy()
        la[raw < 5] = 0
        raw[raw < 5] = 0
        assert_array_equal(la, raw)

        # d) LArray with extra axes
        la = self.larray.copy()
        raw = self.array.copy()
        key = (la < 5).expand([Axis('extra', 2)])
        self.assertEqual(key.ndim, 5)
        # TODO: make this work
        with self.assertRaises(ValueError):
            la[key] = 0

    def test_set(self):
        age, geo, sex, lipro = self.larray.axes

        # 1) using a LGroup key
        ages1_5_9 = age.group([1, 5, 9])

        # a) value has exactly the same shape as the target slice
        la = self.larray.copy()
        raw = self.array.copy()

        la.set(la[ages1_5_9] + 25.0, age=ages1_5_9)
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
        assert_array_equal(la, raw)

        # b) same size but a different shape (extra length-1 axis)
        la = self.larray.copy()
        raw = self.array.copy()

        raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
        fake_axis = Axis('fake', ['label'])
        age_axis = la[ages1_5_9].axes.age
        value = LArray(raw_value, axes=(age_axis, fake_axis, self.geo, self.sex,
                                        self.lipro))
        la.set(value, age=ages1_5_9)
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 26.0
        assert_array_equal(la, raw)

        # dimension of length 1
        # la = self.larray.copy()
        # raw = self.array.copy()
        # raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
        # la.set(la[ages1_5_9].sum(geo=(geo.all(),)), age=ages1_5_9)
        # assert_array_equal(la, raw)

        # c) missing dimension
        la = self.larray.copy()
        raw = self.array.copy()
        la.set(la[ages1_5_9].sum(geo), age=ages1_5_9)
        raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
        assert_array_equal(la, raw)

        # 2) using a raw key
        la = self.larray.copy()
        raw = self.array.copy()
        la.set(la[[2, 7, 3]] + 27.0, age=[1, 5, 9])
        raw[[1, 5, 9]] = raw[[2, 7, 3]] + 27.0
        assert_array_equal(la, raw)

    def test_filter(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        ages1_5_9 = age.group(1, 5, 9)
        ages11 = age.group(11)

        # with LGroup
        self.assertEqual(la.filter(age=ages1_5_9).shape, (3, 44, 2, 15))

        # FIXME: this should raise a comprehensible error!
        # self.assertEqual(la.filter(age=[ages1_5_9]).shape, (3, 44, 2, 15))

        # VG with 1 value => collapse
        self.assertEqual(la.filter(age=ages11).shape, (44, 2, 15))

        # VG with a list of 1 value => do not collapse
        self.assertEqual(la.filter(age=age.group([11])).shape, (1, 44, 2, 15))

        # VG with a list of 1 value defined as a string => do not collapse
        self.assertEqual(la.filter(lipro=lipro.group('P01,')).shape,
                         (116, 44, 2, 1))

        # VG with 1 value
        # XXX: this does not work. Do we want to make this work?
        # filtered = la.filter(age=(ages11,))
        # self.assertEqual(filtered.shape, (1, 44, 2, 15))

        # list
        self.assertEqual(la.filter(age=[1, 5, 9]).shape, (3, 44, 2, 15))

        # string
        self.assertEqual(la.filter(lipro='P01,P02').shape, (116, 44, 2, 2))

        # multiple axes at once
        self.assertEqual(la.filter(age=[1, 5, 9], lipro='P01,P02').shape,
                         (3, 44, 2, 2))

        # multiple axes one after the other
        self.assertEqual(la.filter(age=[1, 5, 9]).filter(lipro='P01,P02').shape,
                         (3, 44, 2, 2))

        # a single value for one dimension => collapse the dimension
        self.assertEqual(la.filter(sex='H').shape, (116, 44, 15))

        # but a list with a single value for one dimension => do not collapse
        self.assertEqual(la.filter(sex=['H']).shape, (116, 44, 1, 15))

        self.assertEqual(la.filter(sex='H,').shape, (116, 44, 1, 15))

        # with duplicate keys
        # XXX: do we want to support this? I don't see any value in that but
        # I might be short-sighted.
        # filtered = la.filter(lipro='P01,P02,P01')

        # XXX: we could abuse python to allow naming groups via Axis.__getitem__
        # (but I doubt it is a good idea).
        # child = age[':17', 'child']

        # slices
        # ------

        # VG slice
        self.assertEqual(la.filter(age=age[:17]).shape, (18, 44, 2, 15))
        # string slice
        self.assertEqual(la.filter(lipro=':P03').shape, (116, 44, 2, 3))
        # raw slice
        self.assertEqual(la.filter(age=slice(17)).shape, (18, 44, 2, 15))

        # filter chain with a slice
        self.assertEqual(la.filter(age=slice(17)).filter(geo='A12,A13').shape,
                         (18, 2, 2, 15))

    def test_filter_multiple_axes(self):
        la = self.larray

        # multiple values in each group
        self.assertEqual(la.filter(age=[1, 5, 9], lipro='P01,P02').shape,
                         (3, 44, 2, 2))
        # with a group of one value
        self.assertEqual(la.filter(age=[1, 5, 9], sex='H,').shape,
                         (3, 44, 1, 15))

        # with a discarded axis (there is a scalar in the key)
        self.assertEqual(la.filter(age=[1, 5, 9], sex='H').shape, (3, 44, 15))

        # with a discarded axis that is not adjacent to the ix_array axis
        # ie with a sliced axis between the scalar axis and the ix_array axis
        # since our array has a axes: age, geo, sex, lipro, any of the
        # following should be tested: age+sex / age+lipro / geo+lipro
        # additionally, if the ix_array axis was first (ie ix_array on age),
        # it worked even before the issue was fixed, since the "indexing"
        # subspace is tacked-on to the beginning (as the first dimension)
        self.assertEqual(la.filter(age=57, sex='H,F').shape,
                         (44, 2, 15))
        self.assertEqual(la.filter(age=57, lipro='P01,P05').shape,
                         (44, 2, 2))
        self.assertEqual(la.filter(geo='A57', lipro='P01,P05').shape,
                         (116, 2, 2))

    def test_sum_full_axes(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        # everything
        self.assertEqual(la.sum(), np.asarray(la).sum())

        # using axes numbers
        self.assertEqual(la.sum(axis=2).shape, (116, 44, 15))
        self.assertEqual(la.sum(axis=(0, 2)).shape, (44, 15))

        # using Axis objects
        self.assertEqual(la.sum(age).shape, (44, 2, 15))
        self.assertEqual(la.sum(age, sex).shape, (44, 15))

        # using axes names
        self.assertEqual(la.sum('age', 'sex').shape, (44, 15))

        # chained sum
        self.assertEqual(la.sum(age, sex).sum(geo).shape, (15,))
        self.assertEqual(la.sum(age, sex).sum(lipro, geo), la.sum())

        # getitem on aggregated
        aggregated = la.sum(age, sex)
        self.assertEqual(aggregated[self.vla_str].shape, (22, 15))

        # filter on aggregated
        self.assertEqual(aggregated.filter(geo=self.vla_str).shape, (22, 15))

    def test_sum_full_axes_with_nan(self):
        la = self.larray.copy()
        la['H', 'P02', 'A12', 0] = np.nan
        raw = la.data

        # everything
        self.assertEqual(la.sum(), np.nansum(raw))
        self.assertTrue(isnan(la.sum(skipna=False)))

        # using Axis objects
        assert_array_nan_equal(la.sum(x.age), np.nansum(raw, 0))
        assert_array_nan_equal(la.sum(x.age, skipna=False), raw.sum(0))

        assert_array_nan_equal(la.sum(x.age, x.sex), np.nansum(raw, (0, 2)))
        assert_array_nan_equal(la.sum(x.age, x.sex, skipna=False),
                               raw.sum((0, 2)))

    def test_sum_full_axes_keep_axes(self):
        la = self.larray
        agg = la.sum(keepaxes=True)
        self.assertEqual(agg.shape, (1, 1, 1, 1))
        for axis in agg.axes:
            self.assertEqual(axis.labels, ['sum'])

        agg = la.sum(keepaxes='total')
        self.assertEqual(agg.shape, (1, 1, 1, 1))
        for axis in agg.axes:
            self.assertEqual(axis.labels, ['total'])

    def test_mean_full_axes(self):
        la = self.larray
        raw = self.array

        self.assertEqual(la.mean(), np.mean(raw))
        assert_array_nan_equal(la.mean(x.age), np.mean(raw, 0))
        assert_array_nan_equal(la.mean(x.age, x.sex), np.mean(raw, (0, 2)))

    def test_mean_groups(self):
        # using int type to test that we get a float in return
        la = self.larray.astype(int)
        raw = self.array
        res = la.mean(x.geo['A11', 'A13', 'A24', 'A31'])
        assert_array_nan_equal(res, np.mean(raw[:, [0, 2, 4, 5]], 1))

    def test_median_full_axes(self):
        la = self.larray
        raw = self.array

        self.assertEqual(la.median(), np.median(raw))
        assert_array_nan_equal(la.median(x.age), np.median(raw, 0))
        assert_array_nan_equal(la.median(x.age, x.sex), np.median(raw, (0, 2)))

    def test_median_groups(self):
        la = self.larray
        raw = self.array

        res = la.median(x.geo['A11', 'A13', 'A24'])
        self.assertEqual(res.shape, (116, 2, 15))
        assert_array_nan_equal(res, np.median(raw[:, [0, 2, 4]], 1))

    def test_percentile_full_axes(self):
        la = self.larray
        raw = self.array

        self.assertEqual(la.percentile(10),
                         np.percentile(raw, 10))
        assert_array_nan_equal(la.percentile(10, x.age),
                               np.percentile(raw, 10, 0))
        assert_array_nan_equal(la.percentile(10, x.age, x.sex),
                               np.percentile(raw, 10, (0, 2)))

    def test_percentile_groups(self):
        la = self.larray
        raw = self.array

        res = la.percentile(10, x.geo['A11', 'A13', 'A24'])
        assert_array_nan_equal(res, np.percentile(raw[:, [0, 2, 4]], 10, 1))

    def test_cumsum(self):
        la = self.larray

        # using Axis objects
        assert_array_equal(la.cumsum(x.age), self.array.cumsum(0))
        assert_array_equal(la.cumsum(x.lipro), self.array.cumsum(3))

        # using axes numbers
        assert_array_equal(la.cumsum(1), self.array.cumsum(1))

        # using axes names
        assert_array_equal(la.cumsum('sex'), self.array.cumsum(2))

    def test_group_agg_kwargs(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        self.assertEqual(la.sum(sex='H').shape, (116, 44, 15))
        self.assertEqual(la.sum(sex='H,F').shape, (116, 44, 15))
        self.assertEqual(la.sum(sex=sex['H']).shape, (116, 44, 15))

        self.assertEqual(la.sum(geo='A11,A21,A25').shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo.group('A11,A21,A25')).shape,
                         (116, 2, 15))

        self.assertEqual(la.sum(geo=':').shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo.all()).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo[':']).shape, (116, 2, 15))
        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the previous
        # tests.
        self.assertEqual(la.sum(geo='A11:A21').shape, (116, 2, 15))
        assert_array_equal(la.sum(geo='A11:A21'), la.sum(geo=':'))
        assert_array_equal(la.sum(geo=geo['A11:A21']), la.sum(geo=':'))

        # a.2) a tuple of one group => do not collapse dimension
        self.assertEqual(la.sum(geo=(geo.all(),)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum(geo=(vla, wal, bru)).shape, (116, 3, 2, 15))
        # with one label in several groups
        self.assertEqual(la.sum(sex=(['H'], ['H', 'F'])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum(sex=('H', 'H,F')).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum(sex='H;H,F').shape, (116, 44, 2, 15))

        aggregated = la.sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(aggregated.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        self.assertEqual(la.sum(lipro='P01,P03;P02,P05;:',
                                geo=(vla, wal, bru, belgium)).shape,
                         (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does
        # not allow non-kwargs after kwargs.
        self.assertEqual(la.sum(age, sex, geo=(vla, wal, bru, belgium)).shape,
                         (4, 15))

        # c) chain group aggregate after axis aggregate
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(reg.shape, (4, 15))

    def test_group_agg_guess_axis(self):
        la = self.larray
        raw = self.array
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        self.assertEqual(la.sum('H').shape, (116, 44, 15))
        self.assertEqual(la.sum('H,').shape, (116, 44, 15))
        self.assertEqual(la.sum('H,F').shape, (116, 44, 15))

        self.assertEqual(la.sum('A11,A21,A25').shape, (116, 2, 15))
        # with a name
        self.assertEqual(la.sum('A11,A21,A25 >> g1').shape, (116, 2, 15))
        self.assertEqual(la.sum(['A11', 'A21', 'A25']).shape, (116, 2, 15))

        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to taking the
        # full axis.
        self.assertEqual(la.sum('A11:A21').shape, (116, 2, 15))
        assert_array_equal(la.sum('A11:A21'), la.sum(geo=':'))
        assert_array_equal(la.sum('A11:A21'), la.sum(geo))

        # a.2) a tuple of one group => do not collapse dimension
        self.assertEqual(la.sum((geo.all(),)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because
        # it gets tricky when we have some other axes into play. For now the
        # error message is unclear because it first aggregates on "vla", then
        # tries to aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((['H'], ['H', 'F'])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum(('H', 'H,F')).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum('H;H,F').shape, (116, 44, 2, 15))
        # with group names
        res = la.sum('H >> men;H,F >> all')
        self.assertEqual(res.shape, (116, 44, 2, 15))
        self.assertTrue('sex' in res.axes)
        men = sex['H'].named('men')
        all_ = sex['H,F'].named('all')
        assert_array_equal(res.axes.sex.labels, ['men', 'all'])
        assert_array_equal(res['men'], raw[:, :, 0, :])
        assert_array_equal(res['all'], raw.sum(2))

        res = la.sum(('H >> men', 'H,F >> all'))
        self.assertEqual(res.shape, (116, 44, 2, 15))
        self.assertTrue('sex' in res.axes)
        assert_array_equal(res.axes.sex.labels, ['men', 'all'])
        assert_array_equal(res['men'], raw[:, :, 0, :])
        assert_array_equal(res['all'], raw.sum(2))

        aggregated = la.sum((vla, wal, bru, belgium))
        self.assertEqual(aggregated.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        self.assertEqual(la.sum('P01,P03;P02,P05;P01:',
                                (vla, wal, bru, belgium)).shape,
                         (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does
        # not allow non-kwargs after kwargs.
        self.assertEqual(la.sum(age, sex, (vla, wal, bru, belgium)).shape,
                         (4, 15))

        # c) chain group aggregate after axis aggregate
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(reg.shape, (4, 15))

    def test_group_agg_label_group(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = geo[self.vla_str], geo[self.wal_str], geo[self.bru_str]
        belgium = geo[self.belgium]

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        men = sex.i[[0]]
        self.assertEqual(la.sum(men).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['H']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['H,']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['H,F']).shape, (116, 44, 15))

        self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[['A11', 'A21', 'A25']]).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo.group('A11,A21,A25')).shape,
                         (116, 2, 15))

        self.assertEqual(la.sum(geo.all()).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[':']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[:]).shape, (116, 2, 15))

        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the previous
        # tests.
        self.assertEqual(la.sum(geo['A11:A21']).shape, (116, 2, 15))
        assert_array_equal(la.sum(geo['A11:A21']), la.sum(geo))
        assert_array_equal(la.sum(geo['A11':'A21']), la.sum(geo))

        # a.2) a tuple of one group => do not collapse dimension
        self.assertEqual(la.sum((geo[:],)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because
        # it gets tricky when we have some other axes into play. For now the
        # error message is unclear because it first aggregates on "vla", then
        # tries to aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((sex['H'], sex[['H', 'F']])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['H'], sex['H', 'F'])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['H'], sex['H,F'])).shape, (116, 44, 2, 15))
        # XXX: do we want to support this?
        # self.assertEqual(la.sum(sex['H;H,F']).shape, (116, 44, 2, 15))

        aggregated = la.sum((vla, wal, bru, belgium))
        self.assertEqual(aggregated.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        # self.assertEqual(la.sum(lipro['P01,P03;P02,P05;P01:'],
        #                         (vla, wal, bru, belgium)).shape,
        #                  (116, 4, 2, 3))
        self.assertEqual(la.sum((lipro['P01,P03'], lipro['P02,P05'], lipro[:]),
                                (vla, wal, bru, belgium)).shape,
                         (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does
        # not allow non-kwargs after kwargs.
        self.assertEqual(la.sum(age, sex, (vla, wal, bru, belgium)).shape,
                         (4, 15))

        # c) chain group aggregate after axis aggregate
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(reg.shape, (4, 15))

    def test_group_agg_label_group_no_axis(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = \
            LGroup(self.vla_str), LGroup(self.wal_str), LGroup(self.bru_str)
        belgium = LGroup(self.belgium)

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        self.assertEqual(la.sum(LGroup('H')).shape, (116, 44, 15))
        self.assertEqual(la.sum(LGroup('H,')).shape, (116, 44, 15))
        self.assertEqual(la.sum(LGroup('H,F')).shape, (116, 44, 15))

        self.assertEqual(la.sum(LGroup('A11,A21,A25')).shape, (116, 2, 15))
        self.assertEqual(la.sum(LGroup(['A11', 'A21', 'A25'])).shape, 
                         (116, 2, 15))

        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the full axis.
        self.assertEqual(la.sum(LGroup('A11:A21')).shape, (116, 2, 15))
        assert_array_equal(la.sum(LGroup('A11:A21')), la.sum(geo))
        assert_array_equal(la.sum(LGroup(slice('A11', 'A21'))), la.sum(geo))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because
        # it gets tricky when we have some other axes into play. For now the
        # error message is unclear because it first aggregates on "vla", then
        # tries to aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((LGroup('H'), LGroup(['H', 'F']))).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum((LGroup('H'), LGroup('H,F'))).shape,
                         (116, 44, 2, 15))
        # XXX: do we want to support this?
        # self.assertEqual(la.sum(sex['H;H,F']).shape, (116, 44, 2, 15))

        aggregated = la.sum((vla, wal, bru, belgium))
        self.assertEqual(aggregated.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        # self.assertEqual(la.sum(lipro['P01,P03;P02,P05;P01:'],
        #                         (vla, wal, bru, belgium)).shape,
        #                  (116, 4, 2, 3))
        self.assertEqual(la.sum((LGroup('P01,P03'), LGroup('P02,P05')),
                                (vla, wal, bru, belgium)).shape,
                         (116, 4, 2, 2))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does
        # not allow non-kwargs after kwargs.
        self.assertEqual(la.sum(age, sex, (vla, wal, bru, belgium)).shape,
                         (4, 15))

        # c) chain group aggregate after axis aggregate
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(reg.shape, (4, 15))

    def test_group_agg_axis_ref_label_group(self):
        la = self.larray
        age, geo, sex, lipro = x.age, x.geo, x.sex, x.lipro
        vla, wal, bru = geo[self.vla_str], geo[self.wal_str], geo[self.bru_str]
        belgium = geo[self.belgium]

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        men = sex.i[[0]]
        self.assertEqual(la.sum(men).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['H']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['H,']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['H,F']).shape, (116, 44, 15))

        self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[['A11', 'A21', 'A25']]).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo.group('A11,A21,A25')).shape,
                         (116, 2, 15))

        self.assertEqual(la.sum(geo.all()).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[':']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[:]).shape, (116, 2, 15))

        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the previous
        # tests.
        self.assertEqual(la.sum(geo['A11:A21']).shape, (116, 2, 15))
        assert_array_equal(la.sum(geo['A11:A21']), la.sum(geo))
        assert_array_equal(la.sum(geo['A11':'A21']), la.sum(geo))

        # a.2) a tuple of one group => do not collapse dimension
        self.assertEqual(la.sum((geo[:],)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because
        # it gets tricky when we have some other axes into play. For now the
        # error message is unclear because it first aggregates on "vla", then
        # tries to aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((sex['H'], sex[['H', 'F']])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['H'], sex['H', 'F'])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['H'], sex['H,F'])).shape, (116, 44, 2, 15))
        # XXX: do we want to support this?
        # self.assertEqual(la.sum(sex['H;H,F']).shape, (116, 44, 2, 15))

        aggregated = la.sum((vla, wal, bru, belgium))
        self.assertEqual(aggregated.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        # self.assertEqual(la.sum(lipro['P01,P03;P02,P05;P01:'],
        #                         (vla, wal, bru, belgium)).shape,
        #                  (116, 4, 2, 3))
        self.assertEqual(la.sum((lipro['P01,P03'], lipro['P02,P05'], lipro[:]),
                                (vla, wal, bru, belgium)).shape,
                         (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does
        # not allow non-kwargs after kwargs.
        self.assertEqual(la.sum(age, sex, (vla, wal, bru, belgium)).shape,
                         (4, 15))

        # c) chain group aggregate after axis aggregate
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(reg.shape, (4, 15))

    def test_group_agg_one_axis(self):
        a = Axis('a', range(3))
        la = ndrange(a)
        raw = np.asarray(la)

        assert_array_equal(la.sum(a[0, 2]), raw[[0, 2]].sum())

    def test_group_agg_anonymous_axis(self):
        la = ndrange((2, 3))
        a1, a2 = la.axes
        raw = np.asarray(la)
        assert_array_equal(la.sum(a2[0, 2]), raw[:, [0, 2]].sum(1))

    # TODO: fix this (and add other tests for references (x.) to anonymous axes
    # def test_group_agg_anonymous_axis_ref(self):
    #     la = ndrange((2, 3))
    #     raw = np.asarray(la)
    #     # this does not work because x[1] refers to an axis with name 1,
    #     # which does not exist. We might want to change this.
    #     assert_array_equal(la.sum(x[1][0, 2]), raw[:, [0, 2]].sum(1))

    # group aggregates on a group-aggregated array
    def test_group_agg_on_group_agg(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))

        # 1) one group => collapse dimension
        self.assertEqual(reg.sum(lipro='P01,P02').shape, (4,))

        # 2) a tuple of one group => do not collapse dimension
        self.assertEqual(reg.sum(lipro=('P01,P02',)).shape, (4, 1))

        # 3) several groups
        self.assertEqual(reg.sum(lipro='P01;P02;:').shape, (4, 3))

        # this is INVALID
        # TODO: raise a nice exception
        # regsum = reg.sum(lipro='P01,P02,:')

        # this is currently allowed even though it can be confusing:
        # P01 and P02 are both groups with one element each.
        self.assertEqual(reg.sum(lipro=('P01', 'P02', ':')).shape, (4, 3))
        self.assertEqual(reg.sum(lipro=('P01', 'P02', lipro.all())).shape,
                         (4, 3))

        # explicit groups are better
        self.assertEqual(reg.sum(lipro=('P01,', 'P02,', ':')).shape, (4, 3))
        self.assertEqual(reg.sum(lipro=(['P01'], ['P02'], ':')).shape, (4, 3))

        # 4) groups on the aggregated dimension

        # self.assertEqual(reg.sum(geo=([vla, bru], [wal, bru])).shape, (2, 3))
        # vla, wal, bru

    # group aggregates on a group-aggregated array
    def test_group_agg_on_group_agg_nokw(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        # XXX: should this be supported too? (it currently fails)
        # reg = la.sum(age, sex).sum(vla, wal, bru, belgium)

        # 1) one group => collapse dimension
        self.assertEqual(reg.sum('P01,P02').shape, (4,))

        # 2) a tuple of one group => do not collapse dimension
        self.assertEqual(reg.sum(('P01,P02',)).shape, (4, 1))

        # 3) several groups
        # : is ambiguous
        # self.assertEqual(reg.sum('P01;P02;:').shape, (4, 3))
        self.assertEqual(reg.sum('P01;P02;P01:').shape, (4, 3))

        # this is INVALID
        # TODO: raise a nice exception
        # regsum = reg.sum(lipro='P01,P02,:')

        # this is currently allowed even though it can be confusing:
        # P01 and P02 are both groups with one element each.
        self.assertEqual(reg.sum(('P01', 'P02', 'P01:')).shape, (4, 3))
        self.assertEqual(reg.sum(('P01', 'P02', lipro.all())).shape,
                         (4, 3))

        # explicit groups are better
        self.assertEqual(reg.sum(('P01,', 'P02,', 'P01:')).shape, (4, 3))
        self.assertEqual(reg.sum((['P01'], ['P02'], 'P01:')).shape, (4, 3))

        # 4) groups on the aggregated dimension

        # self.assertEqual(reg.sum(geo=([vla, bru], [wal, bru])).shape, (2, 3))
        # vla, wal, bru

    def test_getitem_on_group_agg(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))

        # using a string
        vla = self.vla_str
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # one more level...
        self.assertEqual(reg[vla]['P03'], 389049848.0)

        # using an anonymous LGroup
        vla = self.geo.group(self.vla_str)
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # using a named LGroup
        vla = self.geo.group(self.vla_str, name='Vlaanderen')
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

    def test_getitem_on_group_agg_nokw(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        # using a string
        vla = self.vla_str
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # one more level...
        self.assertEqual(reg[vla]['P03'], 389049848.0)

        # using an anonymous LGroup
        vla = self.geo.group(self.vla_str)
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # using a named LGroup
        vla = self.geo.group(self.vla_str, name='Vlaanderen')
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

    def test_filter_on_group_agg(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))

        # using a string
        vla = self.vla_str
        self.assertEqual(reg.filter(geo=vla).shape, (15,))
        # using an anonymous LGroup
        vla = self.geo.group(self.vla_str)
        self.assertEqual(reg.filter(geo=vla).shape, (15,))
        # using a named LGroup
        vla = self.geo.group(self.vla_str, name='Vlaanderen')
        self.assertEqual(reg.filter(geo=vla).shape, (15,))

        # Note that reg.filter(geo=(vla,)) does NOT work. It might be a
        # little confusing for users, because reg[(vla,)] works but it is
        # normal because reg.filter(geo=(vla,)) is equivalent to:
        # reg[((vla,),)] or reg[(vla,), :]

        # mixed VG/string slices
        child = age[:17]
        working = age[18:64]
        retired = age[65:]

        byage = la.sum(age=(child, 5, working, retired))
        self.assertEqual(byage.shape, (4, 44, 2, 15))

        byage = la.sum(age=(child, slice(5, 10), working, retired))
        self.assertEqual(byage.shape, (4, 44, 2, 15))

        # filter on an aggregated larray created with mixed groups
        self.assertEqual(byage.filter(age=child).shape, (44, 2, 15))

        # TODO: make this work
        # self.assertEqual(byage.filter(age=slice(18)).shape, (44, 2, 15))

    # def test_sum_groups_vg_args(self):
    #     la = self.larray
    #     age, geo, sex, lipro = la.axes
    #     vla, wal, bru, belgium = self.vla, self.wal, self.bru, self.belgium
    #
    #     # simple
    #     # ------
    #
    #     # a) one group aggregate (on a fresh array)
    #
    #     # one group => collapse dimension
    #     self.assertEqual(la.sum(sex['H']).shape, (116, 44, 15))
    #     self.assertEqual(la.sum(sex['H,F']).shape, (116, 44, 15))
    #     self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))

    #     # several groups
    #     self.assertEqual(la.sum((vla, wal, belgium)).shape, (116, 3, 2, 15))
    #
    #     # b) group aggregates on several dimensions at the same time
    #
    #     # one group per dim => collapse
    #     self.assertEqual(la.sum(lipro['P01,P03'], vla).shape, (116, 4, 2, 3))
    #     # several groups per dim
    #     lipro_groups = (lipro['P01,P02'], lipro['P05'], lipro['P07,P06'])
    #     self.assertEqual(la.sum(lipro_groups, (vla, wal, bru, belgium)).shape,
    #                      (116, 4, 2, 3))

    #     # c) both axis aggregate and group aggregate at the same time

    #     # In this version we do not need to list "full axes" aggregates first
    #     # since group aggregates are not kwargs

    #     # one group
    #     self.assertEqual(la.sum(age, sex, vla).shape, (15,))
    #     self.assertEqual(la.sum(vla, age, sex).shape, (15,))
    #     self.assertEqual(la.sum(age, vla, sex).shape, (15,))
    #     # tuple of groups
    #     self.assertEqual(la.sum(age, sex, (vla, belgium)).shape, (4, 15))
    #     self.assertEqual(la.sum(age, (vla, belgium), sex).shape, (4, 15))
    #     self.assertEqual(la.sum((vla, belgium), age, sex).shape, (4, 15))
    #     self.assertEqual(la.sum((vla, belgium), age, sex).shape, (4, 15))
    #
    #
    #     # d) mixing arg and kwarg group aggregates
    #     self.assertEqual(la.sum(lipro['P01,P02,P03,P05,P08'],
    #                             geo=(vla, wal, bru)).shape,
    #                      (116, 3, 2, 5))

    def test_sum_several_vg_groups(self):
        la, geo = self.larray, self.geo
        fla = geo.group(self.vla_str, name='Flanders')
        wal = geo.group(self.wal_str, name='Wallonia')
        bru = geo.group(self.bru_str, name='Brussels')

        reg = la.sum(geo=(fla, wal, bru))
        self.assertEqual(reg.shape, (116, 3, 2, 15))

        # the result is indexable
        # a) by VG
        self.assertEqual(reg.filter(geo=fla).shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo=(fla, wal)).shape, (116, 2, 2, 15))

        # b) by string (name of groups)
        self.assertEqual(reg.filter(geo='Flanders').shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo='Flanders,Wallonia').shape,
                         (116, 2, 2, 15))

        # using string groups
        reg = la.sum(geo=(self.vla_str, self.wal_str, self.bru_str))
        self.assertEqual(reg.shape, (116, 3, 2, 15))
        # the result is indexable
        # a) by string (def)
        self.assertEqual(reg.filter(geo=self.vla_str).shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo=(self.vla_str, self.wal_str)).shape,
                         (116, 2, 2, 15))

        # b) by VG
        self.assertEqual(reg.filter(geo=fla).shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo=(fla, wal)).shape,
                         (116, 2, 2, 15))

    def test_sum_with_groups_from_other_axis(self):
        small = self.small

        # use a group from another *compatible* axis
        lipro2 = Axis('lipro', ['P%02d' % i for i in range(1, 16)])
        self.assertEqual(small.sum(lipro=lipro2['P01,P03']).shape, (2,))

        # use (compatible) group from another *incompatible* axis
        # XXX: I am unsure whether or not this should be allowed. Maybe we
        # should simply check that the group is valid in axis, but that
        # will trigger a pretty meaningful error anyway
        lipro3 = Axis('lipro', 'P01,P03,P05')
        self.assertEqual(small.sum(lipro3['P01,P03']).shape, (2,))

        # use a group (from another axis) which is incompatible with the axis of
        # the same name in the array
        lipro4 = Axis('lipro', 'P01,P03,P16')
        with self.assertRaisesRegexp(ValueError,
                                     "lipro\['P01', 'P16'\] is not a valid "
                                     "label for any axis"):
            small.sum(lipro4['P01,P16'])

    def test_ratio(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        regions = (self.vla_str, self.wal_str, self.bru_str, self.belgium)
        reg = la.sum(age, sex, regions)
        self.assertEqual(reg.shape, (4, 15))

        fla = geo.group(self.vla_str, name='Flanders')
        wal = geo.group(self.wal_str, name='Wallonia')
        bru = geo.group(self.bru_str, name='Brussels')
        regions = (fla, wal, bru)
        reg = la.sum(age, sex, regions)

        ratio = reg.ratio()
        assert_array_equal(ratio, reg / reg.sum(geo, lipro))
        self.assertEqual(ratio.shape, (3, 15))

        ratio = reg.ratio(geo)
        assert_array_equal(ratio, reg / reg.sum(geo))
        self.assertEqual(ratio.shape, (3, 15))

        ratio = reg.ratio(geo, lipro)
        assert_array_equal(ratio, reg / reg.sum(geo, lipro))
        self.assertEqual(ratio.shape, (3, 15))
        self.assertEqual(ratio.sum(), 1.0)

    def test_percent(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        regions = (self.vla_str, self.wal_str, self.bru_str, self.belgium)
        reg = la.sum(age, sex, regions)
        self.assertEqual(reg.shape, (4, 15))

        fla = geo.group(self.vla_str, name='Flanders')
        wal = geo.group(self.wal_str, name='Wallonia')
        bru = geo.group(self.bru_str, name='Brussels')
        regions = (fla, wal, bru)
        reg = la.sum(age, sex, regions)

        percent = reg.percent()
        assert_array_equal(percent, reg * 100 / reg.sum(geo, lipro))
        self.assertEqual(percent.shape, (3, 15))

        percent = reg.percent(geo)
        assert_array_equal(percent, reg * 100 / reg.sum(geo))
        self.assertEqual(percent.shape, (3, 15))

        percent = reg.percent(geo, lipro)
        assert_array_equal(percent, reg * 100 / reg.sum(geo, lipro))
        self.assertEqual(percent.shape, (3, 15))
        self.assertAlmostEqual(percent.sum(), 100.0)

    def test_total(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        # la = self.small
        # sex, lipro = la.axes

        self.assertEqual(la.with_total().shape, (117, 45, 3, 16))
        self.assertEqual(la.with_total(sex).shape, (116, 44, 3, 15))
        self.assertEqual(la.with_total(lipro).shape, (116, 44, 2, 16))
        self.assertEqual(la.with_total(sex, lipro).shape, (116, 44, 3, 16))

        fla = geo.group(self.vla_str, name='Flanders')
        wal = geo.group(self.wal_str, name='Wallonia')
        bru = geo.group(self.bru_str, name='Brussels')
        bel = geo.all('Belgium')

        self.assertEqual(la.with_total(geo=(fla, wal, bru), op=mean).shape,
                         (116, 47, 2, 15))
        self.assertEqual(la.with_total((fla, wal, bru), op=mean).shape,
                         (116, 47, 2, 15))
        # works but "wrong" for x.geo (double what is expected because it
        # includes fla wal & bru)
        # TODO: we probably want to display a warning (or even an error?) in
        #       that case. If we really want that behavior, we can still split
        #       the operation: .with_total((fla, wal, bru)).with_total(x.geo)
        # OR we might want to only sum the axis as it was before the op (but
        #    that does not play well when working with multiple axes).
        a1 = la.with_total(x.sex, (fla, wal, bru), x.geo, x.lipro)
        self.assertEqual(a1.shape, (116, 48, 3, 16))

        # correct total but the order is not very nice
        a2 = la.with_total(x.sex, x.geo, (fla, wal, bru), x.lipro)
        self.assertEqual(a2.shape, (116, 48, 3, 16))

        # the correct way to do it
        a3 = la.with_total(x.sex, (fla, wal, bru, bel), x.lipro)
        self.assertEqual(a3.shape, (116, 48, 3, 16))

        # a4 = la.with_total((lipro[':P05'], lipro['P05:']), op=mean)
        a4 = la.with_total((':P05', 'P05:'), op=mean)
        self.assertEqual(a4.shape, (116, 44, 2, 17))

    def test_transpose(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        reordered = la.transpose(geo, age, lipro, sex)
        self.assertEqual(reordered.shape, (44, 116, 15, 2))

        reordered = la.transpose(lipro, age)
        self.assertEqual(reordered.shape, (15, 116, 44, 2))

        self.assertEqual(la.transpose().axes.names,
                         ['lipro', 'sex', 'geo', 'age'])

    def test_transpose_anonymous(self):
        a = ndrange((2, 3, 4))

        # reordered = a.transpose(0, 2, 1)
        # self.assertEqual(reordered.shape, (2, 4, 3))

        # axes = self[1, 2]
        # => union(axes, self)
        # => axes.extend([self[0]])
        # => breaks because self[0] not compatible with axes[0]
        # => breaks because self[0] not compatible with self[1]

        # a real union should not care and should return
        # self[1, 2, 0] but will this break other stuff? My gut feeling is yes

        # when doing a binop between anonymous axes, we use union too (that
        # might be the problem) and we need *that* union to match axes by
        # position
        reordered = a.transpose(1, 2)
        self.assertEqual(reordered.shape, (3, 4, 2))

        reordered = a.transpose(2, 0)
        self.assertEqual(reordered.shape, (4, 2, 3))

        reordered = a.transpose()
        self.assertEqual(reordered.shape, (4, 3, 2))

    def test_binary_ops(self):
        raw = self.small_data
        la = self.small

        assert_array_equal(la + la, raw + raw)
        assert_array_equal(la + 1, raw + 1)
        assert_array_equal(1 + la, 1 + raw)

        assert_array_equal(la - la, raw - raw)
        assert_array_equal(la - 1, raw - 1)
        assert_array_equal(1 - la, 1 - raw)

        assert_array_equal(la * la, raw * raw)
        assert_array_equal(la * 2, raw * 2)
        assert_array_equal(2 * la, 2 * raw)

        assert_array_nan_equal(la / la, raw / raw)
        assert_array_equal(la / 2, raw / 2)
        assert_array_equal(30 / la, 30 / raw)
        assert_array_equal(30 / (la + 1), 30 / (raw + 1))

        raw_int = raw.astype(int)
        la_int = LArray(raw_int, axes=(self.sex, self.lipro))
        assert_array_equal(la_int / 2, raw_int / 2)
        assert_array_equal(la_int // 2, raw_int // 2)

        # test adding two larrays with different axes order
        assert_array_equal(la + la.transpose(), raw * 2)

        # mixed operations
        raw2 = raw / 2
        la_raw2 = la - raw2
        self.assertEqual(la_raw2.axes, la.axes)
        assert_array_equal(la_raw2, raw - raw2)
        raw2_la = raw2 - la
        self.assertEqual(raw2_la.axes, la.axes)
        assert_array_equal(raw2_la, raw2 - raw)

        la_ge_raw2 = la >= raw2
        self.assertEqual(la_ge_raw2.axes, la.axes)
        assert_array_equal(la_ge_raw2, raw >= raw2)

        raw2_ge_la = raw2 >= la
        self.assertEqual(raw2_ge_la.axes, la.axes)
        assert_array_equal(raw2_ge_la, raw2 >= raw)

    def test_binary_ops_no_name_axes(self):
        raw = self.small_data
        raw2 = self.small_data + 1
        la = ndrange(self.small.shape)
        la2 = ndrange(self.small.shape) + 1

        assert_array_equal(la + la2, raw + raw2)
        assert_array_equal(la + 1, raw + 1)
        assert_array_equal(1 + la, 1 + raw)

        assert_array_equal(la - la2, raw - raw2)
        assert_array_equal(la - 1, raw - 1)
        assert_array_equal(1 - la, 1 - raw)

        assert_array_equal(la * la2, raw * raw2)
        assert_array_equal(la * 2, raw * 2)
        assert_array_equal(2 * la, 2 * raw)

        assert_array_nan_equal(la / la2, raw / raw2)
        assert_array_equal(la / 2, raw / 2)
        assert_array_equal(30 / la, 30 / raw)
        assert_array_equal(30 / (la + 1), 30 / (raw + 1))

        raw_int = raw.astype(int)
        la_int = LArray(raw_int)
        assert_array_equal(la_int / 2, raw_int / 2)
        assert_array_equal(la_int // 2, raw_int // 2)

        # adding two larrays with different axes order cannot work with
        # unnamed axes
        # assert_array_equal(la + la.transpose(), raw * 2)

        # mixed operations
        raw2 = raw / 2
        la_raw2 = la - raw2
        self.assertEqual(la_raw2.axes, la.axes)
        assert_array_equal(la_raw2, raw - raw2)
        raw2_la = raw2 - la
        self.assertEqual(raw2_la.axes, la.axes)
        assert_array_equal(raw2_la, raw2 - raw)

        la_ge_raw2 = la >= raw2
        self.assertEqual(la_ge_raw2.axes, la.axes)
        assert_array_equal(la_ge_raw2, raw >= raw2)

        raw2_ge_la = raw2 >= la
        self.assertEqual(raw2_ge_la.axes, la.axes)
        assert_array_equal(raw2_ge_la, raw2 >= raw)

    def test_broadcasting_no_name(self):
        a = ndrange((2, 3))
        b = ndrange(3)
        c = ndrange(2)

        with self.assertRaises(ValueError):
            # ValueError: incompatible axes:
            # Axis(None, [0, 1, 2])
            # vs
            # Axis(None, [0, 1])
            a * b

        d = a * c
        self.assertEqual(d.shape, (2, 3))
        # {0}*\{1}* | 0 | 1 | 2
        #         0 | 0 | 0 | 0
        #         1 | 3 | 4 | 5
        self.assertTrue(np.array_equal(d, [[0, 0, 0],
                                           [3, 4, 5]]))

        # it is unfortunate that the behavior is different from numpy
        # (even though I find our behavior more intuitive)
        d = np.asarray(a) * np.asarray(b)
        self.assertEqual(d.shape, (2, 3))
        self.assertTrue(np.array_equal(d, [[0, 1,  4],
                                           [0, 4, 10]]))

        with self.assertRaises(ValueError):
            # ValueError: operands could not be broadcast together with shapes
            # (2,3) (2,)
            np.asarray(a) * np.asarray(c)

    def test_unary_ops(self):
        raw = self.small_data
        la = self.small

        # using numpy functions
        assert_array_equal(np.abs(la - 10), np.abs(raw - 10))
        assert_array_equal(np.negative(la), np.negative(raw))
        assert_array_equal(np.invert(la), np.invert(raw))

        # using python builtin ops
        assert_array_equal(abs(la - 10), abs(raw - 10))
        assert_array_equal(-la, -raw)
        assert_array_equal(+la, +raw)
        assert_array_equal(~la, ~raw)

    def test_mean(self):
        la = self.small
        raw = self.small_data

        sex, lipro = la.axes
        assert_array_equal(la.mean(lipro), raw.mean(1))

    def test_with_axes(self):
        lipro2 = self.lipro.rename('lipro2').labels = \
            [l.replace('P', 'Q') for l in self.lipro.labels]
        sex2 = self.age.rename('sex2').labels = ['Man', 'Woman']

        la = LArray(self.small_data, axes=(sex2, lipro2),
                    title=self.small_title)
        la2 = self.small.with_axes((sex2, lipro2))
        assert_array_equal(la, la2)
        self.assertEqual(la.title, la2.title, "title of array returned by "
                                              "with_axes should be the same as the original one. "
                                              "We got '{}' instead of '{}'".format(la2.title, la.title))

    def test_append(self):
        la = self.small
        sex, lipro = la.axes

        la = la.append(lipro, la.sum(lipro), label='sum')
        self.assertEqual(la.shape, (2, 16))
        la = la.append(sex, la.sum(sex), label='sum')
        self.assertEqual(la.shape, (3, 16))

        # crap the sex axis is different !!!! we don't have this problem with
        # the kwargs syntax below
        # la = la.append(la.mean(sex), axis=sex, label='mean')
        # self.assertEqual(la.shape, (4, 16))

        # another syntax (which implies we could not have an axis named "label")
        # la = la.append(lipro=la.sum(lipro), label='sum')
        # self.assertEqual(la.shape, (117, 44, 2, 15))

    # the aim of this test is to drop the last value of an axis, but instead
    # of dropping the last axis tick/label, drop the first one.
    def test_shift_axis(self):
        la = self.small
        sex, lipro = la.axes

        # TODO: check how awful the syntax is with an axis that is not last
        # or first
        l2 = LArray(la[:, :'P14'], axes=[sex, Axis('lipro', lipro.labels[1:])])
        l2 = LArray(la[:, :'P14'], axes=[sex, lipro.subaxis(slice(1, None))])

        # We can also modify the axis in-place (dangerous!)
        # lipro.labels = np.append(lipro.labels[1:], lipro.labels[0])
        l2 = la[:, 'P02':]
        l2.axes.lipro.labels = lipro.labels[1:]

    def test_extend(self):
        la = self.small
        sex, lipro = la.axes

        all_lipro = lipro[:]
        tail = la.sum(lipro=(all_lipro,))
        la = la.extend(lipro, tail)
        self.assertEqual(la.shape, (2, 16))
        # test with a string axis
        la = la.extend('sex', la.sum(sex=(sex.all(),)))
        self.assertEqual(la.shape, (3, 16))

    def test_read_csv(self):
        la = read_csv(abspath('test1d.csv'))
        self.assertEqual(la.ndim, 1)
        self.assertEqual(la.shape, (3,))
        self.assertEqual(la.axes.names, ['time'])
        assert_array_equal(la, [3722, 3395, 3347])

        la = read_csv(abspath('test2d.csv'))
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', 'time'])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        la = read_csv(abspath('test3d.csv'))
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (5, 2, 3))
        self.assertEqual(la.axes.names, ['age', 'sex', 'time'])
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

        la = read_csv(abspath('test5d.csv'))
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[x.arr[1], 0, 'F', x.nat[1], :],
                           [3722, 3395, 3347])

    @unittest.skipIf(xw is None, "xlwings is not available")
    def test_read_excel_xlwings(self):
        la = read_excel(abspath('test.xlsx'), '1d')
        self.assertEqual(la.ndim, 1)
        self.assertEqual(la.shape, (3,))
        self.assertEqual(la.axes.names, ['time'])
        assert_array_equal(la, [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '2d')
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', 'time'])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '3d')
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (5, 2, 3))
        self.assertEqual(la.axes.names, ['age', 'sex', 'time'])
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '5d')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[x.arr[1], 0, 'F', x.nat[1], :],
                           [3722, 3395, 3347])

    def test_read_excel_pandas(self):
        la = read_excel(abspath('test.xlsx'), '1d', engine='xlrd')
        self.assertEqual(la.ndim, 1)
        self.assertEqual(la.shape, (3,))
        self.assertEqual(la.axes.names, ['time'])
        assert_array_equal(la, [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '2d', nb_index=1, engine='xlrd')
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', 'time'])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '3d', index_col=[0, 1], engine='xlrd')
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (5, 2, 3))
        self.assertEqual(la.axes.names, ['age', 'sex', 'time'])
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '5d', nb_index=4, engine='xlrd')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[x.arr[1], 0, 'F', x.nat[1], :],
                           [3722, 3395, 3347])

    def test_df_aslarray(self):
        dt = [('age', int), ('sex', 'U1'),
              ('2007', int), ('2010', int), ('2013', int)]
        data = np.array([
            (0, 'F', 3722, 3395, 3347),
            (0, 'H', 338, 316, 323),
            (1, 'F', 2878, 2791, 2822),
            (1, 'H', 1121, 1037, 976),
            (2, 'F', 4073, 4161, 4429),
            (2, 'H', 1561, 1463, 1467),
            (3, 'F', 3507, 3741, 3366),
            (3, 'H', 2052, 2052, 2118),
        ], dtype=dt)
        df = pd.DataFrame(data)
        df.set_index(['age', 'sex'], inplace=True)
        df.columns.name = 'time'

        la = df_aslarray(df)
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (4, 2, 3))
        self.assertEqual(la.axes.names, ['age', 'sex', 'time'])
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

    def test_to_csv(self):
        la = read_csv(abspath('test5d.csv'))
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[x.arr[1], 0, 'F', x.nat[1], :],
                           [3722, 3395, 3347])

        la.to_csv('out.csv')
        result = ['arr,age,sex,nat\\time,2007,2010,2013\n',
                  '1,0,F,1,3722,3395,3347\n',
                  '1,0,F,2,338,316,323\n']
        with open('out.csv') as f:
            self.assertEqual(f.readlines()[:3], result)

        la.to_csv('out.csv', transpose=False)
        result = ['arr,age,sex,nat,time,0\n', '1,0,F,1,2007,3722\n',
                  '1,0,F,1,2010,3395\n']
        with open('out.csv') as f:
            self.assertEqual(f.readlines()[:3], result)

        la = ndrange([Axis('time', range(2015, 2018))])
        la.to_csv('test_out1d.csv')
        result = ['time,2015,2016,2017\n',
                  ',0,1,2\n']
        with open('test_out1d.csv') as f:
            self.assertEqual(f.readlines(), result)

    def test_to_excel(self):
        la = read_excel(abspath('test.xlsx'), '5d', nb_index=4, engine=None)
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[x.arr[1], 0, 'F', x.nat[1], :],
                           [3722, 3395, 3347])

        la.to_excel('out.xlsx', '5d')
        out = read_excel('out.xlsx', '5d', nb_index=4, engine=None)
        assert_array_equal(out, la)

    def test_ufuncs(self):
        la = self.small
        raw = self.small_data

        # simple one-argument ufunc
        assert_array_equal(exp(la), np.exp(raw))

        # with out=
        la_out = zeros(la.axes)
        raw_out = np.zeros(raw.shape)

        la_out2 = exp(la, la_out)
        raw_out2 = np.exp(raw, raw_out)

        # FIXME: this is not the case currently
        # self.assertIs(la_out2, la_out)
        assert_array_equal(la_out2, la_out)
        self.assertIs(raw_out2, raw_out)

        assert_array_equal(la_out, raw_out)

        # with out= and broadcasting
        # we need to put the 'a' axis first because raw numpy only supports that
        la_out = zeros([Axis('a', [0, 1, 2])] + list(la.axes))
        raw_out = np.zeros((3,) + raw.shape)

        la_out2 = exp(la, la_out)
        raw_out2 = np.exp(raw, raw_out)

        # self.assertIs(la_out2, la_out)
        # XXX: why is la_out2 transposed?
        assert_array_equal(la_out2.transpose(x.a), la_out)
        self.assertIs(raw_out2, raw_out)

        assert_array_equal(la_out, raw_out)

        sex, lipro = la.axes

        low = la.sum(sex) // 4 + 3
        raw_low = raw.sum(0) // 4 + 3
        high = la.sum(sex) // 4 + 13
        raw_high = raw.sum(0) // 4 + 13

        # LA + scalars
        assert_array_equal(la.clip(0, 10), raw.clip(0, 10))
        assert_array_equal(clip(la, 0, 10), np.clip(raw, 0, 10))

        # LA + LA (no broadcasting)
        assert_array_equal(clip(la, 21 - la, 9 + la // 2),
                           np.clip(raw, 21 - raw, 9 + raw // 2))

        # LA + LA (with broadcasting)
        assert_array_equal(clip(la, low, high),
                           np.clip(raw, raw_low, raw_high))

        # where (no broadcasting)
        assert_array_equal(where(la < 5, -5, la),
                           np.where(raw < 5, -5, raw))

        # where (transposed no broadcasting)
        assert_array_equal(where(la < 5, -5, la.T),
                           np.where(raw < 5, -5, raw))

        # where (with broadcasting)
        result = where(la['P01'] < 5, -5, la)
        self.assertEqual(result.axes.names, ['sex', 'lipro'])
        assert_array_equal(result, np.where(raw[:,[0]] < 5, -5, raw))

        # round
        small_float = self.small + 0.6
        rounded = round(small_float)
        assert_array_equal(rounded, np.round(self.small_data + 0.6))

    def test_diag(self):
        # 2D -> 1D
        a = ndrange((3, 3))
        d = diag(a)
        self.assertEqual(d.ndim, 1)
        self.assertEqual(d.i[0], a.i[0, 0])
        self.assertEqual(d.i[1], a.i[1, 1])
        self.assertEqual(d.i[2], a.i[2, 2])

        # 1D -> 2D
        a2 = diag(d)
        self.assertEqual(a2.ndim, 2)
        self.assertEqual(a2.i[0, 0], a.i[0, 0])
        self.assertEqual(a2.i[1, 1], a.i[1, 1])
        self.assertEqual(a2.i[2, 2], a.i[2, 2])

        # 3D -> 2D
        a = ndrange((3, 3, 3))
        d = diag(a)
        self.assertEqual(d.ndim, 2)
        self.assertEqual(d.i[0, 0], a.i[0, 0, 0])
        self.assertEqual(d.i[1, 1], a.i[1, 1, 1])
        self.assertEqual(d.i[2, 2], a.i[2, 2, 2])

        # 3D -> 1D
        d = diag(a, axes=(0, 1, 2))
        self.assertEqual(d.ndim, 1)
        self.assertEqual(d.i[0], a.i[0, 0, 0])
        self.assertEqual(d.i[1], a.i[1, 1, 1])
        self.assertEqual(d.i[2], a.i[2, 2, 2])

        # 1D (anon) -> 2D
        d_anon = d.rename(0, None).drop_labels()
        a2 = diag(d_anon)
        self.assertEqual(a2.ndim, 2)

        # 1D (anon) -> 3D
        a3 = diag(d_anon, ndim=3)
        self.assertEqual(a2.ndim, 2)
        self.assertEqual(a3.i[0, 0, 0], a.i[0, 0, 0])
        self.assertEqual(a3.i[1, 1, 1], a.i[1, 1, 1])
        self.assertEqual(a3.i[2, 2, 2], a.i[2, 2, 2])

        # using Axis object
        sex = Axis('sex', 'M,F')
        a = eye(sex)
        d = diag(a)
        self.assertEqual(d.ndim, 1)
        self.assertEqual(d.axes.names, ['sex,sex'])
        assert_array_equal(d.axes.labels, [['M,M', 'F,F']])
        self.assertEqual(d.i[0], 1.0)
        self.assertEqual(d.i[1], 1.0)

    # cannot use @ in the tests because that is an invalid syntax in Python 2
    def test_matmul(self):
        a1 = eye(3) * 2
        a2 = ndrange((3, 3))
        # FIXME: this will break for python 3.10 and later
        if sys.version >= '3.5':
            # LArray value
            assert_array_equal(a1.__matmul__(a2), ndrange((3, 3)) * 2)

            # ndarray value
            assert_array_equal(a1.__matmul__(a2.data), ndrange((3, 3)) * 2)

    def test_rmatmul(self):
        a1 = eye(3) * 2
        a2 = ndrange((3, 3))
        # FIXME: this will break for python 3.10 and later
        if sys.version >= '3.5':
            # equivalent to a1.data @ a2
            res = a2.__rmatmul__(a1.data)
            self.assertIsInstance(res, LArray)
            assert_array_equal(res, ndrange((3, 3)) * 2)

    def test_broadcast_with(self):
        a1 = ndrange((3, 2))
        a2 = ndrange(3)
        b = a2.broadcast_with(a1)
        self.assertEqual(b.ndim, a1.ndim)
        self.assertEqual(b.shape, (3, 1))
        assert_array_equal(b.i[:, 0], a2)

        a1 = ndrange((1, 3))
        a2 = ndrange((3, 1))
        b = a2.broadcast_with(a1)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.shape, (3, 1))
        assert_array_equal(b, a2)

    def test_plot(self):
        pass
        #small_h = small['H']
        #small_h.plot(kind='bar')
        #small_h.plot()
        #small_h.hist()

        #large_data = np.random.randn(1000)
        #tick_v = np.random.randint(ord('a'), ord('z'), size=1000)
        #ticks = [chr(c) for c in tick_v]
        #large_axis = Axis('large', ticks)
        #large = LArray(large_data, axes=[large_axis])
        #large.plot()
        #large.hist()

    # def test_to_excel(self):
    #     a = ndrange('a=a1,a2,a3')
    #     b = ndrange('a=a1,a2,a3;b=b1,b2')
    #
    #     # Book1/Sheet1/A1
    #     a.to_excel()
    #     # Book2/Sheet1/A1
    #     a.to_excel(transpose=True)
    #     # Book1/Sheet2/A1
    #     b.to_excel('Book1')
    #     # Book1/Sheet1/A10
    #     b.to_excel('Book1', 'Sheet1', 'A10')
    #     # b.xlsx/Sheet1/A1
    #     b.to_excel('c:/tmp/b.xlsx', overwrite_file=True)
    #     # b.xlsx/YADA/A1
    #     b.to_excel('c:/tmp/b.xlsx', 'YADA')
    #     # b.xlsx/Sheet1/A10
    #     b.to_excel('c:/tmp/b.xlsx', 'Sheet1', 'A10')

    def test_combine_axes(self):
        arr = ndtest((2, 3, 4, 5))
        res = arr.combine_axes((x.a, x.b))
        self.assertEqual(res.axes.names, ['a_b', 'c', 'd'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2 * 3, 4, 5))
        assert_array_equal(res.axes.a_b.labels[:2], ['a0_b0', 'a0_b1'])
        assert_array_equal(res['a1_b0'], arr['a1', 'b0'])

        res = arr.combine_axes((x.a, x.c))
        self.assertEqual(res.axes.names, ['a_c', 'b', 'd'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2 * 4, 3, 5))
        assert_array_equal(res.axes.a_c.labels[:2], ['a0_c0', 'a0_c1'])
        assert_array_equal(res['a1_c0'], arr['a1', 'c0'])

        res = arr.combine_axes((x.b, x.d))
        self.assertEqual(res.axes.names, ['a', 'b_d', 'c'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2, 3 * 5, 4))
        assert_array_equal(res.axes.b_d.labels[:2], ['b0_d0', 'b0_d1'])
        assert_array_equal(res['b1_d0'], arr['b1', 'd0'])

    def test_split_axis(self):
        arr = ndtest((2, 3, 4, 5))
        res = arr.combine_axes((x.b, x.d))
        self.assertEqual(res.axes.names, ['a', 'b_d', 'c'])
        res = res.split_axis('b_d')
        self.assertEqual(res.axes.names, ['a', 'b', 'd', 'c'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2, 3, 5, 4))
        assert_array_equal(res.transpose(x.a, x.b, x.c, x.d), arr)


if __name__ == "__main__":
    import doctest
    from larray import core
    doctest.testmod(core)
    unittest.main()
