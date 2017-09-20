from __future__ import absolute_import, division, print_function

from unittest import TestCase

import pytest
import numpy as np

from larray.tests.common import abspath, assert_array_equal, assert_array_nan_equal
from larray import Axis, AxisCollection, LGroup, PGroup


class TestAxis(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        sex_tuple = ('M', 'F')
        sex_list = ['M', 'F']
        sex_array = np.array(sex_list)

        # wildcard axis
        axis = Axis(10, 'axis')
        assert len(axis) == 10
        assert list(axis.labels) == list(range(10))
        # tuple of strings
        assert_array_equal(Axis(sex_tuple, 'sex').labels, sex_array)
        # list of strings
        assert_array_equal(Axis(sex_list, 'sex').labels, sex_array)
        # array of strings
        assert_array_equal(Axis(sex_array, 'sex').labels, sex_array)
        # single string
        assert_array_equal(Axis('sex=M,F').labels, sex_array)
        # list of ints
        assert_array_equal(Axis(range(116), 'age').labels, np.arange(116))
        # range-string
        axis = Axis('0..115', 'age')
        assert_array_equal(axis.labels, np.arange(116))
        # another axis group
        group = axis[:10]
        group_axis = Axis(group)
        assert_array_equal(group_axis.labels, np.arange(11))
        assert_array_equal(group_axis.name, 'age')
        # another axis as labels argument
        other = Axis('other=0..10')
        axis = Axis(other, 'age')
        assert_array_equal(axis.labels, other.labels)
        assert_array_equal(axis.name, 'age')

    def test_equals(self):
        self.assertTrue(Axis('sex=M,F').equals(Axis('sex=M,F')))
        self.assertTrue(Axis('sex=M,F').equals(Axis(['M', 'F'], 'sex')))
        self.assertFalse(Axis('sex=M,W').equals(Axis('sex=M,F')))
        self.assertFalse(Axis('sex1=M,F').equals(Axis('sex2=M,F')))
        self.assertFalse(Axis('sex1=M,W').equals(Axis('sex2=M,F')))

    def test_getitem(self):
        age = Axis('age=0..10')
        # a tuple
        a159 = age[1, 5, 9]
        self.assertEqual(a159.key, [1, 5, 9])
        self.assertIs(a159.name, None)
        self.assertIs(a159.axis, age)

        # a normal list
        a159 = age[[1, 5, 9]]
        self.assertEqual(a159.key, [1, 5, 9])
        self.assertIs(a159.name, None)
        self.assertIs(a159.axis, age)

        # a string list
        a159 = age['1,5,9']
        self.assertEqual(a159.key, [1, 5, 9])
        self.assertIs(a159.name, None)
        self.assertIs(a159.axis, age)

        # a normal slice
        a10to20 = age[5:9]
        self.assertEqual(a10to20.key, slice(5, 9))
        self.assertIs(a10to20.axis, age)

        # a string slice
        a10to20 = age['5:9']
        self.assertEqual(a10to20.key, slice(5, 9))
        self.assertIs(a10to20.axis, age)

        # with name
        group = age[[1, 5, 9]] >> 'test'
        self.assertEqual(group.key, [1, 5, 9])
        self.assertEqual(group.name, 'test')
        self.assertIs(group.axis, age)

        # all
        group = age[:] >> 'all'
        self.assertEqual(group.key, slice(None))
        self.assertIs(group.axis, age)

        # an axis
        age2 = Axis('age=0..5')
        group = age[age2]
        assert list(group.key) == list(age2.labels)

    def test_translate(self):
        # an axis with labels having the object dtype
        a = Axis(np.array(["a0", "a1"], dtype=object), 'a')

        self.assertEqual(a.translate('a1'), 1)
        self.assertEqual(a.translate('a1 >> A1'), 1)

    def test_getitem_lgroup_keys(self):
        def group_equal(g1, g2):
            return (g1.key == g2.key and g1.name == g2.name and
                    g1.axis is g2.axis)

        age = Axis(range(100), 'age')
        ages = [1, 5, 9]

        val_only = LGroup(ages)
        self.assertTrue(group_equal(age[val_only], LGroup(ages, axis=age)))
        self.assertTrue(group_equal(age[val_only] >> 'a_name', LGroup(ages, 'a_name', axis=age)))

        val_name = LGroup(ages, 'val_name')
        self.assertTrue(group_equal(age[val_name], LGroup(ages, 'val_name', age)))
        self.assertTrue(group_equal(age[val_name] >> 'a_name', LGroup(ages, 'a_name', age)))

        val_axis = LGroup(ages, axis=age)
        self.assertTrue(group_equal(age[val_axis], LGroup(ages, axis=age)))
        self.assertTrue(group_equal(age[val_axis] >> 'a_name', LGroup(ages, 'a_name', axis=age)))

        val_axis_name = LGroup(ages, 'val_axis_name', age)
        self.assertTrue(group_equal(age[val_axis_name], LGroup(ages, 'val_axis_name', age)))
        self.assertTrue(group_equal(age[val_axis_name] >> 'a_name', LGroup(ages, 'a_name', age)))

    def test_getitem_group_keys(self):
        a = Axis('a=a0..a2')
        alt_a = Axis('a=a1..a3')

        # a) key is a single LGroup
        # -------------------------

        # a.1) containing a scalar
        key = a['a1']
        # use it on the same axis
        g = a[key]
        self.assertEqual(g.key, 'a1')
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertEqual(g.key, 'a1')
        self.assertIs(g.axis, alt_a)

        # a.2) containing a slice
        key = a['a1':'a2']
        # use it on the same axis
        g = a[key]
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, alt_a)

        # a.3) containing a list
        key = a[['a1', 'a2']]
        # use it on the same axis
        g = a[key]
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, alt_a)

        # b) key is a single PGroup
        # -------------------------

        # b.1) containing a scalar
        key = a.i[1]
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, 'a1')
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, 'a1')
        self.assertIs(g.axis, alt_a)

        # b.2) containing a slice
        key = a.i[1:3]
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, alt_a)

        # b.3) containing a list
        key = a.i[[1, 2]]
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(list(g.key), ['a1', 'a2'])
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(list(g.key), ['a1', 'a2'])
        self.assertIs(g.axis, alt_a)

        # c) key is a slice
        # -----------------

        # c.1) with LGroup bounds
        lg_a1 = a['a1']
        lg_a2 = a['a2']
        # use it on the same axis
        g = a[lg_a1:lg_a2]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[lg_a1:lg_a2]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, alt_a)

        # c.2) with PGroup bounds
        pg_a1 = a.i[1]
        pg_a2 = a.i[2]
        # use it on the same axis
        g = a[pg_a1:pg_a2]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[pg_a1:pg_a2]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, slice('a1', 'a2'))
        self.assertIs(g.axis, alt_a)

        # d) key is a list of scalar groups => create a single LGroup
        # ---------------------------------

        # d.1) with LGroup
        key = [a['a1'], a['a2']]
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, alt_a)

        # d.2) with PGroup
        key = [a.i[1], a.i[2]]
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, alt_a)

        # e) key is a list of non-scalar groups => retarget multiple groups to axis
        # -------------------------------------

        # e.1) with LGroup
        key = [a['a1', 'a2'], a['a2', 'a1']]
        # use it on the same axis => nothing happens
        g = a[key]
        self.assertIsInstance(g, list)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(g[0].key, ['a1', 'a2'])
        self.assertEqual(g[1].key, ['a2', 'a1'])
        self.assertIs(g[0].axis, a)
        self.assertIs(g[1].axis, a)
        # use it on a different axis => change axis
        g = alt_a[key]
        self.assertIsInstance(g, list)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(g[0].key, ['a1', 'a2'])
        self.assertEqual(g[1].key, ['a2', 'a1'])
        self.assertIs(g[0].axis, alt_a)
        self.assertIs(g[1].axis, alt_a)

        # e.2) with PGroup
        key = (a.i[1, 2], a.i[2, 1])
        # use it on the same axis => change to LGroup
        g = a[key]
        self.assertIsInstance(g, tuple)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(list(g[0].key), ['a1', 'a2'])
        self.assertEqual(list(g[1].key), ['a2', 'a1'])
        self.assertIs(g[0].axis, a)
        self.assertIs(g[1].axis, a)
        # use it on a different axis => retarget to axis
        g = alt_a[key]
        self.assertIsInstance(g, tuple)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(list(g[0].key), ['a1', 'a2'])
        self.assertEqual(list(g[1].key), ['a2', 'a1'])
        self.assertIs(g[0].axis, alt_a)
        self.assertIs(g[1].axis, alt_a)

        # f) key is a tuple of scalar groups => create a single LGroup
        # ----------------------------------

        # f.1) with LGroups
        key = (a['a1'], a['a2'])
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, alt_a)

        # f.2) with PGroup
        key = (a.i[1], a.i[2])
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, LGroup)
        self.assertEqual(g.key, ['a1', 'a2'])
        self.assertIs(g.axis, alt_a)

        # g) key is a tuple of non-scalar groups => retarget multiple groups to axis
        # --------------------------------------

        # g.1) with LGroups
        key = (a['a1', 'a2'], a['a2', 'a1'])
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, tuple)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(g[0].key, ['a1', 'a2'])
        self.assertEqual(g[1].key, ['a2', 'a1'])
        self.assertIs(g[0].axis, a)
        self.assertIs(g[1].axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, tuple)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(g[0].key, ['a1', 'a2'])
        self.assertEqual(g[1].key, ['a2', 'a1'])
        self.assertIs(g[0].axis, alt_a)
        self.assertIs(g[1].axis, alt_a)

        # g.2) with PGroup
        key = (a.i[1, 2], a.i[2, 1])
        # use it on the same axis
        g = a[key]
        self.assertIsInstance(g, tuple)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(list(g[0].key), ['a1', 'a2'])
        self.assertEqual(list(g[1].key), ['a2', 'a1'])
        self.assertIs(g[0].axis, a)
        self.assertIs(g[1].axis, a)
        # use it on a different axis
        g = alt_a[key]
        self.assertIsInstance(g, tuple)
        self.assertIsInstance(g[0], LGroup)
        self.assertIsInstance(g[1], LGroup)
        self.assertEqual(list(g[0].key), ['a1', 'a2'])
        self.assertEqual(list(g[1].key), ['a2', 'a1'])
        self.assertIs(g[0].axis, alt_a)
        self.assertIs(g[1].axis, alt_a)

    def test_init_from_group(self):
        code = Axis('code=C01..C03')
        code_group = code[:'C02']
        subset_axis = Axis(code_group, 'code_subset')
        assert_array_equal(subset_axis.labels, ['C01', 'C02'])

    def test_match(self):
        sutcode = Axis(['A23', 'A2301', 'A25', 'A2501'], 'sutcode')
        self.assertEqual(sutcode.matches('^...$'), LGroup(['A23', 'A25']))
        self.assertEqual(sutcode.startswith('A23'), LGroup(['A23', 'A2301']))
        self.assertEqual(sutcode.endswith('01'), LGroup(['A2301', 'A2501']))

    def test_iter(self):
        sex = Axis('sex=M,F')
        self.assertEqual(list(sex), [PGroup(0, axis=sex), PGroup(1, axis=sex)])

    def test_positional(self):
        age = Axis('age=0..115')

        # these are NOT equivalent (not translated until used in an LArray
        # self.assertEqual(age.i[:17], age[':17'])
        key = age.i[:-1]
        self.assertEqual(key.key, slice(None, -1))
        self.assertIs(key.axis, age)

    def test_contains(self):
        # normal Axis
        age = Axis('age=0..10')

        age2 = age[2]
        age2bis = age[(2,)]
        age2ter = age[[2]]
        age2qua = '2,'

        age20 = LGroup('20')
        age20bis = LGroup('20,')
        age20ter = LGroup(['20'])
        age20qua = '20,'

        # TODO: move assert to another test
        # self.assertEqual(age2bis, age2ter)

        age247 = age['2,4,7']
        age247bis = age[['2', '4', '7']]
        age359 = age[['3', '5', '9']]
        age468 = age['4,6,8'] >> 'even'

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
        agg = Axis((age2, age247, age359, age468, '2,6', ['3', '5', '7'], ('6', '7', '9')), "agg")
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
        self.assertTrue(age['2,6'] in agg)
        self.assertTrue(age[['2', '6']] in agg)

        self.assertTrue('3,5,7' in agg)
        self.assertTrue(['3', '5', '7'] in agg)
        self.assertTrue(age['3,5,7'] in agg)
        self.assertTrue(age[['3', '5', '7']] in agg)

        self.assertTrue('6,7,9' in agg)
        self.assertTrue(['6', '7', '9'] in agg)
        self.assertTrue(age['6,7,9'] in agg)
        self.assertTrue(age[['6', '7', '9']] in agg)

        self.assertFalse(5 in agg)
        self.assertFalse('5' in agg)
        self.assertFalse(age20 in agg)
        self.assertFalse(age20bis in agg)
        self.assertFalse(age20ter in agg)
        self.assertFalse(age20qua in agg)
        self.assertFalse('2,7' in agg)
        self.assertFalse(['2', '7'] in agg)
        self.assertFalse(age['2,7'] in agg)
        self.assertFalse(age[['2', '7']] in agg)


class TestAxisCollection(TestCase):
    def setUp(self):
        self.lipro = Axis('lipro=P01..P04')
        self.sex = Axis('sex=M,F')
        self.sex2 = Axis('sex=F,M')
        self.age = Axis('age=0..7')
        self.geo = Axis('geo=A11,A12,A13')
        self.value = Axis('value=0..10')
        self.collection = AxisCollection((self.lipro, self.sex, self.age))

    def test_init_from_group(self):
        lipro_subset = self.lipro[:'P03']
        col2 = AxisCollection((lipro_subset, self.sex))
        self.assertEqual(col2.names, ['lipro', 'sex'])
        assert_array_equal(col2.lipro.labels, ['P01', 'P02', 'P03'])
        assert_array_equal(col2.sex.labels, ['M', 'F'])

    def test_init_from_string(self):
        col = AxisCollection('age=10;sex=M,F;year=2000..2017')
        assert col.names == ['age', 'sex', 'year']
        assert list(col.age.labels) == [10]
        assert list(col.sex.labels) == ['M', 'F']
        assert list(col.year.labels) == [y for y in range(2000, 2018)]

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

    def test_setitem_name_axis_def(self):
        col = self.collection[:]
        # replace an axis with one with another name
        col['lipro'] = 'geo=A11,A12,A13'
        self.assertEqual(len(col), 3)
        self.assertEqual(col, [self.geo, self.sex, self.age])
        # replace an axis with one with the same name
        col['sex'] = 'sex=F,M'
        self.assertEqual(col, [self.geo, self.sex2, self.age])
        col['geo'] = 'lipro=P01..P04'
        self.assertEqual(col, [self.lipro, self.sex2, self.age])
        col['age'] = 'geo=A11,A12,A13'
        self.assertEqual(col, [self.lipro, self.sex2, self.geo])
        col['sex'] = 'sex=M,F'
        col['geo'] = 'age=0..7'
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
        self.assertEqual(newcol.shape, (4, 3, 8))
        newcol = newcol.replace(self.geo, self.sex)
        self.assertEqual(len(newcol), 3)
        self.assertEqual(newcol.names, ['lipro', 'sex', 'age'])
        self.assertEqual(newcol.shape, (4, 2, 8))

        # from now on, reuse original collection
        newcol = col.replace(self.sex, 3)
        self.assertEqual(len(newcol), 3)
        self.assertEqual(newcol.names, ['lipro', None, 'age'])
        self.assertEqual(newcol.shape, (4, 3, 8))

        newcol = col.replace(self.sex, ['a', 'b', 'c'])
        self.assertEqual(len(newcol), 3)
        self.assertEqual(newcol.names, ['lipro', None, 'age'])
        self.assertEqual(newcol.shape, (4, 3, 8))

        newcol = col.replace(self.sex, "letters=a,b,c")
        self.assertEqual(len(newcol), 3)
        self.assertEqual(newcol.names, ['lipro', 'letters', 'age'])
        self.assertEqual(newcol.shape, (4, 3, 8))

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
        geo = Axis('geo=A11,A12,A13')
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
        new = col + [Axis('geo=A11,A12,A13'), Axis('age=0..7')]
        self.assertEqual(new, [lipro, sex, age, geo])

        # c) with incompatible dupe
        # XXX: the "new" age axis is ignored. We might want to ignore it if it
        #  is the same but raise an exception if it is different
        with self.assertRaises(ValueError):
            col + [Axis('geo=A11,A12,A13'), Axis('age=0..6')]

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
        assert_array_equal(res.lipro_sex.labels[0], 'P01_M')
        res = col.combine_axes((lipro, age))
        self.assertEqual(res.names, ['lipro_age', 'sex'])
        self.assertEqual(res.size, col.size)
        self.assertEqual(res.shape, (4 * 8, 2))
        assert_array_equal(res.lipro_age.labels[0], 'P01_0')
        res = col.combine_axes((sex, age))
        self.assertEqual(res.names, ['lipro', 'sex_age'])
        self.assertEqual(res.size, col.size)
        self.assertEqual(res.shape, (4, 2 * 8))
        assert_array_equal(res.sex_age.labels[0], 'M_0')

    def test_info(self):
        expected = """\
4 x 2 x 8
 lipro [4]: 'P01' 'P02' 'P03' 'P04'
 sex [2]: 'M' 'F'
 age [8]: 0 1 2 ... 5 6 7"""
        self.assertEqual(self.collection.info, expected)

    def test_str(self):
        self.assertEqual(str(self.collection), "{lipro, sex, age}")

    def test_repr(self):
        self.assertEqual(repr(self.collection), """AxisCollection([
    Axis(['P01', 'P02', 'P03', 'P04'], 'lipro'),
    Axis(['M', 'F'], 'sex'),
    Axis([0, 1, 2, 3, 4, 5, 6, 7], 'age')
])""")


if __name__ == "__main__":
    pytest.main()
