from unittest import TestCase

import numpy as np
import pandas as pd

from larray import (LArray, Axis, ValueGroup, union, to_labels, to_key,
                    srange, larray_equal, read_csv, df_aslarray)
from utils import array_equal

#XXX: maybe we should force value groups to use tuple and families (group of
# groups to use lists, or vice versa, so that we know which is which)
# or use a class, just for that?
# group(a, b, c)
# family(group(a), b, c)


def assert_equal_factory(test_func):
    def assert_equal(a, b):
        assert test_func(a, b), "got: %s\nexpected: %s" % (a, b)
    return assert_equal


assert_array_equal = assert_equal_factory(array_equal)
assert_larray_equal = assert_equal_factory(larray_equal)


class TestValueStrings(TestCase):
    def test_split(self):
        self.assertEqual(to_labels('H,F'), ['H', 'F'])
        self.assertEqual(to_labels('H, F'), ['H', 'F'])

    def test_union(self):
        self.assertEqual(union('A11,A22', 'A12,A22'), ['A11', 'A22', 'A12'])

    def test_range(self):
        #XXX: we might want to return real int instead, because if we ever
        # want to have more complex queries, such as:
        # arr.filter(age > 10 and age < 20)
        # this would break for string values (because '10' < '2')
        self.assertEqual(to_labels('0:115'), srange(116))
        self.assertEqual(to_labels(':115'), srange(116))
        self.assertRaises(ValueError, to_labels, '10:')
        self.assertRaises(ValueError, to_labels, ':')


class TestKeyStrings(TestCase):
    def test_nonstring(self):
        self.assertEqual(to_key(('H', 'F')), ['H', 'F'])
        self.assertEqual(to_key(['H', 'F']), ['H', 'F'])

    def test_split(self):
        self.assertEqual(to_key('H,F'), ['H', 'F'])
        self.assertEqual(to_key('H, F'), ['H', 'F'])
        self.assertEqual(to_key('H,'), ['H'])
        self.assertEqual(to_key('H'), 'H')

    def test_slice_strings(self):
        #XXX: we might want to return real int instead, because if we ever
        # want to have more complex queries, such as:
        # arr.filter(age > 10 and age < 20)
        # this would break for string values (because '10' < '2')
        #XXX: these two examples return different things, do we want that?
        self.assertEqual(to_key('0:115'), slice('0', '115'))
        self.assertEqual(to_key(':115'), slice('115'))
        self.assertEqual(to_key('10:'), slice('10', None))
        self.assertEqual(to_key(':'), slice(None))


class TestAxis(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        sex_list = ['H', 'F']
        sex_array = np.array(sex_list)

        # list of strings
        assert_array_equal(Axis('sex', sex_list).labels, sex_array)
        # array of strings
        assert_array_equal(Axis('sex', sex_array).labels, sex_array)
        # single string
        assert_array_equal(Axis('sex', 'H,F').labels, sex_array)
        # list of ints
        assert_array_equal((Axis('age', range(116))).labels, np.arange(116))
        # range-string
        assert_array_equal((Axis('age', ':115')).labels, np.array(srange(116)))

    def test_eq(self):
        self.assertTrue(Axis('sex', 'H,F') == Axis('sex', 'H,F'))
        self.assertTrue(Axis('sex', 'H,F') == Axis('sex', ['H', 'F']))
        self.assertFalse(Axis('sex', 'M,F') == Axis('sex', 'H,F'))
        self.assertFalse(Axis('sex1', 'H,F') == Axis('sex2', 'H,F'))
        self.assertFalse(Axis('sex1', 'M,F') == Axis('sex2', 'H,F'))

    def test_ne(self):
        self.assertFalse(Axis('sex', 'H,F') != Axis('sex', 'H,F'))
        self.assertFalse(Axis('sex', 'H,F') != Axis('sex', ['H', 'F']))
        self.assertTrue(Axis('sex', 'M,F') != Axis('sex', 'H,F'))
        self.assertTrue(Axis('sex1', 'H,F') != Axis('sex2', 'H,F'))
        self.assertTrue(Axis('sex1', 'M,F') != Axis('sex2', 'H,F'))

    def test_group(self):
        age = Axis('age', ':115')
        ages_list = ['1', '5', '9']
        self.assertEqual(age.group(ages_list), ValueGroup(ages_list, axis=age))
        self.assertEqual(age.group(ages_list), ValueGroup(ages_list))
        self.assertEqual(age.group('1,5,9'), ValueGroup(ages_list))
        self.assertEqual(age.group('1', '5', '9'), ValueGroup(ages_list))

        # with a slice string
        self.assertEqual(age.group('10:20'), ValueGroup(slice('10', '20')))

        # with name
        group = age.group(srange(10, 20), name='teens')
        self.assertEqual(group.key, srange(10, 20))
        self.assertEqual(group.name, 'teens')
        self.assertEqual(group.axis, age)

        #TODO: support more stuff in string groups
        # arr3x = geo.group('A3*') # * match one or more chars
        # arr3x = geo.group('A3?') # ? matches one char (equivalent in this case)
        # arr3x = geo.seq('A31', 'A38') # not equivalent to geo['A31:A38'] !
        #                               # (if A22 is between A31 and A38)

    def test_getitem(self):
        age = Axis('age', ':115')
        vg = age.group(':17')
        # these are equivalent
        self.assertEqual(age[:'17'], vg)
        self.assertEqual(age[':17'], vg)

        group = age[:]
        self.assertEqual(group.key, slice(None))
        self.assertEqual(group.axis, age)

    def test_all(self):
        age = Axis('age', ':115')
        group = age.all()
        self.assertEqual(group.key, slice(None))
        self.assertEqual(group.axis, age)

    def test_contains(self):
        # normal Axis
        age = Axis('age', ':10')

        age2 = age.group('2')
        age2bis = age.group('2,')
        age2ter = age.group(['2'])
        age2qua = '2,'

        age20 = ValueGroup('20')
        age20bis = ValueGroup('20,')
        age20ter = ValueGroup(['20'])
        age20qua = '20,'

        #TODO: move assert to another test
        self.assertEqual(age2bis, age2ter)

        age247 = age.group('2,4,7')
        age247bis = age.group(['2', '4', '7'])
        age359 = age.group(['3', '5', '9'])
        age468 = age.group('4,6,8', name='even')

        self.assertFalse(5 in age)
        self.assertTrue('5' in age)

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
        agg = Axis("agg", (age2, age247, age359, age468,
                           '2,6', ['3', '5', '7']))
        self.assertTrue(age2 in agg)
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
        self.assertTrue('4,6,8' in agg)
        self.assertTrue(['4', '6', '8'] in agg)
        self.assertTrue('even' in agg)

        self.assertTrue('2,6' in agg)
        self.assertTrue(['2', '6'] in agg)
        self.assertTrue(age.group('2,6') in agg)
        self.assertTrue(age.group(['2', '6']) in agg)

        self.assertTrue('3,5,7' in agg)
        self.assertTrue(['3', '5', '7'] in agg)
        self.assertTrue(age.group('3,5,7') in agg)
        self.assertTrue(age.group(['3', '5', '7']) in agg)

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


class TestValueGroup(TestCase):
    def setUp(self):
        self.age = Axis('age', ':115')
        self.lipro = Axis('lipro', ['P%02d' % i for i in range(1, 16)])

        self.slice_full = ValueGroup('1:5', "full", self.age)
        self.slice_named = ValueGroup('1:5', "named")
        self.slice_both = ValueGroup('1:5')
        self.slice_start = ValueGroup('1:')
        self.slice_stop = ValueGroup(':5')
        self.slice_none = ValueGroup(':')

        self.single_value = ValueGroup('P03')
        self.list = ValueGroup('P01,P03,P07')
        self.list_named = ValueGroup('P01,P03,P07', "P137")

    def test_init(self):
        self.assertEqual(self.slice_full.name, "full")
        self.assertEqual(self.slice_full.key, '1:5')
        self.assertEqual(self.slice_full.axis, self.age)
        self.assertEqual(self.slice_named.name, "named")
        self.assertEqual(self.slice_named.key, '1:5')
        self.assertEqual(self.slice_both.key, '1:5')
        self.assertEqual(self.slice_start.key, '1:')
        self.assertEqual(self.slice_stop.key, ':5')
        self.assertEqual(self.slice_none.key, ':')

        self.assertEqual(self.single_value.key, 'P03')
        self.assertEqual(self.list.key, 'P01,P03,P07')

    def test_eq(self):
        self.assertEqual(self.slice_both, self.slice_full)
        self.assertEqual(self.slice_both, self.slice_named)
        self.assertEqual(self.slice_both, ValueGroup(slice('1', '5')))
        self.assertEqual(self.slice_start, ValueGroup(slice('1', None)))
        self.assertEqual(self.slice_stop, ValueGroup(slice('5')))
        self.assertEqual(self.slice_none, ValueGroup(slice(None)))
        self.assertEqual(self.list, ValueGroup(['P01', 'P03', 'P07']))
        # test with raw objects
        self.assertEqual(self.slice_both, '1:5')
        self.assertEqual(self.slice_start, '1:')
        self.assertEqual(self.slice_stop, ':5')
        self.assertEqual(self.slice_none, ':')
        self.assertEqual(self.slice_both, slice('1', '5'))
        self.assertEqual(self.slice_start, slice('1', None))
        self.assertEqual(self.slice_stop, slice('5'))
        self.assertEqual(self.slice_none, slice(None))
        self.assertEqual(self.list, 'P01,P03,P07')
        self.assertEqual(self.list, ' P01 , P03 , P07 ')
        self.assertEqual(self.list, ['P01', 'P03', 'P07'])
        self.assertEqual(self.list, ('P01', 'P03', 'P07'))

    def test_hash(self):
        d = {self.slice_both: 1,
             self.single_value: 2,
             self.list_named: 3}
        # target a ValueGroup with an equivalent ValueGroup
        self.assertEqual(d.get(self.slice_both), 1)
        self.assertEqual(d.get(self.single_value), 2)
        self.assertEqual(d.get(self.list), 3)
        self.assertEqual(d.get(self.list_named), 3)
        # this cannot and will never work!
        # self.assertEqual(d.get("P137"), 3)

        # target a ValueGroup with an equivalent key
        self.assertEqual(d.get('1:5'), 1)
        self.assertEqual(d.get('P03'), 2)
        self.assertEqual(d.get('P01,P03,P07'), 3)

        # this cannot and will never work!
        # hash(str) and hash(tuple) are not special, so ' P01 ,...' and
        # ('P01', ...) do not hash to the same value than 'P01,P03...", which is
        # our "canonical hash"
        # self.assertEqual(d.get(' P01 , P03 , P07 '), 3)
        # self.assertEqual(d.get(('P01', 'P03', 'P07')), 3)

        # target a simple key with an equivalent ValueGroup
        d = {'1:5': 1,
             'P03': 2,
             'P01,P03,P07': 3}
        self.assertEqual(d.get(self.slice_both), 1)
        self.assertEqual(d.get(self.single_value), 2)
        self.assertEqual(d.get(self.list), 3)
        self.assertEqual(d.get(ValueGroup(' P01 , P03 , P07 ')), 3)
        self.assertEqual(d.get(ValueGroup(('P01', 'P03', 'P07'))), 3)

    def test_str(self):
        self.assertEqual(str(self.slice_full), 'full')
        self.assertEqual(str(self.slice_named), 'named')
        self.assertEqual(str(self.slice_both), '1:5')
        self.assertEqual(str(self.slice_start), '1:')
        self.assertEqual(str(self.slice_stop), ':5')
        self.assertEqual(str(self.slice_none), ':')
        self.assertEqual(str(self.single_value), 'P03')
        self.assertEqual(str(self.list), 'P01,P03,P07')

    def test_repr(self):
        #FIXME: add axis
        self.assertEqual(repr(self.slice_full), "ValueGroup('1:5', 'full')")
        self.assertEqual(repr(self.slice_named), "ValueGroup('1:5', 'named')")
        self.assertEqual(repr(self.slice_both), "ValueGroup('1:5')")
        self.assertEqual(repr(self.list), "ValueGroup('P01,P03,P07')")


class TestLArray(TestCase):
    def _assert_equal_raw(self, la, raw):
        assert_array_equal(np.asarray(la), raw)

    def setUp(self):
        self.lipro = Axis('lipro', ['P%02d' % i for i in range(1, 16)])
        self.age = Axis('age', ':115')
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

        #belgium = vla + wal + bru # equivalent
        #wal_bru = belgium - vla
        #wal_bru = wal + bru # equivalent

        self.geo = Axis('geo', self.belgium)

        self.array = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
                                                 .astype(float)
        self.larray = LArray(self.array,
                             axes=(self.age, self.geo, self.sex, self.lipro))

        self.small_data = np.random.randn(2, 15)
        self.small = LArray(self.small_data, axes=(self.sex, self.lipro))

    def test_info(self):
        #XXX: make .info into a property?
        # self.assertEqual(self.bel.info, "abc")
        expected = """\
116 x 44 x 2 x 15
 age [116]: '0' '1' '2' ... '113' '114' '115'
 geo [44]: 'A11' 'A12' 'A13' ... 'A92' 'A93' 'A21'
 sex [2]: 'H' 'F'
 lipro [15]: 'P01' 'P02' 'P03' ... 'P13' 'P14' 'P15'"""
        self.assertEqual(self.larray.info(), expected)

    def test_getitem(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes

        ages1_5_9 = age.group('1,5,9')

        filtered = la[ages1_5_9]
        self.assertEqual(filtered.shape, (3, 44, 2, 15))
        self._assert_equal_raw(filtered, raw[[1, 5, 9]])

    def test_set(self):
        raw = self.array.copy()
        la = self.larray.copy()
        age, geo, sex, lipro = la.axes
        ages1_5_9 = age.group('1,5,9')

        la.set(la[ages1_5_9] + 25.0, age=ages1_5_9)

        #XXX: We could also implement:
        # la.xs[ages1_5_9] = la[ages1_5_9] + 25.0
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
        self._assert_equal_raw(la, raw)

    def test_filter(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        # ages1_5_9 = self.la.axes.age.group('1,5,9')
        # ages1_5_9 = self.la.axes['age'].group('1,5,9')
        ages1_5_9 = age.group('1,5,9')
        ages11 = age.group('11')

        # with ValueGroup
        self.assertEqual(la.filter(age=ages1_5_9).shape, (3, 44, 2, 15))

        #FIXME: this should raise a comprehensible error!
        # self.assertEqual(la.filter(age=[ages1_5_9]).shape, (3, 44, 2, 15))

        # VG with 1 value => collapse
        self.assertEqual(la.filter(age=ages11).shape, (44, 2, 15))

        # VG with a list of 1 value => do not collapse
        self.assertEqual(la.filter(age=age.group(['11'])).shape, (1, 44, 2, 15))

        # VG with a list of 1 value defined as a string => do not collapse
        self.assertEqual(la.filter(age=age.group('11,')).shape, (1, 44, 2, 15))

        # VG with 1 value
        #XXX: this does not work. Do we want to make this work?
        # filtered = la.filter(age=(ages11,))
        # self.assertEqual(filtered.shape, (1, 44, 2, 15))

        # list
        self.assertEqual(la.filter(age=['1', '5', '9']).shape, (3, 44, 2, 15))

        # string
        self.assertEqual(la.filter(age='1,5,9').shape, (3, 44, 2, 15))

        # multiple axes at once
        self.assertEqual(la.filter(age='1,5,9', lipro='P01,P02').shape,
                         (3, 44, 2, 2))

        # multiple axes one after the other
        self.assertEqual((la.filter(age='1,5,9').filter(lipro='P01,P02')).shape,
                         (3, 44, 2, 2))

        # a single value for one dimension => collapse the dimension
        self.assertEqual(la.filter(sex='H').shape, (116, 44, 15))

        # but a list with a single value for one dimension => do not collapse
        self.assertEqual(la.filter(sex=['H']).shape, (116, 44, 1, 15))

        self.assertEqual(la.filter(sex='H,').shape, (116, 44, 1, 15))

        # with duplicate keys
        #XXX: do we want to support this? I don't see any value in that but
        # I might be short-sighted.
        # filtered = la.filter(lipro='P01,P02,P01')

        #XXX: we could abuse python to allow naming groups via Axis.__getitem__
        # (but I doubt it is a good idea).
        # child = age[':17', 'child']

        # slices
        # ------

        # VG slice
        self.assertEqual(la.filter(age=age[':17']).shape, (18, 44, 2, 15))
        # string slice
        self.assertEqual(la.filter(age=':17').shape, (18, 44, 2, 15))
        # raw slice
        self.assertEqual(la.filter(age=slice('17')).shape, (18, 44, 2, 15))

        # filter chain with a slice
        self.assertEqual(la.filter(age=':17').filter(geo='A12,A13').shape,
                         (18, 2, 2, 15))

    def test_filter_multiple_axes(self):
        la = self.larray

        # multiple values in each group
        self.assertEqual(la.filter(age='1,5,9', lipro='P01,P02').shape,
                         (3, 44, 2, 2))
        # with a group of one value
        self.assertEqual(la.filter(age='1,5,9', sex='H,').shape, (3, 44, 1, 15))

        # with a discarded axis (there is a scalar in the key)
        self.assertEqual(la.filter(age='1,5,9', sex='H').shape, (3, 44, 15))

        # with a discarded axis that is not adjacent to the ix_array axis
        # ie with a sliced axis between the scalar axis and the ix_array axis
        # since our array has a axes: age, geo, sex, lipro, any of the
        # following should be tested: age+sex / age+lipro / geo+lipro
        # additionally, if the ix_array axis was first (ie ix_array on age),
        # it worked even before the issue was fixed, since the "indexing"
        # subspace is tacked-on to the beginning (as the first dimension)
        self.assertEqual(la.filter(age='57', sex='H,F').shape,
                         (44, 2, 15))
        self.assertEqual(la.filter(age='57', lipro='P01,P05').shape,
                         (44, 2, 2))
        self.assertEqual(la.filter(geo='A57', lipro='P01,P05').shape,
                         (116, 2, 2))

    def test_sum_full_axes(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        # everything
        self.assertEqual(la.sum(), np.asarray(la).sum())

        # using axes numbers
        self.assertEqual(la.sum(0, 2).shape, (44, 15))

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

    def test_group_agg(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        self.assertEqual(la.sum(sex='H').shape, (116, 44, 15))
        self.assertEqual(la.sum(sex='H,F').shape, (116, 44, 15))

        self.assertEqual(la.sum(geo='A11,A21,A25').shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo.group('A11,A21,A25')).shape,
                         (116, 2, 15))

        self.assertEqual(la.sum(geo=geo.all()).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=':').shape, (116, 2, 15))
        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the previous
        # tests.
        self.assertEqual(la.sum(geo='A11:A21').shape, (116, 2, 15))
        assert_larray_equal(la.sum(geo='A11:A21'), la.sum(geo=':'))

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

        # using an anonymous ValueGroup
        vla = self.geo.group(self.vla_str)
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # using a named ValueGroup
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
        # using an anonymous ValueGroup
        vla = self.geo.group(self.vla_str)
        self.assertEqual(reg.filter(geo=vla).shape, (15,))
        # using a named ValueGroup
        vla = self.geo.group(self.vla_str, name='Vlaanderen')
        self.assertEqual(reg.filter(geo=vla).shape, (15,))

        # Note that reg.filter(geo=(vla,)) does NOT work. It might be a
        # little confusing for users, because reg[(vla,)] works but it is
        # normal because reg.filter(geo=(vla,)) is equivalent to:
        # reg[((vla,),)] or reg[(vla,), :]

        # mixed VG/string slices
        child = age[':17']
        working = age['18:64']
        retired = age['65:']
        byage = la.sum(age=(child, '5', working, retired))
        self.assertEqual(byage.shape, (4, 44, 2, 15))
        byage = la.sum(age=(child, '5:10', working, retired))
        self.assertEqual(byage.shape, (4, 44, 2, 15))

        # filter on an aggregated larray created with mixed groups
        self.assertEqual(byage.filter(age=child).shape, (44, 2, 15))
        self.assertEqual(byage.filter(age=':17').shape, (44, 2, 15))

        #TODO: make this work
        # self.assertEqual(byage.filter(age=slice('17')).shape, (44, 2, 15))
        #TODO: make it work for integer indices
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
        la = self.larray
        age, geo, sex, lipro = la.axes
        fla = geo.group(self.vla_str, name='Flanders')
        wal = geo.group(self.wal_str, name='Wallonia')
        bru = geo.group(self.bru_str, name='Brussel')
        self.assertEqual(la.sum(geo=(fla, wal, bru)).shape, (116, 3, 2, 15))

    def test_sum_named_vg_groups_string_indexable(self):
        """
        an aggregated array (reg) created using *named* groups should also be
        indexable by the group name
        """
        la, geo = self.larray, self.geo
        vla = geo.group(self.vla_str, name='Flanders')
        wal = geo.group(self.wal_str, name='Wallonia')
        bru = geo.group(self.bru_str, name='Brussels')
        bel = geo.all(name='Belgium')
        reg = la.sum(geo=(vla, wal, bru, bel))
        self.assertEqual(reg.filter(geo='Flanders').shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo='Flanders,Wallonia').shape,
                         (116, 2, 2, 15))

    def test_sum_with_groups_from_other_axis(self):
        small = self.small

        # use a group from another *compatible* axis
        lipro2 = Axis('lipro', ['P%02d' % i for i in range(1, 16)])
        self.assertEqual(small.sum(lipro=lipro2.group('P01,P03')).shape, (2,))

        # use group from another *incompatible* axis
        #XXX: I am not sure anymore we should be so precise
        # lipro3 = Axis('lipro', 'P01,P03,P05')
        # self.assertRaises(ValueError, small.sum, lipro=lipro3.group('P01,P03'))

    def test_ratio(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        reg = la.sum(age, sex, geo=(self.vla_str, self.wal_str, self.bru_str,
                                    self.belgium))
        self.assertEqual(reg.shape, (4, 15))

        ratio = reg.ratio(geo, lipro)
        self.assertEqual(ratio.shape, (4, 15))
        assert_array_equal(ratio, reg / reg.sum(geo, lipro))
        self.assertEqual(ratio.sum(), 1.0)

    def test_reorder(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        reordered = la.reorder(geo, age, lipro, sex)
        self.assertEqual(reordered.shape, (44, 116, 15, 2))

        reordered = la.reorder(lipro, age)
        self.assertEqual(reordered.shape, (15, 116, 44, 2))

    def test_arithmetics(self):
        raw = self.small_data
        la = self.small

        self._assert_equal_raw(la + la, raw + raw)
        self._assert_equal_raw(la * 2, raw * 2)
        self._assert_equal_raw(2 * la, 2 * raw)
        self._assert_equal_raw(la + 1, raw + 1)
        self._assert_equal_raw(1 + la, 1 + raw)
        self._assert_equal_raw(30 / la, 30 / raw)
        self._assert_equal_raw(30 / (la + 1), 30 / (raw + 1))
        self._assert_equal_raw(la / la, raw / raw)

        raw_int = raw.astype(int)
        la_int = LArray(raw_int, axes=(self.sex, self.lipro))
        self._assert_equal_raw(la_int / 2, raw_int / 2)
        self._assert_equal_raw(la_int // 2, raw_int // 2)

    def test_mean(self):
        la = self.small
        raw = self.small_data

        sex, lipro = la.axes
        self._assert_equal_raw(la.mean(lipro), raw.mean(1))

    def test_append(self):
        la = self.small
        sex, lipro = la.axes

        la = la.append(lipro=la.sum(lipro), label='sum')
        self.assertEqual(la.shape, (2, 16))
        la = la.append(sex=la.sum(sex), label='sum')
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

        #TODO: check how awful the syntax is with an axis that is not last
        # or first
        l2 = LArray(la[:, :'P14'], axes=[sex, Axis('lipro', lipro.labels[1:])])
        l2 = LArray(la[:, :'P14'],
                    axes=[sex, lipro.subaxis(slice('P02', None))])

        # We can also modify the axis in-place (dangerous!)
        # lipro.labels = np.append(lipro.labels[1:], lipro.labels[0])
        l2 = la[:, 'P02':]
        #FIXME: the mapping is not updated when .labels change
        l2.axes[1].labels = lipro.labels[1:]

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

    # def test_excel_export(self):
    #     la = self.larray
    #     age, geo, sex, lipro = la.axes
    #
    #     reg = la.sum(age, sex, geo=(self.vla, self.wal, self.bru, self.belgium))
    #     self.assertEqual(reg.shape, (4, 15))
    #
    #     print("excel export", end='')
    #     reg.to_excel('c:\\tmp\\reg.xlsx', '_')
    #     #ages.to_excel('c:/tmp/ages.xlsx')
    #     print("done")

    def test_readcsv(self):
        # la = read_csv('test1d.csv')
        # self.assertEqual(la.ndim, 1)
        # self.assertEqual(la.shape, (5,))
        # self.assertEqual(la.axes_names, ['age'])
        # self._assert_equal_raw(la[0], [3722])

        la = read_csv('test2d.csv')
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes_names, ['age', 'time'])
        self._assert_equal_raw(la[0, :], [3722, 3395, 3347])

        la = read_csv('test3d.csv')
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (5, 2, 3))
        self.assertEqual(la.axes_names, ['age', 'sex', 'time'])
        self._assert_equal_raw(la[0, 'F', :], [3722, 3395, 3347])

        la = read_csv('test5d.csv')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes_names, ['arr', 'age', 'sex', 'nat', 'time'])
        self._assert_equal_raw(la[1, 0, 'F', 1, :], [3722, 3395, 3347])

    def test_df_aslarray(self):
        dt = [('age', int), ('sex\\time', 'U1'),
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
        df.set_index(['age', 'sex\\time'], inplace=True)

        la = df_aslarray(df)
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (4, 2, 3))
        self.assertEqual(la.axes_names, ['age', 'sex', 'time'])
        self._assert_equal_raw(la[0, 'F', :], [3722, 3395, 3347])

    def test_to_csv(self):
        la = read_csv('test5d.csv')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes_names, ['arr', 'age', 'sex', 'nat', 'time'])
        self._assert_equal_raw(la[1, 0, 'F', 1, :], [3722, 3395, 3347])

        la.to_csv('out.csv')
        result = ['arr,age,sex,nat\\time,2007,2010,2013\n',
                  '1,0,F,1,3722,3395,3347\n',
                  '1,0,F,2,338,316,323\n']
        self.assertEqual(open('out.csv').readlines()[:3], result)

        la.to_csv('out.csv', transpose=False)
        result = ['arr,age,sex,nat,time,0\n', '1,0,F,1,2007,3722\n',
                  '1,0,F,1,2010,3395\n']
        self.assertEqual(open('out.csv').readlines()[:3], result)

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