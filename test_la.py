from __future__ import division, print_function

from unittest import TestCase

import numpy as np

from larray import LArray, Axis, ValueGroup, union, to_labels, to_key, srange, \
    larray_equal

#XXX: maybe we should force value groups to use tuple and families (group of
# groups to use lists, or vice versa, so that we know which is which)
# or use a class, just for that?
# group(a, b, c)
# family(group(a), b, c)


def array_equal(a, b):
    # np.array_equal is not implemented on strings in numpy < 1.9
    if (np.issubdtype(a.dtype, np.str) and np.issubdtype(b.dtype,
                                                             np.str)):
        return (a == b).all()
    else:
        return np.array_equal(a, b)


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

    def test_group(self):
        age = Axis('age', ':115')
        ages_list = ['1', '5', '9']
        self.assertEqual(age.group(ages_list), ValueGroup(age, ages_list))
        self.assertEqual(age.group('1,5,9'), ValueGroup(age, ages_list))
        self.assertEqual(age.group('1', '5', '9'), ValueGroup(age, ages_list))

        # with a slice string
        self.assertEqual(age.group('10:20'), ValueGroup(age, slice('10', '20')))

        # with name
        self.assertEqual(age.group(srange(10, 20), name='teens'),
                         ValueGroup(age, srange(10, 20), 'teens'))

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

    def test_all(self):
        age = Axis('age', ':115')
        self.assertEqual(age.all(), ValueGroup(age, slice(None)))
        self.assertEqual(age.all(), age[:])


class TestValueGroup(TestCase):
    def setUp(self):
        self.age = Axis('age', ':115')
        self.lipro = Axis('lipro', ['P%02d' % i for i in range(1, 16)])

        self.slice_both = ValueGroup(self.age, '1:5')
        self.slice_start = ValueGroup(self.age, '1:')
        self.slice_stop = ValueGroup(self.age, ':5')
        self.slice_none = ValueGroup(self.age, ':')

        self.single_value = ValueGroup(self.lipro, 'P03')
        self.list = ValueGroup(self.lipro, 'P01,P03,P07')

    def test_init(self):
        self.assertEqual(self.slice_both, ValueGroup(self.age, slice('1', '5')))
        self.assertEqual(self.slice_start,
                         ValueGroup(self.age, slice('1', None)))
        self.assertEqual(self.slice_stop, ValueGroup(self.age, slice('5')))
        self.assertEqual(self.slice_none, ValueGroup(self.age, slice(None)))

    def test_name(self):
        self.assertEqual(self.slice_both.name, '1:5')
        self.assertEqual(self.slice_start.name, '1:')
        self.assertEqual(self.slice_stop.name, ':5')
        self.assertEqual(self.slice_none.name, ':')
        self.assertEqual(self.single_value.name, 'P03')
        self.assertEqual(self.list.name, 'P01,P03,P07')

    def test_str(self):
        self.assertEqual(str(self.list), 'P01,P03,P07')

    def test_repr(self):
        #FIXME: this does not roundtrip correctly
        self.assertEqual(repr(self.list), 'lipro[P01,P03,P07]')

    def test_hash(self):
        d = {self.slice_both: 1,
             self.single_value: 2,
             self.list: 3}
        self.assertEqual(d.get(self.slice_both), 1)
        self.assertEqual(d.get(self.single_value), 2)
        self.assertEqual(d.get(self.list), 3)


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
        self.vla = vla
        self.wal = wal
        # string without commas
        self.bru = bru
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
(116, 44, 2, 15)
 age [116]: '0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' \
'15' '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' \
'30' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40' '41' '42' '43' '44' \
'45' '46' '47' '48' '49' '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' \
'60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' \
'75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' \
'90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100' '101' '102' '103' \
'104' '105' '106' '107' '108' '109' '110' '111' '112' '113' '114' '115'
 geo [44]: 'A11' 'A12' 'A13' 'A23' 'A24' 'A31' 'A32' 'A33' 'A34' 'A35' 'A36' \
'A37' 'A38' 'A41' 'A42' 'A43' 'A44' 'A45' 'A46' 'A71' 'A72' 'A73' 'A25' 'A51' \
'A52' 'A53' 'A54' 'A55' 'A56' 'A57' 'A61' 'A62' 'A63' 'A64' 'A65' 'A81' 'A82' \
'A83' 'A84' 'A85' 'A91' 'A92' 'A93' 'A21'
 sex [2]: 'H' 'F'
 lipro [15]: 'P01' 'P02' 'P03' 'P04' 'P05' 'P06' 'P07' 'P08' 'P09' 'P10' 'P11' \
'P12' 'P13' 'P14' 'P15'"""
        self.assertEqual(self.larray.info(), expected)

    def test_filter(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        # ages1_5_9 = self.la.axes.age.group('1,5,9')
        # ages1_5_9 = self.la.axes['age'].group('1,5,9')
        ages1_5_9 = age.group('1,5,9')
        ages11 = age.group('11')

        # with ValueGroup
        filtered = la.filter(age=ages1_5_9)
        self.assertEqual(filtered.shape, (3, 44, 2, 15))

        # VG with 1 value => collapse
        filtered = la.filter(age=ages11)
        self.assertEqual(filtered.shape, (44, 2, 15))

        # VG with 1 value
        #XXX: this does not work. Do we want to make this work?
        # filtered = la.filter(age=(ages11,))
        # self.assertEqual(filtered.shape, (1, 44, 2, 15))

        # list
        filtered = la.filter(age=['1', '5', '9'])
        self.assertEqual(filtered.shape, (3, 44, 2, 15))

        # string
        filtered = la.filter(age='1,5,9')
        self.assertEqual(filtered.shape, (3, 44, 2, 15))

        # multiple axes at once
        filtered = la.filter(age='1,5,9', lipro='P01,P02')
        self.assertEqual(filtered.shape, (3, 44, 2, 2))

        # multiple axes one after the other
        filtered = la.filter(age='1,5,9').filter(lipro='P01,P02')
        self.assertEqual(filtered.shape, (3, 44, 2, 2))

        # a single value for one dimension => collapse the dimension
        filtered = la.filter(sex='H')
        self.assertEqual(filtered.shape, (116, 44, 15))

        # but a list with a single value for one dimension => do not collapse
        filtered = la.filter(sex=['H'])
        self.assertEqual(filtered.shape, (116, 44, 1, 15))

        filtered = la.filter(sex='H,')
        self.assertEqual(filtered.shape, (116, 44, 1, 15))

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

    def test_sum(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru, belgium = self.vla, self.wal, self.bru, self.belgium

        # full axes reductions
        # ====================

        # everything
        self.assertEqual(la.sum(), np.asarray(la).sum())

        # using axes numbers
        self.assertEqual(la.sum(0, 2).shape, (44, 15))

        # using Axis objects
        self.assertEqual(la.sum(age).shape, (44, 2, 15))
        self.assertEqual(la.sum(age, sex).shape, (44, 15))

        # chained sum
        self.assertEqual(la.sum(age, sex).sum(geo).shape, (15,))
        self.assertEqual(la.sum(age, sex).sum(lipro, geo), la.sum())

        # getitem on aggregated
        aggregated = la.sum(age, sex)
        self.assertEqual(aggregated[self.vla].shape, (22, 15))

        # filter on aggregated
        self.assertEqual(aggregated.filter(geo=self.vla).shape, (22, 15))

        # group aggregates
        # ================

        # simple
        # ------

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        self.assertEqual(la.sum(sex='H').shape, (116, 44, 15))
        self.assertEqual(la.sum(sex='H,F').shape, (116, 44, 15))
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
        self.assertEqual(la.sum(geo=(vla, wal, bru)).shape, (116, 3, 2, 15))
        # with one label in several groups
        self.assertEqual(la.sum(sex=(['H'], ['H', 'F'])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum(sex=('H', 'H,F')).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum(sex='H;H,F').shape, (116, 44, 2, 15))

        aggregated = la.sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(aggregated.shape, (116, 4, 2, 15))
        self.assertTrue(isinstance(aggregated.axes[1].labels[0], ValueGroup))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does
        # not allow non-kwargs after kwargs.
        self.assertEqual(la.sum(age, sex, geo=(vla, wal, bru, belgium)).shape,
                         (4, 15))

        # c) chain group aggregate after axis aggregate
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(reg.shape, (4, 15))

        # group aggregates on a group-aggregated array
        # --------------------------------------------

        # 1) one group => collapse dimension
        self.assertEqual(reg.sum(lipro='P01,P02').shape, (4,))

        # 2) a tuple of one group => do not collapse dimension
        self.assertEqual((reg.sum(lipro=('P01,P02',))).shape, (4, 1))

        # 3) several groups
        self.assertEqual((reg.sum(lipro='P01;P02;:')).shape, (4, 3))

        # this is INVALID
        # TODO: raise a nice exception
        # regsum = reg.sum(lipro='P01,P02,:')

        # this is currently allowed even though it can be confusing:
        # P01 and P02 are both groups with one element each.
        self.assertEqual((reg.sum(lipro=('P01', 'P02', ':'))).shape, (4, 3))
        self.assertEqual((reg.sum(lipro=('P01', 'P02', lipro.all()))).shape,
                         (4, 3))

        # explicit groups are better
        self.assertEqual((reg.sum(lipro=('P01,', 'P02,', ':'))).shape, (4, 3))
        self.assertEqual((reg.sum(lipro=(['P01'], ['P02'], ':'))).shape, (4, 3))

        # getitem on a group-aggregated array
        # -----------------------------------

        # using a string
        vla = self.vla
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # one more level...
        self.assertEqual(reg[vla]['P03'], 389049848.0)

        # using an anonymous ValueGroup
        vla = self.geo.group(self.vla)
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # using a named ValueGroup
        vla = self.geo.group(self.vla, name='Vlaanderen')
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # filter on a group-aggregated array
        # ----------------------------------

        # using a string
        vla = self.vla
        self.assertEqual(reg.filter(geo=vla).shape, (15,))
        # using an anonymous ValueGroup
        vla = self.geo.group(self.vla)
        self.assertEqual(reg.filter(geo=vla).shape, (15,))
        # using a named ValueGroup
        vla = self.geo.group(self.vla, name='Vlaanderen')
        self.assertEqual(reg.filter(geo=vla).shape, (15,))

        # Note that reg.filter(geo=(vla,)) does NOT work. It might be a
        # little confusing for users, because reg[(vla,)] works but it is
        # normal because reg.filter(geo=(vla,)) is equivalent to:
        # reg[((vla,),)] or reg[(vla,), :]

        #TODO: check that *if* the aggregated array (reg) was created using
        # *named* groups, it can also be indexed by the group name
        # vla = self.geo.group(self.vla, name='Vlaanderen')
        # reg = aggregated.sum(geo=(vla, self.wal, self.bru, self.belgium))
        # self.assertEqual(reg['Vlaanderen'].shape, (15,))

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
        self.assertEqual(byage.filter(age=slice('17')).shape, (44, 2, 15))
        #TODO: make it work for integer indices
        # self.assertEqual(byage.filter(age=slice(18)).shape, (44, 2, 15))

    def test_isnan(self):
        self._assert_equal_raw(np.isnan(self.small), np.isnan(self.small_data))

    def test_ratio(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        reg = la.sum(age, sex, geo=(self.vla, self.wal, self.bru, self.belgium))
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