from __future__ import division, print_function

from unittest import TestCase

import numpy as np

from larray import LArray, Axis, union, OrderedSet, to_labels, ValueGroup, \
    srange


def assert_array_equal(first, other):
    assert np.array_equal(first, other), "got: %s\nexpected: %s" % (first,
                                                                    other)


class TestValueStrings(TestCase):
    def test_split(self):
        self.assertEqual(to_labels('H,F'), OrderedSet(['H', 'F']))
        self.assertEqual(to_labels('H, F'), OrderedSet(['H', 'F']))

    def test_union(self):
        self.assertEqual(union('A11,A22', 'A12,A22'),
                         OrderedSet(['A11', 'A22', 'A12']))

    def test_range(self):
        #XXX: we might want to detect the bounds are "int strings" and convert
        # them to int, because if we ever want to have more complex queries,
        # such as: arr.filter(age > 10 and age < 20) this would break for
        # string values (because '10' < '2')
        self.assertEqual(to_labels('0:115'), srange(116))
        self.assertEqual(to_labels(':115'), srange(116))
        self.assertRaises(ValueError, to_labels, '10:')
        self.assertRaises(ValueError, to_labels, ':')


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

        # string
        assert_array_equal(Axis('sex', 'H,F').labels, sex_array)

        # ordered set
        assert_array_equal(Axis('sex', OrderedSet(sex_list)).labels, sex_array)

        # list of ints
        age = Axis('age', range(116))
        assert_array_equal(age.labels, np.arange(116))

    def test_group(self):
        age = Axis('age', ':115')
        ages_list = ['1', '5', '9']
        age_group = age.group(ages_list)
        self.assertEqual(age_group, ValueGroup(age, ages_list))
        self.assertEqual(age.group('1,5,9'), ValueGroup(age, ages_list))

        # with name
        vla = geo.group(vla, name='vla')
        wal = geo.group(wal, name='wal')
        bru = geo.group('A21', name='bru')
        belgium = geo.group(':', name='belgium')
        # belgium = geo[:]
        # belgium = geo.all() #'belgium')


class TestValueGroup(TestCase):
    def setUp(self):
        self.age = Axis('age', ':115')
        self.sex = Axis('sex', 'H,F')
        self.lipro = Axis('lipro', ['P%02d' % i for i in range(1, 16)])

        # list of strings
        # assert_array_equal(.labels, sex_array)

    def test_list(self):
        vg = ValueGroup(self.sex, 'H,')
        vg = ValueGroup(self.sex, '1,5,9')
        self.slice_group = ValueGroup(age, '1:5')

        self.assertEqual(self.list_group.name, "yada")
        self.assertEqual(self.slice_group.name, "yada")

    def test_str(self):
        self.assertEqual(str(self.list_group), "yada")

    def test_repr(self):
        self.assertEqual(repr(self.list_group), "yada")


class TestLArray(TestCase):
    def setUp(self):
        lipro_labels = ['P%02d' % i for i in range(1, 16)]
        lipro = Axis('lipro', ','.join(lipro_labels))
        assert np.array_equal(lipro.labels, np.array(lipro_labels))
        lipro = Axis('lipro', lipro_labels)
        assert np.array_equal(lipro.labels, np.array(lipro_labels))

        age_labels = range(116)
        age = Axis('age', age_labels)
        assert np.array_equal(age.labels, np.arange(116))
        age = Axis('age', '0:115')  # stop bound is inclusive !
        assert np.array_equal(age.labels, np.arange(116))

        sex_labels = ['H', 'F']
        # test with a space
        sex = Axis('sex', 'H, F')
        assert np.array_equal(sex.labels, np.array(sex_labels))

        vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,' \
              'A43,A44,A45,A46,A71,A72,A73'
        wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,' \
              'A83,A84,A85,A91,A92,A93'
        bru = 'A21'

        geo = Axis('geo', union(vla, wal, bru))
        self.array = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
                                                 .astype(float)
        self.larray = LArray(self.array, axes=(age, geo, sex, lipro))

    def test_info(self):
        #XXX: make .info into a property?
        # self.assertEqual(self.bel.info, "abc")
        self.assertEqual(self.larray.info(), "abc")

    def test_filter(self):
        la = self.larray

        # ages1_5_9 = self.bel.axes.age.group('1,5,9')
        # ages1_5_9 = self.bel.axes['age'].group('1,5,9')
        ages1_5_9 = la.axes[0].group('1,5,9')

        # with ValueGroup
        filtered = la.filter(age=ages1_5_9)

        # list
        filtered = la.filter(age=['1', '5', '9'])

        # string
        filtered = la.filter(age='1,5,9')
        #XXX: list of strings?

        # multiple axes at once
        filtered = la.filter(age='1,5,9', lipro='P01,P02')

        # a single value for one dimension => collapse the dimension
        filtered = la.filter(sex='H', lipro='P01,P02')

        # but a list with a single value for one dimension => do not collapse
        filtered = la.filter(sex=['H'], lipro='P01,P02')
        filtered = la.filter(sex='H,', lipro='P01,P02')

        # with duplicate keys
        #XXX: do we want to support this?
        # filtered = la.filter(lipro='P01,P02,P01')

    def test_sum(self):
        bel = self.larray

        # aggbel = bel.sum(0, 2) # 2d (geo, lipro)
        # print "aggbel.shape", aggbel.shape
        aggbel = bel.sum(age, sex) # 2d (geo, lipro)
        print("aggbel.shape", aggbel.shape)

        #belgium = vla + wal + bru # equivalent
        #wal_bru = belgium - vla
        #wal_bru = wal + bru # equivalent

        # aggbelvla = aggbel[vla]
        aggbelvla = bel.filter(age == '10')
        # strings vs numbers: '
        aggbelvla = filteraggbel[age > 10 and age < 20]
        print("aggbel[vla]", aggbelvla.info())
        print(aggbelvla)

        # 2d (geo, lipro) -> 2d (geo', lipro)
        reg = aggbel.sum(geo=(vla, wal, bru, belgium))
        print("reg", reg.info())
        print(reg)

        # regsum = reg.sum(lipro=('P01', 'P02', lipro.all()))
        regsum = reg.sum(lipro='P01,P02,P03,:')
        print(regsum)

        regvla = reg[vla]
        # print reg['vla,P15']
        print("regvla", regvla.info())
        print(regvla)
        print(regvla.info())

        regvlap03 = regvla['P03']
        print("regvlap03 axes", regvlap03.info())
        print(regvlap03)

        # child = age[:17] # stop bound is inclusive !
        child = age.group(':17') # stop bound is inclusive !
        # working = age[18:64] # create a ValueGroup(Axis(age), [18, ..., 64], '18:64')
        working = age.group('18:64') # create a ValueGroup(Axis(age), [18, ..., 64],
        # '18:64')
        # retired = age[65:]
        retired = age.group('65:')
        #arr3x = geo.group('A3*') # * match one or more chars
        #arr3x = geo.group('A3?') # ? matches one char (equivalent in this case)
        #arr3x = geo.seq('A31', 'A38')
        # arr3x = geo['A31':'A38'] # not equivalent! (if A22 is between A31 and A38)

        test = bel.filter(age=child)
        print("test", test.info())
        # test = bel.filter(age=age[:17]).filter(geo=belgium)
        test = bel.filter(age=':17').filter(geo=belgium)
        print(test.info())
        # test = bel.filter(age=range(18))
        # print test.info()

        # ages = bel.sum(age=(child, 5, working, retired))
        ages = bel.sum(age=(child, '5:10', working, retired))
        print(ages.info())
        ages2 = ages.filter(age=child)
        print(ages2.info())

        #print "ages.filter", ages.filter(age=:17) # invalid syntax
        #print "ages.filter(age=':17')", ages.filter(age=':17')
        #print "ages.filter(age=slice(17))", ages.filter(age=slice(17))

        total = reg.sum()            # total (including duplicates like belgium?)
        print("total", total)
        # total (including duplicates like belgium?)
        total = reg.sum(geo, lipro)
        print("total", total)

        x = bel.sum(age) # 3d matrix
        print("sum(age)")
        print(x.info())

        x = bel.sum(lipro, geo) # 2d matrix
        print("sum(lipro, geo)")
        print(x.info())

        x = bel.sum(lipro, geo=geo.all()) # the same 2d matrix?
        x = bel.sum(lipro, geo=':') # the same 2d matrix?
        x = bel.sum(lipro, geo='A11:A33') # include everything between the two labels
        #  (it can include 'A63')
        print("sum(lipro, geo=geo.all())")
        print("sum(lipro, geo=geo.all())")
        print(x.info())

        x = bel.sum(lipro, sex='H') # a 3d matrix (sex dimension of size 1)
        print("sum(lipro, sex='H')")
        print(x.info())

        # x = bel.sum(lipro, sex=(['H'], ['H', 'F'])) # idem
        x = bel.sum(lipro, sex=('H', 'H,F')) # idem
        x = bel.sum(lipro, sex='H;H,F') # <-- abbreviation
        print("sum(lipro, sex=('H',))")
        print(x.info())

        x = bel.sum(lipro, geo=(geo.all(),)) # 3d matrix (geo dimension of size 1)
        print("sum(lipro, geo=(geo.all(),))")
        print(x.info())
        #print bel.sum(lipro, geo=(vla, wal, bru)) # 3d matrix

        #bel.sum(lipro, geo=(vla, wal, bru), sex) # <-- not allowed!!! (I think we can live with that)

    def test_ratio(self):
        ratio = reg.ratio(geo, lipro)
        print("reg.ratio(geo, lipro)")
        print(ratio.info())
        print(ratio)
        print(ratio.sum())

    def test_reorder(self):
        newbel = bel.reorder(age, geo, sex, lipro)

        newbel = bel.reorder(age, geo)

    def test_arithmetics(self):
        small_data = np.random.randn(2, 15)
        small = LArray(small_data, axes=(sex, lipro))
        print(small)
        print(small + small)
        print(small * 2)
        print(2 * small)
        print(small + 1)
        print(1 + small)
        print(30 / small)
        print(30 / (small + 1))
        print(small / small)
        small_int = LArray(small_data, axes=(sex, lipro))
        print("truediv")
        print(small_int / 2)
        print("floordiv")
        print(small_int // 2)

    def test_excel_export(self):
        print("excel export", end='')
        reg.to_excel('c:\\tmp\\reg.xlsx', '_')
        #ages.to_excel('c:/tmp/ages.xlsx')
        print("done")

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