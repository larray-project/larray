from __future__ import absolute_import, division, print_function

import os
import sys
from unittest import TestCase

import pytest
import numpy as np
import pandas as pd

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.tests.common import abspath, assert_array_equal, assert_array_nan_equal, assert_larray_equiv
from larray import (LArray, Axis, LGroup, union, zeros, zeros_like, ndrange, ndtest, ones, eye, diag, stack,
                    clip, exp, where, X, mean, isnan, round, read_hdf, read_csv, read_eurostat, read_excel,
                    from_lists, from_string, open_excel, from_frame, sequence)
from larray.core.axis import _to_ticks, _to_key


class TestValueStrings(TestCase):
    def test_split(self):
        self.assertEqual(_to_ticks('M,F'), ['M', 'F'])
        self.assertEqual(_to_ticks('M, F'), ['M', 'F'])

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
        self.assertEqual(_to_key(('M', 'F')), ['M', 'F'])
        self.assertEqual(_to_key(['M', 'F']), ['M', 'F'])

    def test_split(self):
        self.assertEqual(_to_key('M,F'), ['M', 'F'])
        self.assertEqual(_to_key('M, F'), ['M', 'F'])
        self.assertEqual(_to_key('M,'), ['M'])
        self.assertEqual(_to_key('M'), 'M')

    def test_slice_strings(self):
        # these two examples have different results and this is fine because numeric axes do not necessarily start at 0
        self.assertEqual(_to_key('0:115'), slice(0, 115))
        self.assertEqual(_to_key(':115'), slice(115))
        self.assertEqual(_to_key('10:'), slice(10, None))
        self.assertEqual(_to_key(':'), slice(None))


class TestLArray(TestCase):
    def setUp(self):
        self.title = 'test array'
        self.lipro = Axis(['P%02d' % i for i in range(1, 16)], 'lipro')
        self.age = Axis('age=0..115')
        self.sex = Axis('sex=M,F')

        vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,A43,A44,A45,A46,A71,A72,A73'
        wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,A83,A84,A85,A91,A92,A93'
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

        self.geo = Axis(self.belgium, 'geo')

        self.array = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15).astype(float)
        self.larray = LArray(self.array, axes=(self.age, self.geo, self.sex, self.lipro), title=self.title)

        self.small_title = 'small test array'
        self.small_data = np.arange(30).reshape(2, 15)
        self.small = LArray(self.small_data, axes=(self.sex, self.lipro), title=self.small_title)

    def test_ndrange(self):
        arr = ndrange('a=a0..a2')
        self.assertEqual(arr.shape, (3,))
        self.assertEqual(arr.axes.names, ['a'])
        assert_array_equal(arr.data, np.arange(3))

        # using an explicit Axis object
        a = Axis('a=a0..a2')
        arr = ndrange(a)
        self.assertEqual(arr.shape, (3,))
        self.assertEqual(arr.axes.names, ['a'])
        assert_array_equal(arr.data, np.arange(3))

        # using a group as an axis
        arr = ndrange(a[:'a1'])
        self.assertEqual(arr.shape, (2,))
        self.assertEqual(arr.axes.names, ['a'])
        assert_array_equal(arr.data, np.arange(2))

    def test_getattr(self):
        self.assertEqual(type(self.larray.geo), Axis)
        self.assertIs(self.larray.geo, self.geo)
        with self.assertRaises(AttributeError):
            self.larray.geom

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
        assert_array_equal(l[0], array['M'])
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
 sex [2]: 'M' 'F'
 lipro [15]: 'P01' 'P02' 'P03' ... 'P13' 'P14' 'P15'
dtype: float64"""
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
        self.assertEqual(str(self.small[lipro3, sex['M']]), """\
lipro  P01  P02  P03
         0    1    2""")
        # two dimensions
        self.assertEqual(str(self.small.filter(lipro=lipro3)), """\
sex\lipro  P01  P02  P03
        M    0    1    2
        F   15   16   17""")
        # four dimensions (too many rows)
        self.assertEqual(str(self.larray.filter(lipro=lipro3)), """\
age  geo  sex\lipro       P01       P02       P03
  0  A11          M       0.0       1.0       2.0
  0  A11          F      15.0      16.0      17.0
  0  A12          M      30.0      31.0      32.0
  0  A12          F      45.0      46.0      47.0
  0  A13          M      60.0      61.0      62.0
...  ...        ...       ...       ...       ...
115  A92          F  153045.0  153046.0  153047.0
115  A93          M  153060.0  153061.0  153062.0
115  A93          F  153075.0  153076.0  153077.0
115  A21          M  153090.0  153091.0  153092.0
115  A21          F  153105.0  153106.0  153107.0""")
        # too many columns
        self.assertEqual(str(self.larray['P01', 'A11', 'M']), """\
age    0       1       2       3       4       5       6       7        8  ...  \
     106       107       108       109       110       111       112       113       114       115
     0.0  1320.0  2640.0  3960.0  5280.0  6600.0  7920.0  9240.0  10560.0  ...  \
139920.0  141240.0  142560.0  143880.0  145200.0  146520.0  147840.0  149160.0  150480.0  151800.0""")

    def test_getitem(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = age[[1, 5, 9]]
        lipro159 = lipro['P01,P05,P09']

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis([1, 5, 9], 'age')))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # LGroup at "incorrect" place
        assert_array_equal(la[lipro159], raw[..., [0, 4, 8]])

        # multiple LGroup key (in "incorrect" order)
        res = la[lipro159, age159]
        self.assertEqual(res.axes.names, ['age', 'geo', 'sex', 'lipro'])
        assert_array_equal(res, raw[[1, 5, 9]][..., [0, 4, 8]])

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

        # Ellipsis and LGroup
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # string 'int..int'
        assert_array_equal(la['10..13'], la['10,11,12,13'])
        assert_array_equal(la['8, 10..13, 15'], la['8,10,11,12,13,15'])

        # ambiguous label
        arr = ndrange("a=l0,l1;b=l1,l2")
        res = arr[arr.b['l1']]
        assert_array_equal(res, arr.data[:, 0])

        # key with duplicate axes
        with self.assertRaises(ValueError):
            la[age[1, 2], age[3, 4]]

        # key with invalid axis
        bad = Axis(3, 'bad')
        with self.assertRaises(KeyError):
            la[bad[1, 2], age[3, 4]]

    def test_getitem_abstract_axes(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = X.age[1, 5, 9]
        lipro159 = X.lipro['P01,P05,P09']

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis([1, 5, 9], 'age')))
        assert_array_equal(subset, raw[[1, 5, 9]])

        # LGroup at "incorrect" place
        assert_array_equal(la[lipro159], raw[..., [0, 4, 8]])

        # multiple LGroup key (in "incorrect" order)
        assert_array_equal(la[lipro159, age159], raw[[1, 5, 9]][..., [0, 4, 8]])

        # mixed LGroup/positional key
        assert_array_equal(la[[1, 5, 9], lipro159], raw[[1, 5, 9]][..., [0, 4, 8]])

        # single None slice
        assert_array_equal(la[:], raw)

        # only Ellipsis
        assert_array_equal(la[...], raw)

        # Ellipsis and LGroup
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaises(ValueError):
            la[X.age[1, 2], X.age[3]]

        # key with invalid axis
        with self.assertRaises(KeyError):
            la[X.bad[1, 2], X.age[3, 4]]

    def test_getitem_anonymous_axes(self):
        la = ndrange((3, 4))
        raw = la.data
        assert_array_equal(la[X[0][1:]], raw[1:])
        assert_array_equal(la[X[1][2:]], raw[:, 2:])
        assert_array_equal(la[X[0][2:], X[1][1:]], raw[2:, 1:])
        assert_array_equal(la.i[2:, 1:], raw[2:, 1:])

    def test_getitem_guess_axis(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes

        # key at "correct" place
        assert_array_equal(la[[1, 5, 9]], raw[[1, 5, 9]])
        subset = la[[1, 5, 9]]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis([1, 5, 9], 'age')))
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

        # Ellipsis and LGroup
        assert_array_equal(la[..., 'P01,P05,P09'], raw[..., [0, 4, 8]])
        assert_array_equal(la[..., ['P01', 'P05', 'P09']], raw[..., [0, 4, 8]])

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
        arr = ndrange("a=l0,l1;b=l1,l2")
        with self.assertRaisesRegexp(ValueError, "l1 is ambiguous \(valid in a, b\)"):
            arr['l1']

        # ambiguous key disambiguated via string
        res = arr['b[l1]']
        assert_array_equal(res, arr.data[:, 0])

    def test_getitem_positional_group(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = age.i[1, 5, 9]
        lipro159 = lipro.i[0, 4, 8]

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis([1, 5, 9], 'age')))
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

        # Ellipsis and LGroup
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaisesRegexp(ValueError,
                                     "key has several values for axis: age"):
            la[age.i[1, 2], age.i[3, 4]]

    def test_getitem_abstract_positional(self):
        raw = self.array
        la = self.larray
        age, geo, sex, lipro = la.axes
        age159 = X.age.i[1, 5, 9]
        lipro159 = X.lipro.i[0, 4, 8]

        # LGroup at "correct" place
        subset = la[age159]
        self.assertEqual(subset.axes[1:], (geo, sex, lipro))
        self.assertTrue(subset.axes[0].equals(Axis([1, 5, 9], 'age')))
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

        # Ellipsis and LGroup
        assert_array_equal(la[..., lipro159], raw[..., [0, 4, 8]])

        # key with duplicate axes
        with self.assertRaisesRegexp(ValueError, "key has several values for axis: age"):
            la[X.age.i[2, 3], X.age.i[1, 5]]

    def test_getitem_bool_larray_key(self):
        arr = ndtest((3, 2, 4))
        raw = arr.data

        # all dimensions
        res = arr[arr < 5]
        self.assertTrue(isinstance(res, LArray))
        self.assertEqual(res.ndim, 1)
        assert_array_equal(res, raw[raw < 5])

        # missing dimension
        filt = arr['b1'] % 5 == 0
        res = arr[filt]
        self.assertTrue(isinstance(res, LArray))
        self.assertEqual(res.ndim, 2)
        self.assertEqual(res.shape, (3, 2))
        raw_key = raw[:, 1, :] % 5 == 0
        raw_d1, raw_d3 = raw_key.nonzero()
        assert_array_equal(res, raw[raw_d1, :, raw_d3])

    def test_getitem_bool_larray_and_group_key(self):
        arr = ndtest((3, 6, 4)).set_labels('b', '0..5')

        # using axis
        res = arr['a0,a2', arr.b < 3, 'c0:c3']
        assert isinstance(res, LArray)
        assert res.ndim == 3
        assert_array_equal(res, arr['a0,a2', '0:2', 'c0:c3'])

        # using axis reference
        res = arr['a0,a2', X.b < 3, 'c0:c3']
        assert isinstance(res, LArray)
        assert res.ndim == 3
        assert_array_equal(res, arr['a0,a2', '0:2', 'c0:c3'])

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

        # XXX: we might want to transpose the result to always move combined axes to the front
        a = ndrange((2, 3, 4, 5))
        mask = ones(a.axes[1, 2], dtype=bool)
        res = a[mask]
        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape, (2, 12, 5))

    def test_getitem_igroup_on_int_axis(self):
        a = Axis('a=1..3')
        arr = ndrange(a)
        self.assertEqual(arr[a.i[1]], 1)

    def test_getitem_int_larray_lgroup_key(self):
        # e axis go from 0 to 3
        arr = ndrange((2, 2, 4)).rename(0, 'c').rename(1, 'd').rename(2, 'e')
        # key values go from 0 to 3
        key = ndrange((2, 2)).rename(0, 'a').rename(1, 'b')
        # this replaces 'e' axis by 'a' and 'b' axes
        res = arr[X.e[key]]
        self.assertEqual(res.shape, (2, 2, 2, 2))
        self.assertEqual(res.axes.names, ['c', 'd', 'a', 'b'])

    def test_getitem_structured_key_with_groups(self):
        arr = ndtest((3, 2))
        expected = arr['a1':]

        a, b = arr.axes
        alt_a = Axis('a=a1..a3')

        # a) slice with lgroup
        # a.1) LGroup.axis from array.axes
        assert_array_equal(arr[a['a1']:a['a2']], expected)

        # a.2) LGroup.axis not from array.axes
        assert_array_equal((arr[alt_a['a1']:alt_a['a2']]), expected)

        # b) slice with igroup
        # b.1) IGroup.axis from array.axes
        assert_array_equal((arr[a.i[1]:a.i[2]]), expected)

        # b.2) IGroup.axis not from array.axes
        assert_array_equal((arr[alt_a.i[0]:alt_a.i[1]]), expected)

        # c) list with LGroup
        # c.1) LGroup.axis from array.axes
        assert_array_equal((arr[[a['a1'], a['a2']]]), expected)

        # c.2) LGroup.axis not from array.axes
        assert_array_equal((arr[[alt_a['a1'], alt_a['a2']]]), expected)

        # d) list with IGroup
        # d.1) IGroup.axis from array.axes
        assert_array_equal((arr[[a.i[1], a.i[2]]]), expected)

        # d.2) IGroup.axis not from array.axes
        assert_array_equal((arr[[alt_a.i[0], alt_a.i[1]]]), expected)

    def test_getitem_single_larray_key_guess(self):
        a = Axis(['a1', 'a2'], 'a')
        b = Axis(['b1', 'b2', 'b3'], 'b')
        c = Axis(['c1', 'c2', 'c3', 'c4'], 'c')

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
        b_bis = Axis(['b1', 'b2'], 'b')
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
    #     # TODO: [0, 1, 2] is enough in this case (thanks to broadcasting) because in numpy missing dimensions are
    #     #       filled by adding length 1 dimensions to the left. Ie it works because b is the last dimension.
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
        self.assertEqual(res.axes, la.axes.replace(X.lipro, Axis(keys, 'lipro')))
        assert_array_equal(res, raw[:, :, :, [3, 0, 2, 1]])

    def test_getitem_int_larray_key_guess(self):
        a = Axis([0, 1], 'a')
        b = Axis([2, 3], 'b')
        c = Axis([4, 5], 'c')
        d = Axis([6, 7], 'd')
        e = Axis([8, 9, 10, 11], 'e')

        arr = ndrange([c, d, e])
        key = LArray([[8, 9], [10, 11]], [a, b])
        self.assertEqual(arr[key].axes, [c, d, a, b])

    def test_getitem_int_ndarray_key_guess(self):
        c = Axis([4, 5], 'c')
        d = Axis([6, 7], 'd')
        e = Axis([8, 9, 10, 11], 'e')

        arr = ndrange([c, d, e])
        # ND keys do not work yet
        # key = np.array([[8, 11], [10, 9]])
        key = np.array([8, 11, 10])
        res = arr[key]
        self.assertEqual(res.axes, [c, d, Axis([8, 11, 10], 'e')])

    def test_getitem_axis_object(self):
        arr = ndtest((2, 3))
        a, b = arr.axes

        assert_array_equal(arr[a], arr)
        assert_array_equal(arr[b], arr)

        b2 = Axis('b=b0,b2')

        assert_array_equal(arr[b2], from_string("""a\\b  b0  b2
                                                     a0   0   2
                                                     a1   3   5"""))

    def test_positional_indexer_getitem(self):
        raw = self.array
        la = self.larray
        for key in [0, (0, 5, 1, 2), (slice(None), 5, 1), (0, 5), [1, 0], ([1, 0], 5)]:
            assert_array_equal(la.i[key], raw[key])
        assert_array_equal(la.i[[1, 0], [5, 4]], raw[np.ix_([1, 0], [5, 4])])
        with self.assertRaises(IndexError):
            la.i[0, 0, 0, 0, 0]

    def test_positional_indexer_setitem(self):
        for key in [0, (0, 2, 1, 2), (slice(None), 2, 1), (0, 2), [1, 0], ([1, 0], 2)]:
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

    def test_points_indexer_getitem(self):
        arr = ndtest((2, 3, 3))
        raw = arr.data

        keys = [
            ('a0',
                0),
            (('a0', 'c2'),
                (0, slice(None), 2)),
            (('a0', 'b1', 'c2'),
                (0, 1, 2)),
            # key in the "correct" order
            ((['a1', 'a0', 'a1', 'a0'], 'b1', ['c1', 'c0', 'c1', 'c0']),
                ([1, 0, 1, 0], 1, [1, 0, 1, 0])),
            # key in the "wrong" order
            ((['a1', 'a0', 'a1', 'a0'], ['c1', 'c0', 'c1', 'c0'], 'b1'),
                ([1, 0, 1, 0], 1, [1, 0, 1, 0])),
            # advanced key with a missing dimension
            ((['a1', 'a0', 'a1', 'a0'], ['c1', 'c0', 'c1', 'c0']),
                ([1, 0, 1, 0], slice(None), [1, 0, 1, 0])),
        ]
        for label_key, index_key in keys:
            assert_array_equal(arr.points[label_key], raw[index_key])

        # XXX: we might want to raise KeyError or IndexError instead?
        with self.assertRaises(ValueError):
            arr.points['a0', 'b1', 'c2', 'd0']

    def test_points_indexer_setitem(self):
        keys = [
            ('a0',
                0),
            (('a0', 'c2'),
                (0, slice(None), 2)),
            (('a0', 'b1', 'c2'),
                (0, 1, 2)),
            # key in the "correct" order
            ((['a1', 'a0', 'a1', 'a0'], 'b1', ['c1', 'c0', 'c1', 'c0']),
                ([1, 0, 1, 0], 1, [1, 0, 1, 0])),
            # key in the "wrong" order
            ((['a1', 'a0', 'a1', 'a0'], ['c1', 'c0', 'c1', 'c0'], 'b1'),
                ([1, 0, 1, 0], 1, [1, 0, 1, 0])),
            # advanced key with a missing dimension
            ((['a1', 'a0', 'a1', 'a0'], ['c1', 'c0', 'c1', 'c0']),
                ([1, 0, 1, 0], slice(None), [1, 0, 1, 0])),
        ]
        for label_key, index_key in keys:
            arr = ndtest((2, 3, 3))
            raw = arr.data.copy()
            arr.points[label_key] = 42
            raw[index_key] = 42
            assert_array_equal(arr, raw)

        arr = ndtest(2)
        # XXX: we might want to raise KeyError or IndexError instead?
        with self.assertRaises(ValueError):
            arr.points['a0', 'b1'] = 42

        # test when broadcasting is involved
        arr = ndtest((2, 3, 4))
        data = arr.data.copy()
        value = data[:, 0, 0].reshape(2, 1)
        data[:, [0, 1, 2], [0, 1, 2]] = value
        arr.points['b0,b1,b2', 'c0,c1,c2'] = arr['b0', 'c0']
        assert_array_equal(arr, data)

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

        # b) value has exactly the same shape but LGroup at a "wrong" positions
        la = self.larray.copy()
        la[geo[:], ages1_5_9] = la[ages1_5_9] + 25.0
        # same raw as previous test
        assert_array_equal(la, raw)

        # c) value has an extra length-1 axis
        la = self.larray.copy()
        raw = self.array.copy()

        raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
        fake_axis = Axis(['label'], 'fake')
        age_axis = la[ages1_5_9].axes.age
        value = LArray(raw_value, axes=(age_axis, fake_axis, self.geo, self.sex, self.lipro))
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
        value = la[age[1, 5, 8], sex['M']] + 25.0
        la[age[1, 5, 8], sex['M']] = value
        raw[[1, 5, 8], :, 0] = raw[[1, 5, 8], :, 0] + 25.0
        assert_array_equal(la, raw)

        # 3) using a string key
        la = self.larray.copy()
        raw = self.array.copy()
        la['1, 5, 9'] = la['1, 5, 9'] + 27.0
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 27.0
        assert_array_equal(la, raw)

        # 4) using ellipsis keys
        # only Ellipsis
        la = self.larray.copy()
        la[...] = 0
        assert_array_equal(la, np.zeros_like(raw))

        # Ellipsis and LGroup
        la = self.larray.copy()
        raw = self.array.copy()
        la[..., lipro['P01,P05,P09']] = 0
        raw[..., [0, 4, 8]] = 0
        assert_array_equal(la, raw)

        # 5) using a single slice(None) key
        la = self.larray.copy()
        la[:] = 0
        assert_array_equal(la, np.zeros_like(raw))

        # 6) incompatible axes
        la = self.small.copy()
        la2 = la.copy()

        with pytest.raises(ValueError, match="Value {!s} axis is not present in target subset {!s}. "
                                             "A value can only have the same axes or fewer axes than the subset "
                                             "being targeted".format(la2.axes - la['P01'].axes, la['P01'].axes)):
           la['P01'] = la2

        la2 = la.rename('sex', 'gender')
        with pytest.raises(ValueError, match="Value {!s} axis is not present in target subset {!s}. "
                                             "A value can only have the same axes or fewer axes than the subset "
                                             "being targeted".format(la2['P01'].axes - la['P01'].axes, la['P01'].axes)):
           la['P01'] = la2['P01']

    def test_setitem_ndarray(self):
        """
        tests LArray.__setitem__(key, value) where value is a raw ndarray.
        In that case, value.shape is more restricted as we rely on numpy broadcasting.
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
        la[0, 'P02', 'A12', 'M'] = 42
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
        key = la[sex['M']] < 5
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
        key = (la < 5).expand([Axis(2, 'extra')])
        self.assertEqual(key.ndim, 5)
        # TODO: make this work
        with self.assertRaises(ValueError):
            la[key] = 0

    def test_set(self):
        age, geo, sex, lipro = self.larray.axes

        # 1) using a LGroup key
        ages1_5_9 = age[[1, 5, 9]]

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
        fake_axis = Axis(['label'], 'fake')
        age_axis = la[ages1_5_9].axes.age
        value = LArray(raw_value, axes=(age_axis, fake_axis, self.geo, self.sex, self.lipro))
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
        la.set(la[[1, 5, 9]] + 27.0, age=[1, 5, 9])
        raw[[1, 5, 9]] = raw[[1, 5, 9]] + 27.0
        assert_array_equal(la, raw)

    def test_filter(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        ages1_5_9 = age[(1, 5, 9)]
        ages11 = age[11]

        # with LGroup
        self.assertEqual(la.filter(age=ages1_5_9).shape, (3, 44, 2, 15))

        # FIXME: this should raise a comprehensible error!
        # self.assertEqual(la.filter(age=[ages1_5_9]).shape, (3, 44, 2, 15))

        # LGroup with 1 value => collapse
        self.assertEqual(la.filter(age=ages11).shape, (44, 2, 15))

        # LGroup with a list of 1 value => do not collapse
        self.assertEqual(la.filter(age=age[[11]]).shape, (1, 44, 2, 15))

        # LGroup with a list of 1 value defined as a string => do not collapse
        self.assertEqual(la.filter(lipro=lipro['P01,']).shape, (116, 44, 2, 1))

        # LGroup with 1 value
        # XXX: this does not work. Do we want to make this work?
        # filtered = la.filter(age=(ages11,))
        # self.assertEqual(filtered.shape, (1, 44, 2, 15))

        # list
        self.assertEqual(la.filter(age=[1, 5, 9]).shape, (3, 44, 2, 15))

        # string
        self.assertEqual(la.filter(lipro='P01,P02').shape, (116, 44, 2, 2))

        # multiple axes at once
        self.assertEqual(la.filter(age=[1, 5, 9], lipro='P01,P02').shape, (3, 44, 2, 2))

        # multiple axes one after the other
        self.assertEqual(la.filter(age=[1, 5, 9]).filter(lipro='P01,P02').shape, (3, 44, 2, 2))

        # a single value for one dimension => collapse the dimension
        self.assertEqual(la.filter(sex='M').shape, (116, 44, 15))

        # but a list with a single value for one dimension => do not collapse
        self.assertEqual(la.filter(sex=['M']).shape, (116, 44, 1, 15))

        self.assertEqual(la.filter(sex='M,').shape, (116, 44, 1, 15))

        # with duplicate keys
        # XXX: do we want to support this? I don't see any value in that but I might be short-sighted.
        # filtered = la.filter(lipro='P01,P02,P01')

        # XXX: we could abuse python to allow naming groups via Axis.__getitem__
        # (but I doubt it is a good idea).
        # child = age[':17', 'child']

        # slices
        # ------

        # LGroup slice
        self.assertEqual(la.filter(age=age[:17]).shape, (18, 44, 2, 15))
        # string slice
        self.assertEqual(la.filter(lipro=':P03').shape, (116, 44, 2, 3))
        # raw slice
        self.assertEqual(la.filter(age=slice(17)).shape, (18, 44, 2, 15))

        # filter chain with a slice
        self.assertEqual(la.filter(age=slice(17)).filter(geo='A12,A13').shape, (18, 2, 2, 15))

    def test_filter_multiple_axes(self):
        la = self.larray

        # multiple values in each group
        self.assertEqual(la.filter(age=[1, 5, 9], lipro='P01,P02').shape, (3, 44, 2, 2))
        # with a group of one value
        self.assertEqual(la.filter(age=[1, 5, 9], sex='M,').shape, (3, 44, 1, 15))

        # with a discarded axis (there is a scalar in the key)
        self.assertEqual(la.filter(age=[1, 5, 9], sex='M').shape, (3, 44, 15))

        # with a discarded axis that is not adjacent to the ix_array axis ie with a sliced axis between the scalar axis
        # and the ix_array axis since our array has a axes: age, geo, sex, lipro, any of the following should be tested:
        # age+sex / age+lipro / geo+lipro
        # additionally, if the ix_array axis was first (ie ix_array on age), it worked even before the issue was fixed,
        # since the "indexing" subspace is tacked-on to the beginning (as the first dimension)
        self.assertEqual(la.filter(age=57, sex='M,F').shape, (44, 2, 15))
        self.assertEqual(la.filter(age=57, lipro='P01,P05').shape, (44, 2, 2))
        self.assertEqual(la.filter(geo='A57', lipro='P01,P05').shape, (116, 2, 2))

    def test_contains(self):
        arr = ndrange('a=0..2;b=b0..b2;c=2..4')
        # string label
        assert 'b1' in arr
        assert not 'b4' in arr
        # int label
        assert 1 in arr
        assert 5 not in arr
        # duplicate label
        assert 2 in arr
        # slice
        assert not slice('b0', 'b2') in arr

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
        la['M', 'P02', 'A12', 0] = np.nan
        raw = la.data

        # everything
        self.assertEqual(la.sum(), np.nansum(raw))
        self.assertTrue(isnan(la.sum(skipna=False)))

        # using Axis objects
        assert_array_nan_equal(la.sum(X.age), np.nansum(raw, 0))
        assert_array_nan_equal(la.sum(X.age, skipna=False), raw.sum(0))

        assert_array_nan_equal(la.sum(X.age, X.sex), np.nansum(raw, (0, 2)))
        assert_array_nan_equal(la.sum(X.age, X.sex, skipna=False), raw.sum((0, 2)))

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
        assert_array_nan_equal(la.mean(X.age), np.mean(raw, 0))
        assert_array_nan_equal(la.mean(X.age, X.sex), np.mean(raw, (0, 2)))

    def test_mean_groups(self):
        # using int type to test that we get a float in return
        la = self.larray.astype(int)
        raw = self.array
        res = la.mean(X.geo['A11', 'A13', 'A24', 'A31'])
        assert_array_nan_equal(res, np.mean(raw[:, [0, 2, 4, 5]], 1))

    def test_median_full_axes(self):
        la = self.larray
        raw = self.array

        self.assertEqual(la.median(), np.median(raw))
        assert_array_nan_equal(la.median(X.age), np.median(raw, 0))
        assert_array_nan_equal(la.median(X.age, X.sex), np.median(raw, (0, 2)))

    def test_median_groups(self):
        la = self.larray
        raw = self.array

        res = la.median(X.geo['A11', 'A13', 'A24'])
        self.assertEqual(res.shape, (116, 2, 15))
        assert_array_nan_equal(res, np.median(raw[:, [0, 2, 4]], 1))

    def test_percentile_full_axes(self):
        arr = ndtest((2, 3, 4))
        raw = arr.data
        self.assertEqual(arr.percentile(10), np.percentile(raw, 10))
        assert_array_nan_equal(arr.percentile(10, X.a), np.percentile(raw, 10, 0))
        assert_array_nan_equal(arr.percentile(10, X.c, X.a), np.percentile(raw, 10, (2, 0)))

    def test_percentile_groups(self):
        arr = ndtest((2, 5, 3))
        raw = arr.data

        res = arr.percentile(10, X.b['b0', 'b2', 'b4'])
        assert_array_nan_equal(res, np.percentile(raw[:, [0, 2, 4]], 10, 1))

    def test_cumsum(self):
        la = self.larray

        # using Axis objects
        assert_array_equal(la.cumsum(X.age), self.array.cumsum(0))
        assert_array_equal(la.cumsum(X.lipro), self.array.cumsum(3))

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
        self.assertEqual(la.sum(sex='M').shape, (116, 44, 15))
        self.assertEqual(la.sum(sex='M,F').shape, (116, 44, 15))
        self.assertEqual(la.sum(sex=sex['M']).shape, (116, 44, 15))

        self.assertEqual(la.sum(geo='A11,A21,A25').shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo['A11,A21,A25']).shape, (116, 2, 15))

        self.assertEqual(la.sum(geo=':').shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo[:]).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo=geo[':']).shape, (116, 2, 15))
        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the previous
        # tests.
        self.assertEqual(la.sum(geo='A11:A21').shape, (116, 2, 15))
        assert_array_equal(la.sum(geo='A11:A21'), la.sum(geo=':'))
        assert_array_equal(la.sum(geo=geo['A11:A21']), la.sum(geo=':'))

        # a.2) a tuple of one group => do not collapse dimension
        self.assertEqual(la.sum(geo=(geo[:],)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum(geo=(vla, wal, bru)).shape, (116, 3, 2, 15))
        # with one label in several groups
        self.assertEqual(la.sum(sex=(['M'], ['M', 'F'])).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum(sex=('M', 'M,F')).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum(sex='M;M,F').shape, (116, 44, 2, 15))

        res = la.sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        res = la.sum(lipro='P01,P03;P02,P05;:', geo=(vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does not allow non-kwargs after kwargs.
        res = la.sum(age, sex, geo=(vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

        # c) chain group aggregate after axis aggregate
        res = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

    def test_group_agg_guess_axis(self):
        la = self.larray
        raw = self.array
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        self.assertEqual(la.sum('M').shape, (116, 44, 15))
        self.assertEqual(la.sum('M,').shape, (116, 44, 15))
        self.assertEqual(la.sum('M,F').shape, (116, 44, 15))

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
        self.assertEqual(la.sum((geo[:],)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
        # axes into play. For now the error message is unclear because it first aggregates on "vla", then tries to
        # aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((['M'], ['M', 'F'])).shape,
                         (116, 44, 2, 15))
        self.assertEqual(la.sum(('M', 'M,F')).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum('M;M,F').shape, (116, 44, 2, 15))
        # with group names
        res = la.sum('M >> men;M,F >> all')
        self.assertEqual(res.shape, (116, 44, 2, 15))
        self.assertTrue('sex' in res.axes)
        men = sex['M'].named('men')
        all_ = sex['M,F'].named('all')
        assert_array_equal(res.axes.sex.labels, ['men', 'all'])
        assert_array_equal(res['men'], raw[:, :, 0, :])
        assert_array_equal(res['all'], raw.sum(2))

        res = la.sum(('M >> men', 'M,F >> all'))
        self.assertEqual(res.shape, (116, 44, 2, 15))
        self.assertTrue('sex' in res.axes)
        assert_array_equal(res.axes.sex.labels, ['men', 'all'])
        assert_array_equal(res['men'], raw[:, :, 0, :])
        assert_array_equal(res['all'], raw.sum(2))

        res = la.sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        res = la.sum('P01,P03;P02,P05;P01:', (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        res = la.sum(age, sex, (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

        # c) chain group aggregate after axis aggregate
        res = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

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
        self.assertEqual(la.sum(sex['M']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['M,']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['M,F']).shape, (116, 44, 15))

        self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[['A11', 'A21', 'A25']]).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))

        self.assertEqual(la.sum(geo[:]).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[':']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[:]).shape, (116, 2, 15))

        # Include everything between two labels. Since A11 is the first label and A21 is the last one, this should be
        # equivalent to the previous tests.
        self.assertEqual(la.sum(geo['A11:A21']).shape, (116, 2, 15))
        assert_array_equal(la.sum(geo['A11:A21']), la.sum(geo))
        assert_array_equal(la.sum(geo['A11':'A21']), la.sum(geo))

        # a.2) a tuple of one group => do not collapse dimension
        self.assertEqual(la.sum((geo[:],)).shape, (116, 1, 2, 15))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
        # axes into play. For now the error message is unclear because it first aggregates on "vla", then tries to
        # aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((sex['M'], sex[['M', 'F']])).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['M'], sex['M', 'F'])).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['M'], sex['M,F'])).shape, (116, 44, 2, 15))
        # XXX: do we want to support this?
        # self.assertEqual(la.sum(sex['M;H,F']).shape, (116, 44, 2, 15))

        res = la.sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        # self.assertEqual(la.sum(lipro['P01,P03;P02,P05;P01:'], (vla, wal, bru, belgium)).shape,
        #                  (116, 4, 2, 3))
        res = la.sum((lipro['P01,P03'], lipro['P02,P05'], lipro[:]), (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        res = la.sum(age, sex, (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

        # c) chain group aggregate after axis aggregate
        res = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

    def test_group_agg_label_group_no_axis(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = LGroup(self.vla_str), LGroup(self.wal_str), LGroup(self.bru_str)
        belgium = LGroup(self.belgium)

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        self.assertEqual(la.sum(LGroup('M')).shape, (116, 44, 15))
        self.assertEqual(la.sum(LGroup('M,')).shape, (116, 44, 15))
        self.assertEqual(la.sum(LGroup('M,F')).shape, (116, 44, 15))

        self.assertEqual(la.sum(LGroup('A11,A21,A25')).shape, (116, 2, 15))
        self.assertEqual(la.sum(LGroup(['A11', 'A21', 'A25'])).shape, (116, 2, 15))

        # Include everything between two labels. Since A11 is the first label
        # and A21 is the last one, this should be equivalent to the full axis.
        self.assertEqual(la.sum(LGroup('A11:A21')).shape, (116, 2, 15))
        assert_array_equal(la.sum(LGroup('A11:A21')), la.sum(geo))
        assert_array_equal(la.sum(LGroup(slice('A11', 'A21'))), la.sum(geo))

        # a.3) several groups
        # string groups
        self.assertEqual(la.sum((vla, wal, bru)).shape, (116, 3, 2, 15))

        # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
        # axes into play. For now the error message is unclear because it first aggregates on "vla", then tries to
        # aggregate on "wal", but there is no "geo" dimension anymore.
        # self.assertEqual(la.sum(vla, wal, bru).shape, (116, 3, 2, 15))

        # with one label in several groups
        self.assertEqual(la.sum((LGroup('M'), LGroup(['M', 'F']))).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum((LGroup('M'), LGroup('M,F'))).shape, (116, 44, 2, 15))
        # XXX: do we want to support this?
        # self.assertEqual(la.sum(sex['M;M,F']).shape, (116, 44, 2, 15))

        res = la.sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        # self.assertEqual(la.sum(lipro['P01,P03;P02,P05;P01:'], (vla, wal, bru, belgium)).shape,
        #                  (116, 4, 2, 3))
        res = la.sum((LGroup('P01,P03'), LGroup('P02,P05')), (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 2))

        # b) both axis aggregate and group aggregate at the same time
        res = la.sum(age, sex, (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

        # c) chain group aggregate after axis aggregate
        res = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

    def test_group_agg_axis_ref_label_group(self):
        la = self.larray
        age, geo, sex, lipro = X.age, X.geo, X.sex, X.lipro
        vla, wal, bru = geo[self.vla_str], geo[self.wal_str], geo[self.bru_str]
        belgium = geo[self.belgium]

        # a) group aggregate on a fresh array

        # a.1) one group => collapse dimension
        # not sure I should support groups with a single item in an aggregate
        men = sex.i[[0]]
        self.assertEqual(la.sum(men).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['M']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['M,']).shape, (116, 44, 15))
        self.assertEqual(la.sum(sex['M,F']).shape, (116, 44, 15))

        self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo[['A11', 'A21', 'A25']]).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo['A11', 'A21', 'A25']).shape, (116, 2, 15))
        self.assertEqual(la.sum(geo['A11,A21,A25']).shape, (116, 2, 15))

        self.assertEqual(la.sum(geo[:]).shape, (116, 2, 15))
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
        self.assertEqual(la.sum((sex['M'], sex[['M', 'F']])).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['M'], sex['M', 'F'])).shape, (116, 44, 2, 15))
        self.assertEqual(la.sum((sex['M'], sex['M,F'])).shape, (116, 44, 2, 15))
        # XXX: do we want to support this?
        # self.assertEqual(la.sum(sex['M;M,F']).shape, (116, 44, 2, 15))

        res = la.sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 15))

        # a.4) several dimensions at the same time
        # self.assertEqual(la.sum(lipro['P01,P03;P02,P05;P01:'],
        #                         (vla, wal, bru, belgium)).shape,
        #                  (116, 4, 2, 3))
        res = la.sum((lipro['P01,P03'], lipro['P02,P05'], lipro[:]), (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (116, 4, 2, 3))

        # b) both axis aggregate and group aggregate at the same time
        res = la.sum(age, sex, (vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

        # c) chain group aggregate after axis aggregate
        res = la.sum(age, sex).sum((vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 15))

    def test_group_agg_one_axis(self):
        a = Axis(range(3), 'a')
        la = ndrange(a)
        raw = np.asarray(la)

        assert_array_equal(la.sum(a[0, 2]), raw[[0, 2]].sum())

    def test_group_agg_anonymous_axis(self):
        la = ndrange((2, 3))
        a1, a2 = la.axes
        raw = np.asarray(la)
        assert_array_equal(la.sum(a2[0, 2]), raw[:, [0, 2]].sum(1))

    def test_group_agg_on_int_array(self):
        # issue 193
        arr = ndrange('year=2014..2018')
        group = arr.year[:2016]
        self.assertEqual(arr.mean(group), 1.0)
        self.assertEqual(arr.median(group), 1.0)
        self.assertEqual(arr.percentile(90, group), 1.8)
        self.assertEqual(arr.std(group), 1.0)
        self.assertEqual(arr.var(group), 1.0)

    def test_group_agg_on_bool_array(self):
        # issue 194
        a = ndtest((2, 3))
        b = a > 1
        expected = from_string("""a,a0,a1
                                   , 1, 2""", sep=',')
        assert_array_equal(b.sum('b1:'), expected)

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
        self.assertEqual(reg.sum(lipro=('P01', 'P02', lipro[:])).shape, (4, 3))

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
        self.assertEqual(reg.sum(('P01', 'P02', lipro[:])).shape, (4, 3))

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

        # using a string
        vla = self.vla_str
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # one more level...
        self.assertEqual(reg[vla]['P03'], 389049848.0)

        # using an anonymous LGroup
        vla = self.geo[self.vla_str]
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # using a named LGroup
        vla = self.geo[self.vla_str] >> 'Vlaanderen'
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
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

        # using a string
        vla = self.vla_str
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # one more level...
        self.assertEqual(reg[vla]['P03'], 389049848.0)

        # using an anonymous LGroup
        vla = self.geo[self.vla_str]
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
        # the following are all equivalent
        self.assertEqual(reg[vla].shape, (15,))
        self.assertEqual(reg[(vla,)].shape, (15,))
        self.assertEqual(reg[(vla, slice(None))].shape, (15,))
        self.assertEqual(reg[vla, slice(None)].shape, (15,))
        self.assertEqual(reg[vla, :].shape, (15,))

        # using a named LGroup
        vla = self.geo[self.vla_str] >> 'Vlaanderen'
        reg = la.sum(age, sex).sum((vla, wal, bru, belgium))
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

        # using a string
        vla = self.vla_str
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(reg.filter(geo=vla).shape, (15,))

        # using a named LGroup
        vla = self.geo[self.vla_str] >> 'Vlaanderen'
        reg = la.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
        self.assertEqual(reg.filter(geo=vla).shape, (15,))

        # Note that reg.filter(geo=(vla,)) does NOT work. It might be a
        # little confusing for users, because reg[(vla,)] works but it is
        # normal because reg.filter(geo=(vla,)) is equivalent to:
        # reg[((vla,),)] or reg[(vla,), :]

        # mixed LGroup/string slices
        child = age[:17]
        child_named = age[:17] >> 'child'
        working = age[18:64]
        retired = age[65:]

        byage = la.sum(age=(child, 5, working, retired))
        self.assertEqual(byage.shape, (4, 44, 2, 15))

        byage = la.sum(age=(child, slice(5, 10), working, retired))
        self.assertEqual(byage.shape, (4, 44, 2, 15))

        # filter on an aggregated larray created with mixed groups
        self.assertEqual(byage.filter(age=':17').shape, (44, 2, 15))

        byage = la.sum(age=(child_named, 5, working, retired))
        self.assertEqual(byage.filter(age=child_named).shape, (44, 2, 15))

    def test_sum_several_lg_groups(self):
        la, geo = self.larray, self.geo
        fla = geo[self.vla_str] >> 'Flanders'
        wal = geo[self.wal_str] >> 'Wallonia'
        bru = geo[self.bru_str] >> 'Brussels'

        reg = la.sum(geo=(fla, wal, bru))
        self.assertEqual(reg.shape, (116, 3, 2, 15))

        # the result is indexable
        # a) by LGroup
        self.assertEqual(reg.filter(geo=fla).shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo=(fla, wal)).shape, (116, 2, 2, 15))

        # b) by string (name of groups)
        self.assertEqual(reg.filter(geo='Flanders').shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo='Flanders,Wallonia').shape, (116, 2, 2, 15))

        # using string groups
        reg = la.sum(geo=(self.vla_str, self.wal_str, self.bru_str))
        self.assertEqual(reg.shape, (116, 3, 2, 15))
        # the result is indexable
        # a) by string (def)
        self.assertEqual(reg.filter(geo=self.vla_str).shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo=(self.vla_str, self.wal_str)).shape, (116, 2, 2, 15))

        # b) by LGroup
        self.assertEqual(reg.filter(geo=self.vla_str).shape, (116, 2, 15))
        self.assertEqual(reg.filter(geo=(self.vla_str, self.wal_str)).shape, (116, 2, 2, 15))

    def test_sum_with_groups_from_other_axis(self):
        small = self.small

        # use a group from another *compatible* axis
        lipro2 = Axis('lipro=P01..P15')
        self.assertEqual(small.sum(lipro=lipro2['P01,P03']).shape, (2,))

        # use (compatible) group from another *incompatible* axis
        # XXX: I am unsure whether or not this should be allowed. Maybe we
        # should simply check that the group is valid in axis, but that
        # will trigger a pretty meaningful error anyway
        lipro3 = Axis('lipro=P01,P03,P05')
        self.assertEqual(small.sum(lipro3['P01,P03']).shape, (2,))

        # use a group (from another axis) which is incompatible with the axis of
        # the same name in the array
        lipro4 = Axis('lipro=P01,P03,P16')
        with self.assertRaisesRegexp(ValueError, "lipro\['P01', 'P16'\] is not a valid label for any axis"):
            small.sum(lipro4['P01,P16'])

    def test_agg_kwargs(self):
        la = self.larray
        data = self.array

        # dtype
        self.assertEqual(la.sum(dtype=int), data.sum(dtype=int))

        # ddof
        self.assertEqual(la.std(ddof=0), data.std(ddof=0))

        # out
        res = la.std(X.sex)
        out = zeros_like(res)
        la.std(X.sex, out=out)
        assert_array_equal(res, out)

    def test_agg_by(self):
        la = self.larray
        age, geo, sex, lipro = la.axes
        vla, wal, bru = self.vla_str, self.wal_str, self.bru_str
        belgium = self.belgium

        # no group or axis
        self.assertEqual(la.sum_by().shape, ())
        self.assertEqual(la.sum_by(), la.sum())

        # a) group aggregate on a fresh array

        # a.1) one group
        res = la.sum_by(geo='A11,A21,A25')
        self.assertEqual(res.shape, ())
        self.assertEqual(res, la.sum(geo='A11,A21,A25').sum())

        # a.2) a tuple of one group
        res = la.sum_by(geo=(geo[:],))
        self.assertEqual(res.shape, (1,))
        assert_array_equal(res, la.sum(age, sex, lipro, geo=(geo[:],)))

        # a.3) several groups
        # string groups
        res = la.sum_by(geo=(vla, wal, bru))
        self.assertEqual(res.shape, (3,))
        assert_array_equal(res, la.sum(age, sex, lipro, geo=(vla, wal, bru)))

        # with one label in several groups
        self.assertEqual(la.sum_by(sex=(['M'], ['M', 'F'])).shape, (2,))
        self.assertEqual(la.sum_by(sex=('M', 'M,F')).shape, (2,))

        res = la.sum_by(sex='M;M,F')
        self.assertEqual(res.shape, (2,))
        assert_array_equal(res, la.sum(age, geo, lipro, sex='M;M,F'))

        # a.4) several dimensions at the same time
        res = la.sum_by(geo=(vla, wal, bru, belgium), lipro='P01,P03;P02,P05;:')
        self.assertEqual(res.shape, (4, 3))
        assert_array_equal(res, la.sum(age, sex, geo=(vla, wal, bru, belgium), lipro='P01,P03;P02,P05;:'))

        # b) both axis aggregate and group aggregate at the same time
        # Note that you must list "full axes" aggregates first (Python does not allow non-kwargs after kwargs.
        res = la.sum_by(sex, geo=(vla, wal, bru, belgium))
        self.assertEqual(res.shape, (4, 2))
        assert_array_equal(res, la.sum(age, lipro, geo=(vla, wal, bru, belgium)))

        # c) chain group aggregate after axis aggregate
        res = la.sum_by(geo, sex)
        self.assertEqual(res.shape, (44, 2))
        assert_array_equal(res, la.sum(age, lipro))

        res2 = res.sum_by(geo=(vla, wal, bru, belgium))
        self.assertEqual(res2.shape, (4,))
        assert_array_equal(res2, res.sum(sex, geo=(vla, wal, bru, belgium)))

    def test_agg_igroup(self):
        arr = ndtest(3)
        res = arr.sum((X.a.i[:2], X.a.i[1:]))
        assert_array_equal(res.a.labels, [':a1', 'a1:'])

    def test_ratio(self):
        la = self.larray
        age, geo, sex, lipro = la.axes

        regions = (self.vla_str, self.wal_str, self.bru_str, self.belgium)
        reg = la.sum(age, sex, regions)
        self.assertEqual(reg.shape, (4, 15))

        fla = geo[self.vla_str] >> 'Flanders'
        wal = geo[self.wal_str] >> 'Wallonia'
        bru = geo[self.bru_str] >> 'Brussels'
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

        fla = geo[self.vla_str] >> 'Flanders'
        wal = geo[self.wal_str] >> 'Wallonia'
        bru = geo[self.bru_str] >> 'Brussels'
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

        fla = geo[self.vla_str] >> 'Flanders'
        wal = geo[self.wal_str] >> 'Wallonia'
        bru = geo[self.bru_str] >> 'Brussels'
        bel = geo[:] >> 'Belgium'

        self.assertEqual(la.with_total(geo=(fla, wal, bru), op=mean).shape, (116, 47, 2, 15))
        self.assertEqual(la.with_total((fla, wal, bru), op=mean).shape, (116, 47, 2, 15))
        # works but "wrong" for x.geo (double what is expected because it includes fla wal & bru)
        # TODO: we probably want to display a warning (or even an error?) in that case.
        # If we really want that behavior, we can still split the operation:
        # .with_total((fla, wal, bru)).with_total(x.geo)
        # OR we might want to only sum the axis as it was before the op (but that does not play well when working with
        #    multiple axes).
        a1 = la.with_total(X.sex, (fla, wal, bru), X.geo, X.lipro)
        self.assertEqual(a1.shape, (116, 48, 3, 16))

        # correct total but the order is not very nice
        a2 = la.with_total(X.sex, X.geo, (fla, wal, bru), X.lipro)
        self.assertEqual(a2.shape, (116, 48, 3, 16))

        # the correct way to do it
        a3 = la.with_total(X.sex, (fla, wal, bru, bel), X.lipro)
        self.assertEqual(a3.shape, (116, 48, 3, 16))

        # a4 = la.with_total((lipro[':P05'], lipro['P05:']), op=mean)
        a4 = la.with_total((':P05', 'P05:'), op=mean)
        self.assertEqual(a4.shape, (116, 44, 2, 17))

    def test_transpose(self):
        arr = ndtest((2, 3, 4))
        a, b, c = arr.axes
        res = arr.transpose()
        self.assertEqual(res.axes, [c, b, a])
        res = arr.transpose('b', 'c', 'a')
        self.assertEqual(res.axes, [b, c, a])
        res = arr.transpose('b')
        self.assertEqual(res.axes, [b, a, c])

        # using Ellipsis instead of ... to avoid a syntax error on Python 2 (where ... is only available within [])
        res = arr.transpose(Ellipsis, 'a')
        self.assertEqual(res.axes, [b, c, a])
        res = arr.transpose('c', Ellipsis, 'a')
        self.assertEqual(res.axes, [c, b, a])

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

        # when doing a binop between anonymous axes, we use union too (that might be the problem) and we need *that*
        # union to match axes by position
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

        with np.errstate(invalid='ignore'):
            raw_res = raw / raw
        with pytest.warns(RuntimeWarning) as caught_warnings:
            res = la / la
        assert_array_nan_equal(res, raw_res)
        assert len(caught_warnings) == 1
        warn_msg = "invalid value (NaN) encountered during operation (this is typically caused by a 0 / 0)"
        assert caught_warnings[0].message.args[0] == warn_msg
        assert caught_warnings[0].filename == __file__

        assert_array_equal(la / 2, raw / 2)

        with np.errstate(divide='ignore'):
            raw_res = 30 / raw
        with pytest.warns(RuntimeWarning) as caught_warnings:
            res = 30 / la
        assert_array_equal(res, raw_res)
        assert len(caught_warnings) == 1
        assert caught_warnings[0].message.args[0] == "divide by zero encountered during operation"
        assert caught_warnings[0].filename == __file__

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

        with np.errstate(divide='ignore'):
            raw_res = 30 / raw
        with pytest.warns(RuntimeWarning) as caught_warnings:
            res = 30 / la
        assert_array_equal(res, raw_res)
        assert len(caught_warnings) == 1
        assert caught_warnings[0].message.args[0] == "divide by zero encountered during operation"
        assert caught_warnings[0].filename == __file__

        assert_array_equal(30 / (la + 1), 30 / (raw + 1))

        raw_int = raw.astype(int)
        la_int = LArray(raw_int)
        assert_array_equal(la_int / 2, raw_int / 2)
        assert_array_equal(la_int // 2, raw_int // 2)

        # adding two larrays with different axes order cannot work with unnamed axes
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
        # {0}*\{1}*  0  1  2
        #         0  0  0  0
        #         1  3  4  5
        self.assertTrue(np.array_equal(d, [[0, 0, 0],
                                           [3, 4, 5]]))

        # it is unfortunate that the behavior is different from numpy (even though I find our behavior more intuitive)
        d = np.asarray(a) * np.asarray(b)
        self.assertEqual(d.shape, (2, 3))
        self.assertTrue(np.array_equal(d, [[0, 1,  4],
                                           [0, 4, 10]]))

        with self.assertRaises(ValueError):
            # ValueError: operands could not be broadcast together with shapes (2,3) (2,)
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

    def test_sequence(self):
        res = sequence('b=b0..b2', ndtest(3) * 3, 1.0)
        assert_array_equal(ndtest((3, 3), dtype=float), res)

    def test_set_labels(self):
        la = self.small.copy()
        la.set_labels(X.sex, ['Man', 'Woman'], inplace=True)
        assert_array_equal(la, self.small.set_labels(X.sex, ['Man', 'Woman']))

    def test_replace_axes(self):
        lipro2 = Axis([l.replace('P', 'Q') for l in self.lipro.labels], 'lipro2')
        sex2 = Axis(['Man', 'Woman'], 'sex2')

        la = LArray(self.small_data, axes=(self.sex, lipro2),
                    title=self.small_title)
        # replace one axis
        la2 = self.small.set_axes(X.lipro, lipro2)
        assert_array_equal(la, la2)
        self.assertEqual(la.title, la2.title, "title of array returned by replace_axes should be the same as the "
                                              "original one. We got '{}' instead of '{}'".format(la2.title, la.title))

        la = LArray(self.small_data, axes=(sex2, lipro2), title=self.small_title)
        # all at once
        la2 = self.small.set_axes([sex2, lipro2])
        assert_array_equal(la, la2)
        # using keywrods args
        la2 = self.small.set_axes(sex=sex2, lipro=lipro2)
        assert_array_equal(la, la2)
        # using dict
        la2 = self.small.set_axes({X.sex: sex2, X.lipro: lipro2})
        assert_array_equal(la, la2)
        # using list of pairs (axis_to_replace, new_axis)
        la2 = self.small.set_axes([(X.sex, sex2), (X.lipro, lipro2)])
        assert_array_equal(la, la2)

    def test_reindex(self):
        arr = ndtest((2, 2))
        res = arr.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=-1)
        assert_array_equal(res, from_string("""a\\b  b1  b2  b0
                                                 a0   1  -1   0
                                                 a1   3  -1   2"""))

        arr2 = ndtest((2, 2))
        arr2.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=-1, inplace=True)
        assert_array_equal(arr2, from_string("""a\\b  b1  b2  b0
                                                  a0   1  -1   0
                                                  a1   3  -1   2"""))

        # LArray fill value
        filler = ndrange(arr.a)
        res = arr.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=filler)
        assert_array_equal(res, from_string("""a\\b  b1  b2  b0
                                                 a0   1   0   0
                                                 a1   3   1   2"""))

        # using labels from another array
        arr = ndrange('a=v0..v2;b=v0,v2,v1,v3')
        res = arr.reindex('a', arr.b.labels, fill_value=-1)
        assert_array_equal(res, from_string("""a\\b  v0  v2  v1  v3
                                                 v0   0   1   2   3
                                                 v2   8   9  10  11
                                                 v1   4   5   6   7
                                                 v3  -1  -1  -1  -1"""))
        res = arr.reindex('a', arr.b, fill_value=-1)
        assert_array_equal(res, from_string("""a\\b  v0  v2  v1  v3
                                                 v0   0   1   2   3
                                                 v2   8   9  10  11
                                                 v1   4   5   6   7
                                                 v3  -1  -1  -1  -1"""))

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
        l2 = LArray(la[:, :'P14'], axes=[sex, Axis(lipro.labels[1:], 'lipro')])
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
        la = la.extend('sex', la.sum(sex=(sex[:],)))
        self.assertEqual(la.shape, (3, 16))

    def test_hdf_roundtrip(self):
        a = ndtest((2, 3))
        a.to_hdf(abspath('test.h5'), 'a')
        res = read_hdf(abspath('test.h5'), 'a')

        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.axes.names, ['a', 'b'])
        assert_array_equal(res, a)

        # issue 72: int-like strings should not be parsed (should round-trip correctly)
        a = from_lists([['axis', '10', '20'],
                        ['',        0,    1]])
        a.to_hdf(abspath('issue72.h5'), 'a')
        res = read_hdf(abspath('issue72.h5'), 'a')
        self.assertEqual(res.ndim, 1)
        axis = res.axes[0]
        self.assertEqual(axis.name, 'axis')
        assert_array_equal(axis.labels, ['10', '20'])

        # passing group as key to to_hdf
        a3 = ndtest((4, 3, 4))
        fpath = abspath('test.h5')
        os.remove(fpath)
        # single element group
        for label in a3.a:
            a3[label].to_hdf(fpath, label)
        # unnamed group
        group = a3.c['c0,c2']
        a3[group].to_hdf(fpath, group)
        # unnamed group + slice
        group = a3.c['c0::2']
        a3[group].to_hdf(fpath, group)
        # named group
        group = a3.c['c0,c2'] >> 'even'
        a3[group].to_hdf(fpath, group)
        # group with name containing special characters (replaced by _)
        group = a3.c['c0,c2'] >> ':name?with*special/\[characters]'
        a3[group].to_hdf(fpath, group)

        # passing group as key to read_hdf
        for label in a3.a:
            subset = read_hdf(fpath, label)
            assert_array_equal(subset, a3[label])

        # load Session
        from  larray.core.session import Session
        s = Session(fpath)
        assert s.names == sorted(['a0', 'a1', 'a2', 'a3', 'c0,c2', 'c0::2', 'even', ':name?with*special__[characters]'])

    def test_from_string(self):
        expected = ndrange("sex=M,F")

        res = from_string('''sex  M  F
                             \t   0  1''')
        assert_array_equal(res, expected)

        res = from_string('''sex  M  F
                             nan  0  1''')
        assert_array_equal(res, expected)

        res = from_string('''sex  M  F
                             NaN  0  1''')
        assert_array_equal(res, expected)

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
        assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :],
                           [3722, 3395, 3347])

        la = read_csv(abspath('test2d_classic.csv'))
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', None])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        la = read_csv(abspath('test1d_liam2.csv'), dialect='liam2')
        self.assertEqual(la.ndim, 1)
        self.assertEqual(la.shape, (3,))
        self.assertEqual(la.axes.names, ['time'])
        assert_array_equal(la, [3722, 3395, 3347])

        la = read_csv(abspath('test5d_liam2.csv'), dialect='liam2')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :],
                           [3722, 3395, 3347])

    def test_read_eurostat(self):
        la = read_eurostat(abspath('test5d_eurostat.csv'))
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        # FIXME: integer labels should be parsed as such
        assert_array_equal(la[X.arr['1'], '0', 'F', X.nat['1'], :],
                           [3722, 3395, 3347])

    @pytest.mark.skipif(xw is None, reason="xlwings is not available")
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
        assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :],
                           [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '2d_classic')
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', None])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        # passing a Group as sheetname arg
        axis = Axis('dim=1d,2d,3d,5d')

        la = read_excel(abspath('test.xlsx'), axis['1d'])
        self.assertEqual(la.ndim, 1)
        self.assertEqual(la.shape, (3,))
        self.assertEqual(la.axes.names, ['time'])
        assert_array_equal(la, [3722, 3395, 3347])

        # fill_value argument
        la = read_excel(abspath('test.xlsx'), 'missing_values', fill_value=42)
        assert la.ndim == 3
        assert la.shape == (5, 2, 3)
        assert la.axes.names == ['age', 'sex', 'time']
        assert_array_equal(la[1, 'H', :], [42, 42, 42])
        assert_array_equal(la[4, 'F', :], [42, 42, 42])

        # invalid keyword argument
        with self.assertRaisesRegexp(TypeError, "'dtype' is an invalid keyword argument for this function when using "
                                                "the xlwings backend"):
            read_excel(abspath('test.xlsx'), engine='xlwings', dtype=float)

        # Excel sheet with blank cells on right/bottom border of the array to read
        fpath = abspath('test_blank_cells.xlsx')
        good_la = read_excel(fpath, 'good')
        bad_la = read_excel(fpath, 'bad')
        assert_array_equal(good_la, bad_la)
        # with additional empty column in the middle of the array to read
        good_la2 = ndrange('a=a0,a1;b=2003..2006').astype(object)
        good_la2[2005] = None
        good_la2 = good_la2.set_axes('b', Axis([2003, 2004, None, 2006], 'b'))
        bad_la2 = read_excel(fpath, 'bad2')
        assert_array_equal(good_la2, bad_la2)

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

        la = read_excel(abspath('test.xlsx'), '2d', engine='xlrd')
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', 'time'])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '3d', index_col=[0, 1], engine='xlrd')
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (5, 2, 3))
        self.assertEqual(la.axes.names, ['age', 'sex', 'time'])
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '3d', engine='xlrd')
        self.assertEqual(la.ndim, 3)
        self.assertEqual(la.shape, (5, 2, 3))
        self.assertEqual(la.axes.names, ['age', 'sex', 'time'])
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '5d', nb_index=4, engine='xlrd')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :],
                           [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '5d', engine='xlrd')
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :],
                           [3722, 3395, 3347])

        la = read_excel(abspath('test.xlsx'), '2d_classic', engine='xlrd')
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.shape, (5, 3))
        self.assertEqual(la.axes.names, ['age', None])
        assert_array_equal(la[0, :], [3722, 3395, 3347])

        # passing a Group as sheetname arg
        axis = Axis('dim=1d,2d,3d,5d')

        la = read_excel(abspath('test.xlsx'), axis['1d'], engine='xlrd')
        self.assertEqual(la.ndim, 1)
        self.assertEqual(la.shape, (3,))
        self.assertEqual(la.axes.names, ['time'])
        assert_array_equal(la, [3722, 3395, 3347])

        # Excel sheet with blank cells on right/bottom border of the array to read
        fpath = abspath('test_blank_cells.xlsx')
        good_la = read_excel(fpath, 'good', engine='xlrd')
        bad_la = read_excel(fpath, 'bad', engine='xlrd')
        assert_array_equal(good_la, bad_la)
        # with additional empty column in the middle of the array to read
        good_la2 = ndrange('a=a0,a1;b=2003..2006').astype(float)
        good_la2[2005] = np.nan
        good_la2 = good_la2.set_axes('b', Axis([2003, 2004, 'Unnamed: 3', 2006], 'b'))
        bad_la2 = read_excel(fpath, 'bad2', engine='xlrd')
        assert_array_nan_equal(good_la2, bad_la2)

    def test_from_lists(self):
        # sort_rows
        arr = from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
                          ['F', 'BE', 0, 0, 1],
                          ['F', 'FO', 0, 0, 2],
                          ['M', 'BE', 1, 0, 0],
                          ['M', 'FO', 2, 0, 0]])
        sorted_arr = from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
                                 ['M', 'BE', 1, 0, 0],
                                 ['M', 'FO', 2, 0, 0],
                                 ['F', 'BE', 0, 0, 1],
                                 ['F', 'FO', 0, 0, 2]], sort_rows=True)
        assert_array_equal(sorted_arr, arr)

        # sort_columns
        arr = from_lists([['sex', 'nat\\year', 1991, 1992, 1993],
                          ['M', 'BE', 1, 0, 0],
                          ['M', 'FO', 2, 0, 0],
                          ['F', 'BE', 0, 0, 1],
                          ['F', 'FO', 0, 0, 2]])
        sorted_arr = from_lists([['sex', 'nat\\year', 1992, 1991, 1993],
                                 ['M', 'BE', 0, 1, 0],
                                 ['M', 'FO', 0, 2, 0],
                                 ['F', 'BE', 0, 0, 1],
                                 ['F', 'FO', 0, 0, 2]], sort_columns=True)
        assert_array_equal(sorted_arr, arr)

    def test_from_frame(self):
        # 1) data = scalar
        # ================
        # Dataframe becomes 1D LArray
        data = np.array([10])
        index = ['i0']
        columns = ['c0']
        axis_index, axis_columns = Axis(index), Axis(columns)

        df = pd.DataFrame(data, index=index, columns=columns)
        assert df.index.name is None
        assert df.columns.name is None
        assert list(df.index.values) == index
        assert list(df.columns.values) == columns

        # anonymous indexes/columns
        # input dataframe:
        # ----------------
        #     c0
        # i0  10
        # output LArray:
        # --------------
        # {0}\{1}  c0
        #      i0  10
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, 1)
        assert la.axes.names == [None, None]
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, 1)), [axis_index, axis_columns])
        assert_array_equal(la, expected_la)

        # anonymous columns
        # input dataframe:
        # ----------------
        #        c0
        # index
        # i0     10
        # output LArray:
        # --------------
        # index\{1}  c0
        #        i0  10
        df.index.name, df.columns.name = 'index', None
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, 1)
        assert la.axes.names == ['index', None]
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, 1)), [axis_index.rename('index'), axis_columns])
        assert_array_equal(la, expected_la)

        # anonymous index
        # input dataframe:
        # ----------------
        # columns  c0
        # i0       10
        # output LArray:
        # --------------
        # {0}\columns  c0
        #          i0  10
        df.index.name, df.columns.name = None, 'columns'
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, 1)
        assert la.axes.names == [None, 'columns']
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, 1)), [axis_index, axis_columns.rename('columns')])
        assert_array_equal(la, expected_la)

        # index and columns with name
        # input dataframe:
        # ----------------
        # columns  c0
        # index
        # i0       10
        # output LArray:
        # --------------
        # index\columns  c0
        #            i0  10
        df.index.name, df.columns.name = 'index', 'columns'
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, 1)
        assert la.axes.names == ['index', 'columns']
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, 1)), [axis_index.rename('index'), axis_columns.rename('columns')])
        assert_array_equal(la, expected_la)

        # 2) data = vector
        # ================
        size = 3

        # 2A) data = horizontal vector (1 x N)
        # ====================================
        # Dataframe becomes 1D LArray
        data = np.arange(size)
        indexes = ['i0']
        columns = ['c{}'.format(i) for i in range(size)]
        axis_index, axis_columns = Axis(indexes), Axis(columns)

        df = pd.DataFrame(data.reshape(1, size), index=indexes, columns=columns)
        assert df.index.name is None
        assert df.columns.name is None
        assert list(df.index.values) == indexes
        assert list(df.columns.values) == columns

        # anonymous indexes/columns
        # input dataframe:
        # ----------------
        #     c0  c1  c2
        # i0   0   1   2
        # output LArray:
        # --------------
        # {0}\{1}  c0  c1  c2
        #      i0   0   1   2
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, size)
        assert la.axes.names == [None, None]
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, size)),[axis_index, axis_columns])
        assert_array_equal(la, expected_la)

        # anonymous columns
        # input dataframe:
        # ----------------
        #        c0  c1  c2
        # index
        # i0      0   1   2
        # output LArray:
        # --------------
        # index\{1}  c0  c1  c2
        #        i0   0   1   2
        df.index.name, df.columns.name = 'index', None
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, size)
        assert la.axes.names == ['index', None]
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, size)),[axis_index.rename('index'), axis_columns])
        assert_array_equal(la, expected_la)

        # anonymous index
        # input dataframe:
        # ----------------
        # columns  c0  c1  c2
        # i0        0   1   2
        # output LArray:
        # --------------
        # {0}\columns  c0  c1  c2
        #          i0   0   1   2
        df.index.name, df.columns.name = None, 'columns'
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, size)
        assert la.axes.names == [None, 'columns']
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, size)),[axis_index, axis_columns.rename('columns')])
        assert_array_equal(la, expected_la)

        # index and columns with name
        # input dataframe:
        # ----------------
        # columns  c0  c1  c2
        # index
        # i0        0   1   2
        # output LArray:
        # --------------
        # index\columns  c0  c1  c2
        #            i0   0   1   2
        df.index.name, df.columns.name = 'index', 'columns'
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (1, size)
        assert la.axes.names == ['index', 'columns']
        assert list(la.axes.labels[0]) == index
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data.reshape((1, size)),[axis_index.rename('index'), axis_columns.rename('columns')])
        assert_array_equal(la, expected_la)

        # 2B) data = vertical vector (N x 1)
        # ==================================
        # Dataframe becomes 2D LArray
        data = data.reshape(size, 1)
        indexes = ['i{}'.format(i) for i in range(size)]
        columns = ['c0']
        axis_index, axis_columns = Axis(indexes), Axis(columns)

        df = pd.DataFrame(data, index=indexes, columns=columns)
        assert df.index.name is None
        assert df.columns.name is None
        assert list(df.index.values) == indexes
        assert list(df.columns.values) == columns

        # anonymous indexes/columns
        # input dataframe:
        # ----------------
        #     c0
        # i0   0
        # i1   1
        # i2   2
        # output LArray:
        # --------------
        # {0}\{1}  c0
        #      i0   0
        #      i1   1
        #      i2   2
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (size, 1)
        assert la.axes.names == [None, None]
        assert list(la.axes.labels[0]) == indexes
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data, [axis_index, axis_columns])
        assert_array_equal(la, expected_la)

        # anonymous columns
        # input dataframe:
        # ----------------
        #        c0
        # index
        # i0      0
        # i1      1
        # i2      2
        # output LArray:
        # --------------
        # index\{1}  c0
        #        i0   0
        #        i1   1
        #        i2   2
        df.index.name, df.columns.name = 'index', None
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (size, 1)
        assert la.axes.names == ['index', None]
        assert list(la.axes.labels[0]) == indexes
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data, [axis_index.rename('index'), axis_columns])
        assert_array_equal(la, expected_la)

        # anonymous index
        # input dataframe:
        # ----------------
        # columns  c0
        # i0        0
        # i1        1
        # i2        2
        # output LArray:
        # --------------
        # {0}\columns  c0
        #          i0   0
        #          i1   1
        #          i2   2
        df.index.name, df.columns.name = None, 'columns'
        la = from_frame(df)
        assert la.ndim == 2
        assert la.shape == (size, 1)
        assert la.axes.names == [None, 'columns']
        assert list(la.axes.labels[0]) == indexes
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data, [axis_index, axis_columns.rename('columns')])
        assert_array_equal(la, expected_la)

        # index and columns with name
        # input dataframe:
        # ----------------
        # columns  c0
        # index
        # i0        0
        # i1        1
        # i2        2
        # output LArray:
        # --------------
        # {0}\columns  c0
        #          i0   0
        #          i1   1
        #          i2   2
        df.index.name, df.columns.name = 'index', 'columns'
        assert la.ndim == 2
        assert la.shape == (size, 1)
        assert la.axes.names == [None, 'columns']
        assert list(la.axes.labels[0]) == indexes
        assert list(la.axes.labels[1]) == columns
        expected_la = LArray(data, [axis_index, axis_columns.rename('columns')])
        assert_array_equal(la, expected_la)

        # 3) 3D array
        # ===========

        # 3A) Dataframe with 2 index columns
        # ==================================
        dt = [('age', int), ('sex', 'U1'),
              ('2007', int), ('2010', int), ('2013', int)]
        data = np.array([
            (0, 'F', 3722, 3395, 3347),
            (0, 'M', 338, 316, 323),
            (1, 'F', 2878, 2791, 2822),
            (1, 'M', 1121, 1037, 976),
            (2, 'F', 4073, 4161, 4429),
            (2, 'M', 1561, 1463, 1467),
            (3, 'F', 3507, 3741, 3366),
            (3, 'M', 2052, 2052, 2118),
        ], dtype=dt)
        df = pd.DataFrame(data)
        df.set_index(['age', 'sex'], inplace=True)
        df.columns.name = 'time'

        la = from_frame(df)
        assert la.ndim == 3
        assert la.shape == (4, 2, 3)
        assert la.axes.names == ['age', 'sex', 'time']
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])

        # 3B) Dataframe with columns.name containing \\
        # =============================================
        dt = [('age', int), ('sex\\time', 'U1'),
              ('2007', int), ('2010', int), ('2013', int)]
        data = np.array([
            (0, 'F', 3722, 3395, 3347),
            (0, 'M', 338, 316, 323),
            (1, 'F', 2878, 2791, 2822),
            (1, 'M', 1121, 1037, 976),
            (2, 'F', 4073, 4161, 4429),
            (2, 'M', 1561, 1463, 1467),
            (3, 'F', 3507, 3741, 3366),
            (3, 'M', 2052, 2052, 2118),
        ], dtype=dt)
        df = pd.DataFrame(data)
        df.set_index(['age', 'sex\\time'], inplace=True)

        la = from_frame(df, unfold_last_axis_name=True)
        assert la.ndim == 3
        assert la.shape == (4, 2, 3)
        assert la.axes.names == ['age', 'sex', 'time']
        assert_array_equal(la[0, 'F', :], [3722, 3395, 3347])


    def test_to_csv(self):
        la = read_csv(abspath('test5d.csv'))
        self.assertEqual(la.ndim, 5)
        self.assertEqual(la.shape, (2, 5, 2, 2, 3))
        self.assertEqual(la.axes.names, ['arr', 'age', 'sex', 'nat', 'time'])
        assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :],
                           [3722, 3395, 3347])

        la.to_csv(abspath('out.csv'))
        result = ['arr,age,sex,nat\\time,2007,2010,2013\n',
                  '1,0,F,1,3722,3395,3347\n',
                  '1,0,F,2,338,316,323\n']
        with open(abspath('out.csv')) as f:
            self.assertEqual(f.readlines()[:3], result)

        la.to_csv(abspath('out.csv'), transpose=False)
        result = ['arr,age,sex,nat,time,0\n',
                  '1,0,F,1,2007,3722\n',
                  '1,0,F,1,2010,3395\n']
        with open(abspath('out.csv')) as f:
            self.assertEqual(f.readlines()[:3], result)

        la = ndrange([Axis('time=2015..2017')])
        la.to_csv(abspath('test_out1d.csv'))
        result = ['time,2015,2016,2017\n',
                  ',0,1,2\n']
        with open(abspath('test_out1d.csv')) as f:
            self.assertEqual(f.readlines(), result)

    def test_to_excel_xlsxwriter(self):
        fpath = abspath('test_to_excel_xlsxwriter.xlsx')

        # 1D
        a1 = ndtest(3)

        # fpath/Sheet1/A1
        a1.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a1)

        # fpath/Sheet1/A1(transposed)
        a1.to_excel(fpath, transpose=True, engine='xlsxwriter')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a1)

        # 2D
        a2 = ndtest((2, 3))

        # fpath/Sheet1/A1
        a2.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a2)

        # fpath/Sheet1/A10
        # TODO: this is currently not supported (though we would only need to translate A10 to startrow=0 and startcol=0
        # a2.to_excel('fpath', 'Sheet1', 'A10', engine='xlsxwriter')
        # res = read_excel('fpath', 'Sheet1', engine='xlrd', skiprows=9)
        # assert_array_equal(res, a2)

        # fpath/other/A1
        a2.to_excel(fpath, 'other', engine='xlsxwriter')
        res = read_excel(fpath, 'other', engine='xlrd')
        assert_array_equal(res, a2)

        # 3D
        a3 = ndtest((2, 3, 4))

        # fpath/Sheet1/A1
        # FIXME: merge_cells=False should be the default (until Pandas is fixed to read its format)
        a3.to_excel(fpath, overwrite_file=True, engine='xlsxwriter', merge_cells=False)
        # a3.to_excel('fpath', overwrite_file=True, engine='openpyxl')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a3)

        # fpath/Sheet1/A20
        # TODO: implement position (see above)
        # a3.to_excel('fpath', 'Sheet1', 'A20', engine='xlsxwriter', merge_cells=False)
        # res = read_excel('fpath', 'Sheet1', engine='xlrd', skiprows=19)
        # assert_array_equal(res, a3)

        # fpath/other/A1
        a3.to_excel(fpath, 'other', engine='xlsxwriter', merge_cells=False)
        res = read_excel(fpath, 'other', engine='xlrd')
        assert_array_equal(res, a3)

        # 1D
        a1 = ndtest(3)

        # fpath/Sheet1/A1
        a1.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a1)

        # fpath/Sheet1/A1(transposed)
        a1.to_excel(fpath, transpose=True, engine='xlsxwriter')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a1)

        # 2D
        a2 = ndtest((2, 3))

        # fpath/Sheet1/A1
        a2.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a2)

        # fpath/Sheet1/A10
        # TODO: this is currently not supported (though we would only need to translate A10 to startrow=0 and startcol=0
        # a2.to_excel(fpath, 'Sheet1', 'A10', engine='xlsxwriter')
        # res = read_excel('fpath', 'Sheet1', engine='xlrd', skiprows=9)
        # assert_array_equal(res, a2)

        # fpath/other/A1
        a2.to_excel(fpath, 'other', engine='xlsxwriter')
        res = read_excel(fpath, 'other', engine='xlrd')
        assert_array_equal(res, a2)

        # 3D
        a3 = ndtest((2, 3, 4))

        # fpath/Sheet1/A1
        # FIXME: merge_cells=False should be the default (until Pandas is fixed to read its format)
        a3.to_excel(fpath, overwrite_file=True, engine='xlsxwriter', merge_cells=False)
        # a3.to_excel('fpath', overwrite_file=True, engine='openpyxl')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a3)

        # fpath/Sheet1/A20
        # TODO: implement position (see above)
        # a3.to_excel('fpath', 'Sheet1', 'A20', engine='xlsxwriter', merge_cells=False)
        # res = read_excel('fpath', 'Sheet1', engine='xlrd', skiprows=19)
        # assert_array_equal(res, a3)

        # fpath/other/A1
        a3.to_excel(fpath, 'other', engine='xlsxwriter', merge_cells=False)
        res = read_excel(fpath, 'other', engine='xlrd')
        assert_array_equal(res, a3)

        # passing group as sheet_name
        a3 = ndtest((4, 3, 4))
        os.remove(fpath)
        # single element group
        for label in a3.a:
            a3[label].to_excel(fpath, label, engine='xlsxwriter')
        # unnamed group
        group = a3.c['c0,c2']
        a3[group].to_excel(fpath, group, engine='xlsxwriter')
        # unnamed group + slice
        group = a3.c['c0::2']
        a3[group].to_excel(fpath, group, engine='xlsxwriter')
        # named group
        group = a3.c['c0,c2'] >> 'even'
        a3[group].to_excel(fpath, group, engine='xlsxwriter')
        # group with name containing special characters (replaced by _)
        group = a3.c['c0,c2'] >> ':name?with*special/\[char]'
        a3[group].to_excel(fpath, group, engine='xlsxwriter')

    @pytest.mark.skipif(xw is None, reason="xlwings is not available")
    def test_to_excel_xlwings(self):
        fpath = abspath('test_to_excel_xlwings.xlsx')

        # 1D
        a1 = ndtest(3)

        # live book/Sheet1/A1
        # a1.to_excel()

        # fpath/Sheet1/A1 (create a new file if does not exist)
        if os.path.isfile(fpath):
            os.remove(fpath)
        a1.to_excel(fpath, engine='xlwings')
        # we use xlrd to read back instead of xlwings even if that should work, to make the test faster
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a1)

        # fpath/Sheet1/A1(transposed)
        a1.to_excel(fpath, transpose=True, engine='xlwings')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a1)

        # 2D
        a2 = ndtest((2, 3))

        # fpath/Sheet1/A1
        a2.to_excel(fpath, overwrite_file=True, engine='xlwings')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a2)

        # fpath/Sheet1/A10
        a2.to_excel(fpath, 'Sheet1', 'A10', engine='xlwings')
        res = read_excel(fpath, 'Sheet1', engine='xlrd', skiprows=9)
        assert_array_equal(res, a2)

        # fpath/other/A1
        a2.to_excel(fpath, 'other', engine='xlwings')
        res = read_excel(fpath, 'other', engine='xlrd')
        assert_array_equal(res, a2)

        # 3D
        a3 = ndtest((2, 3, 4))

        # fpath/Sheet1/A1
        a3.to_excel(fpath, overwrite_file=True, engine='xlwings')
        res = read_excel(fpath, engine='xlrd')
        assert_array_equal(res, a3)

        # fpath/Sheet1/A20
        a3.to_excel(fpath, 'Sheet1', 'A20', engine='xlwings')
        res = read_excel(fpath, 'Sheet1', engine='xlrd', skiprows=19)
        assert_array_equal(res, a3)

        # fpath/other/A1
        a3.to_excel(fpath, 'other', engine='xlwings')
        res = read_excel(fpath, 'other', engine='xlrd')
        assert_array_equal(res, a3)

        # passing group as sheet_name
        a3 = ndtest((4, 3, 4))
        os.remove(fpath)
        # single element group
        for label in a3.a:
            a3[label].to_excel(fpath, label, engine='xlwings')
        # unnamed group
        group = a3.c['c0,c2']
        a3[group].to_excel(fpath, group, engine='xlwings')
        # unnamed group + slice
        group = a3.c['c0::2']
        a3[group].to_excel(fpath, group, engine='xlwings')
        # named group
        group = a3.c['c0,c2'] >> 'even'
        a3[group].to_excel(fpath, group, engine='xlwings')
        # group with name containing special characters (replaced by _)
        group = a3.c['c0,c2'] >> ':name?with*special/\[char]'
        a3[group].to_excel(fpath, group, engine='xlwings')
        # checks sheet names
        sheet_names = sorted(open_excel(fpath).sheet_names())
        assert sheet_names == sorted(['a0', 'a1', 'a2', 'a3', 'c0,c2', 'c0__2', 'even',
                                      '_name_with_special___char_'])

    @pytest.mark.skipif(xw is None, reason="xlwings is not available")
    def test_open_excel(self):
        # 1) Create new file
        # ==================
        fpath = abspath('should_not_extist.xlsx')
        # overwrite_file must be set to True to create a new file
        with pytest.raises(ValueError):
            open_excel(fpath)

        # 2) with headers
        # ===============
        with open_excel(visible=False) as wb:
            # 1D
            a1 = ndtest(3)

            # Sheet1/A1
            wb['Sheet1'] = a1.dump()
            res = wb['Sheet1'].load()
            assert_array_equal(res, a1)

            wb[0] = a1.dump()
            res = wb[0].load()
            assert_array_equal(res, a1)

            # Sheet1/A1(transposed)
            # TODO: implement .options on Sheet so that one can write:
            # wb[0].options(transpose=True).value = a1.dump()
            wb[0]['A1'].options(transpose=True).value = a1.dump()
            # TODO: implement .options on Range so that you can write:
            # res = wb[0]['A1:B4'].options(transpose=True).load()
            # res = from_lists(wb[0]['A1:B4'].options(transpose=True).value)
            # assert_array_equal(res, a1)

            # 2D
            a2 = ndtest((2, 3))

            # Sheet1/A1
            wb[0] = a2.dump()
            res = wb[0].load()
            assert_array_equal(res, a2)

            # Sheet1/A10
            wb[0]['A10'] = a2.dump()
            res = wb[0]['A10:D12'].load()
            assert_array_equal(res, a2)

            # other/A1
            wb['other'] = a2.dump()
            res = wb['other'].load()
            assert_array_equal(res, a2)

            # new/A10
            # we need to create the sheet first
            wb['new'] = ''
            wb['new']['A10'] = a2.dump()
            res = wb['new']['A10:D12'].load()
            assert_array_equal(res, a2)

            # new2/A10
            # cannot store the return value of "add" because that's a raw xlwings Sheet
            wb.sheets.add('new2')
            wb['new2']['A10'] = a2.dump()
            res = wb['new2']['A10:D12'].load()
            assert_array_equal(res, a2)

            # 3D
            a3 = ndtest((2, 3, 4))

            # 3D/A1
            wb['3D'] = a3.dump()
            res = wb['3D'].load()
            assert_array_equal(res, a3)

            # 3D/A20
            wb['3D']['A20'] = a3.dump()
            res = wb['3D']['A20:F26'].load()
            assert_array_equal(res, a3)

            # 3D/A20 without name for columns
            wb['3D']['A20'] = a3.dump()
            # assume we have no name for the columns axis (ie change b\c to b)
            wb['3D']['B20'] = 'b'
            res = wb['3D']['A20:F26'].load(nb_index=2)
            assert_array_equal(res, a3.data)
            # the two first axes should be the same
            self.assertEqual(res.axes[:2], a3.axes[:2])
            # the third axis should have the same labels (but not the same name obviously)
            assert_array_equal(res.axes[2].labels, a3.axes[2].labels)

        with open_excel(abspath('test.xlsx')) as wb:
            res = wb['2d_classic'].load()
            assert res.ndim == 2
            assert res.shape == (5, 3)
            assert res.axes.names == ['age', None]
            assert_array_equal(res[0, :], [3722, 3395, 3347])

        # 3) without headers
        # ==================
        with open_excel(visible=False) as wb:
            # 1D
            a1 = ndtest(3)

            # Sheet1/A1
            wb['Sheet1'] = a1
            res = wb['Sheet1'].load(header=False)
            assert_array_equal(res, a1.data)

            wb[0] = a1
            res = wb[0].load(header=False)
            assert_array_equal(res, a1.data)

            # Sheet1/A1(transposed)
            # FIXME: we need to .dump(header=False) explicitly because otherwise we go via LArrayConverter which
            #        includes labels. for consistency's sake we should either change LArrayConverter to not include
            #        labels, or change wb[0] = a1 to include them (and use wb[0] = a1.data to avoid them?) but that
            #        would be heavily backward incompatible and how would I load them back?
            # wb[0]['A1'].options(transpose=True).value = a1
            wb[0]['A1'].options(transpose=True).value = a1.dump(header=False)
            res = wb[0]['A1:A3'].load(header=False)
            assert_array_equal(res, a1.data)

            # 2D
            a2 = ndtest((2, 3))

            # Sheet1/A1
            wb[0] = a2
            res = wb[0].load(header=False)
            assert_array_equal(res, a2.data)

            # Sheet1/A10
            wb[0]['A10'] = a2
            res = wb[0]['A10:C11'].load(header=False)
            assert_array_equal(res, a2.data)

            # other/A1
            wb['other'] = a2
            res = wb['other'].load(header=False)
            assert_array_equal(res, a2.data)

            # new/A10
            # we need to create the sheet first
            wb['new'] = ''
            wb['new']['A10'] = a2
            res = wb['new']['A10:C11'].load(header=False)
            assert_array_equal(res, a2.data)

            # 3D
            a3 = ndtest((2, 3, 4))

            # 3D/A1
            wb['3D'] = a3
            res = wb['3D'].load(header=False)
            assert_array_equal(res, a3.data.reshape((6, 4)))

            # 3D/A20
            wb['3D']['A20'] = a3
            res = wb['3D']['A20:D25'].load(header=False)
            assert_array_equal(res, a3.data.reshape((6, 4)))

        # 4) Blank cells
        # ========================
        # Excel sheet with blank cells on right/bottom border of the array to read
        fpath = abspath('test_blank_cells.xlsx')
        with open_excel(fpath) as wb:
            good_la = wb['good'].load()
            bad_la = wb['bad'].load()
            # with additional empty column in the middle of the array to read
            good_la2 = wb['bad2']['A1:E3'].load()
            bad_la2 = wb['bad2'].load()
        assert_array_equal(good_la, bad_la)
        assert_array_equal(good_la2, bad_la2)

        # 5) crash test
        # =============
        arr = ndtest((2, 2))
        fpath = abspath('temporary_test_file.xlsx')
        # create and save a test file
        with open_excel(fpath, overwrite_file=True) as wb:
            wb['arr'] = arr.dump()
            wb.save()
        # raise exception when the file is open
        try:
            with open_excel(fpath, overwrite_file=True) as wb:
                raise ValueError("")
        except ValueError:
            pass
        # check if file is still available
        with open_excel(fpath) as wb:
            assert wb.sheet_names() == ['arr']
            assert_array_equal(wb['arr'].load(), arr)
        # remove file
        if os.path.exists(fpath):
            os.remove(fpath)

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
        la_out = zeros([Axis([0, 1, 2], 'a')] + list(la.axes))
        raw_out = np.zeros((3,) + raw.shape)

        la_out2 = exp(la, la_out)
        raw_out2 = np.exp(raw, raw_out)

        # self.assertIs(la_out2, la_out)
        # XXX: why is la_out2 transposed?
        assert_array_equal(la_out2.transpose(X.a), la_out)
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
        sex = Axis('sex=M,F')
        a = eye(sex)
        d = diag(a)
        self.assertEqual(d.ndim, 1)
        self.assertEqual(d.axes.names, ['sex,sex'])
        assert_array_equal(d.axes.labels, [['M,M', 'F,F']])
        self.assertEqual(d.i[0], 1.0)
        self.assertEqual(d.i[1], 1.0)

    @pytest.mark.skipif(sys.version_info < (3, 5), reason="@ unavailable (Python < 3.5)")
    def test_matmul(self):
        # 2D / anonymous axes
        a1 = eye(3) * 2
        a2 = ndrange((3, 3))
        # cannot use @ in the tests because that is an invalid syntax in Python 2
        # LArray value
        assert_array_equal(a1.__matmul__(a2), ndrange((3, 3)) * 2)

        # ndarray value
        assert_array_equal(a1.__matmul__(a2.data), ndrange((3, 3)) * 2)

        # non anonymous axes (N <= 2)
        arr1d = ndtest(3)
        arr2d = ndtest((3, 3))

        # 1D @ 1D
        self.assertEqual(arr1d.__matmul__(arr1d), 5)

        # 1D @ 2D
        assert_array_equal(arr1d.__matmul__(arr2d),
                           LArray([15, 18, 21], 'b=b0..b2'))

        # 2D @ 1D
        assert_array_equal(arr2d.__matmul__(arr1d),
                           LArray([5, 14, 23], 'a=a0..a2'))

        # 2D(a,b) @ 2D(a,b) -> 2D(a,b)
        res = from_lists([['a\\b', 'b0', 'b1', 'b2'],
                          ['a0', 15, 18, 21],
                          ['a1', 42, 54, 66],
                          ['a2', 69, 90, 111]])
        assert_array_equal(arr2d.__matmul__(arr2d), res)

        # 2D(a,b) @ 2D(b,a) -> 2D(a,a)
        res = from_lists([['a\\a', 'a0', 'a1', 'a2'],
                          ['a0', 5, 14, 23],
                          ['a1', 14, 50, 86],
                          ['a2', 23, 86, 149]])
        assert_array_equal(arr2d.__matmul__(arr2d.T), res)

        # ndarray value
        assert_array_equal(arr1d.__matmul__(arr2d.data),
                           LArray([15, 18, 21]))
        assert_array_equal(arr2d.data.__matmul__(arr2d.T.data),
                           res.data)

        # different axes
        a1 = ndtest('a=a0..a1;b=b0..b2')
        a2 = ndrange('b=b0..b2;c=c0..c3')
        res = from_lists([['a\c', 'c0', 'c1', 'c2', 'c3'],
                          ['a0', 20, 23, 26, 29],
                          ['a1', 56, 68, 80, 92]])
        assert_array_equal(a1.__matmul__(a2), res)

        # non anonymous axes (N >= 2)
        arr2d = ndtest((2, 2))
        arr3d = ndtest((2, 2, 2))
        arr4d = ndtest((2, 2, 2, 2))
        a, b, c, d = arr4d.axes
        e = Axis('e=e0,e1')
        f = Axis('f=f0,f1')

        # 4D(a, b, c, d) @ 3D(e, d, f) -> 5D(a, b, e, c, f)
        arr3d = arr3d.set_axes([e, d, f])
        res = from_lists([['a', 'b', 'e', 'c\\f', 'f0', 'f1'],
                          ['a0', 'b0', 'e0', 'c0', 2, 3],
                          ['a0', 'b0', 'e0', 'c1', 6, 11],
                          ['a0', 'b0', 'e1', 'c0', 6, 7],
                          ['a0', 'b0', 'e1', 'c1', 26, 31],
                          ['a0', 'b1', 'e0', 'c0', 10, 19],
                          ['a0', 'b1', 'e0', 'c1', 14, 27],
                          ['a0', 'b1', 'e1', 'c0', 46, 55],
                          ['a0', 'b1', 'e1', 'c1', 66, 79],
                          ['a1', 'b0', 'e0', 'c0', 18, 35],
                          ['a1', 'b0', 'e0', 'c1', 22, 43],
                          ['a1', 'b0', 'e1', 'c0', 86, 103],
                          ['a1', 'b0', 'e1', 'c1', 106, 127],
                          ['a1', 'b1', 'e0', 'c0', 26, 51],
                          ['a1', 'b1', 'e0', 'c1', 30, 59],
                          ['a1', 'b1', 'e1', 'c0', 126, 151],
                          ['a1', 'b1', 'e1', 'c1', 146, 175]])
        assert_array_equal(arr4d.__matmul__(arr3d), res)

        # 3D(e, d, f) @ 4D(a, b, c, d) -> 5D(e, a, b, d, d)
        res = from_lists([['e', 'a', 'b', 'd\\d', 'd0', 'd1'],
                          ['e0', 'a0', 'b0', 'd0', 2, 3],
                          ['e0', 'a0', 'b0', 'd1', 6, 11],
                          ['e0', 'a0', 'b1', 'd0', 6, 7],
                          ['e0', 'a0', 'b1', 'd1', 26, 31],
                          ['e0', 'a1', 'b0', 'd0', 10, 11],
                          ['e0', 'a1', 'b0', 'd1', 46, 51],
                          ['e0', 'a1', 'b1', 'd0', 14, 15],
                          ['e0', 'a1', 'b1', 'd1', 66, 71],
                          ['e1', 'a0', 'b0', 'd0', 10, 19],
                          ['e1', 'a0', 'b0', 'd1', 14, 27],
                          ['e1', 'a0', 'b1', 'd0', 46, 55],
                          ['e1', 'a0', 'b1', 'd1', 66, 79],
                          ['e1', 'a1', 'b0', 'd0', 82, 91],
                          ['e1', 'a1', 'b0', 'd1', 118, 131],
                          ['e1', 'a1', 'b1', 'd0', 118, 127],
                          ['e1', 'a1', 'b1', 'd1', 170, 183]])
        assert_array_equal(arr3d.__matmul__(arr4d), res)

        # 4D(a, b, c, d) @ 3D(b, d, f) -> 4D(a, b, c, f)
        arr3d = arr3d.set_axes([b, d, f])
        res = from_lists([['a', 'b', 'c\\f', 'f0', 'f1'],
                          ['a0', 'b0', 'c0', 2, 3],
                          ['a0', 'b0', 'c1', 6, 11],
                          ['a0', 'b1', 'c0', 46, 55],
                          ['a0', 'b1', 'c1', 66, 79],
                          ['a1', 'b0', 'c0', 18, 35],
                          ['a1', 'b0', 'c1', 22, 43],
                          ['a1', 'b1', 'c0', 126, 151],
                          ['a1', 'b1', 'c1', 146, 175]])
        assert_array_equal(arr4d.__matmul__(arr3d), res)

        # 3D(b, d, f) @ 4D(a, b, c, d) -> 4D(b, a, d, d)
        res = from_lists([['b', 'a', 'd\\d', 'd0', 'd1'],
                          ['b0', 'a0', 'd0', 2, 3],
                          ['b0', 'a0', 'd1', 6, 11],
                          ['b0', 'a1', 'd0', 10, 11],
                          ['b0', 'a1', 'd1', 46, 51],
                          ['b1', 'a0', 'd0', 46, 55],
                          ['b1', 'a0', 'd1', 66, 79],
                          ['b1', 'a1', 'd0', 118, 127],
                          ['b1', 'a1', 'd1', 170, 183]])
        assert_array_equal(arr3d.__matmul__(arr4d), res)

        # 4D(a, b, c, d) @ 2D(d, f) -> 5D(a, b, c, f)
        arr2d = arr2d.set_axes([d, f])
        res = from_lists([['a', 'b', 'c\\f', 'f0', 'f1'],
                          ['a0', 'b0', 'c0', 2, 3],
                          ['a0', 'b0', 'c1', 6, 11],
                          ['a0', 'b1', 'c0', 10, 19],
                          ['a0', 'b1', 'c1', 14, 27],
                          ['a1', 'b0', 'c0', 18, 35],
                          ['a1', 'b0', 'c1', 22, 43],
                          ['a1', 'b1', 'c0', 26, 51],
                          ['a1', 'b1', 'c1', 30, 59]])
        assert_array_equal(arr4d.__matmul__(arr2d), res)

        # 2D(d, f) @ 4D(a, b, c, d) -> 5D(a, b, d, d)
        res = from_lists([['a', 'b', 'd\\d', 'd0', 'd1'],
                          ['a0', 'b0', 'd0', 2, 3],
                          ['a0', 'b0', 'd1', 6, 11],
                          ['a0', 'b1', 'd0', 6, 7],
                          ['a0', 'b1', 'd1', 26, 31],
                          ['a1', 'b0', 'd0', 10, 11],
                          ['a1', 'b0', 'd1', 46, 51],
                          ['a1', 'b1', 'd0', 14, 15],
                          ['a1', 'b1', 'd1', 66, 71]])
        assert_array_equal(arr2d.__matmul__(arr4d), res)

    @pytest.mark.skipif(sys.version_info < (3, 5), reason="@ unavailable (Python < 3.5)")
    def test_rmatmul(self):
        a1 = eye(3) * 2
        a2 = ndrange((3, 3))

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
        # common axes are reordered according to target (a1 in this case)
        self.assertEqual(b.shape, (1, 3))
        assert_larray_equiv(b, a2)

        a1 = ndrange((2, 3))
        a2 = ndrange((3, 2))
        b = a2.broadcast_with(a1)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.shape, (2, 3))
        assert_larray_equiv(b, a2)

    def test_plot(self):
        pass
        # small_h = small['M']
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

    def test_combine_axes(self):
        # combine N axes into 1
        # =====================
        arr = ndtest((2, 3, 4, 5))
        res = arr.combine_axes((X.a, X.b))
        self.assertEqual(res.axes.names, ['a_b', 'c', 'd'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2 * 3, 4, 5))
        assert_array_equal(res.axes.a_b.labels[:2], ['a0_b0', 'a0_b1'])
        assert_array_equal(res['a1_b0'], arr['a1', 'b0'])

        res = arr.combine_axes((X.a, X.c))
        self.assertEqual(res.axes.names, ['a_c', 'b', 'd'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2 * 4, 3, 5))
        assert_array_equal(res.axes.a_c.labels[:2], ['a0_c0', 'a0_c1'])
        assert_array_equal(res['a1_c0'], arr['a1', 'c0'])

        res = arr.combine_axes((X.b, X.d))
        self.assertEqual(res.axes.names, ['a', 'b_d', 'c'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2, 3 * 5, 4))
        assert_array_equal(res.axes.b_d.labels[:2], ['b0_d0', 'b0_d1'])
        assert_array_equal(res['b1_d0'], arr['b1', 'd0'])

        # combine M axes into N
        # =====================
        arr = ndtest((2, 3, 4, 4, 3, 2))

        # using a list of tuples
        res = arr.combine_axes([('a', 'c'), ('b', 'f'), ('d', 'e')])
        assert res.axes.names == ['a_c', 'b_f', 'd_e']
        assert res.size == arr.size
        assert res.shape == (2 * 4, 3 * 2, 4 * 3)
        assert list(res.axes.a_c.labels[:2]) == ['a0_c0', 'a0_c1']
        assert list(res.axes.b_f.labels[:2]) == ['b0_f0', 'b0_f1']
        assert list(res.axes.d_e.labels[:2]) == ['d0_e0', 'd0_e1']
        assert res['a0_c2', 'b1_f1', 'd3_e2'] == arr['a0', 'b1', 'c2', 'd3', 'e2', 'f1']

        res = arr.combine_axes([('a', 'c'), ('b', 'e', 'f')])
        assert res.axes.names == ['a_c', 'b_e_f', 'd']
        assert res.size == arr.size
        assert res.shape == (2 * 4, 3 * 3 * 2, 4)
        assert list(res.axes.b_e_f.labels[:4]) == ['b0_e0_f0', 'b0_e0_f1', 'b0_e1_f0', 'b0_e1_f1']
        assert_array_equal(res['a0_c2', 'b1_e2_f1'], arr['a0', 'b1', 'c2', 'e2', 'f1'])

        # using a dict (-> user defined axes names)
        res = arr.combine_axes({('a', 'c'): 'AC', ('b', 'f'): 'BF', ('d', 'e'): 'DE'})
        assert res.axes.names == ['AC', 'BF', 'DE']
        assert res.size == arr.size
        assert res.shape == (2 * 4, 3 * 2, 4 * 3)

        res = arr.combine_axes({('a', 'c'): 'AC', ('b', 'e', 'f'): 'BEF'})
        assert res.axes.names == ['AC', 'BEF', 'd']
        assert res.size == arr.size
        assert res.shape == (2 * 4, 3 * 3 * 2, 4)

    def test_split_axes(self):
        # split one axis
        # ==============
        arr = ndtest((2, 3, 4, 5))
        comb = arr.combine_axes((X.b, X.d))
        self.assertEqual(comb.axes.names, ['a', 'b_d', 'c'])
        # default delimiter '_'
        res = comb.split_axes('b_d')
        self.assertEqual(res.axes.names, ['a', 'b', 'd', 'c'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2, 3, 5, 4))
        assert_array_equal(res.transpose(X.a, X.b, X.c, X.d), arr)
        # regex
        names = ['b', 'd']
        regex = '(\w+)_(\w+)'
        res = comb.split_axes('b_d', names=names, regex=regex)
        self.assertEqual(res.axes.names, ['a', 'b', 'd', 'c'])
        self.assertEqual(res.size, arr.size)
        self.assertEqual(res.shape, (2, 3, 5, 4))
        assert_array_equal(res.transpose(X.a, X.b, X.c, X.d), arr)

        # split several axes at once
        # ==========================
        arr = ndrange('a_b=a0_b0..a1_b2; c=c0..c3; d=d0..d3; e_f=e0_f0..e2_f1')

        # using a list of tuples
        res = arr.split_axes(['a_b', 'e_f'])
        assert res.axes.names == ['a', 'b', 'c', 'd', 'e', 'f']
        assert res.size == arr.size
        assert res.shape == (2, 3, 4, 4, 3, 2)
        assert list(res.axes.a.labels) == ['a0', 'a1']
        assert list(res.axes.b.labels) == ['b0', 'b1', 'b2']
        assert list(res.axes.e.labels) == ['e0', 'e1', 'e2']
        assert list(res.axes.f.labels) == ['f0', 'f1']
        assert res['a0', 'b1', 'c2', 'd3', 'e2', 'f1'] == arr['a0_b1', 'c2', 'd3', 'e2_f1']

        # default to all axes with name containing the delimiter _
        assert_array_equal(arr.split_axes(), res)

        # using a dict (-> user defined axes names)
        res = arr.split_axes({'a_b': ('A', 'B'), 'e_f': ('E', 'F')})
        assert res.axes.names == ['A', 'B', 'c', 'd', 'E', 'F']
        assert res.size == arr.size
        assert res.shape == (2, 3, 4, 4, 3, 2)

        # split an axis in more than 2 axes
        arr = ndrange('a_b_c=a0_b0_c0..a1_b2_c3; d=d0..d3; e_f=e0_f0..e2_f1')
        res = arr.split_axes(['a_b_c', 'e_f'])
        assert res.axes.names == ['a', 'b', 'c', 'd', 'e', 'f']
        assert res.size == arr.size
        assert res.shape == (2, 3, 4, 4, 3, 2)
        assert list(res.axes.a.labels) == ['a0', 'a1']
        assert list(res.axes.b.labels) == ['b0', 'b1', 'b2']
        assert list(res.axes.e.labels) == ['e0', 'e1', 'e2']
        assert list(res.axes.f.labels) == ['f0', 'f1']
        assert res['a0', 'b1', 'c2', 'd3', 'e2', 'f1'] == arr['a0_b1_c2', 'd3', 'e2_f1']

        # split an axis in more than 2 axes + passing a dict
        res = arr.split_axes({'a_b_c': ('A', 'B', 'C'), 'e_f': ('E', 'F')})
        assert res.axes.names == ['A', 'B', 'C', 'd', 'E', 'F']
        assert res.size == arr.size
        assert res.shape == (2, 3, 4, 4, 3, 2)

        # using regex
        arr = ndrange('ab=a0b0..a1b2; c=c0..c3; d=d0..d3; ef=e0f0..e2f1')
        res = arr.split_axes({'ab': ('a', 'b'), 'ef': ('e', 'f')}, regex='(\w{2})(\w{2})')
        assert res.axes.names == ['a', 'b', 'c', 'd', 'e', 'f']
        assert res.size == arr.size
        assert res.shape == (2, 3, 4, 4, 3, 2)
        assert list(res.axes.a.labels) == ['a0', 'a1']
        assert list(res.axes.b.labels) == ['b0', 'b1', 'b2']
        assert list(res.axes.e.labels) == ['e0', 'e1', 'e2']
        assert list(res.axes.f.labels) == ['f0', 'f1']
        assert res['a0', 'b1', 'c2', 'd3', 'e2', 'f1'] == arr['a0b1', 'c2', 'd3', 'e2f1']

    def test_stack(self):
        # simple
        arr0 = ndtest(3)
        arr1 = ndtest(3, start=-1)
        a = arr0.axes[0]
        b = Axis('b=b0,b1')
        expected = LArray([[0, -1],
                           [1,  0],
                           [2,  1]], [a, b])
        res = stack((arr0, arr1), b)
        assert_array_equal(res, expected)

        # simple with anonymous axis
        arr0 = ndrange(3)
        arr1 = ndrange(3, start=-1)
        a = arr0.axes[0]
        b = Axis('b=b0,b1')
        expected = LArray([[0, -1],
                           [1,  0],
                           [2,  1]], [a, b])
        res = stack((arr0, arr1), b)
        assert_array_equal(res, expected)

        # nd
        sex = Axis('sex=M,F')
        arr1 = ones('nat=BE, FO')
        # not using the same length as nat, otherwise numpy gets confused :(
        arr2 = zeros('type=1..3')
        nd = LArray([arr1, arr2], sex)
        res = stack(nd, sex)
        expected = from_string("""nat  type\\sex   M    F
                                   BE         1  1.0  0.0
                                   BE         2  1.0  0.0
                                   BE         3  1.0  0.0
                                   FO         1  1.0  0.0
                                   FO         2  1.0  0.0
                                   FO         3  1.0  0.0""")
        assert_array_equal(res, expected)

    def test_0darray_convert(self):
        int_arr = LArray(1)
        assert int(int_arr) == 1
        assert float(int_arr) == 1.0
        assert int_arr.__index__() == 1

        float_arr = LArray(1.0)
        assert int(float_arr) == 1
        assert float(float_arr) == 1.0
        with pytest.raises(TypeError) as e_info:
            float_arr.__index__()

        msg = e_info.value.args[0]
        expected_np11 = "only integer arrays with one element can be converted to an index"
        expected_np12 = "only integer scalar arrays can be converted to a scalar index"
        assert msg in {expected_np11, expected_np12}

    def test_deprecated_methods(self):
        with pytest.warns(FutureWarning) as caught_warnings:
            ndtest((2, 2)).with_axes('a', 'd=d0,d1')
        assert len(caught_warnings) == 1
        assert caught_warnings[0].message.args[0] == "with_axes() is deprecated. Use set_axes() instead."
        assert caught_warnings[0].filename == __file__

        with pytest.warns(FutureWarning) as caught_warnings:
            ndtest((2, 2)).combine_axes().split_axis()
        assert len(caught_warnings) == 1
        assert caught_warnings[0].message.args[0] == "split_axis() is deprecated. Use split_axes() instead."
        assert caught_warnings[0].filename == __file__


if __name__ == "__main__":
    # import doctest
    # import unittest
    # from larray import core
    # doctest.testmod(core)
    # unittest.main()
    pytest.main()
