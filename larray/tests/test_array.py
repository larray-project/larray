import os
import re
import sys

import pytest
import numpy as np
import pandas as pd

from io import StringIO
from collections import OrderedDict

from larray.tests.common import meta                        # noqa: F401
from larray.tests.common import (inputpath, tmp_path,
                                 assert_array_equal, assert_array_nan_equal, assert_larray_equiv, assert_larray_equal,
                                 needs_xlwings, needs_pytables, needs_xlsxwriter, needs_openpyxl, needs_python37,
                                 must_warn)
from larray import (Array, LArray, Axis, LGroup, union, zeros, zeros_like, ndtest, empty, ones, eye, diag, stack,
                    clip, exp, where, X, mean, isnan, round, read_hdf, read_csv, read_eurostat, read_excel,
                    from_lists, from_string, open_excel, from_frame, sequence, nan, IGroup)
from larray.inout.pandas import from_series
from larray.core.axis import _to_ticks, _to_key
from larray.util.misc import LHDFStore
from larray.core.metadata import Metadata


# ================== #
# Test Value Strings #
# ================== #


def test_value_string_split():
    assert_array_equal(_to_ticks('M,F'), np.asarray(['M', 'F']))
    assert_array_equal(_to_ticks('M, F'), np.asarray(['M', 'F']))


def test_value_string_union():
    assert union('A11,A22', 'A12,A22') == ['A11', 'A22', 'A12']


def test_value_string_range():
    assert_array_equal(_to_ticks('0..115'), np.asarray(range(116)))
    assert_array_equal(_to_ticks('..115'), np.asarray(range(116)))
    with pytest.raises(ValueError):
        _to_ticks('10..')
    with pytest.raises(ValueError):
        _to_ticks('..')


# ================ #
# Test Key Strings #
# ================ #

def test_key_string_nonstring():
    assert _to_key(('M', 'F')) == ['M', 'F']
    assert _to_key(['M', 'F']) == ['M', 'F']


def test_key_string_split():
    assert _to_key('M,F') == ['M', 'F']
    assert _to_key('M, F') == ['M', 'F']
    assert _to_key('M,') == ['M']
    assert _to_key('M') == 'M'


def test_key_string_slice_strings():
    # these two examples have different results and this is fine because numeric axes do not necessarily start at 0
    assert _to_key('0:115') == slice(0, 115)
    assert _to_key(':115') == slice(115)
    assert _to_key('10:') == slice(10, None)
    assert _to_key(':') == slice(None)


# =================== #
#    Test Metadata    #
# =================== #

def test_read_set_update_delete_metadata(meta, tmpdir):
    # __eq__
    meta2 = meta.copy()
    assert meta2 == meta

    # set/get metadata to/from an array
    arr = ndtest((3, 3))
    arr.meta = meta
    assert arr.meta == meta

    # access item
    assert arr.meta.date == meta.date

    # add new item
    arr.meta.city = 'London'
    assert arr.meta.city == 'London'

    # update item
    arr.meta.city = 'Berlin'
    assert arr.meta.city == 'Berlin'

    # __contains__
    assert 'city' in arr.meta

    # delete item
    del arr.meta.city
    assert arr.meta == meta

    # __reduce__ and __reduce_ex__
    import pickle
    fname = os.path.join(tmpdir.strpath, 'test_metadata.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(meta, f)
    with open(fname, 'rb') as f:
        meta2 = Metadata(pickle.load(f))
    assert meta2 == meta


@needs_pytables
def test_metadata_hdf(meta, tmpdir):
    key = 'meta'
    fname = os.path.join(tmpdir.strpath, 'test_metadata.hdf')
    with LHDFStore(fname) as store:
        ndtest(3).to_hdf(store, key)
        meta.to_hdf(store, key)
        meta2 = Metadata.from_hdf(store, key)
        assert meta2 == meta


def test_meta_arg_array_creation(array):
    meta_list = [('title', 'test array'), ('description', 'Array used for testing'),
                 ('author', 'John Cleese')]
    meta = Metadata(meta_list)

    # meta as list
    arr = Array(array.data, array.axes, meta=meta_list)
    assert arr.meta == meta
    # meta as OrderedDict
    arr = Array(array.data, array.axes, meta=OrderedDict(meta_list))
    assert arr.meta == meta


# ================ #
#    Test Array    #
# ================ #

# AXES
lipro = Axis([f'P{i:02d}' for i in range(1, 16)], 'lipro')
age = Axis('age=0..115')
sex = Axis('sex=M,F')
vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,A43,A44,A45,A46,A71,A72,A73'
wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,A83,A84,A85,A91,A92,A93'
bru = 'A21'
vla_str = vla
wal_str = wal
bru_str = bru
belgium = union(vla, wal, bru)
geo = Axis(belgium, 'geo')


# ARRAYS
@pytest.fixture()
def array():
    data = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15).astype(float)
    return Array(data, axes=(age, geo, sex, lipro))


@pytest.fixture()
def small_array():
    small_data = np.arange(30).reshape(2, 15)
    return Array(small_data, axes=(sex, lipro))


io_1d = ndtest(3)
io_2d = ndtest("a=1..3; b=b0,b1")
io_3d = ndtest("a=1..3; b=b0,b1; c=c0..c2")
io_int_labels = ndtest("a=0..2; b=0..2; c=0..2")
io_unsorted = ndtest("a=3..1; b=b1,b0; c=c2..c0")
io_missing_values = ndtest("a=1..3; b=b0,b1; c=c0..c2", dtype=float)
io_missing_values[2, 'b0'] = nan
io_missing_values[3, 'b1'] = nan
io_narrow_missing_values = io_missing_values.copy()
io_narrow_missing_values[2, 'b1', 'c1'] = nan


def test_larray_renamed_as_array():
    with must_warn(FutureWarning, msg="LArray has been renamed as Array."):
        arr = LArray([0, 1, 2, 3], 'a=a0..a3')


def test_ndtest():
    arr = ndtest('a=a0..a2')
    assert arr.shape == (3,)
    assert arr.axes.names == ['a']
    assert_array_equal(arr.data, np.arange(3))

    # using an explicit Axis object
    a = Axis('a=a0..a2')
    arr = ndtest(a)
    assert arr.shape == (3,)
    assert arr.axes.names == ['a']
    assert_array_equal(arr.data, np.arange(3))

    # using a group as an axis
    arr = ndtest(a[:'a1'])
    assert arr.shape == (2,)
    assert arr.axes.names == ['a']
    assert_array_equal(arr.data, np.arange(2))


def test_getattr(array):
    assert type(array.geo) == Axis
    assert array.geo is geo
    with pytest.raises(AttributeError):
        array.geom


def test_zeros():
    la = zeros((geo, age))
    assert la.shape == (44, 116)
    assert_array_equal(la, np.zeros((44, 116)))


def test_zeros_like(array):
    la = zeros_like(array)
    assert la.shape == (116, 44, 2, 15)
    assert_array_equal(la, np.zeros((116, 44, 2, 15)))


def test_bool():
    a = ones([2])
    # ValueError: The truth value of an array with more than one element
    #             is ambiguous. Use a.any() or a.all()
    with pytest.raises(ValueError):
        bool(a)

    a = ones([1])
    assert bool(a)

    a = zeros([1])
    assert not bool(a)

    a = Array(np.array(2), [])
    assert bool(a)

    a = Array(np.array(0), [])
    assert not bool(a)


def test_iter(small_array):
    list_ = list(small_array)
    assert_array_equal(list_[0], small_array['M'])
    assert_array_equal(list_[1], small_array['F'])


def test_keys():
    arr = ndtest((2, 2))
    a, b = arr.axes

    keys = arr.keys()
    assert list(keys) == [(a.i[0], b.i[0]), (a.i[0], b.i[1]), (a.i[1], b.i[0]), (a.i[1], b.i[1])]
    assert keys[0] == (a.i[0], b.i[0])
    assert keys[-1] == (a.i[1], b.i[1])

    keys = arr.keys(ascending=False)
    assert list(keys) == [(a.i[1], b.i[1]), (a.i[1], b.i[0]), (a.i[0], b.i[1]), (a.i[0], b.i[0])]
    assert keys[0] == (a.i[1], b.i[1])
    assert keys[-1] == (a.i[0], b.i[0])

    keys = arr.keys(('b', 'a'))
    assert list(keys) == [(b.i[0], a.i[0]), (b.i[0], a.i[1]), (b.i[1], a.i[0]), (b.i[1], a.i[1])]
    assert keys[1] == (b.i[0], a.i[1])
    assert keys[2] == (b.i[1], a.i[0])

    keys = arr.keys(('b', 'a'), ascending=False)
    assert list(keys) == [(b.i[1], a.i[1]), (b.i[1], a.i[0]), (b.i[0], a.i[1]), (b.i[0], a.i[0])]
    assert keys[1] == (b.i[1], a.i[0])
    assert keys[2] == (b.i[0], a.i[1])

    keys = arr.keys('b')
    assert list(keys) == [(b.i[0],), (b.i[1],)]
    assert keys[0] == (b.i[0],)
    assert keys[-1] == (b.i[1],)

    keys = arr.keys('b', ascending=False)
    assert list(keys) == [(b.i[1],), (b.i[0],)]
    assert keys[0] == (b.i[1],)
    assert keys[-1] == (b.i[0],)


def test_values():
    arr = ndtest((2, 2))
    a, b = arr.axes

    values = arr.values()
    assert list(values) == [0, 1, 2, 3]
    assert values[0] == 0
    assert values[-1] == 3

    values = arr.values(ascending=False)
    assert list(values) == [3, 2, 1, 0]
    assert values[0] == 3
    assert values[-1] == 0

    values = arr.values(('b', 'a'))
    assert list(values) == [0, 2, 1, 3]
    assert values[1] == 2
    assert values[2] == 1

    values = arr.values(('b', 'a'), ascending=False)
    assert list(values) == [3, 1, 2, 0]
    assert values[1] == 1
    assert values[2] == 2

    values = arr.values('b')
    res = list(values)
    assert_larray_equal(res[0], arr['b0'])
    assert_larray_equal(res[1], arr['b1'])
    assert_larray_equal(values[0], arr['b0'])
    assert_larray_equal(values[-1], arr['b1'])

    values = arr.values('b', ascending=False)
    res = list(values)
    assert_larray_equal(res[0], arr['b1'])
    assert_larray_equal(res[1], arr['b0'])
    assert_larray_equal(values[0], arr['b1'])
    assert_larray_equal(values[-1], arr['b0'])


def test_items():
    arr = ndtest((2, 2))
    a, b = arr.axes

    items = arr.items()
    assert list(items) == [((a.i[0], b.i[0]), 0), ((a.i[0], b.i[1]), 1), ((a.i[1], b.i[0]), 2), ((a.i[1], b.i[1]), 3)]
    assert items[0] == ((a.i[0], b.i[0]), 0)
    assert items[-1] == ((a.i[1], b.i[1]), 3)

    items = arr.items(ascending=False)
    assert list(items) == [((a.i[1], b.i[1]), 3), ((a.i[1], b.i[0]), 2), ((a.i[0], b.i[1]), 1), ((a.i[0], b.i[0]), 0)]
    assert items[0] == ((a.i[1], b.i[1]), 3)
    assert items[-1] == ((a.i[0], b.i[0]), 0)

    items = arr.items(('b', 'a'))
    assert list(items) == [((b.i[0], a.i[0]), 0), ((b.i[0], a.i[1]), 2), ((b.i[1], a.i[0]), 1), ((b.i[1], a.i[1]), 3)]
    assert items[1] == ((b.i[0], a.i[1]), 2)
    assert items[2] == ((b.i[1], a.i[0]), 1)

    items = arr.items(('b', 'a'), ascending=False)
    assert list(items) == [((b.i[1], a.i[1]), 3), ((b.i[1], a.i[0]), 1), ((b.i[0], a.i[1]), 2), ((b.i[0], a.i[0]), 0)]
    assert items[1] == ((b.i[1], a.i[0]), 1)
    assert items[2] == ((b.i[0], a.i[1]), 2)

    items = arr.items('b')
    items_list = list(items)

    key, value = items[0]
    assert key == (b.i[0],)
    assert_larray_equal(value, arr['b0'])

    key, value = items_list[0]
    assert key == (b.i[0],)
    assert_larray_equal(value, arr['b0'])

    key, value = items[-1]
    assert key == (b.i[1],)
    assert_larray_equal(value, arr['b1'])

    key, value = items_list[-1]
    assert key == (b.i[1],)
    assert_larray_equal(value, arr['b1'])

    items = arr.items('b', ascending=False)
    items_list = list(items)

    key, value = items[0]
    assert key == (b.i[1],)
    assert_larray_equal(value, arr['b1'])

    key, value = items_list[0]
    assert key == (b.i[1],)
    assert_larray_equal(value, arr['b1'])

    key, value = items[-1]
    assert key == (b.i[0],)
    assert_larray_equal(value, arr['b0'])

    key, value = items_list[-1]
    assert key == (b.i[0],)
    assert_larray_equal(value, arr['b0'])


def test_rename(array):
    new_array = array.rename('sex', 'gender')
    # old array axes names not modified
    assert array.axes.names == ['age', 'geo', 'sex', 'lipro']
    assert new_array.axes.names == ['age', 'geo', 'gender', 'lipro']

    new_array = array.rename(sex, 'gender')
    # old array axes names not modified
    assert array.axes.names == ['age', 'geo', 'sex', 'lipro']
    assert new_array.axes.names == ['age', 'geo', 'gender', 'lipro']


def test_info(array, meta):
    array.meta = meta
    expected = """\
title: test array
description: Array used for testing
author: John Cleese
location: Ministry of Silly Walks
office_number: 42
score: 9.7
date: 1970-03-21 00:00:00
116 x 44 x 2 x 15
 age [116]: 0 1 2 ... 113 114 115
 geo [44]: 'A11' 'A12' 'A13' ... 'A92' 'A93' 'A21'
 sex [2]: 'M' 'F'
 lipro [15]: 'P01' 'P02' 'P03' ... 'P13' 'P14' 'P15'
dtype: float64
memory used: 1.17 Mb"""
    assert array.info == expected


def test_str(small_array, array):
    lipro3 = lipro['P01:P03']

    # zero dimension / scalar
    assert str(small_array[lipro['P01'], sex['F']]) == "15"

    # empty / len 0 first dimension
    assert str(small_array[sex[[]]]) == "Array([])"

    # one dimension
    assert str(small_array[lipro3, sex['M']]) == """\
lipro  P01  P02  P03
         0    1    2"""

    # two dimensions
    assert str(small_array.filter(lipro=lipro3)) == """\
sex\\lipro  P01  P02  P03
        M    0    1    2
        F   15   16   17"""

    # four dimensions (too many rows)
    assert str(array.filter(lipro=lipro3)) == """\
age  geo  sex\\lipro       P01       P02       P03
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
115  A21          F  153105.0  153106.0  153107.0"""

    # too many columns
    assert str(array['P01', 'A11', 'M']) == """\
age    0       1       2  ...       112       113       114       115
     0.0  1320.0  2640.0  ...  147840.0  149160.0  150480.0  151800.0"""

    arr = Array([0, ''], Axis(['a0', ''], 'a'))
    assert str(arr) == "a  a0  \n    0  "


def test_getitem(array):
    raw = array.data
    age, geo, sex, lipro = array.axes
    age159 = age[[1, 5, 9]]
    lipro159 = lipro['P01,P05,P09']

    # LGroup at "correct" place
    subset = array[age159]
    assert subset.axes[1:] == (geo, sex, lipro)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'age'))
    assert_array_equal(subset, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_array_equal(array[lipro159], raw[..., [0, 4, 8]])

    # multiple LGroup key (in "incorrect" order)
    res = array[lipro159, age159]
    assert res.axes.names == ['age', 'geo', 'sex', 'lipro']
    assert_array_equal(res, raw[[1, 5, 9]][..., [0, 4, 8]])

    # LGroup key and scalar
    res = array[lipro159, 5]
    assert res.axes.names == ['geo', 'sex', 'lipro']
    assert_array_equal(res, raw[..., [0, 4, 8]][5])

    # mixed LGroup/positional key
    assert_array_equal(array[[1, 5, 9], lipro159],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # single None slice
    assert_array_equal(array[:], raw)

    # only Ellipsis
    assert_array_equal(array[...], raw)

    # Ellipsis and LGroup
    assert_array_equal(array[..., lipro159], raw[..., [0, 4, 8]])

    # string 'int..int'
    assert_array_equal(array['10..13'], array['10,11,12,13'])
    assert_array_equal(array['8, 10..13, 15'], array['8,10,11,12,13,15'])

    # ambiguous label
    arr = ndtest("a=l0,l1;b=l1,l2")
    res = arr[arr.b['l1']]
    assert_array_equal(res, arr.data[:, 0])

    # scalar group on another axis
    arr = ndtest((3, 2))
    alt_a = Axis("alt_a=a1..a2")
    lgroup = alt_a['a1']
    assert_array_equal(arr[lgroup], arr['a1'])
    pgroup = alt_a.i[0]
    assert_array_equal(arr[pgroup], arr['a1'])

    # key with duplicate axes
    with pytest.raises(ValueError):
        array[age[1, 2], age[3, 4]]

    # key with lgroup from another axis leading to duplicate axis
    bad = Axis(3, 'bad')
    with pytest.raises(ValueError):
        array[bad[1, 2], age[3, 4]]


def test_getitem_abstract_axes(array):
    raw = array.data
    age, geo, sex, lipro = array.axes
    age159 = X.age[1, 5, 9]
    lipro159 = X.lipro['P01,P05,P09']

    # LGroup at "correct" place
    subset = array[age159]
    assert subset.axes[1:] == (geo, sex, lipro)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'age'))
    assert_array_equal(subset, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_array_equal(array[lipro159], raw[..., [0, 4, 8]])

    # multiple LGroup key (in "incorrect" order)
    assert_array_equal(array[lipro159, age159], raw[[1, 5, 9]][..., [0, 4, 8]])

    # mixed LGroup/positional key
    assert_array_equal(array[[1, 5, 9], lipro159], raw[[1, 5, 9]][..., [0, 4, 8]])

    # single None slice
    assert_array_equal(array[:], raw)

    # only Ellipsis
    assert_array_equal(array[...], raw)

    # Ellipsis and LGroup
    assert_array_equal(array[..., lipro159], raw[..., [0, 4, 8]])

    # key with duplicate axes
    with pytest.raises(ValueError):
        array[X.age[1, 2], X.age[3]]

    # key with invalid axis
    with pytest.raises(ValueError):
        array[X.bad[1, 2], X.age[3, 4]]


def test_getitem_anonymous_axes():
    arr = ndtest([Axis(3), Axis(4)])
    raw = arr.data
    assert_array_equal(arr[X[0][1:]], raw[1:])
    assert_array_equal(arr[X[1][2:]], raw[:, 2:])
    assert_array_equal(arr[X[0][2:], X[1][1:]], raw[2:, 1:])
    assert_array_equal(arr.i[2:, 1:], raw[2:, 1:])


def test_getitem_guess_axis(array):
    raw = array.data
    age, geo, sex, lipro = array.axes

    # key at "correct" place
    assert_array_equal(array[[1, 5, 9]], raw[[1, 5, 9]])
    subset = array[[1, 5, 9]]
    assert subset.axes[1:] == (geo, sex, lipro)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'age'))
    assert_array_equal(subset, raw[[1, 5, 9]])

    # key at "incorrect" place
    assert_array_equal(array['P01,P05,P09'], raw[..., [0, 4, 8]])
    assert_array_equal(array[['P01', 'P05', 'P09']], raw[..., [0, 4, 8]])

    # multiple keys (in "incorrect" order)
    assert_array_equal(array['P01,P05,P09', [1, 5, 9]],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # mixed LGroup/key
    assert_array_equal(array[lipro['P01,P05,P09'], [1, 5, 9]],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # single None slice
    assert_array_equal(array[:], raw)

    # only Ellipsis
    assert_array_equal(array[...], raw)

    # Ellipsis and LGroup
    assert_array_equal(array[..., 'P01,P05,P09'], raw[..., [0, 4, 8]])
    assert_array_equal(array[..., ['P01', 'P05', 'P09']], raw[..., [0, 4, 8]])

    # LGroup without axis (which also needs to be guessed)
    g = LGroup(['P01', 'P05', 'P09'])
    assert_array_equal(array[g], raw[..., [0, 4, 8]])

    # key with duplicate axes
    with pytest.raises(ValueError, match="key has several values for axis: age"):
        array[[1, 2], [3, 4]]

    # key with invalid label (ie label not found on any axis)
    with pytest.raises(ValueError, match="999 is not a valid label for any axis"):
        array[[1, 2], 999]

    # key with invalid label list (ie list of labels not found on any axis)
    with pytest.raises(ValueError, match=r"\[998, 999\] is not a valid label for any axis"):
        array[[1, 2], [998, 999]]

    # key with partial invalid list (ie list containing a label not found
    # on any axis)
    # FIXME: the message should be the same as for 999, 4 (ie it should NOT mention age).
    with pytest.raises(ValueError, match=r"age\[3, 999\] is not a valid label for any axis"):
        array[[1, 2], [3, 999]]

    with pytest.raises(ValueError, match=r"\[999, 4\] is not a valid label for any axis"):
        array[[1, 2], [999, 4]]

    # ambiguous key
    arr = ndtest("a=l0,l1;b=l1,l2")
    with pytest.raises(ValueError, match=r"l1 is ambiguous \(valid in a, b\)"):
        arr['l1']

    # ambiguous key disambiguated via string
    res = arr['b[l1]']
    assert_array_equal(res, arr.data[:, 0])


def test_getitem_positional_group(array):
    raw = array.data
    age, geo, sex, lipro = array.axes
    age159 = age.i[1, 5, 9]
    lipro159 = lipro.i[0, 4, 8]

    # LGroup at "correct" place
    subset = array[age159]
    assert subset.axes[1:] == (geo, sex, lipro)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'age'))
    assert_array_equal(subset, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_array_equal(array[lipro159], raw[..., [0, 4, 8]])

    # multiple LGroup key (in "incorrect" order)
    assert_array_equal(array[lipro159, age159],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # mixed LGroup/positional key
    assert_array_equal(array[[1, 5, 9], lipro159],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # single None slice
    assert_array_equal(array[:], raw)

    # only Ellipsis
    assert_array_equal(array[...], raw)

    # Ellipsis and LGroup
    assert_array_equal(array[..., lipro159], raw[..., [0, 4, 8]])

    # key with duplicate axes
    with pytest.raises(ValueError, match="key has several values for axis: age"):
        array[age.i[1, 2], age.i[3, 4]]


def test_getitem_str_positional_group():
    arr = ndtest('a=l0..l2;b=l0..l2')
    a, b = arr.axes
    res = arr['b.i[1]']
    expected = Array([1, 4, 7], 'a=l0..l2')
    assert_array_equal(res, expected)


def test_getitem_abstract_positional(array):
    raw = array.data
    age, geo, sex, lipro = array.axes
    age159 = X.age.i[1, 5, 9]
    lipro159 = X.lipro.i[0, 4, 8]

    # LGroup at "correct" place
    subset = array[age159]
    assert subset.axes[1:] == (geo, sex, lipro)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'age'))
    assert_array_equal(subset, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_array_equal(array[lipro159], raw[..., [0, 4, 8]])

    # multiple LGroup key (in "incorrect" order)
    assert_array_equal(array[lipro159, age159],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # mixed LGroup/positional key
    assert_array_equal(array[[1, 5, 9], lipro159],
                       raw[[1, 5, 9]][..., [0, 4, 8]])

    # single None slice
    assert_array_equal(array[:], raw)

    # only Ellipsis
    assert_array_equal(array[...], raw)

    # Ellipsis and LGroup
    assert_array_equal(array[..., lipro159], raw[..., [0, 4, 8]])

    # key with duplicate axes
    with pytest.raises(ValueError, match="key has several values for axis: age"):
        array[X.age.i[2, 3], X.age.i[1, 5]]


def test_getitem_bool_larray_key_arr_whout_bool_axis():
    arr = ndtest((3, 2, 4))
    raw = arr.data

    # all dimensions
    res = arr[arr < 5]
    assert isinstance(res, Array)
    assert res.ndim == 1
    assert_array_equal(res, raw[raw < 5])

    # missing dimension
    filter_ = arr['b1'] % 5 == 0
    res = arr[filter_]
    assert isinstance(res, Array)
    assert res.ndim == 2
    assert res.shape == (3, 2)
    raw_key = raw[:, 1, :] % 5 == 0
    raw_d1, raw_d3 = raw_key.nonzero()
    assert_array_equal(res, raw[raw_d1, :, raw_d3])

    # using an Axis object
    arr = ndtest('a=a0,a1;b=0..3')
    raw = arr.data
    res = arr[arr.b < 2]
    assert_array_equal(res, raw[:, :2])

    # using an AxisReference (ExprNode)
    res = arr[X.b < 2]
    assert_array_equal(res, raw[:, :2])


def test_getitem_bool_larray_key_arr_wh_bool_axis():
    gender = Axis([False, True], 'gender')
    arr = Array([0.1, 0.2], gender)
    id_axis = Axis('id=0..3')
    key = Array([True, False, True, True], id_axis)
    expected = Array([0.2, 0.1, 0.2, 0.2], id_axis)

    # LGroup using the real axis
    assert_larray_equal(arr[gender[key]], expected)

    # LGroup using an AxisReference
    assert_larray_equal(arr[X.gender[key]], expected)

    # this test checks that the current behavior does not change unintentionally...
    # ... but I am unsure the current behavior is what we actually want
    msg = re.escape("boolean subset key contains more axes ({id}) than array ({gender})")
    with pytest.raises(ValueError, match=msg):
        arr[key]


def test_getitem_bool_larray_and_group_key():
    arr = ndtest((3, 6, 4)).set_labels('b', '0..5')

    # using axis
    res = arr['a0,a2', arr.b < 3, 'c0:c3']
    assert isinstance(res, Array)
    assert res.ndim == 3
    expected = arr['a0,a2', '0:2', 'c0:c3']
    assert_array_equal(res, expected)

    # using axis reference
    res = arr['a0,a2', X.b < 3, 'c0:c3']
    assert isinstance(res, Array)
    assert res.ndim == 3
    assert_array_equal(res, expected)


def test_getitem_bool_ndarray_key_arr_whout_bool_axis(array):
    raw = array.data
    res = array[raw < 5]
    assert isinstance(res, Array)
    assert res.ndim == 1
    assert_array_equal(res, raw[raw < 5])


def test_getitem_bool_ndarray_key_arr_wh_bool_axis():
    gender = Axis([False, True], 'gender')
    arr = Array([0.1, 0.2], gender)
    key = np.array([True, False, True, True])
    expected = arr.i[[1, 0, 1, 1]]

    # LGroup using the real axis
    assert_larray_equal(arr[gender[key]], expected)

    # LGroup using an AxisReference
    assert_larray_equal(arr[X.gender[key]], expected)

    # raw key => ???
    # this test checks that the current behavior does not change unintentionally...
    # ... but I am unsure the current behavior is what we actually want
    # L? is to account for Python2 where shape can be 'long' integers
    msg = r"boolean key with a different shape \(\(4L?,\)\) than array \(\(2,\)\)"
    with pytest.raises(ValueError, match=msg):
        arr[key]


def test_getitem_bool_anonymous_axes():
    a = ndtest([Axis(2), Axis(3), Axis(4), Axis(5)])
    mask = ones(a.axes[1, 3], dtype=bool)
    res = a[mask]
    assert res.ndim == 3
    assert res.shape == (15, 2, 4)

    # XXX: we might want to transpose the result to always move combined axes to the front
    a = ndtest([Axis(2), Axis(3), Axis(4), Axis(5)])
    mask = ones(a.axes[1, 2], dtype=bool)
    res = a[mask]
    assert res.ndim == 3
    assert res.shape == (2, 12, 5)


def test_getitem_igroup_on_int_axis():
    a = Axis('a=1..3')
    arr = ndtest(a)
    assert arr[a.i[1]] == 1


def test_getitem_integer_string_axes():
    arr = ndtest((5, 5))
    a, b = arr.axes

    assert_array_equal(arr['0[a0, a2]'], arr[a['a0', 'a2']])
    assert_array_equal(arr['0[a0:a2]'], arr[a['a0:a2']])
    with pytest.raises(ValueError):
        arr['1[a0, a2]']

    assert_array_equal(arr['0.i[0, 2]'], arr[a.i[0, 2]])
    assert_array_equal(arr['0.i[0:2]'], arr[a.i[0:2]])
    with pytest.raises(ValueError):
        arr['3.i[0, 2]']


def test_getitem_int_larray_lgroup_key():
    # e axis go from 0 to 3
    arr = ndtest("c=0,1; d=0,1; e=0..3")
    # key values go from 0 to 3
    key = ndtest("a=0,1; b=0,1")
    # this replaces 'e' axis by 'a' and 'b' axes
    res = arr[X.e[key]]
    assert res.shape == (2, 2, 2, 2)
    assert res.axes.names == ['c', 'd', 'a', 'b']


def test_getitem_structured_key_with_groups():
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


def test_getitem_single_larray_key_guess():
    # TODO: we really need another way to get test axes, e.g. testaxes(2, 3, 4) or testaxes((2, 3, 4))
    a, b, c = ndtest((2, 3, 4)).axes
    arr = ndtest((a, b))
    # >>> arr
    # a\b b0 b1 b2
    #  a0  0  1  2
    #  a1  3  4  5

    # 1) key with extra axis
    key = Array(['a0', 'a1', 'a1', 'a0'], c)
    # replace the target axis by the extra axis
    expected = from_string(r"""
c\b  b0  b1  b2
 c0   0   1   2
 c1   3   4   5
 c2   3   4   5
 c3   0   1   2""")
    assert_array_equal(arr[key], expected)

    # 2) key with the target axis (the one being replaced)
    key = Array(['b1', 'b0', 'b2'], b)
    # axis stays the same but data should be flipped/shuffled
    expected = from_string(r"""
a\b  b0  b1  b2
 a0   1   0   2
 a1   4   3   5""")
    assert_array_equal(arr[key], expected)

    # 2bis) key with part of the target axis (the one being replaced)
    key = Array(['b2', 'b1'], 'b=b0,b1')
    expected = from_string(r"""
a\b  b0  b1
 a0   2   1
 a1   5   4""")
    assert_array_equal(arr[key], expected)

    # 3) key with another existing axis (not the target axis)
    key = Array(['a0', 'a1', 'a0'], b)
    expected = from_string("""
b  b0  b1  b2
\t  0   4   2""")
    assert_array_equal(arr[key], expected)

    # TODO: this does not work yet but should be much easier to implement with "align" in make_np_broadcastable
    # 3bis) key with *part* of another existing axis (not the target axis)
    # key = Array(['a1', 'a0'], 'b=b0,b1')
    # expected = from_string("""
    # b  b0  b1
    # \t  3   1""")
    # assert_array_equal(arr[key], expected)

    # 4) key has both the target axis and another existing axis
    # TODO: maybe we should make this work without requiring astype!
    key = from_string(r"""
a\b b0 b1 b2
 a0 a0 a1 a0
 a1 a1 a0 a1""").astype(str)
    expected = from_string(r"""
a\b  b0  b1  b2
 a0   0   4   2
 a1   3   1   5""")
    assert_array_equal(arr[key], expected)

    # 5) key has both the target axis and an extra axis
    key = from_string(r"""
a\c  c0  c1  c2  c3
 a0  a0  a1  a1  a0
 a1  a1  a0  a0  a1""").astype(str)
    expected = from_string(r"""
 a  c\b  b0  b1  b2
a0   c0   0   1   2
a0   c1   3   4   5
a0   c2   3   4   5
a0   c3   0   1   2
a1   c0   3   4   5
a1   c1   0   1   2
a1   c2   0   1   2
a1   c3   3   4   5""")
    assert_array_equal(arr[key], expected)

    # 6) key has both another existing axis (not target) and an extra axis
    key = from_string(r"""
a\c  c0  c1  c2  c3
 a0  b0  b1  b0  b1
 a1  b2  b1  b2  b1""").astype(str)
    expected = from_string(r"""
a\c  c0  c1  c2  c3
 a0   0   1   0   1
 a1   5   4   5   4""")
    assert_array_equal(arr[key], expected)

    # 7) key has the target axis, another existing axis and an extra axis
    key = from_string(r"""
 a  b\c  c0  c1  c2  c3
a0   b0  a0  a1  a0  a1
a0   b1  a1  a0  a1  a0
a0   b2  a0  a1  a0  a1
a1   b0  a0  a1  a1  a0
a1   b1  a1  a1  a1  a1
a1   b2  a0  a1  a1  a0""").astype(str)
    expected = from_string(r"""
 a  b\c  c0  c1  c2  c3
a0   b0   0   3   0   3
a0   b1   4   1   4   1
a0   b2   2   5   2   5
a1   b0   0   3   3   0
a1   b1   4   4   4   4
a1   b2   2   5   5   2""")
    assert_array_equal(arr[key], expected)


def test_getitem_multiple_larray_key_guess():
    a, b, c, d, e = ndtest((2, 3, 2, 3, 2)).axes
    arr = ndtest((a, b))
    # >>> arr
    # a\b  b0  b1  b2
    #  a0   0   1   2
    #  a1   3   4   5

    # 1) keys with each a different existing axis
    k1 = from_string(""" a  a1  a0
                        \t  b2  b0""")
    k2 = from_string(""" b  b1  b2  b3
                        \t  a0  a1  a0""")
    expected = from_string(r"""b\a  a1  a0
                                b1   2   0
                                b2   5   3
                                b3   2   0""")
    assert_array_equal(arr[k1, k2], expected)

    # 2) keys with a common existing axis
    k1 = from_string(""" b  b0  b1  b2
                        \t  a1  a0  a1""")
    k2 = from_string(""" b  b0  b1  b2
                        \t  b1  b2  b0""")
    expected = from_string(""" b  b0  b1  b2
                              \t   4   2   3""")
    assert_array_equal(arr[k1, k2], expected)

    # 3) keys with each a different extra axis
    k1 = from_string(""" c  c0  c1
                        \t  a1  a0""")
    k2 = from_string(""" d  d0  d1  d2
                        \t  b1  b2  b0""")
    expected = from_string(r"""c\d  d0  d1  d2
                                c0   4   5   3
                                c1   1   2   0""")
    assert_array_equal(arr[k1, k2], expected)

    # 4) keys with a common extra axis
    k1 = from_string(r"""c\d  d0  d1  d2
                          c0  a1  a0  a1
                          c1  a0  a1  a0""").astype(str)
    k2 = from_string(r"""c\e  e0  e1
                          c0  b1  b2
                          c1  b0  b1""").astype(str)
    expected = from_string(r""" c  d\e  e0  e1
                               c0   d0   4   5
                               c0   d1   1   2
                               c0   d2   4   5
                               c1   d0   0   1
                               c1   d1   3   4
                               c1   d2   0   1""")
    assert_array_equal(arr[k1, k2], expected)


def test_getitem_ndarray_key_guess(array):
    raw = array.data
    keys = ['P04', 'P01', 'P03', 'P02']
    key = np.array(keys)
    res = array[key]
    assert isinstance(res, Array)
    assert res.axes == array.axes.replace(X.lipro, Axis(keys, 'lipro'))
    assert_array_equal(res, raw[:, :, :, [3, 0, 2, 1]])


def test_getitem_int_larray_key_guess():
    a = Axis([0, 1], 'a')
    b = Axis([2, 3], 'b')
    c = Axis([4, 5], 'c')
    d = Axis([6, 7], 'd')
    e = Axis([8, 9, 10, 11], 'e')

    arr = ndtest([c, d, e])
    key = Array([[8, 9], [10, 11]], [a, b])
    assert arr[key].axes == [c, d, a, b]


def test_getitem_int_ndarray_key_guess():
    c = Axis([4, 5], 'c')
    d = Axis([6, 7], 'd')
    e = Axis([8, 9, 10, 11], 'e')

    arr = ndtest([c, d, e])
    # ND keys do not work yet
    # key = nparray([[8, 11], [10, 9]])
    key = np.array([8, 11, 10])
    res = arr[key]
    assert res.axes == [c, d, Axis([8, 11, 10], 'e')]


def test_getitem_axis_object():
    arr = ndtest((2, 3))
    a, b = arr.axes

    assert_array_equal(arr[a], arr)
    assert_array_equal(arr[b], arr)

    b2 = Axis('b=b0,b2')

    assert_array_equal(arr[b2], from_string("""a\\b  b0  b2
                                                 a0   0   2
                                                 a1   3   5"""))


def test_getitem_empty_tuple():
    # an empty tuple should return a view on the original array
    arr = ndtest((2, 3))
    res = arr[()]
    assert_array_equal(res, arr)
    assert res is not arr

    z = Array(0)
    res = z[()]
    assert res == z
    assert res is not z


def test_positional_indexer_getitem(array):
    raw = array.data
    for key in [0, (0, 5, 1, 2), (slice(None), 5, 1), (0, 5), [1, 0], ([1, 0], 5)]:
        assert_array_equal(array.i[key], raw[key])
    assert_array_equal(array.i[[1, 0], [5, 4]], raw[np.ix_([1, 0], [5, 4])])
    with pytest.raises(IndexError):
        array.i[0, 0, 0, 0, 0]


def test_positional_indexer_setitem(array):
    for key in [0, (0, 2, 1, 2), (slice(None), 2, 1), (0, 2), [1, 0], ([1, 0], 2)]:
        arr = array.copy()
        raw = array.data.copy()
        arr.i[key] = 42
        raw[key] = 42
        assert_array_equal(arr, raw)

    raw = array.data
    array.i[[1, 0], [5, 4]] = 42
    raw[np.ix_([1, 0], [5, 4])] = 42
    assert_array_equal(array, raw)


def test_points_indexer_getitem():
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
    with pytest.raises(ValueError):
        arr.points['a0', 'b1', 'c2', 'd0']


def test_points_indexer_setitem():
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
    with pytest.raises(ValueError):
        arr.points['a0', 'b1'] = 42

    # test when broadcasting is involved
    arr = ndtest((2, 3, 4))
    raw = arr.data.copy()
    raw_value = raw[:, 0, 0].reshape(2, 1)
    raw[:, [0, 1, 2], [0, 1, 2]] = raw_value
    arr.points['b0,b1,b2', 'c0,c1,c2'] = arr['b0', 'c0']
    assert_array_equal(arr, raw)


def test_setitem_larray(array, small_array):
    """
    tests Array.__setitem__(key, value) where value is an Array
    """
    age, geo, sex, lipro = array.axes

    # 1) using a LGroup key
    ages1_5_9 = age[[1, 5, 9]]

    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()

    arr[ages1_5_9] = arr[ages1_5_9] + 25.0
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
    assert_array_equal(arr, raw)

    # b) value has exactly the same shape but LGroup at a "wrong" positions
    arr = array.copy()
    arr[geo[:], ages1_5_9] = arr[ages1_5_9] + 25.0
    # same raw as previous test
    assert_array_equal(arr, raw)

    # c) value has an extra length-1 axis
    arr = array.copy()
    raw = array.data.copy()

    raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
    fake_axis = Axis(['label'], 'fake')
    age_axis = arr[ages1_5_9].axes.age
    value = Array(raw_value, axes=(age_axis, fake_axis, geo, sex, lipro))
    arr[ages1_5_9] = value
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 26.0
    assert_array_equal(arr, raw)

    # d) value has the same axes than target but one has length 1
    # arr = array.copy()
    # raw = array.data.copy()
    # raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    # arr[ages1_5_9] = arr[ages1_5_9].sum(geo=(geo.all(),))
    # assert_array_equal(arr, raw)

    # e) value has a missing dimension
    arr = array.copy()
    raw = array.data.copy()
    arr[ages1_5_9] = arr[ages1_5_9].sum(geo)
    raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    assert_array_equal(arr, raw)

    # 2) using a LGroup and scalar key (triggers advanced indexing/cross)

    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()

    # using 1, 5, 8 and not 9 so that the list is not collapsed to slice
    value = arr[age[1, 5, 8], sex['M']] + 25.0
    arr[age[1, 5, 8], sex['M']] = value
    raw[[1, 5, 8], :, 0] = raw[[1, 5, 8], :, 0] + 25.0
    assert_array_equal(arr, raw)

    # 3) using a string key
    arr = array.copy()
    raw = array.data.copy()
    arr['1, 5, 9'] = arr['1, 5, 9'] + 27.0
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 27.0
    assert_array_equal(arr, raw)

    # 4) using ellipsis keys
    # only Ellipsis
    arr = array.copy()
    arr[...] = 0
    assert_array_equal(arr, np.zeros_like(raw))

    # Ellipsis and LGroup
    arr = array.copy()
    raw = array.data.copy()
    arr[..., lipro['P01,P05,P09']] = 0
    raw[..., [0, 4, 8]] = 0
    assert_array_equal(arr, raw)

    # 5) using a single slice(None) key
    arr = array.copy()
    arr[:] = 0
    assert_array_equal(arr, np.zeros_like(raw))

    # 6) incompatible axes
    arr = small_array.copy()
    subset_axes = arr['P01'].axes
    value = small_array.copy()
    expected_msg = f"Value {value.axes - subset_axes!s} axis is not present in target subset {subset_axes!s}. " \
                   f"A value can only have the same axes or fewer axes than the subset being targeted"
    with pytest.raises(ValueError, match=expected_msg):
        arr['P01'] = value

    value = arr.rename('sex', 'gender')['P01']
    expected_msg = f"Value {value.axes - subset_axes!s} axis is not present in target subset {subset_axes!s}. " \
                   f"A value can only have the same axes or fewer axes than the subset being targeted"
    with pytest.raises(ValueError, match=expected_msg):
        arr['P01'] = value

    # 7) incompatible labels
    sex2 = Axis('sex=F,M')
    la2 = Array(small_array.data, axes=(sex2, lipro))
    with pytest.raises(ValueError, match="incompatible axes:"):
        arr[:] = la2

    # key has multiple Arrays (this is used within .points indexing)
    # ==============================================================
    # first some setup
    a = Axis(['a0', 'a1'], None)
    b = Axis(['b0', 'b1', 'b2'], None)
    expected = ndtest((a, b))
    value = expected.combine_axes()

    # a) with anonymous axes
    combined_axis = value.axes[0]
    a_key = Array([0, 0, 0, 1, 1, 1], combined_axis)
    b_key = Array([0, 1, 2, 0, 1, 2], combined_axis)
    key = (a.i[a_key], b.i[b_key])
    array = empty((a, b))
    array[key] = value
    assert_array_equal(array, expected)

    # b) with wildcard combined_axis
    wild_combined_axis = combined_axis.ignore_labels()
    wild_a_key = Array([0, 0, 0, 1, 1, 1], wild_combined_axis)
    wild_b_key = Array([0, 1, 2, 0, 1, 2], wild_combined_axis)
    wild_key = (a.i[wild_a_key], b.i[wild_b_key])
    array = empty((a, b))
    array[wild_key] = value
    assert_array_equal(array, expected)

    # c) with a wildcard value
    wild_value = value.ignore_labels()
    array = empty((a, b))
    array[key] = wild_value
    assert_array_equal(array, expected)

    # d) with a wildcard combined axis and wildcard value
    array = empty((a, b))
    array[wild_key] = wild_value
    assert_array_equal(array, expected)


def test_setitem_ndarray(array):
    """
    tests Array.__setitem__(key, value) where value is a raw ndarray.
    In that case, value.shape is more restricted as we rely on numpy broadcasting.
    """
    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()
    value = raw[[1, 5, 9]] + 25.0
    arr[[1, 5, 9]] = value
    raw[[1, 5, 9]] = value
    assert_array_equal(arr, raw)

    # b) value has the same axes than target but one has length 1
    arr = array.copy()
    raw = array.data.copy()
    value = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    arr[[1, 5, 9]] = value
    raw[[1, 5, 9]] = value
    assert_array_equal(arr, raw)


def test_setitem_scalar(array):
    """
    tests Array.__setitem__(key, value) where value is a scalar
    """
    # a) list key (one dimension)
    arr = array.copy()
    raw = array.data.copy()
    arr[[1, 5, 9]] = 42
    raw[[1, 5, 9]] = 42
    assert_array_equal(arr, raw)

    # b) full scalar key (ie set one cell)
    arr = array.copy()
    raw = array.data.copy()
    arr[0, 'P02', 'A12', 'M'] = 42
    raw[0, 1, 0, 1] = 42
    assert_array_equal(arr, raw)


def test_setitem_bool_array_key(array):
    # XXX: this test is awfully slow (more than 1s)
    age, geo, sex, lipro = array.axes

    # Array key
    # a1) same shape, same order
    arr = array.copy()
    raw = array.data.copy()
    arr[arr < 5] = 0
    raw[raw < 5] = 0
    assert_array_equal(arr, raw)

    # a2) same shape, different order
    arr = array.copy()
    raw = array.data.copy()
    key = (arr < 5).T
    arr[key] = 0
    raw[raw < 5] = 0
    assert_array_equal(arr, raw)

    # b) numpy-broadcastable shape
    # arr = array.copy()
    # raw = array.data.copy()
    # key = arr[sex['F,']] < 5
    # self.assertEqual(key.ndim, 4)
    # arr[key] = 0
    # raw[raw[:, :, [1]] < 5] = 0
    # assert_array_equal(arr, raw)

    # c) Array-broadcastable shape (missing axis)
    arr = array.copy()
    raw = array.data.copy()
    key = arr[sex['M']] < 5
    assert key.ndim == 3
    arr[key] = 0

    raw_key = raw[:, :, 0, :] < 5
    raw_d1, raw_d2, raw_d4 = raw_key.nonzero()
    raw[raw_d1, raw_d2, :, raw_d4] = 0
    assert_array_equal(arr, raw)

    # ndarray key
    arr = array.copy()
    raw = array.data.copy()
    arr[raw < 5] = 0
    raw[raw < 5] = 0
    assert_array_equal(arr, raw)

    # d) Array with extra axes
    arr = array.copy()
    key = (arr < 5).expand([Axis(2, 'extra')])
    assert key.ndim == 5
    # TODO: make this work
    with pytest.raises(ValueError):
        arr[key] = 0


def test_set(array):
    age, geo, sex, lipro = array.axes

    # 1) using a LGroup key
    ages1_5_9 = age[[1, 5, 9]]

    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()

    arr.set(arr[ages1_5_9] + 25.0, age=ages1_5_9)
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
    assert_array_equal(arr, raw)

    # b) same size but a different shape (extra length-1 axis)
    arr = array.copy()
    raw = array.data.copy()

    raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
    fake_axis = Axis(['label'], 'fake')
    age_axis = arr[ages1_5_9].axes.age
    value = Array(raw_value, axes=(age_axis, fake_axis, geo, sex, lipro))
    arr.set(value, age=ages1_5_9)
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 26.0
    assert_array_equal(arr, raw)

    # dimension of length 1
    # arr = array.copy()
    # raw = array.data.copy()
    # raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    # arr.set(arr[ages1_5_9].sum(geo=(geo.all(),)), age=ages1_5_9)
    # assert_array_equal(arr, raw)

    # c) missing dimension
    arr = array.copy()
    raw = array.data.copy()
    arr.set(arr[ages1_5_9].sum(geo), age=ages1_5_9)
    raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    assert_array_equal(arr, raw)

    # 2) using a raw key
    arr = array.copy()
    raw = array.data.copy()
    arr.set(arr[[1, 5, 9]] + 27.0, age=[1, 5, 9])
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 27.0
    assert_array_equal(arr, raw)


def test_filter(array):
    age, geo, sex, lipro = array.axes

    ages1_5_9 = age[(1, 5, 9)]
    ages11 = age[11]

    # with LGroup
    assert array.filter(age=ages1_5_9).shape == (3, 44, 2, 15)

    # FIXME: this should raise a comprehensible error!
    # self.assertEqual(array.filter(age=[ages1_5_9]).shape, (3, 44, 2, 15))

    # LGroup with 1 value => collapse
    assert array.filter(age=ages11).shape == (44, 2, 15)

    # LGroup with a list of 1 value => do not collapse
    assert array.filter(age=age[[11]]).shape == (1, 44, 2, 15)

    # LGroup with a list of 1 value defined as a string => do not collapse
    assert array.filter(lipro=lipro['P01,']).shape == (116, 44, 2, 1)

    # LGroup with 1 value
    # XXX: this does not work. Do we want to make this work?
    # filtered = array.filter(age=(ages11,))
    # self.assertEqual(filtered.shape, (1, 44, 2, 15))

    # list
    assert array.filter(age=[1, 5, 9]).shape == (3, 44, 2, 15)

    # string
    assert array.filter(lipro='P01,P02').shape == (116, 44, 2, 2)

    # multiple axes at once
    assert array.filter(age=[1, 5, 9], lipro='P01,P02').shape == (3, 44, 2, 2)

    # multiple axes one after the other
    assert array.filter(age=[1, 5, 9]).filter(lipro='P01,P02').shape == (3, 44, 2, 2)

    # a single value for one dimension => collapse the dimension
    assert array.filter(sex='M').shape == (116, 44, 15)

    # but a list with a single value for one dimension => do not collapse
    assert array.filter(sex=['M']).shape == (116, 44, 1, 15)

    assert array.filter(sex='M,').shape == (116, 44, 1, 15)

    # with duplicate keys
    # XXX: do we want to support this? I don't see any value in that but I might be short-sighted.
    # filtered = array.filter(lipro='P01,P02,P01')

    # XXX: we could abuse python to allow naming groups via Axis.__getitem__
    # (but I doubt it is a good idea).
    # child = age[':17', 'child']

    # slices
    # ------

    # LGroup slice
    assert array.filter(age=age[:17]).shape == (18, 44, 2, 15)
    # string slice
    assert array.filter(lipro=':P03').shape == (116, 44, 2, 3)
    # raw slice
    assert array.filter(age=slice(17)).shape == (18, 44, 2, 15)

    # filter chain with a slice
    assert array.filter(age=slice(17)).filter(geo='A12,A13').shape == (18, 2, 2, 15)


def test_filter_multiple_axes(array):
    # multiple values in each group
    assert array.filter(age=[1, 5, 9], lipro='P01,P02').shape == (3, 44, 2, 2)
    # with a group of one value
    assert array.filter(age=[1, 5, 9], sex='M,').shape == (3, 44, 1, 15)

    # with a discarded axis (there is a scalar in the key)
    assert array.filter(age=[1, 5, 9], sex='M').shape == (3, 44, 15)

    # with a discarded axis that is not adjacent to the ix_array axis ie with a sliced axis between the scalar axis
    # and the ix_array axis since our array has a axes: age, geo, sex, lipro, any of the following should be tested:
    # age+sex / age+lipro / geo+lipro
    # additionally, if the ix_array axis was first (ie ix_array on age), it worked even before the issue was fixed,
    # since the "indexing" subspace is tacked-on to the beginning (as the first dimension)
    assert array.filter(age=57, sex='M,F').shape == (44, 2, 15)
    assert array.filter(age=57, lipro='P01,P05').shape == (44, 2, 2)
    assert array.filter(geo='A57', lipro='P01,P05').shape == (116, 2, 2)


def test_nonzero():
    arr = ndtest((2, 3))
    a, b = arr.axes
    cond = arr > 1
    assert_array_equal(cond, from_string(r"""a\b     b0     b1    b2
                                              a0  False  False  True
                                              a1   True   True  True"""))
    a_group, b_group = cond.nonzero()
    assert isinstance(a_group, IGroup)
    assert a_group.axis is a
    assert a_group.key.equals(from_string("""a_b  a0_b2  a1_b0  a1_b1  a1_b2
                                              \t      0      1      1      1"""))
    assert isinstance(b_group, IGroup)
    assert b_group.axis is b
    assert b_group.key.equals(from_string("""a_b  a0_b2  a1_b0  a1_b1  a1_b2
                                              \t      2      0      1      2"""))

    expected = from_string("""a_b  a0_b2  a1_b0  a1_b1  a1_b2
                               \t      2      3      4      5""")
    assert_array_equal(arr[a_group, b_group], expected)
    assert_array_equal(arr.points[a_group, b_group], expected)
    assert_array_equal(arr[cond], expected)


def test_contains():
    arr = ndtest('a=0..2;b=b0..b2;c=2..4')
    # string label
    assert 'b1' in arr
    assert 'b4' not in arr
    # int label
    assert 1 in arr
    assert 5 not in arr
    # duplicate label
    assert 2 in arr
    # slice
    assert not slice('b0', 'b2') in arr


def test_sum_full_axes(array):
    age, geo, sex, lipro = array.axes

    # everything
    assert array.sum() == np.asarray(array).sum()

    # using axes numbers
    assert array.sum(axis=2).shape == (116, 44, 15)
    assert array.sum(axis=(0, 2)).shape == (44, 15)

    # using Axis objects
    assert array.sum(age).shape == (44, 2, 15)
    assert array.sum(age, sex).shape == (44, 15)

    # using axes names
    assert array.sum('age', 'sex').shape == (44, 15)

    # chained sum
    assert array.sum(age, sex).sum(geo).shape == (15,)
    assert array.sum(age, sex).sum(lipro, geo) == array.sum()

    # getitem on aggregated
    aggregated = array.sum(age, sex)
    assert aggregated[vla_str].shape == (22, 15)

    # filter on aggregated
    assert aggregated.filter(geo=vla_str).shape == (22, 15)


def test_sum_full_axes_with_nan(array):
    array['M', 'P02', 'A12', 0] = nan
    raw = array.data

    # everything
    assert array.sum() == np.nansum(raw)
    assert isnan(array.sum(skipna=False))

    # using Axis objects
    assert_array_nan_equal(array.sum(X.age), np.nansum(raw, 0))
    assert_array_nan_equal(array.sum(X.age, skipna=False), raw.sum(0))

    assert_array_nan_equal(array.sum(X.age, X.sex), np.nansum(raw, (0, 2)))
    assert_array_nan_equal(array.sum(X.age, X.sex, skipna=False), raw.sum((0, 2)))


def test_sum_full_axes_keep_axes(array):
    agg = array.sum(keepaxes=True)
    assert agg.shape == (1, 1, 1, 1)
    for axis in agg.axes:
        assert axis.labels == ['sum']

    agg = array.sum(keepaxes='total')
    assert agg.shape == (1, 1, 1, 1)
    for axis in agg.axes:
        assert axis.labels == ['total']


def test_mean_full_axes(array):
    raw = array.data

    assert array.mean() == np.mean(raw)
    assert_array_nan_equal(array.mean(X.age), np.mean(raw, 0))
    assert_array_nan_equal(array.mean(X.age, X.sex), np.mean(raw, (0, 2)))


def test_mean_groups(array):
    # using int type to test that we get a float in return
    arr = array.astype(int)
    raw = array.data
    res = arr.mean(X.geo['A11', 'A13', 'A24', 'A31'])
    assert_array_nan_equal(res, np.mean(raw[:, [0, 2, 4, 5]], 1))


def test_median_full_axes(array):
    raw = array.data

    assert array.median() == np.median(raw)
    assert_array_nan_equal(array.median(X.age), np.median(raw, 0))
    assert_array_nan_equal(array.median(X.age, X.sex), np.median(raw, (0, 2)))


def test_median_groups(array):
    raw = array.data
    res = array.median(X.geo['A11', 'A13', 'A24'])
    assert res.shape == (116, 2, 15)
    assert_array_nan_equal(res, np.median(raw[:, [0, 2, 4]], 1))


def test_percentile_full_axes():
    arr = ndtest((2, 3, 4))
    raw = arr.data
    assert arr.percentile(10) == np.percentile(raw, 10)
    assert_array_nan_equal(arr.percentile(10, X.a), np.percentile(raw, 10, 0))
    assert_array_nan_equal(arr.percentile(10, X.c, X.a), np.percentile(raw, 10, (2, 0)))


def test_percentile_groups():
    arr = ndtest((2, 5, 3))
    raw = arr.data

    res = arr.percentile(10, X.b['b0', 'b2', 'b4'])
    assert_array_nan_equal(res, np.percentile(raw[:, [0, 2, 4]], 10, 1))


def test_cumsum(array):
    raw = array.data

    # using Axis objects
    assert_array_equal(array.cumsum(X.age), raw.cumsum(0))
    assert_array_equal(array.cumsum(X.lipro), raw.cumsum(3))

    # using axes numbers
    assert_array_equal(array.cumsum(1), raw.cumsum(1))

    # using axes names
    assert_array_equal(array.cumsum('sex'), raw.cumsum(2))


def test_group_agg_kwargs(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    assert array.sum(sex='M').shape == (116, 44, 15)
    assert array.sum(sex='M,F').shape == (116, 44, 15)
    assert array.sum(sex=sex['M']).shape == (116, 44, 15)

    assert array.sum(geo='A11,A21,A25').shape == (116, 2, 15)
    assert array.sum(geo=['A11', 'A21', 'A25']).shape == (116, 2, 15)
    assert array.sum(geo=geo['A11,A21,A25']).shape == (116, 2, 15)

    assert array.sum(geo=':').shape == (116, 2, 15)
    assert array.sum(geo=geo[:]).shape == (116, 2, 15)
    assert array.sum(geo=geo[':']).shape == (116, 2, 15)
    # Include everything between two labels. Since A11 is the first label
    # and A21 is the last one, this should be equivalent to the previous
    # tests.
    assert array.sum(geo='A11:A21').shape == (116, 2, 15)
    assert_array_equal(array.sum(geo='A11:A21'), array.sum(geo=':'))
    assert_array_equal(array.sum(geo=geo['A11:A21']), array.sum(geo=':'))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum(geo=(geo[:],)).shape == (116, 1, 2, 15)

    # a.3) several groups
    # string groups
    assert array.sum(geo=(vla, wal, bru)).shape == (116, 3, 2, 15)
    # with one label in several groups
    assert array.sum(sex=(['M'], ['M', 'F'])).shape == (116, 44, 2, 15)
    assert array.sum(sex=('M', 'M,F')).shape == (116, 44, 2, 15)
    assert array.sum(sex='M;M,F').shape == (116, 44, 2, 15)

    res = array.sum(geo=(vla, wal, bru, belgium))
    assert res.shape == (116, 4, 2, 15)

    # a.4) several dimensions at the same time
    res = array.sum(lipro='P01,P03;P02,P05;:', geo=(vla, wal, bru, belgium))
    assert res.shape == (116, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    # Note that you must list "full axes" aggregates first (Python does not allow non-kwargs after kwargs.
    res = array.sum(age, sex, geo=(vla, wal, bru, belgium))
    assert res.shape == (4, 15)

    # c) chain group aggregate after axis aggregate
    res = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
    assert res.shape == (4, 15)


def test_group_agg_guess_axis(array):
    raw = array.data.copy()
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    assert array.sum('M').shape == (116, 44, 15)
    assert array.sum('M,').shape == (116, 44, 15)
    assert array.sum('M,F').shape == (116, 44, 15)

    assert array.sum('A11,A21,A25').shape == (116, 2, 15)
    # with a name
    assert array.sum('A11,A21,A25 >> g1').shape == (116, 2, 15)
    assert array.sum(['A11', 'A21', 'A25']).shape == (116, 2, 15)

    # Include everything between two labels. Since A11 is the first label
    # and A21 is the last one, this should be equivalent to taking the
    # full axis.
    assert array.sum('A11:A21').shape == (116, 2, 15)
    assert_array_equal(array.sum('A11:A21'), array.sum(geo=':'))
    assert_array_equal(array.sum('A11:A21'), array.sum(geo))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum((geo[:],)).shape == (116, 1, 2, 15)

    # a.3) several groups
    # string groups
    assert array.sum((vla, wal, bru)).shape == (116, 3, 2, 15)

    # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
    # axes into play. For now the error message is unclear because it first aggregates on "vla", then tries to
    # aggregate on "wal", but there is no "geo" dimension anymore.
    # self.assertEqual(array.sum(vla, wal, bru).shape, (116, 3, 2, 15))

    # with one label in several groups
    assert array.sum((['M'], ['M', 'F'])).shape == (116, 44, 2, 15)
    assert array.sum(('M', 'M,F')).shape == (116, 44, 2, 15)
    assert array.sum('M;M,F').shape == (116, 44, 2, 15)
    # with group names
    res = array.sum('M >> men;M,F >> all')
    assert res.shape == (116, 44, 2, 15)
    assert 'sex' in res.axes
    assert_array_equal(res.axes.sex.labels, ['men', 'all'])
    assert_array_equal(res['men'], raw[:, :, 0, :])
    assert_array_equal(res['all'], raw.sum(2))

    res = array.sum(('M >> men', 'M,F >> all'))
    assert res.shape == (116, 44, 2, 15)
    assert 'sex' in res.axes
    assert_array_equal(res.axes.sex.labels, ['men', 'all'])
    assert_array_equal(res['men'], raw[:, :, 0, :])
    assert_array_equal(res['all'], raw.sum(2))

    res = array.sum((vla, wal, bru, belgium))
    assert res.shape == (116, 4, 2, 15)

    # a.4) several dimensions at the same time
    res = array.sum('P01,P03;P02,P05;P01:', (vla, wal, bru, belgium))
    assert res.shape == (116, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(age, sex, (vla, wal, bru, belgium))
    assert res.shape == (4, 15)

    # c) chain group aggregate after axis aggregate
    res = array.sum(age, sex).sum((vla, wal, bru, belgium))
    assert res.shape == (4, 15)

    # issue #868: labels in reverse order with a "step" between them > index of last label
    arr = ndtest(4)
    assert arr.sum('a3,a1') == 4


def test_group_agg_label_group(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = geo[vla_str], geo[wal_str], geo[bru_str]
    lg_belgium = geo[belgium]

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    men = sex.i[[0]]
    assert array.sum(men).shape == (116, 44, 15)
    assert array.sum(sex['M']).shape == (116, 44, 15)
    assert array.sum(sex['M,']).shape == (116, 44, 15)
    assert array.sum(sex['M,F']).shape == (116, 44, 15)

    assert array.sum(geo['A11,A21,A25']).shape == (116, 2, 15)
    assert array.sum(geo[['A11', 'A21', 'A25']]).shape == (116, 2, 15)
    assert array.sum(geo['A11', 'A21', 'A25']).shape == (116, 2, 15)
    assert array.sum(geo['A11,A21,A25']).shape == (116, 2, 15)

    assert array.sum(geo[:]).shape == (116, 2, 15)
    assert array.sum(geo[':']).shape == (116, 2, 15)
    assert array.sum(geo[:]).shape == (116, 2, 15)

    # Include everything between two labels. Since A11 is the first label and A21 is the last one, this should be
    # equivalent to the previous tests.
    assert array.sum(geo['A11:A21']).shape == (116, 2, 15)
    assert_array_equal(array.sum(geo['A11:A21']), array.sum(geo))
    assert_array_equal(array.sum(geo['A11':'A21']), array.sum(geo))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum((geo[:],)).shape == (116, 1, 2, 15)

    # a.3) several groups
    # string groups
    assert array.sum((vla, wal, bru)).shape == (116, 3, 2, 15)

    # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
    # axes into play. For now the error message is unclear because it first aggregates on "vla", then tries to
    # aggregate on "wal", but there is no "geo" dimension anymore.
    # self.assertEqual(array.sum(vla, wal, bru).shape, (116, 3, 2, 15))

    # with one label in several groups
    assert array.sum((sex['M'], sex[['M', 'F']])).shape == (116, 44, 2, 15)
    assert array.sum((sex['M'], sex['M', 'F'])).shape == (116, 44, 2, 15)
    assert array.sum((sex['M'], sex['M,F'])).shape == (116, 44, 2, 15)
    # XXX: do we want to support this?
    # self.assertEqual(array.sum(sex['M;H,F']).shape, (116, 44, 2, 15))

    res = array.sum((vla, wal, bru, lg_belgium))
    assert res.shape == (116, 4, 2, 15)

    # a.4) several dimensions at the same time
    # self.assertEqual(array.sum(lipro['P01,P03;P02,P05;P01:'], (vla, wal, bru, lg_belgium)).shape,
    #                  (116, 4, 2, 3))
    res = array.sum((lipro['P01,P03'], lipro['P02,P05'], lipro[:]), (vla, wal, bru, lg_belgium))
    assert res.shape == (116, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(age, sex, (vla, wal, bru, lg_belgium))
    assert res.shape == (4, 15)

    # c) chain group aggregate after axis aggregate
    res = array.sum(age, sex).sum((vla, wal, bru, lg_belgium))
    assert res.shape == (4, 15)


def test_group_agg_label_group_no_axis(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = LGroup(vla_str), LGroup(wal_str), LGroup(bru_str)
    lg_belgium = LGroup(belgium)

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    assert array.sum(LGroup('M')).shape == (116, 44, 15)
    assert array.sum(LGroup('M,')).shape == (116, 44, 15)
    assert array.sum(LGroup('M,F')).shape == (116, 44, 15)

    assert array.sum(LGroup('A11,A21,A25')).shape == (116, 2, 15)
    assert array.sum(LGroup(['A11', 'A21', 'A25'])).shape == (116, 2, 15)

    # Include everything between two labels. Since A11 is the first label
    # and A21 is the last one, this should be equivalent to the full axis.
    assert array.sum(LGroup('A11:A21')).shape == (116, 2, 15)
    assert_array_equal(array.sum(LGroup('A11:A21')), array.sum(geo))
    assert_array_equal(array.sum(LGroup(slice('A11', 'A21'))), array.sum(geo))

    # a.3) several groups
    # string groups
    assert array.sum((vla, wal, bru)).shape == (116, 3, 2, 15)

    # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
    # axes into play. For now the error message is unclear because it first aggregates on "vla", then tries to
    # aggregate on "wal", but there is no "geo" dimension anymore.
    # self.assertEqual(array.sum(vla, wal, bru).shape, (116, 3, 2, 15))

    # with one label in several groups
    assert array.sum((LGroup('M'), LGroup(['M', 'F']))).shape == (116, 44, 2, 15)
    assert array.sum((LGroup('M'), LGroup('M,F'))).shape == (116, 44, 2, 15)
    # XXX: do we want to support this?
    # self.assertEqual(array.sum(sex['M;M,F']).shape, (116, 44, 2, 15))

    res = array.sum((vla, wal, bru, lg_belgium))
    assert res.shape == (116, 4, 2, 15)

    # a.4) several dimensions at the same time
    # self.assertEqual(array.sum(lipro['P01,P03;P02,P05;P01:'], (vla, wal, bru, lg_belgium)).shape,
    #                  (116, 4, 2, 3))
    res = array.sum((LGroup('P01,P03'), LGroup('P02,P05')), (vla, wal, bru, lg_belgium))
    assert res.shape == (116, 4, 2, 2)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(age, sex, (vla, wal, bru, lg_belgium))
    assert res.shape == (4, 15)

    # c) chain group aggregate after axis aggregate
    res = array.sum(age, sex).sum((vla, wal, bru, lg_belgium))
    assert res.shape == (4, 15)


def test_group_agg_axis_ref_label_group(array):
    age, geo, sex, lipro = X.age, X.geo, X.sex, X.lipro
    vla, wal, bru = geo[vla_str], geo[wal_str], geo[bru_str]
    lg_belgium = geo[belgium]

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    men = sex.i[[0]]
    assert array.sum(men).shape == (116, 44, 15)
    assert array.sum(sex['M']).shape == (116, 44, 15)
    assert array.sum(sex['M,']).shape == (116, 44, 15)
    assert array.sum(sex['M,F']).shape == (116, 44, 15)

    assert array.sum(geo['A11,A21,A25']).shape == (116, 2, 15)
    assert array.sum(geo[['A11', 'A21', 'A25']]).shape == (116, 2, 15)
    assert array.sum(geo['A11', 'A21', 'A25']).shape == (116, 2, 15)
    assert array.sum(geo['A11,A21,A25']).shape == (116, 2, 15)

    assert array.sum(geo[:]).shape == (116, 2, 15)
    assert array.sum(geo[':']).shape == (116, 2, 15)
    assert array.sum(geo[:]).shape == (116, 2, 15)

    # Include everything between two labels. Since A11 is the first label
    # and A21 is the last one, this should be equivalent to the previous
    # tests.
    assert array.sum(geo['A11:A21']).shape == (116, 2, 15)
    assert_array_equal(array.sum(geo['A11:A21']), array.sum(geo))
    assert_array_equal(array.sum(geo['A11':'A21']), array.sum(geo))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum((geo[:],)).shape == (116, 1, 2, 15)

    # a.3) several groups
    # string groups
    assert array.sum((vla, wal, bru)).shape == (116, 3, 2, 15)

    # XXX: do we also want to support this? I do not really like it because
    # it gets tricky when we have some other axes into play. For now the
    # error message is unclear because it first aggregates on "vla", then
    # tries to aggregate on "wal", but there is no "geo" dimension anymore.
    # self.assertEqual(array.sum(vla, wal, bru).shape, (116, 3, 2, 15))

    # with one label in several groups
    assert array.sum((sex['M'], sex[['M', 'F']])).shape == (116, 44, 2, 15)
    assert array.sum((sex['M'], sex['M', 'F'])).shape == (116, 44, 2, 15)
    assert array.sum((sex['M'], sex['M,F'])).shape == (116, 44, 2, 15)
    # XXX: do we want to support this?
    # self.assertEqual(array.sum(sex['M;M,F']).shape, (116, 44, 2, 15))

    res = array.sum((vla, wal, bru, lg_belgium))
    assert res.shape == (116, 4, 2, 15)

    # a.4) several dimensions at the same time
    # self.assertEqual(array.sum(lipro['P01,P03;P02,P05;P01:'],
    #                         (vla, wal, bru, belgium)).shape,
    #                  (116, 4, 2, 3))
    res = array.sum((lipro['P01,P03'], lipro['P02,P05'], lipro[:]), (vla, wal, bru, lg_belgium))
    assert res.shape == (116, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(age, sex, (vla, wal, bru, lg_belgium))
    assert res.shape == (4, 15)

    # c) chain group aggregate after axis aggregate
    res = array.sum(age, sex).sum((vla, wal, bru, lg_belgium))
    assert res.shape == (4, 15)


def test_group_agg_one_axis():
    a = Axis(range(3), 'a')
    la = ndtest(a)
    raw = np.asarray(la)

    assert_array_equal(la.sum(a[0, 2]), raw[[0, 2]].sum())


def test_group_agg_anonymous_axis():
    la = ndtest([Axis(2), Axis(3)])
    a1, a2 = la.axes
    raw = np.asarray(la)
    assert_array_equal(la.sum(a2[0, 2]), raw[:, [0, 2]].sum(1))


def test_group_agg_zero_padded_label():
    arr = ndtest("a=01,02,03,10,11; b=b0..b2")
    expected = Array([36, 30, 39], "a=01_03,10,11")
    assert_array_equal(arr.sum("01,02,03 >> 01_03; 10; 11", "b"), expected)


def test_group_agg_on_int_array():
    # issue 193
    arr = ndtest('year=2014..2018')
    group = arr.year[:2016]
    assert arr.mean(group) == 1.0
    assert arr.median(group) == 1.0
    assert arr.percentile(90, group) == 1.8
    assert arr.std(group) == 1.0
    assert arr.var(group) == 1.0


def test_group_agg_on_bool_array():
    # issue 194
    a = ndtest((2, 3))
    b = a > 1
    expected = from_string("""a,a0,a1
                               , 1, 2""", sep=',')
    assert_array_equal(b.sum('b1:'), expected)


# TODO: fix this (and add other tests for references (X.) to anonymous axes
# def test_group_agg_anonymous_axis_ref():
#     la = ndtest([Axis(2), Axis(3)])
#     raw = np.asarray(la)
#     # this does not work because x[1] refers to an axis with name 1,
#     # which does not exist. We might want to change this.
#     assert_array_equal(la.sum(x[1][0, 2]), raw[:, [0, 2]].sum(1))


# group aggregates on a group-aggregated array
def test_group_agg_on_group_agg(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    reg = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))

    # 1) one group => collapse dimension
    assert reg.sum(lipro='P01,P02').shape == (4,)

    # 2) a tuple of one group => do not collapse dimension
    assert reg.sum(lipro=('P01,P02',)).shape == (4, 1)

    # 3) several groups
    assert reg.sum(lipro='P01;P02;:').shape == (4, 3)

    # this is INVALID
    # TODO: raise a nice exception
    # regsum = reg.sum(lipro='P01,P02,:')

    # this is currently allowed even though it can be confusing:
    # P01 and P02 are both groups with one element each.
    assert reg.sum(lipro=('P01', 'P02', ':')).shape == (4, 3)
    assert reg.sum(lipro=('P01', 'P02', lipro[:])).shape == (4, 3)

    # explicit groups are better
    assert reg.sum(lipro=('P01,', 'P02,', ':')).shape == (4, 3)
    assert reg.sum(lipro=(['P01'], ['P02'], ':')).shape == (4, 3)

    # 4) groups on the aggregated dimension

    # self.assertEqual(reg.sum(geo=([vla, bru], [wal, bru])).shape, (2, 3))
    # vla, wal, bru


# group aggregates on a group-aggregated array
def test_group_agg_on_group_agg_nokw(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    reg = array.sum(age, sex).sum((vla, wal, bru, belgium))
    # XXX: should this be supported too? (it currently fails)
    # reg = array.sum(age, sex).sum(vla, wal, bru, belgium)

    # 1) one group => collapse dimension
    assert reg.sum('P01,P02').shape == (4,)

    # 2) a tuple of one group => do not collapse dimension
    assert reg.sum(('P01,P02',)).shape == (4, 1)

    # 3) several groups
    # : is ambiguous
    # self.assertEqual(reg.sum('P01;P02;:').shape, (4, 3))
    assert reg.sum('P01;P02;P01:').shape == (4, 3)

    # this is INVALID
    # TODO: raise a nice exception
    # regsum = reg.sum(lipro='P01,P02,:')

    # this is currently allowed even though it can be confusing:
    # P01 and P02 are both groups with one element each.
    assert reg.sum(('P01', 'P02', 'P01:')).shape == (4, 3)
    assert reg.sum(('P01', 'P02', lipro[:])).shape == (4, 3)

    # explicit groups are better
    assert reg.sum(('P01,', 'P02,', 'P01:')).shape == (4, 3)
    assert reg.sum((['P01'], ['P02'], 'P01:')).shape == (4, 3)

    # 4) groups on the aggregated dimension

    # self.assertEqual(reg.sum(geo=([vla, bru], [wal, bru])).shape, (2, 3))
    # vla, wal, bru


def test_getitem_on_group_agg(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    # using a string
    vla = vla_str
    reg = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
    # the following are all equivalent
    assert reg[vla].shape == (15,)
    assert reg[(vla,)].shape == (15,)
    assert reg[(vla, slice(None))].shape == (15,)
    assert reg[vla, slice(None)].shape == (15,)
    assert reg[vla, :].shape == (15,)

    # one more level...
    assert reg[vla]['P03'] == 389049848.0

    # using an anonymous LGroup
    vla = geo[vla_str]
    reg = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
    # the following are all equivalent
    assert reg[vla].shape == (15,)
    assert reg[(vla,)].shape == (15,)
    assert reg[(vla, slice(None))].shape == (15,)
    assert reg[vla, slice(None)].shape == (15,)
    assert reg[vla, :].shape == (15,)

    # using a named LGroup
    vla = geo[vla_str] >> 'Vlaanderen'
    reg = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
    # the following are all equivalent
    assert reg[vla].shape == (15,)
    assert reg[(vla,)].shape == (15,)
    assert reg[(vla, slice(None))].shape == (15,)
    assert reg[vla, slice(None)].shape == (15,)
    assert reg[vla, :].shape == (15,)


def test_getitem_on_group_agg_nokw(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    # using a string
    vla = vla_str
    reg = array.sum(age, sex).sum((vla, wal, bru, belgium))
    # the following are all equivalent
    assert reg[vla].shape == (15,)
    assert reg[(vla,)].shape == (15,)
    assert reg[(vla, slice(None))].shape == (15,)
    assert reg[vla, slice(None)].shape == (15,)
    assert reg[vla, :].shape == (15,)

    # one more level...
    assert reg[vla]['P03'] == 389049848.0

    # using an anonymous LGroup
    vla = geo[vla_str]
    reg = array.sum(age, sex).sum((vla, wal, bru, belgium))
    # the following are all equivalent
    assert reg[vla].shape == (15,)
    assert reg[(vla,)].shape == (15,)
    assert reg[(vla, slice(None))].shape == (15,)
    assert reg[vla, slice(None)].shape == (15,)
    assert reg[vla, :].shape == (15,)

    # using a named LGroup
    vla = geo[vla_str] >> 'Vlaanderen'
    reg = array.sum(age, sex).sum((vla, wal, bru, belgium))
    # the following are all equivalent
    assert reg[vla].shape == (15,)
    assert reg[(vla,)].shape == (15,)
    assert reg[(vla, slice(None))].shape == (15,)
    assert reg[vla, slice(None)].shape == (15,)
    assert reg[vla, :].shape == (15,)


def test_filter_on_group_agg(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    # using a string
    # vla = vla_str
    # reg = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
    # assert reg.filter(geo=vla).shape == (15,)

    # using a named LGroup
    vla = geo[vla_str] >> 'Vlaanderen'
    reg = array.sum(age, sex).sum(geo=(vla, wal, bru, belgium))
    assert reg.filter(geo=vla).shape == (15,)

    # Note that reg.filter(geo=(vla,)) does NOT work. It might be a
    # little confusing for users, because reg[(vla,)] works but it is
    # normal because reg.filter(geo=(vla,)) is equivalent to:
    # reg[((vla,),)] or reg[(vla,), :]

    # mixed LGroup/string slices
    child = age[:17]
    child_named = age[:17] >> 'child'
    working = age[18:64]
    retired = age[65:]

    byage = array.sum(age=(child, 5, working, retired))
    assert byage.shape == (4, 44, 2, 15)

    byage = array.sum(age=(child, slice(5, 10), working, retired))
    assert byage.shape == (4, 44, 2, 15)

    # filter on an aggregated larray created with mixed groups
    # assert byage.filter(age=':17').shape == (44, 2, 15)

    byage = array.sum(age=(child_named, 5, working, retired))
    assert byage.filter(age=child_named).shape == (44, 2, 15)


def test_sum_several_lg_groups(array):
    # 1) aggregated array created using LGroups
    # -----------------------------------------
    fla = geo[vla_str] >> 'Flanders'
    wal = geo[wal_str] >> 'Wallonia'
    bru = geo[bru_str] >> 'Brussels'

    reg = array.sum(geo=(fla, wal, bru))
    assert reg.shape == (116, 3, 2, 15)

    # the result is indexable
    # 1.a) by LGroup
    assert reg.filter(geo=fla).shape == (116, 2, 15)
    assert reg.filter(geo=(fla, wal)).shape == (116, 2, 2, 15)

    # 1.b) by string (name of groups)
    assert reg.filter(geo='Flanders').shape == (116, 2, 15)
    assert reg.filter(geo='Flanders,Wallonia').shape == (116, 2, 2, 15)

    # 2) aggregated array created using string groups
    # -----------------------------------------------
    reg = array.sum(geo=(vla_str, wal_str, bru_str))
    assert reg.shape == (116, 3, 2, 15)

    # the result is indexable
    # 2.a) by string (def)
    # assert reg.filter(geo=vla_str).shape == (116, 2, 15)
    assert reg.filter(geo=(vla_str, wal_str)).shape == (116, 2, 2, 15)

    # 2.b) by LGroup
    # assert reg.filter(geo=fla).shape == (116, 2, 15)
    # assert reg.filter(geo=(fla, wal)).shape == (116, 2, 2, 15)


def test_sum_with_groups_from_other_axis(small_array):
    # use a group from another *compatible* axis
    lipro2 = Axis('lipro=P01..P15')
    assert small_array.sum(lipro=lipro2['P01,P03']).shape == (2,)

    # use (compatible) group from another *incompatible* axis
    # XXX: I am unsure whether or not this should be allowed. Maybe we
    # should simply check that the group is valid in axis, but that
    # will trigger a pretty meaningful error anyway
    lipro3 = Axis('lipro=P01,P03,P05')
    assert small_array.sum(lipro3['P01,P03']).shape == (2,)

    # use a group (from another axis) which is incompatible with the axis of
    # the same name in the array
    lipro4 = Axis('lipro=P01,P03,P16')
    with pytest.raises(ValueError, match=r"lipro\['P01', 'P16'\] is not a valid label for any axis"):
        small_array.sum(lipro4['P01,P16'])


def test_agg_kwargs(array):
    raw = array.data

    # dtype
    assert array.sum(dtype=int) == raw.sum(dtype=int)

    # ddof
    assert array.std(ddof=0) == raw.std(ddof=0)

    # out
    res = array.std(X.sex)
    out = zeros_like(res)
    array.std(X.sex, out=out)
    assert_array_equal(res, out)


def test_agg_by(array):
    age, geo, sex, lipro = array.axes
    vla, wal, bru = vla_str, wal_str, bru_str

    # no group or axis
    assert array.sum_by().shape == ()
    assert array.sum_by() == array.sum()

    # all axes
    assert array.sum_by(geo, age, lipro, sex).equals(array)
    assert array.sum_by(age, geo, sex, lipro).equals(array)

    # a) group aggregate on a fresh array

    # a.1) one group
    res = array.sum_by(geo='A11,A21,A25')
    assert res.shape == ()
    assert res == array.sum(geo='A11,A21,A25').sum()

    # a.2) a tuple of one group
    res = array.sum_by(geo=(geo[:],))
    assert res.shape == (1,)
    assert_array_equal(res, array.sum(age, sex, lipro, geo=(geo[:],)))

    # a.3) several groups
    # string groups
    res = array.sum_by(geo=(vla, wal, bru))
    assert res.shape == (3,)
    assert_array_equal(res, array.sum(age, sex, lipro, geo=(vla, wal, bru)))

    # with one label in several groups
    assert array.sum_by(sex=(['M'], ['M', 'F'])).shape == (2,)
    assert array.sum_by(sex=('M', 'M,F')).shape == (2,)

    res = array.sum_by(sex='M;M,F')
    assert res.shape == (2,)
    assert_array_equal(res, array.sum(age, geo, lipro, sex='M;M,F'))

    # a.4) several dimensions at the same time
    res = array.sum_by(geo=(vla, wal, bru, belgium), lipro='P01,P03;P02,P05;:')
    assert res.shape == (4, 3)
    assert_array_equal(res, array.sum(age, sex, geo=(vla, wal, bru, belgium), lipro='P01,P03;P02,P05;:'))

    # b) both axis aggregate and group aggregate at the same time
    # Note that you must list "full axes" aggregates first (Python does not allow non-kwargs after kwargs.
    res = array.sum_by(sex, geo=(vla, wal, bru, belgium))
    assert res.shape == (4, 2)
    assert_array_equal(res, array.sum(age, lipro, geo=(vla, wal, bru, belgium)))

    # c) chain group aggregate after axis aggregate
    res = array.sum_by(geo, sex)
    assert res.shape == (44, 2)
    assert_array_equal(res, array.sum(age, lipro))

    res2 = res.sum_by(geo=(vla, wal, bru, belgium))
    assert res2.shape == (4,)
    assert_array_equal(res2, res.sum(sex, geo=(vla, wal, bru, belgium)))


def test_agg_igroup():
    arr = ndtest(3)
    res = arr.sum((X.a.i[:2], X.a.i[1:]))
    assert_array_equal(res.a.labels, [':a1', 'a1:'])


def test_ratio(array):
    age, geo, sex, lipro = array.axes

    regions = (vla_str, wal_str, bru_str, belgium)
    reg = array.sum(age, sex, regions)
    assert reg.shape == (4, 15)

    fla = geo[vla_str] >> 'Flanders'
    wal = geo[wal_str] >> 'Wallonia'
    bru = geo[bru_str] >> 'Brussels'
    regions = (fla, wal, bru)
    reg = array.sum(age, sex, regions)

    ratio = reg.ratio()
    assert_array_equal(ratio, reg / reg.sum(geo, lipro))
    assert ratio.shape == (3, 15)

    ratio = reg.ratio(geo)
    assert_array_equal(ratio, reg / reg.sum(geo))
    assert ratio.shape == (3, 15)

    ratio = reg.ratio(geo, lipro)
    assert_array_equal(ratio, reg / reg.sum(geo, lipro))
    assert ratio.shape == (3, 15)
    assert ratio.sum() == 1.0


def test_percent(array):
    age, geo, sex, lipro = array.axes

    regions = (vla_str, wal_str, bru_str, belgium)
    reg = array.sum(age, sex, regions)
    assert reg.shape == (4, 15)

    fla = geo[vla_str] >> 'Flanders'
    wal = geo[wal_str] >> 'Wallonia'
    bru = geo[bru_str] >> 'Brussels'
    regions = (fla, wal, bru)
    reg = array.sum(age, sex, regions)

    percent = reg.percent()
    assert_array_equal(percent, (reg * 100.0 / reg.sum(geo, lipro)))
    assert percent.shape == (3, 15)

    percent = reg.percent(geo)
    assert_array_equal(percent, (reg * 100.0 / reg.sum(geo)))
    assert percent.shape == (3, 15)

    percent = reg.percent(geo, lipro)
    assert_array_equal(percent, (reg * 100.0 / reg.sum(geo, lipro)))
    assert percent.shape == (3, 15)
    assert round(abs(percent.sum() - 100.0), 7) == 0


def test_total(array):
    age, geo, sex, lipro = array.axes
    # array = small_array
    # sex, lipro = array.axes

    assert array.with_total().shape == (117, 45, 3, 16)
    assert array.with_total(sex).shape == (116, 44, 3, 15)
    assert array.with_total(lipro).shape == (116, 44, 2, 16)
    assert array.with_total(sex, lipro).shape == (116, 44, 3, 16)

    fla = geo[vla_str] >> 'Flanders'
    wal = geo[wal_str] >> 'Wallonia'
    bru = geo[bru_str] >> 'Brussels'
    bel = geo[:] >> 'Belgium'

    assert array.with_total(geo=(fla, wal, bru), op=mean).shape == (116, 47, 2, 15)
    assert array.with_total((fla, wal, bru), op=mean).shape == (116, 47, 2, 15)
    # works but "wrong" for X.geo (double what is expected because it includes fla wal & bru)
    # TODO: we probably want to display a warning (or even an error?) in that case.
    # If we really want that behavior, we can still split the operation:
    # .with_total((fla, wal, bru)).with_total(X.geo)
    # OR we might want to only sum the axis as it was before the op (but that does not play well when working with
    #    multiple axes).
    a1 = array.with_total(X.sex, (fla, wal, bru), X.geo, X.lipro)
    assert a1.shape == (116, 48, 3, 16)

    # correct total but the order is not very nice
    a2 = array.with_total(X.sex, X.geo, (fla, wal, bru), X.lipro)
    assert a2.shape == (116, 48, 3, 16)

    # the correct way to do it
    a3 = array.with_total(X.sex, (fla, wal, bru, bel), X.lipro)
    assert a3.shape == (116, 48, 3, 16)

    # a4 = array.with_total((lipro[':P05'], lipro['P05:']), op=mean)
    a4 = array.with_total((':P05', 'P05:'), op=mean)
    assert a4.shape == (116, 44, 2, 17)


def test_transpose():
    arr = ndtest((2, 3, 4))
    a, b, c = arr.axes
    res = arr.transpose()
    assert res.axes == [c, b, a]
    res = arr.transpose('b', 'c', 'a')
    assert res.axes == [b, c, a]
    res = arr.transpose('b')
    assert res.axes == [b, a, c]

    res = arr.transpose(..., 'a')
    assert res.axes == [b, c, a]
    res = arr.transpose('c', ..., 'a')
    assert res.axes == [c, b, a]


def test_transpose_anonymous():
    a = ndtest([Axis(2), Axis(3), Axis(4)])

    # reordered = a.transpose(0, 2, 1)
    # self.assertEqual(reordered.shape, (2, 4, 3))

    # axes = [1, 2]
    # => union(axes, )
    # => axes.extend([[0]])
    # => breaks because [0] not compatible with axes[0]
    # => breaks because [0] not compatible with [1]

    # a real union should not care and should return
    # [1, 2, 0] but will this break other stuff? My gut feeling is yes

    # when doing a binop between anonymous axes, we use union too (that might be the problem) and we need *that*
    # union to match axes by position
    reordered = a.transpose(1, 2)
    assert reordered.shape == (3, 4, 2)

    reordered = a.transpose(2, 0)
    assert reordered.shape == (4, 2, 3)

    reordered = a.transpose()
    assert reordered.shape == (4, 3, 2)


def test_binary_ops(small_array):
    raw = small_array.data

    assert_array_equal(small_array + small_array, raw + raw)
    assert_array_equal(small_array + 1, raw + 1)
    assert_array_equal(1 + small_array, 1 + raw)

    assert_array_equal(small_array - small_array, raw - raw)
    assert_array_equal(small_array - 1, raw - 1)
    assert_array_equal(1 - small_array, 1 - raw)

    assert_array_equal(small_array * small_array, raw * raw)
    assert_array_equal(small_array * 2, raw * 2)
    assert_array_equal(2 * small_array, 2 * raw)

    with np.errstate(invalid='ignore'):
        raw_res = raw / raw

    warn_msg = "invalid value (NaN) encountered during operation (this is typically caused by a 0 / 0)"
    with must_warn(RuntimeWarning, msg=warn_msg):
        res = small_array / small_array
    assert_array_nan_equal(res, raw_res)

    assert_array_equal(small_array / 2, raw / 2)

    with np.errstate(divide='ignore'):
        raw_res = 30 / raw
    with must_warn(RuntimeWarning, msg="divide by zero encountered during operation"):
        res = 30 / small_array
    assert_array_equal(res, raw_res)

    assert_array_equal(30 / (small_array + 1), 30 / (raw + 1))

    raw_int = raw.astype(int)
    la_int = Array(raw_int, axes=(sex, lipro))
    assert_array_equal(la_int / 2, raw_int / 2)
    assert_array_equal(la_int // 2, raw_int // 2)

    # test adding two larrays with different axes order
    assert_array_equal(small_array + small_array.transpose(), raw * 2)

    # mixed operations
    raw2 = raw / 2
    la_raw2 = small_array - raw2
    assert la_raw2.axes == small_array.axes
    assert_array_equal(la_raw2, raw - raw2)
    raw2_la = raw2 - small_array
    assert raw2_la.axes == small_array.axes
    assert_array_equal(raw2_la, raw2 - raw)

    la_ge_raw2 = small_array >= raw2
    assert la_ge_raw2.axes == small_array.axes
    assert_array_equal(la_ge_raw2, raw >= raw2)

    raw2_ge_la = raw2 >= small_array
    assert raw2_ge_la.axes == small_array.axes
    assert_array_equal(raw2_ge_la, raw2 >= raw)


def test_binary_ops_no_name_axes(small_array):
    raw = small_array.data
    raw2 = small_array.data + 1
    la = ndtest([Axis(label) for label in small_array.shape])
    la2 = ndtest([Axis(label) for label in small_array.shape]) + 1

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
    with must_warn(RuntimeWarning, msg="divide by zero encountered during operation"):
        res = 30 / la
    assert_array_equal(res, raw_res)

    assert_array_equal(30 / (la + 1), 30 / (raw + 1))

    raw_int = raw.astype(int)
    la_int = Array(raw_int)
    assert_array_equal(la_int / 2, raw_int / 2)
    assert_array_equal(la_int // 2, raw_int // 2)

    # adding two larrays with different axes order cannot work with unnamed axes
    # assert_array_equal(la + la.transpose(), raw * 2)

    # mixed operations
    raw2 = raw / 2
    la_raw2 = la - raw2
    assert la_raw2.axes == la.axes
    assert_array_equal(la_raw2, raw - raw2)
    raw2_la = raw2 - la
    assert raw2_la.axes == la.axes
    assert_array_equal(raw2_la, raw2 - raw)

    la_ge_raw2 = la >= raw2
    assert la_ge_raw2.axes == la.axes
    assert_array_equal(la_ge_raw2, raw >= raw2)

    raw2_ge_la = raw2 >= la
    assert raw2_ge_la.axes == la.axes
    assert_array_equal(raw2_ge_la, raw2 >= raw)


def test_broadcasting_no_name():
    a = ndtest([Axis(2), Axis(3)])
    b = ndtest(Axis(3))
    c = ndtest(Axis(2))

    with pytest.raises(ValueError):
        # ValueError: incompatible axes:
        # Axis(None, [0, 1, 2])
        # vs
        # Axis(None, [0, 1])
        a * b

    d = a * c
    assert d.shape == (2, 3)
    # {0}*\{1}*  0  1  2
    #         0  0  0  0
    #         1  3  4  5
    assert np.array_equal(d, [[0, 0, 0],
                              [3, 4, 5]])

    # it is unfortunate that the behavior is different from numpy (even though I find our behavior more intuitive)
    d = np.asarray(a) * np.asarray(b)
    assert d.shape == (2, 3)
    assert np.array_equal(d, [[0, 1,  4],
                              [0, 4, 10]])

    with pytest.raises(ValueError):
        # ValueError: operands could not be broadcast together with shapes (2,3) (2,)
        np.asarray(a) * np.asarray(c)


def test_binary_ops_with_scalar_group():
    time = Axis('time=2015..2019')
    arr = ndtest(3)
    expected = arr + 2015
    assert_larray_equal(time.i[0] + arr, expected)
    assert_larray_equal(arr + time.i[0], expected)


def test_unary_ops(small_array):
    raw = small_array.data

    # using numpy functions
    assert_array_equal(np.abs(small_array - 10), np.abs(raw - 10))
    assert_array_equal(np.negative(small_array), np.negative(raw))
    assert_array_equal(np.invert(small_array), np.invert(raw))

    # using python builtin ops
    assert_array_equal(abs(small_array - 10), abs(raw - 10))
    assert_array_equal(-small_array, -raw)
    assert_array_equal(+small_array, +raw)
    assert_array_equal(~small_array, ~raw)


def test_mean(small_array):
    raw = small_array.data
    sex, lipro = small_array.axes
    assert_array_equal(small_array.mean(lipro), raw.mean(1))


def test_sequence():
    res = sequence('b=b0..b2', ndtest(3) * 3, 1.0)
    assert_array_equal(ndtest((3, 3), dtype=float), res)


def test_sort_values():
    # 1D arrays
    arr = Array([0, 1, 6, 3, -1], "a=a0..a4")
    res = arr.sort_values()
    expected = Array([-1, 0, 1, 3, 6], "a=a4,a0,a1,a3,a2")
    assert_array_equal(res, expected)
    # ascending arg
    res = arr.sort_values(ascending=False)
    expected = Array([6, 3, 1, 0, -1], "a=a2,a3,a1,a0,a4")
    assert_array_equal(res, expected)

    # 3D arrays
    arr = Array([[[10, 2, 4], [3, 7, 1]], [[5, 1, 6], [2, 8, 9]]],
                 'a=a0,a1; b=b0,b1; c=c0..c2')
    res = arr.sort_values(axis='c')
    expected = Array([[[2, 4, 10], [1, 3, 7]], [[1, 5, 6], [2, 8, 9]]],
                     [Axis('a=a0,a1'), Axis('b=b0,b1'), Axis(3, 'c')])
    assert_array_equal(res, expected)


def test_set_labels(small_array):
    small_array.set_labels(X.sex, ['Man', 'Woman'], inplace=True)
    expected = small_array.set_labels(X.sex, ['Man', 'Woman'])
    assert_array_equal(small_array, expected)


def test_set_axes(small_array):
    lipro2 = Axis([label.replace('P', 'Q') for label in lipro.labels], 'lipro2')
    sex2 = Axis(['Man', 'Woman'], 'sex2')

    la = Array(small_array.data, axes=(sex, lipro2))
    # replace one axis
    la2 = small_array.set_axes(X.lipro, lipro2)
    assert_array_equal(la, la2)

    la = Array(small_array.data, axes=(sex2, lipro2))
    # all at once
    la2 = small_array.set_axes([sex2, lipro2])
    assert_array_equal(la, la2)
    # using keywrods args
    la2 = small_array.set_axes(sex=sex2, lipro=lipro2)
    assert_array_equal(la, la2)
    # using dict
    la2 = small_array.set_axes({X.sex: sex2, X.lipro: lipro2})
    assert_array_equal(la, la2)
    # using list of pairs (axis_to_replace, new_axis)
    la2 = small_array.set_axes([(X.sex, sex2), (X.lipro, lipro2)])
    assert_array_equal(la, la2)


def test_reindex():
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

    # Array fill value
    filler = ndtest(arr.a)
    res = arr.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=filler)
    assert_array_equal(res, from_string("""a\\b  b1  b2  b0
                                             a0   1   0   0
                                             a1   3   1   2"""))

    # using labels from another array
    arr = ndtest('a=v0..v2;b=v0,v2,v1,v3')
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

    # passing a list of Axis
    arr = ndtest((2, 2))
    res = arr.reindex([Axis("a=a0,a1"), Axis("c=c0"), Axis("b=b1,b2")], fill_value=-1)
    assert_array_equal(res, from_string(""" a  b\\c  c0
                                           a0   b1   1
                                           a0   b2  -1
                                           a1   b1   3
                                           a1   b2  -1"""))


def test_expand():
    country = Axis("country=BE,FR,DE")
    arr = ndtest(country)

    out1 = empty((sex, country))
    arr.expand(out=out1)

    out2 = empty((sex, country))
    out2[:] = arr

    assert_array_equal(out1, out2)


def test_append(small_array):
    sex, lipro = small_array.axes

    small_array = small_array.append(lipro, small_array.sum(lipro), label='sum')
    assert small_array.shape == (2, 16)
    small_array = small_array.append(sex, small_array.sum(sex), label='sum')
    assert small_array.shape == (3, 16)

    # crap the sex axis is different !!!! we don't have this problem with
    # the kwargs syntax below
    # small_array = small_array.append(small_array.mean(sex), axis=sex, label='mean')
    # self.assertEqual(small_array.shape, (4, 16))

    # another syntax (which implies we could not have an axis named "label")
    # small_array = small_array.append(lipro=small_array.sum(lipro), label='sum')
    # self.assertEqual(small_array.shape, (117, 44, 2, 15))


def test_insert():
    # simple tests are in the docstring
    arr1 = ndtest((2, 3))

    # insert at multiple places at once

    # we cannot use from_string in these tests because it deduplicates ambiguous (column) labels automatically
    res = arr1.insert([42, 43], before='b1', label='new')
    assert_array_equal(res, from_lists([
    ['a\\b', 'b0', 'new', 'new', 'b1', 'b2'],
    ['a0',      0,    42,    43,    1,    2],
    ['a1',      3,    42,    43,    4,    5]]))

    res = arr1.insert(42, before=['b1', 'b2'], label='new')
    assert_array_equal(res, from_lists([
    ['a\\b', 'b0', 'new', 'b1', 'new', 'b2'],
    ['a0',      0,    42,    1,    42,    2],
    ['a1',      3,    42,    4,    42,    5]]))

    res = arr1.insert(42, before='b1', label=['b0.1', 'b0.2'])
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.1  b0.2  b1  b2
     a0   0    42    42   1   2
     a1   3    42    42   4   5"""))

    res = arr1.insert(42, before=['b1', 'b2'], label=['b0.5', 'b1.5'])
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    42   2
     a1   3    42   4    42   5"""))

    res = arr1.insert([42, 43], before='b1', label=['b0.1', 'b0.2'])
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.1  b0.2  b1  b2
     a0   0    42    43   1   2
     a1   3    42    43   4   5"""))

    res = arr1.insert([42, 43], before=['b1', 'b2'], label='new')
    assert_array_equal(res, from_lists([
    ['a\\b', 'b0', 'new', 'b1', 'new', 'b2'],
    [  'a0',    0,    42,    1,    43,    2],
    [  'a1',    3,    42,    4,    43,    5]]))

    res = arr1.insert([42, 43], before=['b1', 'b2'], label=['b0.5', 'b1.5'])
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    43   2
     a1   3    42   4    43   5"""))

    res = arr1.insert([42, 43], before='b1,b2', label=['b0.5', 'b1.5'])
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    43   2
     a1   3    42   4    43   5"""))

    arr2 = ndtest(2)
    res = arr1.insert([arr2 + 42, arr2 + 43], before=['b1', 'b2'], label=['b0.5', 'b1.5'])
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    43   2
     a1   3    43   4    44   5"""))

    arr3 = ndtest('a=a0,a1;b=b0.1,b0.2') + 42
    res = arr1.insert(arr3, before='b1,b2')
    assert_array_equal(res, from_string(r"""
    a\b  b0  b0.1  b1  b0.2  b2
     a0   0    42   1    43   2
     a1   3    44   4    45   5"""))

    # with ambiguous labels
    arr4 = ndtest('a=v0,v1;b=v0,v1')
    expected = from_string(r"""
    a\b  v0  v0.5  v1
     v0   0    42   1
     v1   2    42   3""")

    res = arr4.insert(42, before='b[v1]', label='v0.5')
    assert_array_equal(res, expected)

    res = arr4.insert(42, before=X.b['v1'], label='v0.5')
    assert_array_equal(res, expected)

    res = arr4.insert(42, before=arr4.b['v1'], label='v0.5')
    assert_array_equal(res, expected)


def test_drop():
    arr1 = ndtest(3)
    expected = Array([0, 2], 'a=a0,a2')

    # indices
    res = arr1.drop('a.i[1]')
    assert_array_equal(res, expected)

    res = arr1.drop(X.a.i[1])
    assert_array_equal(res, expected)

    # labels
    res = arr1.drop(X.a['a1'])
    assert_array_equal(res, expected)

    res = arr1.drop('a[a1]')
    assert_array_equal(res, expected)

    # 2D array
    arr2 = ndtest((2, 4))
    expected = from_string(r"""
    a\b  b0  b2
     a0   0   2
     a1   4   6""")
    res = arr2.drop(['b1', 'b3'])
    assert_array_equal(res, expected)

    res = arr2.drop(X.b['b1', 'b3'])
    assert_array_equal(res, expected)

    res = arr2.drop('b.i[1, 3]')
    assert_array_equal(res, expected)

    res = arr2.drop(X.b.i[1, 3])
    assert_array_equal(res, expected)

    a = Axis('a=label0..label2')
    b = Axis('b=label0..label2')
    arr3 = ndtest((a, b))

    res = arr3.drop('a[label1]')
    assert_array_equal(res, from_string(r"""
       a\b  label0  label1  label2
    label0       0       1       2
    label2       6       7       8"""))

    # XXX: implement the following (#671)?
    # res = arr3.drop('0[label1]')
    res = arr3.drop(X[0]['label1'])
    assert_array_equal(res, from_string(r"""
       a\b  label0  label1  label2
    label0       0       1       2
    label2       6       7       8"""))

    res = arr3.drop(a['label1'])
    assert_array_equal(res, from_string(r"""
       a\b  label0  label1  label2
    label0       0       1       2
    label2       6       7       8"""))


# the aim of this test is to drop the last value of an axis, but instead
# of dropping the last axis tick/label, drop the first one.
def test_shift_axis(small_array):
    sex, lipro = small_array.axes

    # TODO: check how awful the syntax is with an axis that is not last
    # or first
    l2 = Array(small_array[:, :'P14'], axes=[sex, Axis(lipro.labels[1:], 'lipro')])
    l2 = Array(small_array[:, :'P14'], axes=[sex, lipro.subaxis(slice(1, None))])

    # We can also modify the axis in-place (dangerous!)
    # lipro.labels = np.append(lipro.labels[1:], lipro.labels[0])
    l2 = small_array[:, 'P02':]
    l2.axes.lipro.labels = lipro.labels[1:]


def test_unique():
    arr = Array([[[0, 2, 0, 0],
                  [1, 1, 1, 0]],
                 [[0, 2, 0, 0],
                   [2, 1, 2, 0]]], 'a=a0,a1;b=b0,b1;c=c0..c3')
    assert_array_equal(arr.unique('a'), arr)
    assert_array_equal(arr.unique('b'), arr)
    assert_array_equal(arr.unique('c'), arr['c0,c1,c3'])
    expected = from_string("""\
a_b\\c  c0  c1  c2  c3
a0_b0   0   2   0   0
a0_b1   1   1   1   0
a1_b1   2   1   2   0""")
    assert_array_equal(arr.unique(('a', 'b')), expected)


def test_extend(small_array):
    sex, lipro = small_array.axes

    all_lipro = lipro[:]
    tail = small_array.sum(lipro=(all_lipro,))
    small_array = small_array.extend(lipro, tail)
    assert small_array.shape == (2, 16)
    # test with a string axis
    small_array = small_array.extend('sex', small_array.sum(sex=(sex[:],)))
    assert small_array.shape == (3, 16)


@needs_pytables
def test_hdf_roundtrip(tmpdir, meta):
    import tables

    a = ndtest((2, 3), meta=meta)
    fpath = tmp_path(tmpdir, 'test.h5')
    a.to_hdf(fpath, 'a')
    res = read_hdf(fpath, 'a')

    assert a.ndim == 2
    assert a.shape == (2, 3)
    assert a.axes.names == ['a', 'b']
    assert_array_equal(res, a)
    assert res.meta == a.meta

    # issue 72: int-like strings should not be parsed (should round-trip correctly)
    fpath = tmp_path(tmpdir, 'issue72.h5')
    a = from_lists([['axis', '10', '20'],
                    ['',        0,    1]])
    a.to_hdf(fpath, 'a')
    res = read_hdf(fpath, 'a')
    assert res.ndim == 1
    axis = res.axes[0]
    assert axis.name == 'axis'
    assert_array_equal(axis.labels, ['10', '20'])

    # passing group as key to to_hdf
    a3 = ndtest((4, 3, 4))
    fpath = tmp_path(tmpdir, 'test.h5')
    os.remove(fpath)

    # single element group
    for label in a3.a:
        a3[label].to_hdf(fpath, label)

    # unnamed group
    group = a3.c['c0,c2']
    with must_warn(tables.NaturalNameWarning, check_file=False):
        a3[group].to_hdf(fpath, group)

    # unnamed group + slice
    group = a3.c['c0::2']
    with must_warn(tables.NaturalNameWarning, check_file=False):
        a3[group].to_hdf(fpath, group)

    # named group
    group = a3.c['c0,c2'] >> 'even'
    a3[group].to_hdf(fpath, group)

    # group with name containing special characters (replaced by _)
    group = a3.c['c0,c2'] >> r':name?with*special/\[characters]'
    with must_warn(tables.NaturalNameWarning, check_file=False):
        a3[group].to_hdf(fpath, group)

    # passing group as key to read_hdf
    for label in a3.a:
        subset = read_hdf(fpath, label)
        assert_array_equal(subset, a3[label])

    # load Session
    from larray.core.session import Session
    s = Session(fpath)
    assert s.names == sorted(['a0', 'a1', 'a2', 'a3', 'c0,c2', 'c0::2', 'even', ':name?with*special__[characters]'])


def test_from_string():
    expected = ndtest("sex=M,F")

    res = from_string('''sex  M  F
                         \t   0  1''')
    assert_array_equal(res, expected)

    res = from_string('''sex  M  F
                         nan  0  1''')
    assert_array_equal(res, expected)

    res = from_string('''sex  M  F
                         NaN  0  1''')
    assert_array_equal(res, expected)


def test_read_csv():
    res = read_csv(inputpath('test1d.csv'))
    assert_array_equal(res, io_1d)

    res = read_csv(inputpath('test2d.csv'))
    assert_array_equal(res, io_2d)

    res = read_csv(inputpath('test3d.csv'))
    assert_array_equal(res, io_3d)

    res = read_csv(inputpath('testint_labels.csv'))
    assert_array_equal(res, io_int_labels)

    res = read_csv(inputpath('test2d_classic.csv'))
    assert_array_equal(res, ndtest("a=a0..a2; b0..b2"))

    la = read_csv(inputpath('test1d_liam2.csv'), dialect='liam2')
    assert la.ndim == 1
    assert la.shape == (3,)
    assert la.axes.names == ['time']
    assert_array_equal(la, [3722, 3395, 3347])

    la = read_csv(inputpath('test5d_liam2.csv'), dialect='liam2')
    assert la.ndim == 5
    assert la.shape == (2, 5, 2, 2, 3)
    assert la.axes.names == ['arr', 'age', 'sex', 'nat', 'time']
    assert_array_equal(la[X.arr[1], 0, 'F', X.nat[1], :], [3722, 3395, 3347])

    # missing values
    res = read_csv(inputpath('testmissing_values.csv'))
    assert_array_nan_equal(res, io_missing_values)

    # test StringIO
    res = read_csv(StringIO('a,1,2\n,0,1\n'))
    assert_array_equal(res, ndtest('a=1,2'))

    # sort_columns=True
    res = read_csv(StringIO('a,a2,a0,a1\n,2,0,1\n'), sort_columns=True)
    assert_array_equal(res, ndtest(3))

    #################
    # narrow format #
    #################
    res = read_csv(inputpath('test1d_narrow.csv'), wide=False)
    assert_array_equal(res, io_1d)

    res = read_csv(inputpath('test2d_narrow.csv'), wide=False)
    assert_array_equal(res, io_2d)

    res = read_csv(inputpath('test3d_narrow.csv'), wide=False)
    assert_array_equal(res, io_3d)

    # missing values
    res = read_csv(inputpath('testmissing_values_narrow.csv'), wide=False)
    assert_array_nan_equal(res, io_narrow_missing_values)

    # unsorted values
    res = read_csv(inputpath('testunsorted_narrow.csv'), wide=False)
    assert_array_equal(res, io_unsorted)


def test_read_eurostat():
    la = read_eurostat(inputpath('test5d_eurostat.csv'))
    assert la.ndim == 5
    assert la.shape == (2, 5, 2, 2, 3)
    assert la.axes.names == ['arr', 'age', 'sex', 'nat', 'time']
    # FIXME: integer labels should be parsed as such
    assert_array_equal(la[X.arr['1'], '0', 'F', X.nat['1'], :],
                       [3722, 3395, 3347])


@needs_xlwings
def test_read_excel_xlwings():
    arr = read_excel(inputpath('test.xlsx'), '1d')
    assert_array_equal(arr, io_1d)

    arr = read_excel(inputpath('test.xlsx'), '2d')
    assert_array_equal(arr, io_2d)

    arr = read_excel(inputpath('test.xlsx'), '2d_classic')
    assert_array_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '2d_classic', nb_axes=2)
    assert_array_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '3d')
    assert_array_equal(arr, io_3d)

    # for > 2d, specifying nb_axes is required if there is no name for the horizontal axis
    arr = read_excel(inputpath('test.xlsx'), '3d_classic', nb_axes=3)
    assert_array_equal(arr, ndtest("a=1..3; b=b0,b1; c0..c2"))

    arr = read_excel(inputpath('test.xlsx'), 'int_labels')
    assert_array_equal(arr, io_int_labels)

    # passing a Group as sheet arg
    axis = Axis('dim=1d,2d,3d,5d')

    arr = read_excel(inputpath('test.xlsx'), axis['1d'])
    assert_array_equal(arr, io_1d)

    # missing rows, default fill_value
    arr = read_excel(inputpath('test.xlsx'), 'missing_values')
    expected = ndtest("a=1..3; b=b0,b1; c=c0..c2", dtype=float)
    expected[2, 'b0'] = nan
    expected[3, 'b1'] = nan
    assert_array_nan_equal(arr, expected)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test.xlsx'), 'missing_values', fill_value=42)
    expected = ndtest("a=1..3; b=b0,b1; c=c0..c2", dtype=float)
    expected[2, 'b0'] = 42
    expected[3, 'b1'] = 42
    assert_array_equal(arr, expected)

    # range
    arr = read_excel(inputpath('test.xlsx'), 'position', range='D3:H9')
    assert_array_equal(arr, io_3d)

    #################
    # narrow format #
    #################
    arr = read_excel(inputpath('test_narrow.xlsx'), '1d', wide=False)
    assert_array_equal(arr, io_1d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '2d', wide=False)
    assert_array_equal(arr, io_2d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '3d', wide=False)
    assert_array_equal(arr, io_3d)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test_narrow.xlsx'), 'missing_values', fill_value=42, wide=False)
    expected = io_narrow_missing_values.copy()
    expected[isnan(expected)] = 42
    assert_array_equal(arr, expected)

    # unsorted values
    arr = read_excel(inputpath('test_narrow.xlsx'), 'unsorted', wide=False)
    assert_array_equal(arr, io_unsorted)

    # range
    arr = read_excel(inputpath('test_narrow.xlsx'), 'position', range='D3:G21', wide=False)
    assert_array_equal(arr, io_3d)

    ##############################
    #  invalid keyword argument  #
    ##############################

    with pytest.raises(TypeError, match="'dtype' is an invalid keyword argument for this function "
                                        "when using the xlwings backend"):
        read_excel(inputpath('test.xlsx'), engine='xlwings', dtype=float)

    #################
    #  blank cells  #
    #################

    # Excel sheet with blank cells on right/bottom border of the array to read
    fpath = inputpath('test_blank_cells.xlsx')
    good = read_excel(fpath, 'good')
    bad1 = read_excel(fpath, 'blanksafter_morerowsthancols')
    bad2 = read_excel(fpath, 'blanksafter_morecolsthanrows')
    assert_array_equal(bad1, good)
    assert_array_equal(bad2, good)
    # with additional empty column in the middle of the array to read
    good2 = ndtest('a=a0,a1;b=2003..2006').astype(object)
    good2[2005] = None
    good2 = good2.set_axes('b', Axis([2003, 2004, None, 2006], 'b'))
    bad3 = read_excel(fpath, 'middleblankcol')
    bad4 = read_excel(fpath, '16384col')
    assert_array_equal(bad3, good2)
    assert_array_equal(bad4, good2)


@needs_openpyxl
def test_read_excel_pandas():
    arr = read_excel(inputpath('test.xlsx'), '1d', engine='openpyxl')
    assert_array_equal(arr, io_1d)

    arr = read_excel(inputpath('test.xlsx'), '2d', engine='openpyxl')
    assert_array_equal(arr, io_2d)

    arr = read_excel(inputpath('test.xlsx'), '2d', nb_axes=2, engine='openpyxl')
    assert_array_equal(arr, io_2d)

    arr = read_excel(inputpath('test.xlsx'), '2d_classic', engine='openpyxl')
    assert_array_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '2d_classic', nb_axes=2, engine='openpyxl')
    assert_array_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '3d', index_col=[0, 1], engine='openpyxl')
    assert_array_equal(arr, io_3d)

    arr = read_excel(inputpath('test.xlsx'), '3d', engine='openpyxl')
    assert_array_equal(arr, io_3d)

    # for > 2d, specifying nb_axes is required if there is no name for the horizontal axis
    arr = read_excel(inputpath('test.xlsx'), '3d_classic', nb_axes=3, engine='openpyxl')
    assert_array_equal(arr, ndtest("a=1..3; b=b0,b1; c0..c2"))

    arr = read_excel(inputpath('test.xlsx'), 'int_labels', engine='openpyxl')
    assert_array_equal(arr, io_int_labels)

    # passing a Group as sheet arg
    axis = Axis('dim=1d,2d,3d,5d')

    arr = read_excel(inputpath('test.xlsx'), axis['1d'], engine='openpyxl')
    assert_array_equal(arr, io_1d)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test.xlsx'), 'missing_values', fill_value=42, engine='openpyxl')
    expected = io_missing_values.copy()
    expected[isnan(expected)] = 42
    assert_array_equal(arr, expected)

    #################
    # narrow format #
    #################
    arr = read_excel(inputpath('test_narrow.xlsx'), '1d', wide=False, engine='openpyxl')
    assert_array_equal(arr, io_1d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '2d', wide=False, engine='openpyxl')
    assert_array_equal(arr, io_2d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '3d', wide=False, engine='openpyxl')
    assert_array_equal(arr, io_3d)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test_narrow.xlsx'), 'missing_values',
                     fill_value=42, wide=False, engine='openpyxl')
    expected = io_narrow_missing_values.copy()
    expected[isnan(expected)] = 42
    assert_array_equal(arr, expected)

    # unsorted values
    arr = read_excel(inputpath('test_narrow.xlsx'), 'unsorted', wide=False, engine='openpyxl')
    assert_array_equal(arr, io_unsorted)


def test_from_lists():
    simple_arr = ndtest((2, 2, 3))

    # simple
    arr_list = [['a', 'b\\c', 'c0', 'c1', 'c2'],
                ['a0', 'b0', 0, 1, 2],
                ['a0', 'b1', 3, 4, 5],
                ['a1', 'b0', 6, 7, 8],
                ['a1', 'b1', 9, 10, 11]]
    res = from_lists(arr_list)
    assert_array_equal(res, simple_arr)

    # simple (using dump). This should be the same test than above.
    # We just make sure dump() and from_lists() round-trip correctly.
    arr_list = simple_arr.dump()
    res = from_lists(arr_list)
    assert_array_equal(res, simple_arr)

    # with anonymous axes
    arr_anon = simple_arr.rename({0: None, 1: None, 2: None})
    arr_list = arr_anon.dump()
    assert arr_list == [[None, None, 'c0', 'c1', 'c2'],
                        ['a0', 'b0',    0,    1,    2],
                        ['a0', 'b1',    3,    4,    5],
                        ['a1', 'b0',    6,    7,    8],
                        ['a1', 'b1',    9,   10,   11]]
    res = from_lists(arr_list, nb_axes=3)
    assert_array_equal(res, arr_anon)

    # with empty ('') axes names
    arr_empty_names = simple_arr.rename({0: '', 1: '', 2: ''})
    arr_list = arr_empty_names.dump()
    assert arr_list == [[  '',   '', 'c0', 'c1', 'c2'],
                        ['a0', 'b0',    0,    1,    2],
                        ['a0', 'b1',    3,    4,    5],
                        ['a1', 'b0',    6,    7,    8],
                        ['a1', 'b1',    9,   10,   11]]
    res = from_lists(arr_list, nb_axes=3)
    # this is purposefully NOT arr_empty_names because from_lists (via df_asarray) transforms '' axes to None
    assert_array_equal(res, arr_anon)

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


def test_from_series():
    # Series with Index as index
    expected = ndtest(3)
    s = pd.Series([0, 1, 2], index=pd.Index(['a0', 'a1', 'a2'], name='a'))
    assert_array_equal(from_series(s), expected)

    s = pd.Series([2, 0, 1], index=pd.Index(['a2', 'a0', 'a1'], name='a'))
    assert_array_equal(from_series(s, sort_rows=True), expected)

    expected = ndtest(3)[['a2', 'a0', 'a1']]
    assert_array_equal(from_series(s), expected)

    # Series with MultiIndex as index
    age = Axis('age=0..3')
    gender = Axis('gender=M,F')
    time = Axis('time=2015..2017')
    expected = ndtest((age, gender, time))

    index = pd.MultiIndex.from_product(expected.axes.labels, names=expected.axes.names)
    data = expected.data.flatten()
    s = pd.Series(data, index)

    res = from_series(s)
    assert_array_equal(res, expected)

    res = from_series(s, sort_rows=True)
    assert_array_equal(res, expected.sort_axes())

    expected[0, 'F'] = -1
    s = s.reset_index().drop([3, 4, 5]).set_index(['age', 'gender', 'time'])[0]
    res = from_series(s, fill_value=-1)
    assert_array_equal(res, expected)


def test_from_frame():
    # 1) data = scalar
    # ================
    # Dataframe becomes 1D Array
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
    # output Array:
    # -------------
    # {0}\{1}  c0
    #      i0  10
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, 1)
    assert la.axes.names == [None, None]
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, 1)), [axis_index, axis_columns])
    assert_array_equal(la, expected_la)

    # anonymous columns
    # input dataframe:
    # ----------------
    #        c0
    # index
    # i0     10
    # output Array:
    # -------------
    # index\{1}  c0
    #        i0  10
    df.index.name, df.columns.name = 'index', None
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, 1)
    assert la.axes.names == ['index', None]
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, 1)), [axis_index.rename('index'), axis_columns])
    assert_array_equal(la, expected_la)

    # anonymous columns/non string row axis name
    # input dataframe:
    # ----------------
    #        c0
    # 0
    # i0     10
    # output Array:
    # -------------
    # 0\{1}  c0
    #    i0  10
    df = pd.DataFrame([10], index=pd.Index(['i0'], name=0), columns=['c0'])
    res = from_frame(df)
    expected = Array([[10]], [Axis(['i0'], name=0), Axis(['c0'])])
    assert res.ndim == 2
    assert res.shape == (1, 1)
    assert res.axes.names == [0, None]
    assert_array_equal(res, expected)

    # anonymous index
    # input dataframe:
    # ----------------
    # columns  c0
    # i0       10
    # output Array:
    # -------------
    # {0}\columns  c0
    #          i0  10
    df.index.name, df.columns.name = None, 'columns'
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, 1)
    assert la.axes.names == [None, 'columns']
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, 1)), [axis_index, axis_columns.rename('columns')])
    assert_array_equal(la, expected_la)

    # index and columns with name
    # input dataframe:
    # ----------------
    # columns  c0
    # index
    # i0       10
    # output Array:
    # -------------
    # index\columns  c0
    #            i0  10
    df.index.name, df.columns.name = 'index', 'columns'
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, 1)
    assert la.axes.names == ['index', 'columns']
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, 1)), [axis_index.rename('index'), axis_columns.rename('columns')])
    assert_array_equal(la, expected_la)

    # 2) data = vector
    # ================
    size = 3

    # 2A) data = horizontal vector (1 x N)
    # ====================================
    # Dataframe becomes 1D Array
    data = np.arange(size)
    indexes = ['i0']
    columns = [f'c{i}' for i in range(size)]
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
    # output Array:
    # -------------
    # {0}\{1}  c0  c1  c2
    #      i0   0   1   2
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, size)
    assert la.axes.names == [None, None]
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, size)), [axis_index, axis_columns])
    assert_array_equal(la, expected_la)

    # anonymous columns
    # input dataframe:
    # ----------------
    #        c0  c1  c2
    # index
    # i0      0   1   2
    # output Array:
    # -------------
    # index\{1}  c0  c1  c2
    #        i0   0   1   2
    df.index.name, df.columns.name = 'index', None
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, size)
    assert la.axes.names == ['index', None]
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, size)), [axis_index.rename('index'), axis_columns])
    assert_array_equal(la, expected_la)

    # anonymous index
    # input dataframe:
    # ----------------
    # columns  c0  c1  c2
    # i0        0   1   2
    # output Array:
    # -------------
    # {0}\columns  c0  c1  c2
    #          i0   0   1   2
    df.index.name, df.columns.name = None, 'columns'
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, size)
    assert la.axes.names == [None, 'columns']
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, size)), [axis_index, axis_columns.rename('columns')])
    assert_array_equal(la, expected_la)

    # index and columns with name
    # input dataframe:
    # ----------------
    # columns  c0  c1  c2
    # index
    # i0        0   1   2
    # output Array:
    # -------------
    # index\columns  c0  c1  c2
    #            i0   0   1   2
    df.index.name, df.columns.name = 'index', 'columns'
    la = from_frame(df)
    assert la.ndim == 2
    assert la.shape == (1, size)
    assert la.axes.names == ['index', 'columns']
    assert list(la.axes.labels[0]) == index
    assert list(la.axes.labels[1]) == columns
    expected_la = Array(data.reshape((1, size)), [axis_index.rename('index'), axis_columns.rename('columns')])
    assert_array_equal(la, expected_la)

    # 2B) data = vertical vector (N x 1)
    # ==================================
    # Dataframe becomes 2D Array
    data = data.reshape(size, 1)
    indexes = [f'i{i}' for i in range(size)]
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
    # output Array:
    # -------------
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
    expected_la = Array(data, [axis_index, axis_columns])
    assert_array_equal(la, expected_la)

    # anonymous columns
    # input dataframe:
    # ----------------
    #        c0
    # index
    # i0      0
    # i1      1
    # i2      2
    # output Array:
    # -------------
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
    expected_la = Array(data, [axis_index.rename('index'), axis_columns])
    assert_array_equal(la, expected_la)

    # anonymous index
    # input dataframe:
    # ----------------
    # columns  c0
    # i0        0
    # i1        1
    # i2        2
    # output Array:
    # -------------
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
    expected_la = Array(data, [axis_index, axis_columns.rename('columns')])
    assert_array_equal(la, expected_la)

    # index and columns with name
    # input dataframe:
    # ----------------
    # columns  c0
    # index
    # i0        0
    # i1        1
    # i2        2
    # output Array:
    # -------------
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
    expected_la = Array(data, [axis_index, axis_columns.rename('columns')])
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

    # 3C) Dataframe with no axe names (names are None)
    # ===============================
    arr_no_names = ndtest("a0,a1;b0..b2;c0..c3")
    df_no_names = arr_no_names.df
    res = from_frame(df_no_names)
    assert_array_equal(res, arr_no_names)

    # 3D) Dataframe with empty axe names (names are '')
    # ==================================
    arr_empty_names = ndtest("=a0,a1;=b0..b2;=c0..c3")
    assert arr_empty_names.axes.names == ['', '', '']
    df_no_names = arr_empty_names.df
    res = from_frame(df_no_names)
    assert_array_equal(res, arr_empty_names)

    # 4) test sort_rows and sort_columns arguments
    # ============================================
    age = Axis('age=2,0,1,3')
    gender = Axis('gender=M,F')
    time = Axis('time=2016,2015,2017')
    columns = pd.Index(time.labels, name=time.name)

    # df.index is an Index instance
    expected = ndtest((gender, time))
    index = pd.Index(gender.labels, name=gender.name)
    data = expected.data
    df = pd.DataFrame(data, index=index, columns=columns)

    expected = expected.sort_axes()
    res = from_frame(df, sort_rows=True, sort_columns=True)
    assert_array_equal(res, expected)

    # df.index is a MultiIndex instance
    expected = ndtest((age, gender, time))
    index = pd.MultiIndex.from_product(expected.axes[:-1].labels, names=expected.axes[:-1].names)
    data = expected.data.reshape(len(age) * len(gender), len(time))
    df = pd.DataFrame(data, index=index, columns=columns)

    res = from_frame(df, sort_rows=True, sort_columns=True)
    assert_array_equal(res, expected.sort_axes())

    # 5) test fill_value
    # ==================
    expected[0, 'F'] = -1
    df = df.reset_index().drop([3]).set_index(['age', 'gender'])
    res = from_frame(df, fill_value=-1)
    assert_array_equal(res, expected)


def test_to_csv(tmpdir):
    arr = io_3d.copy()

    arr.to_csv(tmp_path(tmpdir, 'out.csv'))
    result = ['a,b\\c,c0,c1,c2\n',
              '1,b0,0,1,2\n',
              '1,b1,3,4,5\n']
    with open(tmp_path(tmpdir, 'out.csv')) as f:
        assert f.readlines()[:3] == result

    # stacked data (one column containing all the values and another column listing the context of the value)
    arr.to_csv(tmp_path(tmpdir, 'out.csv'), wide=False)
    result = ['a,b,c,value\n',
              '1,b0,c0,0\n',
              '1,b0,c1,1\n']
    with open(tmp_path(tmpdir, 'out.csv')) as f:
        assert f.readlines()[:3] == result

    arr = io_1d.copy()
    arr.to_csv(tmp_path(tmpdir, 'test_out1d.csv'))
    result = ['a,a0,a1,a2\n',
              ',0,1,2\n']
    with open(tmp_path(tmpdir, 'test_out1d.csv')) as f:
        assert f.readlines() == result


@needs_xlsxwriter
@needs_openpyxl
def test_to_excel_xlsxwriter(tmpdir):
    fpath = tmp_path(tmpdir, 'test_to_excel_xlsxwriter.xlsx')

    # 1D
    a1 = ndtest(3)

    # fpath/Sheet1/A1
    a1.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # fpath/Sheet1/A1(transposed)
    a1.to_excel(fpath, transpose=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # fpath/Sheet1/A1
    # stacked data (one column containing all the values and another column listing the context of the value)
    a1.to_excel(fpath, wide=False, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    stacked_a1 = a1.reshape([a1.a, Axis(['value'])])
    assert_array_equal(res, stacked_a1)

    # 2D
    a2 = ndtest((2, 3))

    # fpath/Sheet1/A1
    a2.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a2)

    # fpath/Sheet1/A10
    # TODO: this is currently not supported (though we would only need to translate A10 to startrow=0 and startcol=0
    # a2.to_excel('fpath', 'Sheet1', 'A10', engine='xlsxwriter')
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=9)
    # assert_array_equal(res, a2)

    # fpath/other/A1
    a2.to_excel(fpath, 'other', engine='xlsxwriter')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_array_equal(res, a2)

    # 3D
    a3 = ndtest((2, 3, 4))

    # fpath/Sheet1/A1
    # FIXME: merge_cells=False should be the default (until Pandas is fixed to read its format)
    a3.to_excel(fpath, overwrite_file=True, engine='xlsxwriter', merge_cells=False)
    # a3.to_excel('fpath', overwrite_file=True, engine='openpyxl')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a3)

    # fpath/Sheet1/A20
    # TODO: implement position (see above)
    # a3.to_excel('fpath', 'Sheet1', 'A20', engine='xlsxwriter', merge_cells=False)
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=19)
    # assert_array_equal(res, a3)

    # fpath/other/A1
    a3.to_excel(fpath, 'other', engine='xlsxwriter', merge_cells=False)
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_array_equal(res, a3)

    # 1D
    a1 = ndtest(3)

    # fpath/Sheet1/A1
    a1.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # fpath/Sheet1/A1(transposed)
    a1.to_excel(fpath, transpose=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # fpath/Sheet1/A1
    # stacked data (one column containing all the values and another column listing the context of the value)
    a1.to_excel(fpath, wide=False, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    stacked_a1 = a1.reshape([a1.a, Axis(['value'])])
    assert_array_equal(res, stacked_a1)

    # 2D
    a2 = ndtest((2, 3))

    # fpath/Sheet1/A1
    a2.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a2)

    # fpath/Sheet1/A10
    # TODO: this is currently not supported (though we would only need to translate A10 to startrow=0 and startcol=0
    # a2.to_excel(fpath, 'Sheet1', 'A10', engine='xlsxwriter')
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=9)
    # assert_array_equal(res, a2)

    # fpath/other/A1
    a2.to_excel(fpath, 'other', engine='xlsxwriter')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_array_equal(res, a2)

    # 3D
    a3 = ndtest((2, 3, 4))

    # fpath/Sheet1/A1
    # FIXME: merge_cells=False should be the default (until Pandas is fixed to read its format)
    a3.to_excel(fpath, overwrite_file=True, engine='xlsxwriter', merge_cells=False)
    # a3.to_excel('fpath', overwrite_file=True, engine='openpyxl')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a3)

    # fpath/Sheet1/A20
    # TODO: implement position (see above)
    # a3.to_excel('fpath', 'Sheet1', 'A20', engine='xlsxwriter', merge_cells=False)
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=19)
    # assert_array_equal(res, a3)

    # fpath/other/A1
    a3.to_excel(fpath, 'other', engine='xlsxwriter', merge_cells=False)
    res = read_excel(fpath, 'other', engine='openpyxl')
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
    group = a3.c['c0,c2'] >> r':name?with*special/\[char]'
    a3[group].to_excel(fpath, group, engine='xlsxwriter')


@needs_xlwings
@needs_openpyxl
def test_to_excel_xlwings(tmpdir):
    fpath = tmp_path(tmpdir, 'test_to_excel_xlwings.xlsx')

    # 1D
    a1 = ndtest(3)

    # live book/Sheet1/A1
    # a1.to_excel()

    # fpath/Sheet1/A1 (create a new file if does not exist)
    if os.path.isfile(fpath):
        os.remove(fpath)
    a1.to_excel(fpath, engine='xlwings')
    # we use openpyxl to read back instead of xlwings even if that should work, to make the test faster
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # fpath/Sheet1/A1(transposed)
    a1.to_excel(fpath, transpose=True, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # fpath/Sheet1/A1
    # stacked data (one column containing all the values and another column listing the context of the value)
    a1.to_excel(fpath, wide=False, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a1)

    # 2D
    a2 = ndtest((2, 3))

    # fpath/Sheet1/A1
    a2.to_excel(fpath, overwrite_file=True, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a2)

    # fpath/Sheet1/A10
    a2.to_excel(fpath, 'Sheet1', 'A10', engine='xlwings')
    res = read_excel(fpath, 'Sheet1', engine='openpyxl', skiprows=9)
    assert_array_equal(res, a2)

    # fpath/other/A1
    a2.to_excel(fpath, 'other', engine='xlwings')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_array_equal(res, a2)

    # transpose
    a2.to_excel(fpath, 'transpose', transpose=True, engine='xlwings')
    res = read_excel(fpath, 'transpose', engine='openpyxl')
    assert_array_equal(res, a2.T)

    # 3D
    a3 = ndtest((2, 3, 4))

    # fpath/Sheet1/A1
    a3.to_excel(fpath, overwrite_file=True, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_array_equal(res, a3)

    # fpath/Sheet1/A20
    a3.to_excel(fpath, 'Sheet1', 'A20', engine='xlwings')
    res = read_excel(fpath, 'Sheet1', engine='openpyxl', skiprows=19)
    assert_array_equal(res, a3)

    # fpath/other/A1
    a3.to_excel(fpath, 'other', engine='xlwings')
    res = read_excel(fpath, 'other', engine='openpyxl')
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
    group = a3.c['c0,c2'] >> r':name?with*special/\[char]'
    a3[group].to_excel(fpath, group, engine='xlwings')
    # checks sheet names
    sheet_names = sorted(open_excel(fpath).sheet_names())
    assert sheet_names == sorted(['a0', 'a1', 'a2', 'a3', 'c0,c2', 'c0__2', 'even',
                                  '_name_with_special___char_'])

    # sheet name of 31 characters (= maximum authorized length)
    a3.to_excel(fpath, "sheetname_of_exactly_31_chars__", engine='xlwings')
    # sheet name longer than 31 characters
    with pytest.raises(ValueError, match="Sheet names cannot exceed 31 characters"):
        a3.to_excel(fpath, "sheetname_longer_than_31_characters", engine='xlwings')


def test_dump():
    # narrow format
    res = list(ndtest(3).dump(wide=False, value_name='data'))
    assert res == [['a', 'data'],
                   ['a0', 0],
                   ['a1', 1],
                   ['a2', 2]]
    # array with an anonymous axis and a wildcard axis
    arr = ndtest((Axis('a0,a1'), Axis(2, 'b')))
    res = arr.dump()
    assert res == [['\\b', 0, 1],
                   ['a0', 0, 1],
                   ['a1', 2, 3]]
    res = arr.dump(_axes_display_names=True)
    assert res == [['{0}\\b*', 0, 1],
                   ['a0', 0, 1],
                   ['a1', 2, 3]]


@needs_xlwings
def test_open_excel(tmpdir):
    # 1) Create new file
    # ==================
    fpath = inputpath('should_not_exist.xlsx')
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
        res = wb['3D']['A20:F26'].load(nb_axes=3)
        assert_array_equal(res, a3.data)
        # the two first axes should be the same
        assert res.axes[:2] == a3.axes[:2]
        # the third axis should have the same labels (but not the same name obviously)
        assert_array_equal(res.axes[2].labels, a3.axes[2].labels)

    with open_excel(inputpath('test.xlsx')) as wb:
        expected = ndtest("a=a0..a2; b0..b2")
        res = wb['2d_classic'].load()
        assert_array_equal(res, expected)

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
        # FIXME: we need to .dump(header=False) explicitly because otherwise we go via ArrayConverter which
        #        includes labels. for consistency's sake we should either change ArrayConverter to not include
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
    # ==============
    # Excel sheet with blank cells on right/bottom border of the array to read
    fpath = inputpath('test_blank_cells.xlsx')
    with open_excel(fpath) as wb:
        good = wb['good'].load()
        bad1 = wb['blanksafter_morerowsthancols'].load()
        bad2 = wb['blanksafter_morecolsthanrows'].load()
        # with additional empty column in the middle of the array to read
        good2 = wb['middleblankcol']['A1:E3'].load()
        bad3 = wb['middleblankcol'].load()
        bad4 = wb['16384col'].load()
    assert_array_equal(bad1, good)
    assert_array_equal(bad2, good)
    assert_array_equal(bad3, good2)
    assert_array_equal(bad4, good2)

    # 5) anonymous and wilcard axes
    # =============================
    arr = ndtest((Axis('a0,a1'), Axis(2, 'b')))
    fpath = tmp_path(tmpdir, 'anonymous_and_wildcard_axes.xlsx')
    with open_excel(fpath, overwrite_file=True) as wb:
        wb[0] = arr.dump()
        res = wb[0].load()
        # the result should be identical to the original array except we lost the information about
        # the wildcard axis being a wildcard axis
        expected = arr.set_axes('b', Axis([0, 1], 'b'))
        assert_array_equal(res, expected)

    # 6) crash test
    # =============
    arr = ndtest((2, 2))
    fpath = tmp_path(tmpdir, 'temporary_test_file.xlsx')
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


def test_ufuncs(small_array):
    raw = small_array.data

    # simple one-argument ufunc
    assert_array_equal(exp(small_array), np.exp(raw))

    # with out=
    la_out = zeros(small_array.axes)
    raw_out = np.zeros(raw.shape)

    la_out2 = exp(small_array, la_out)
    raw_out2 = np.exp(raw, raw_out)

    # FIXME: this is not the case currently
    # self.assertIs(la_out2, la_out)
    assert_array_equal(la_out2, la_out)
    assert raw_out2 is raw_out

    assert_array_equal(la_out, raw_out)

    # with out= and broadcasting
    # we need to put the 'a' axis first because array numpy only supports that
    la_out = zeros([Axis([0, 1, 2], 'a')] + list(small_array.axes))
    raw_out = np.zeros((3,) + raw.shape)

    la_out2 = exp(small_array, la_out)
    raw_out2 = np.exp(raw, raw_out)

    # self.assertIs(la_out2, la_out)
    # XXX: why is la_out2 transposed?
    assert_array_equal(la_out2.transpose(X.a), la_out)
    assert raw_out2 is raw_out

    assert_array_equal(la_out, raw_out)

    sex, lipro = small_array.axes

    low = small_array.sum(sex) // 4 + 3
    raw_low = raw.sum(0) // 4 + 3
    high = small_array.sum(sex) // 4 + 13
    raw_high = raw.sum(0) // 4 + 13

    # LA + scalars
    assert_array_equal(small_array.clip(0, 10), raw.clip(0, 10))
    assert_array_equal(clip(small_array, 0, 10), np.clip(raw, 0, 10))

    # LA + LA (no broadcasting)
    assert_array_equal(clip(small_array, 21 - small_array, 9 + small_array // 2),
                       np.clip(raw, 21 - raw, 9 + raw // 2))

    # LA + LA (with broadcasting)
    assert_array_equal(clip(small_array, low, high),
                       np.clip(raw, raw_low, raw_high))

    # where (no broadcasting)
    assert_array_equal(where(small_array < 5, -5, small_array),
                       np.where(raw < 5, -5, raw))

    # where (transposed no broadcasting)
    assert_array_equal(where(small_array < 5, -5, small_array.T),
                       np.where(raw < 5, -5, raw))

    # where (with broadcasting)
    result = where(small_array['P01'] < 5, -5, small_array)
    assert result.axes.names == ['sex', 'lipro']
    assert_array_equal(result, np.where(raw[:, [0]] < 5, -5, raw))

    # round
    small_float = small_array + 0.6
    rounded = round(small_float)
    assert_array_equal(rounded, np.round(raw + 0.6))


def test_diag():
    # 2D -> 1D
    a = ndtest((3, 3))
    d = diag(a)
    assert d.ndim == 1
    assert d.i[0] == a.i[0, 0]
    assert d.i[1] == a.i[1, 1]
    assert d.i[2] == a.i[2, 2]

    # 1D -> 2D
    a2 = diag(d)
    assert a2.ndim == 2
    assert a2.i[0, 0] == a.i[0, 0]
    assert a2.i[1, 1] == a.i[1, 1]
    assert a2.i[2, 2] == a.i[2, 2]

    # 3D -> 2D
    a = ndtest((3, 3, 3))
    d = diag(a)
    assert d.ndim == 2
    assert d.i[0, 0] == a.i[0, 0, 0]
    assert d.i[1, 1] == a.i[1, 1, 1]
    assert d.i[2, 2] == a.i[2, 2, 2]

    # 3D -> 1D
    d = diag(a, axes=(0, 1, 2))
    assert d.ndim == 1
    assert d.i[0] == a.i[0, 0, 0]
    assert d.i[1] == a.i[1, 1, 1]
    assert d.i[2] == a.i[2, 2, 2]

    # 1D (anon) -> 2D
    d_anon = d.rename(0, None).ignore_labels()
    a2 = diag(d_anon)
    assert a2.ndim == 2

    # 1D (anon) -> 3D
    a3 = diag(d_anon, ndim=3)
    assert a2.ndim == 2
    assert a3.i[0, 0, 0] == a.i[0, 0, 0]
    assert a3.i[1, 1, 1] == a.i[1, 1, 1]
    assert a3.i[2, 2, 2] == a.i[2, 2, 2]

    # using Axis object
    sex = Axis('sex=M,F')
    a = eye(sex)
    d = diag(a)
    assert d.ndim == 1
    assert d.axes.names == ['sex_sex']
    assert_array_equal(d.axes.labels, [['M_M', 'F_F']])
    assert d.i[0] == 1.0
    assert d.i[1] == 1.0


def test_matmul():
    # 2D / anonymous axes
    a1 = ndtest([Axis(3), Axis(3)])
    a2 = eye(3, 3) * 2

    # Array value
    assert_array_equal(a1 @ a2, ndtest([Axis(3), Axis(3)]) * 2)

    # ndarray value
    assert_array_equal(a1 @ a2.data, ndtest([Axis(3), Axis(3)]) * 2)

    # non anonymous axes (N <= 2)
    arr1d = ndtest(3)
    arr2d = ndtest((3, 3))

    # 1D @ 1D
    res = arr1d @ arr1d
    assert isinstance(res, np.integer)
    assert res == 5

    # 1D @ 2D
    assert_array_equal(arr1d @ arr2d,
                       Array([15, 18, 21], 'b=b0..b2'))

    # 2D @ 1D
    assert_array_equal(arr2d @ arr1d,
                       Array([5, 14, 23], 'a=a0..a2'))

    # 2D(a,b) @ 2D(a,b) -> 2D(a,b)
    res = from_lists([['a\\b', 'b0', 'b1', 'b2'],
                      ['a0', 15, 18, 21],
                      ['a1', 42, 54, 66],
                      ['a2', 69, 90, 111]])
    assert_array_equal(arr2d @ arr2d, res)

    # 2D(a,b) @ 2D(b,a) -> 2D(a,a)
    res = from_lists([['a\\a', 'a0', 'a1', 'a2'],
                      ['a0', 5, 14, 23],
                      ['a1', 14, 50, 86],
                      ['a2', 23, 86, 149]])
    assert_array_equal(arr2d @ arr2d.T, res)

    # ndarray value
    assert_array_equal(arr1d @ arr2d.data, Array([15, 18, 21]))
    assert_array_equal(arr2d.data @ arr2d.T.data, res.data)

    # different axes
    a1 = ndtest('a=a0..a1;b=b0..b2')
    a2 = ndtest('b=b0..b2;c=c0..c3')
    res = from_lists([[r'a\c', 'c0', 'c1', 'c2', 'c3'],
                      ['a0', 20, 23, 26, 29],
                      ['a1', 56, 68, 80, 92]])
    assert_array_equal(a1 @ a2, res)

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
    assert_array_equal(arr4d @ arr3d, res)

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
    assert_array_equal(arr3d @ arr4d, res)

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
    assert_array_equal(arr4d @ arr3d, res)

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
    assert_array_equal(arr3d @ arr4d, res)

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
    assert_array_equal(arr4d @ arr2d, res)

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
    assert_array_equal(arr2d @ arr4d, res)


def test_rmatmul():
    a1 = eye(3) * 2
    a2 = ndtest([Axis(3), Axis(3)])

    # equivalent to a1.data @ a2
    res = a2.__rmatmul__(a1.data)
    assert isinstance(res, Array)
    assert_array_equal(res, ndtest([Axis(3), Axis(3)]) * 2)


def test_broadcast_with():
    a1 = ndtest((3, 2))
    a2 = ndtest(3)
    b = a2.broadcast_with(a1)
    assert b.ndim == a1.ndim
    assert b.shape == (3, 1)
    assert_array_equal(b.i[:, 0], a2)

    # anonymous axes
    a1 = ndtest([Axis(3), Axis(2)])
    a2 = ndtest(Axis(3))
    b = a2.broadcast_with(a1)
    assert b.ndim == a1.ndim
    assert b.shape == (3, 1)
    assert_array_equal(b.i[:, 0], a2)

    a1 = ndtest([Axis(1), Axis(3)])
    a2 = ndtest([Axis(3), Axis(1)])
    b = a2.broadcast_with(a1)
    assert b.ndim == 2
    # common axes are reordered according to target (a1 in this case)
    assert b.shape == (1, 3)
    assert_larray_equiv(b, a2)

    a1 = ndtest([Axis(2), Axis(3)])
    a2 = ndtest([Axis(3), Axis(2)])
    b = a2.broadcast_with(a1)
    assert b.ndim == 2
    assert b.shape == (2, 3)
    assert_larray_equiv(b, a2)


def test_plot():
    pass
    # small_h = small['M']
    # small_h.plot(kind='bar')
    # small_h.plot()
    # small_h.hist()

    # large_data = np.random.randn(1000)
    # tick_v = np.random.randint(ord('a'), ord('z'), size=1000)
    # ticks = [chr(c) for c in tick_v]
    # large_axis = Axis('large', ticks)
    # large = Array(large_data, axes=[large_axis])
    # large.plot()
    # large.hist()


def test_combine_axes():
    # combine N axes into 1
    # =====================
    arr = ndtest((2, 3, 4, 5))
    res = arr.combine_axes((X.a, X.b))
    assert res.axes.names == ['a_b', 'c', 'd']
    assert res.size == arr.size
    assert res.shape == (2 * 3, 4, 5)
    assert_array_equal(res.axes.a_b.labels[:2], ['a0_b0', 'a0_b1'])
    assert_array_equal(res['a1_b0'], arr['a1', 'b0'])

    res = arr.combine_axes((X.a, X.c))
    assert res.axes.names == ['a_c', 'b', 'd']
    assert res.size == arr.size
    assert res.shape == (2 * 4, 3, 5)
    assert_array_equal(res.axes.a_c.labels[:2], ['a0_c0', 'a0_c1'])
    assert_array_equal(res['a1_c0'], arr['a1', 'c0'])

    res = arr.combine_axes((X.b, X.d))
    assert res.axes.names == ['a', 'b_d', 'c']
    assert res.size == arr.size
    assert res.shape == (2, 3 * 5, 4)
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

    # combine with wildcard=True
    arr = ndtest((2, 3))
    res = arr.combine_axes(wildcard=True)
    assert res.axes.names == ['a_b']
    assert res.size == arr.size
    assert res.shape == (6,)
    assert_array_equal(res.axes[0].labels, np.arange(6))


def test_split_axes():
    # split one axis
    # ==============

    # default sep
    arr = ndtest((2, 3, 4, 5))
    combined = arr.combine_axes(('b', 'd'))
    assert combined.axes.names == ['a', 'b_d', 'c']
    res = combined.split_axes('b_d')
    assert res.axes.names == ['a', 'b', 'd', 'c']
    assert res.shape == (2, 3, 5, 4)
    assert_array_equal(res.transpose('a', 'b', 'c', 'd'), arr)

    # with specified names
    res = combined.rename(b_d='bd').split_axes('bd', names=('b', 'd'))
    assert res.axes.names == ['a', 'b', 'd', 'c']
    assert res.shape == (2, 3, 5, 4)
    assert_array_equal(res.transpose('a', 'b', 'c', 'd'), arr)

    # regex
    res = combined.split_axes('b_d', names=['b', 'd'], regex=r'(\w+)_(\w+)')
    assert res.axes.names == ['a', 'b', 'd', 'c']
    assert res.shape == (2, 3, 5, 4)
    assert_array_equal(res.transpose('a', 'b', 'c', 'd'), arr)

    # custom sep
    combined = ndtest('a|b=a0|b0,a0|b1')
    res = combined.split_axes(sep='|')
    assert_array_equal(res, ndtest('a=a0;b=b0,b1'))

    # split several axes at once
    # ==========================
    arr = ndtest('a_b=a0_b0..a1_b2; c=c0..c3; d=d0..d3; e_f=e0_f0..e2_f1')

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
    arr = ndtest('a_b_c=a0_b0_c0..a1_b2_c3; d=d0..d3; e_f=e0_f0..e2_f1')
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
    arr = ndtest('ab=a0b0..a1b2; c=c0..c3; d=d0..d3; ef=e0f0..e2f1')
    res = arr.split_axes({'ab': ('a', 'b'), 'ef': ('e', 'f')}, regex=r'(\w{2})(\w{2})')
    assert res.axes.names == ['a', 'b', 'c', 'd', 'e', 'f']
    assert res.size == arr.size
    assert res.shape == (2, 3, 4, 4, 3, 2)
    assert list(res.axes.a.labels) == ['a0', 'a1']
    assert list(res.axes.b.labels) == ['b0', 'b1', 'b2']
    assert list(res.axes.e.labels) == ['e0', 'e1', 'e2']
    assert list(res.axes.f.labels) == ['f0', 'f1']
    assert res['a0', 'b1', 'c2', 'd3', 'e2', 'f1'] == arr['a0b1', 'c2', 'd3', 'e2f1']

    # labels with object dtype
    arr = ndtest((2, 2, 2)).combine_axes(('a', 'b'))
    arr = arr.set_axes([a.astype(object) for a in arr.axes])

    res = arr.split_axes()
    expected_kind = 'U' if sys.version_info[0] >= 3 else 'S'
    assert res.a.labels.dtype.kind == expected_kind
    assert res.b.labels.dtype.kind == expected_kind
    assert res.c.labels.dtype.kind == 'O'
    assert_array_equal(res, ndtest((2, 2, 2)))

    # not sorted by first part then second part (issue #364)
    arr = ndtest((2, 3))
    combined = arr.combine_axes()['a0_b0, a1_b0, a0_b1, a1_b1, a0_b2, a1_b2']
    assert_array_equal(combined.split_axes('a_b'), arr)

    # another weirdly sorted test
    combined = arr.combine_axes()['a0_b1, a0_b0, a0_b2, a1_b1, a1_b0, a1_b2']
    assert_array_equal(combined.split_axes('a_b'), arr['b1,b0,b2'])

    # combined does not contain all combinations of labels (issue #369)
    combined_partial = combined[['a0_b0', 'a0_b1', 'a1_b1', 'a0_b2', 'a1_b2']]
    expected = arr.astype(float)
    expected['a1', 'b0'] = nan
    assert_array_nan_equal(combined_partial.split_axes('a_b'), expected)

    # split labels are ambiguous (issue #485)
    combined = ndtest('a_b=a0_b0..a1_b1;c_d=a0_b0..a1_b1')
    expected = ndtest('a=a0,a1;b=b0,b1;c=a0,a1;d=b0,b1')
    assert_array_equal(combined.split_axes(('a_b', 'c_d')), expected)

    # anonymous axes
    combined = ndtest('a0_b0,a0_b1,a0_b2,a1_b0,a1_b1,a1_b2')
    expected = ndtest('a0,a1;b0,b1,b2')
    assert_array_equal(combined.split_axes(0), expected)

    # when no axis is specified and no axis contains the sep, split_axes is a no-op.
    assert_array_equal(combined.split_axes(), combined)


def test_stack():
    # stack along a single axis
    # =========================

    # simple
    a = Axis('a=a0,a1,a2')
    b = Axis('b=b0,b1')

    arr0 = ndtest(a)
    arr1 = ndtest(a, start=-1)

    res = stack((arr0, arr1), b)
    expected = Array([[0, -1],
                      [1,  0],
                      [2,  1]], [a, b])
    assert_array_equal(res, expected)

    # same but using a group as the stacking axis
    larger_b = Axis('b=b0..b3')
    res = stack((arr0, arr1), larger_b[:'b1'])
    assert_array_equal(res, expected)

    # simple with anonymous axis
    axis0 = Axis(3)
    arr0 = ndtest(axis0)
    arr1 = ndtest(axis0, start=-1)
    res = stack((arr0, arr1), b)
    expected = Array([[0, -1],
                      [1,  0],
                      [2,  1]], [axis0, b])
    assert_array_equal(res, expected)

    # using res_axes
    res = stack({'b0': 0, 'b1': 1}, axes=b, res_axes=(a, b))
    expected = Array([[0, 1],
                      [0, 1],
                      [0, 1]], [a, b])
    assert_array_equal(res, expected)

    # giving elements as an Array containing Arrays
    sex = Axis('sex=M,F')
    # not using the same length for nat and type, otherwise numpy gets confused :(
    arr1 = ones('nat=BE, FO')
    arr2 = zeros('type=1..3')
    array_of_arrays = Array([arr1, arr2], sex, dtype=object)
    res = stack(array_of_arrays, sex)
    expected = from_string(r"""nat  type\sex    M    F
                                BE         1  1.0  0.0
                                BE         2  1.0  0.0
                                BE         3  1.0  0.0
                                FO         1  1.0  0.0
                                FO         2  1.0  0.0
                                FO         3  1.0  0.0""")
    assert_array_equal(res, expected)

    # non scalar/non Array
    res = stack(([1, 2, 3], [4, 5, 6]))
    expected = Array([[1, 4],
                      [2, 5],
                      [3, 6]])
    assert_array_equal(res, expected)

    # stack along multiple axes
    # =========================
    # a) simple
    res = stack({('a0', 'b0'): 0,
                 ('a0', 'b1'): 1,
                 ('a1', 'b0'): 2,
                 ('a1', 'b1'): 3,
                 ('a2', 'b0'): 4,
                 ('a2', 'b1'): 5},
                (a, b))
    expected = ndtest((a, b))
    assert_array_equal(res, expected)

    # b) keys not given in axes iteration order
    res = stack({('a0', 'b0'): 0,
                 ('a1', 'b0'): 2,
                 ('a2', 'b0'): 4,
                 ('a0', 'b1'): 1,
                 ('a1', 'b1'): 3,
                 ('a2', 'b1'): 5},
                (a, b))
    expected = ndtest((a, b))
    assert_array_equal(res, expected)

    # c) key parts not given in the order of axes (ie key part for b before key part for a)
    res = stack({('a0', 'b0'): 0,
                 ('a1', 'b0'): 1,
                 ('a2', 'b0'): 2,
                 ('a0', 'b1'): 3,
                 ('a1', 'b1'): 4,
                 ('a2', 'b1'): 5},
                (b, a))
    expected = ndtest((b, a))
    assert_array_equal(res, expected)

    # d) same as c) but with a key-value sequence
    res = stack([(('a0', 'b0'), 0),
                 (('a1', 'b0'), 1),
                 (('a2', 'b0'), 2),
                 (('a0', 'b1'), 3),
                 (('a1', 'b1'), 4),
                 (('a2', 'b1'), 5)],
                (b, a))
    expected = ndtest((b, a))
    assert_array_equal(res, expected)


def test_stack_kwargs_no_axis_labels():
    # these tests rely on kwargs ordering, hence python 3.6

    # 1) using scalars
    # ----------------
    # a) with an axis name
    res = stack(a0=0, a1=1, axes='a')
    expected = Array([0, 1], 'a=a0,a1')
    assert_array_equal(res, expected)

    # b) without an axis name
    res = stack(a0=0, a1=1)
    expected = Array([0, 1], 'a0,a1')
    assert_array_equal(res, expected)

    # 2) dict of arrays
    # -----------------
    a = Axis('a=a0,a1,a2')
    arr0 = ndtest(a)
    arr1 = ndtest(a, start=-1)

    # a) with an axis name
    res = stack(b0=arr0, b1=arr1, axes='b')
    expected = Array([[0, -1],
                      [1,  0],
                      [2,  1]], [a, 'b=b0,b1'])
    assert_array_equal(res, expected)

    # b) without an axis name
    res = stack(b0=arr0, b1=arr1)
    expected = Array([[0, -1],
                      [1,  0],
                      [2,  1]], [a, 'b0,b1'])
    assert_array_equal(res, expected)


@needs_python37
def test_stack_dict_no_axis_labels():
    # these tests rely on dict ordering, hence python 3.7

    # 1) dict of scalars
    # ------------------
    # a) with an axis name
    res = stack({'a0': 0, 'a1': 1}, 'a')
    expected = Array([0, 1], 'a=a0,a1')
    assert_array_equal(res, expected)

    # b) without an axis name
    res = stack({'a0': 0, 'a1': 1})
    expected = Array([0, 1], 'a0,a1')
    assert_array_equal(res, expected)

    # 2) dict of arrays
    # -----------------
    a = Axis('a=a0,a1,a2')
    arr0 = ndtest(a)
    arr1 = ndtest(a, start=-1)

    # a) with an axis name
    res = stack({'b0': arr0, 'b1': arr1}, 'b')
    expected = Array([[0, -1],
                      [1,  0],
                      [2,  1]], [a, 'b=b0,b1'])
    assert_array_equal(res, expected)

    # b) without an axis name
    res = stack({'b0': arr0, 'b1': arr1})
    expected = Array([[0, -1],
                      [1,  0],
                      [2,  1]], [a, 'b0,b1'])
    assert_array_equal(res, expected)


def test_0darray_convert():
    int_arr = Array(1)
    assert int(int_arr) == 1
    assert float(int_arr) == 1.0
    assert int_arr.__index__() == 1

    float_arr = Array(1.0)
    assert int(float_arr) == 1
    assert float(float_arr) == 1.0
    with pytest.raises(TypeError) as e_info:
        float_arr.__index__()

    msg = e_info.value.args[0]
    expected_np11 = "only integer arrays with one element can be converted to an index"
    expected_np12 = "only integer scalar arrays can be converted to a scalar index"
    assert msg in {expected_np11, expected_np12}


def test_deprecated_methods():
    with must_warn(FutureWarning, msg="with_axes() is deprecated. Use set_axes() instead."):
        ndtest((2, 2)).with_axes('a', 'd=d0,d1')

    with must_warn(FutureWarning, msg="split_axis() is deprecated. Use split_axes() instead."):
        ndtest((2, 2)).combine_axes().split_axis()


def test_eq():
    a = ndtest((2, 3, 4))
    ao = a.astype(object)
    assert_array_equal(ao.eq(ao['c0'], nans_equal=True), a == a['c0'])


if __name__ == "__main__":
    # import doctest
    # import unittest
    # from larray import core
    # doctest.testmod(core)
    # unittest.main()
    pytest.main()
