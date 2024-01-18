import os

import pytest
import numpy as np
import pandas as pd

from io import StringIO

from larray.tests.common import (meta, inputpath,
                                 assert_larray_equal, assert_larray_nan_equal, assert_larray_equiv,
                                 needs_xlwings, needs_pytables, needs_xlsxwriter, needs_openpyxl, must_warn, must_raise,
                                 assert_nparray_equal, assert_nparray_nan_equal)
from larray import (Array, LArray, Axis, AxisCollection, LGroup, IGroup, Metadata,
                    zeros, zeros_like, ndtest, empty, ones, full, eye, diag, stack, sequence,
                    asarray, union, clip, exp, where, X, mean, inf, nan, isnan, round,
                    read_hdf, read_csv, read_eurostat, read_excel, open_excel,
                    from_lists, from_string, from_frame, from_series,
                    zip_array_values, zip_array_items)
from larray.core.axis import _to_ticks, _to_tick, _to_key
from larray.util.misc import LHDFStore

# avoid flake8 errors
meta = meta

# Flake8 codes reference (for noqa codes)
# =======================================
# E201: whitespace after '['
# E241: multiple spaces after ','


GROUP_AS_AGGREGATED_LABEL_MSG_TEMPLATE = "Using a Group object which was used to create an aggregate to " \
      "target its aggregated label is deprecated. " \
      "Please use the aggregated label directly instead. " \
      "In this case, you should use {potential_tick!r} instead of " \
      "using {key!r}."

def group_as_aggregated_label_msg(key):
    return GROUP_AS_AGGREGATED_LABEL_MSG_TEMPLATE.format(potential_tick=_to_tick(key), key=key)


# ================== #
# Test Value Strings #
# ================== #


def test_value_string_split():
    assert_nparray_equal(_to_ticks('c0,c1'), np.asarray(['c0', 'c1']))
    assert_nparray_equal(_to_ticks('c0, c1'), np.asarray(['c0', 'c1']))


def test_value_string_union():
    assert union('a1,a3', 'a2,a3') == ['a1', 'a3', 'a2']


def test_value_string_range():
    assert_nparray_equal(_to_ticks('0..15'), np.arange(16))
    assert_nparray_equal(_to_ticks('..15'), np.arange(16))
    with must_raise(ValueError, "no stop bound provided in range: '10..'"):
        _to_ticks('10..')
    with must_raise(ValueError, "no stop bound provided in range: '..'"):
        _to_ticks('..')


# ================ #
# Test Key Strings #
# ================ #

def test_key_string_nonstring():
    assert _to_key(('c0', 'c1')) == ['c0', 'c1']
    assert _to_key(['c0', 'c1']) == ['c0', 'c1']


def test_key_string_split():
    assert _to_key('c0,c1') == ['c0', 'c1']
    assert _to_key('c0, c1') == ['c0', 'c1']
    assert _to_key('c0,') == ['c0']
    assert _to_key('c0') == 'c0'


def test_key_string_slice_strings():
    # these two examples have different results and this is fine because numeric axes do not necessarily start at 0
    assert _to_key('0:16') == slice(0, 16)
    assert _to_key(':16') == slice(16)
    assert _to_key('10:') == slice(10, None)
    assert _to_key(':') == slice(None)


# =================== #
#    Test Metadata    #
# =================== #

def test_read_set_update_delete_metadata(meta, tmp_path):
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
    fname = tmp_path / 'test_metadata.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(meta, f)
    with open(fname, 'rb') as f:
        meta2 = Metadata(pickle.load(f))
    assert meta2 == meta


@needs_pytables
def test_metadata_hdf(meta, tmp_path):
    key = 'meta'
    fname = tmp_path / 'test_metadata.hdf'
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
    # meta as dict
    arr = Array(array.data, array.axes, meta=dict(meta_list))
    assert arr.meta == meta


# ================ #
#    Test Array    #
# ================ #

# AXES
a = Axis('a=0..18')
b_group1 = 'b0,b1,b2,b4,b5,b7,b8'
b_group2 = 'b6,b9,b10,b11'
b_group3 = 'b3'
b_groups = (b_group1, b_group2, b_group3)
all_b = union(*b_groups)
b_groups_all = (b_group1, b_group2, b_group3, all_b)
b = Axis(all_b, 'b')
c = Axis('c=c0,c1')
d = Axis('d=d1..d6')


# ARRAYS
@pytest.fixture()
def array():
    return ndtest((a, b, c, d)).astype(float)


@pytest.fixture()
def small_array():
    return ndtest((c, d))


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
        _ = LArray([0, 1, 2, 3], 'a=a0..a3')


def test_ndtest():
    arr = ndtest('a=a0..a2')
    assert arr.shape == (3,)
    assert arr.axes.names == ['a']
    assert_nparray_equal(arr.data, np.arange(3))

    # using an explicit Axis object
    a = Axis('a=a0..a2')
    arr = ndtest(a)
    assert arr.shape == (3,)
    assert arr.axes.names == ['a']
    assert_nparray_equal(arr.data, np.arange(3))

    # using a group as an axis
    arr = ndtest(a[:'a1'])
    assert arr.shape == (2,)
    assert arr.axes.names == ['a']
    assert_nparray_equal(arr.data, np.arange(2))


def test_getattr(array):
    assert type(array.b) == Axis
    assert array.b is b
    with must_raise(AttributeError, msg="'Array' object has no attribute 'bm'"):
        _ = array.bm


def test_zeros():
    arr = zeros((b, a))
    assert arr.shape == (12, 19)
    assert_nparray_equal(arr.data, np.zeros((12, 19)))


def test_zeros_like(array):
    arr = zeros_like(array)
    assert arr.shape == (19, 12, 2, 6)
    assert_nparray_equal(arr.data, np.zeros((19, 12, 2, 6)))


def test_bool():
    arr = ones([2])
    with must_raise(ValueError, msg="The truth value of an array with more than one element is ambiguous. "
                                    "Use a.any() or a.all()"):
        bool(arr)

    arr = ones([1])
    assert bool(arr)

    arr = zeros([1])
    assert not bool(arr)

    arr = Array(np.array(2), [])
    assert bool(arr)

    arr = Array(np.array(0), [])
    assert not bool(arr)


def test_iter(small_array):
    list_ = list(small_array)
    assert_larray_equal(list_[0], small_array['c0'])
    assert_larray_equal(list_[1], small_array['c1'])


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

    values = arr.values(())
    res = list(values)
    assert_larray_equal(res[0], arr)


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
    new_array = array.rename('c', 'c2')
    # old array axes names not modified
    assert array.axes.names == ['a', 'b', 'c', 'd']
    assert new_array.axes.names == ['a', 'b', 'c2', 'd']

    new_array = array.rename(c, 'c2')
    # old array axes names not modified
    assert array.axes.names == ['a', 'b', 'c', 'd']
    assert new_array.axes.names == ['a', 'b', 'c2', 'd']


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
19 x 12 x 2 x 6
 a [19]: 0 1 2 ... 16 17 18
 b [12]: 'b0' 'b1' 'b2' ... 'b10' 'b11' 'b3'
 c [2]: 'c0' 'c1'
 d [6]: 'd1' 'd2' 'd3' 'd4' 'd5' 'd6'
dtype: float64
memory used: 21.38 Kb"""
    assert array.info == expected


def test_str(small_array, array):
    d3 = d['d1:d3']

    # zero dimension / scalar Array
    scalar_array = Array(42)
    assert str(scalar_array) == "42"

    # empty / len 0 first dimension
    assert str(small_array[c[[]]]) == "Array([])"

    # one dimension
    assert str(small_array[d3, c['c0']]) == """\
d  d1  d2  d3
    0   1   2"""

    # two dimensions
    assert str(small_array.filter(d=d3)) == """\
c\\d  d1  d2  d3
 c0   0   1   2
 c1   6   7   8"""

    # four dimensions (too many rows)
    assert str(array.filter(d=d3)) == """\
  a    b  c\\d      d1      d2      d3
  0   b0   c0     0.0     1.0     2.0
  0   b0   c1     6.0     7.0     8.0
  0   b1   c0    12.0    13.0    14.0
  0   b1   c1    18.0    19.0    20.0
  0   b2   c0    24.0    25.0    26.0
...  ...  ...     ...     ...     ...
 18  b10   c1  2706.0  2707.0  2708.0
 18  b11   c0  2712.0  2713.0  2714.0
 18  b11   c1  2718.0  2719.0  2720.0
 18   b3   c0  2724.0  2725.0  2726.0
 18   b3   c1  2730.0  2731.0  2732.0"""
    # too many columns
    assert str(array['d1', 'b0', 'c0']) == """\
a    0      1      2      3  ...      14      15      16      17      18
   0.0  144.0  288.0  432.0  ...  2016.0  2160.0  2304.0  2448.0  2592.0"""
    arr = Array([0, ''], Axis(['a0', ''], 'a'))
    assert str(arr) == "a  a0  \n    0  "


def test_getitem(array):
    raw = array.data
    a, b, c, d = array.axes
    a159 = a[[1, 5, 9]]
    d124 = d['d1,d2,d4']

    # LGroup at "correct" place
    res = array[a159]
    assert res.axes[1:] == (b, c, d)
    assert res.axes[0].equals(Axis([1, 5, 9], 'a'))
    assert_nparray_equal(res.data, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    res = array[d124]
    assert_nparray_equal(res.data, raw[..., [0, 1, 3]])

    # multiple LGroup key (in "incorrect" order)
    res = array[d124, a159]
    assert res.axes.names == ['a', 'b', 'c', 'd']
    assert_nparray_equal(res.data, raw[[1, 5, 9]][..., [0, 1, 3]])

    # LGroup key and scalar
    res = array[d124, 5]
    assert res.axes.names == ['b', 'c', 'd']
    assert_nparray_equal(res.data, raw[..., [0, 1, 3]][5])

    # mixed LGroup/positional key
    res = array[[1, 5, 9], d124]
    assert_nparray_equal(res.data, raw[[1, 5, 9]][..., [0, 1, 3]])

    # single None slice
    res = array[:]
    assert_nparray_equal(res.data, raw)

    # only Ellipsis
    res = array[...]
    assert_nparray_equal(res.data, raw)

    # Ellipsis and LGroup
    res = array[..., d124]
    assert_nparray_equal(res.data, raw[..., [0, 1, 3]])

    # string 'int..int'
    assert_larray_equal(array['10..13'], array['10,11,12,13'])
    assert_larray_equal(array['8, 10..13, 6'], array['8,10,11,12,13,6'])

    # ambiguous label
    arr = ndtest("a=l0,l1;b=l1,l2")
    res = arr[arr.b['l1']]
    assert_nparray_equal(res.data, arr.data[:, 0])

    # scalar group on another axis
    arr = ndtest((3, 2))
    alt_a = Axis("alt_a=a1..a2")
    lgroup = alt_a['a1']
    assert_larray_equal(arr[lgroup], arr['a1'])
    pgroup = alt_a.i[0]
    assert_larray_equal(arr[pgroup], arr['a1'])

    # key with duplicate axes
    with must_raise(ValueError, msg="key has several values for axis: a\nkey: (a[1, 2], a[3, 4])"):
        _ = array[a[1, 2], a[3, 4]]

    # key with lgroup from another axis leading to duplicate axis
    bad = Axis(3, 'bad')
    with must_raise(ValueError, msg='key has several values for axis: a\nkey: (bad[1, 2], a[3, 4])'):
        _ = array[bad[1, 2], a[3, 4]]


def test_getitem_abstract_axes(array):
    raw = array.data
    a, b, c, d = array.axes
    a159 = X.a[1, 5, 9]
    d124 = X.d['d1,d2,d4']

    # LGroup at "correct" place
    subset = array[a159]
    assert subset.axes[1:] == (b, c, d)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'a'))
    assert_nparray_equal(subset.data, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_nparray_equal(array[d124].data,
                         raw[..., [0, 1, 3]])

    # multiple LGroup key (in "incorrect" order)
    assert_nparray_equal(array[d124, a159].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # mixed LGroup/positional key
    assert_nparray_equal(array[[1, 5, 9], d124].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # single None slice
    assert_nparray_equal(array[:].data, raw)

    # only Ellipsis
    assert_nparray_equal(array[...].data, raw)

    # Ellipsis and LGroup
    assert_nparray_equal(array[..., d124].data,
                         raw[..., [0, 1, 3]])

    # key with duplicate axes
    with must_raise(ValueError, msg="key has several values for axis: a\nkey: (X.a[1, 2], X.a[3])"):
        _ = array[X.a[1, 2], X.a[3]]

    # key with invalid axis
    with must_raise(ValueError, msg='key has several values for axis: a\nkey: (X.bad[1, 2], X.a[3, 4])'):
        _ = array[X.bad[1, 2], X.a[3, 4]]


def test_getitem_anonymous_axes():
    arr = ndtest([Axis(3), Axis(4)])
    raw = arr.data
    assert_nparray_equal(arr[X[0][1:]].data, raw[1:])
    assert_nparray_equal(arr[X[1][2:]].data, raw[:, 2:])
    assert_nparray_equal(arr[X[0][2:], X[1][1:]].data, raw[2:, 1:])
    assert_nparray_equal(arr.i[2:, 1:].data, raw[2:, 1:])


def test_getitem_guess_axis(array):
    raw = array.data
    a, b, c, d = array.axes

    # key at "correct" place
    subset = array[[1, 5, 9]]
    assert subset.axes[1:] == (b, c, d)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'a'))
    assert_nparray_equal(subset.data, raw[[1, 5, 9]])

    # key at "incorrect" place
    assert_nparray_equal(array['d1,d2,d4'].data,
                         raw[..., [0, 1, 3]])
    assert_nparray_equal(array[['d1', 'd2', 'd4']].data,
                         raw[..., [0, 1, 3]])

    # multiple keys (in "incorrect" order)
    assert_nparray_equal(array['d1,d2,d4', [1, 5, 9]].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # mixed LGroup/key
    assert_nparray_equal(array[d['d1,d2,d4'], [1, 5, 9]].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # single None slice
    assert_nparray_equal(array[:].data, raw)

    # only Ellipsis
    assert_nparray_equal(array[...].data, raw)

    # Ellipsis and LGroup
    assert_nparray_equal(array[..., 'd1,d2,d4'].data,
                         raw[..., [0, 1, 3]])
    assert_nparray_equal(array[..., ['d1', 'd2', 'd4']].data,
                         raw[..., [0, 1, 3]])

    # LGroup without axis (which also needs to be guessed)
    g = LGroup(['d1', 'd2', 'd4'])
    assert_nparray_equal(array[g].data, raw[..., [0, 1, 3]])

    # key with duplicate axes
    with must_raise(ValueError, """key has several values for axis: a
key: ([1, 2], [3, 4])"""):
        _ = array[[1, 2], [3, 4]]

    # key with invalid label (ie label not found on any axis)
    with must_raise(ValueError, """999 is not a valid label for any axis:
 a [19]: 0 1 2 ... 16 17 18
 b [12]: 'b0' 'b1' 'b2' ... 'b10' 'b11' 'b3'
 c [2]: 'c0' 'c1'
 d [6]: 'd1' 'd2' 'd3' 'd4' 'd5' 'd6'"""):
        _ = array[[1, 2], 999]

    # key with invalid label list (ie list of labels not found on any axis)
    with must_raise(ValueError, """[998, 999] is not a valid label for any axis:
 a [19]: 0 1 2 ... 16 17 18
 b [12]: 'b0' 'b1' 'b2' ... 'b10' 'b11' 'b3'
 c [2]: 'c0' 'c1'
 d [6]: 'd1' 'd2' 'd3' 'd4' 'd5' 'd6'"""):
        _ = array[[1, 2], [998, 999]]

    # key with partial invalid list (ie list containing a label not found
    # on any axis)
    # FIXME: this should not mention the a axis specifically (this is due to the chunking code)
    with must_raise(ValueError, "a[3, 999] is not a valid label for the 'a' axis with labels: 0, 1, 2, 3, 4, 5, 6, "
                                "7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18"):
        _ = array[[1, 2], [3, 999]]

    with must_raise(ValueError, """[999, 4] is not a valid label for any axis:
 a [19]: 0 1 2 ... 16 17 18
 b [12]: 'b0' 'b1' 'b2' ... 'b10' 'b11' 'b3'
 c [2]: 'c0' 'c1'
 d [6]: 'd1' 'd2' 'd3' 'd4' 'd5' 'd6'"""):
        _ = array[[1, 2], [999, 4]]

    # ambiguous key
    arr = ndtest("a=l0,l1;b=l1,l2")
    with must_raise(ValueError, """'l1' is ambiguous, it is valid in the following axes:
 a [2]: 'l0' 'l1'
 b [2]: 'l1' 'l2'"""):
        _ = arr['l1']

    # ambiguous key disambiguated via string
    res = arr['b[l1]']
    assert_nparray_equal(res.data, arr.data[:, 0])


def test_getitem_positional_group(array):
    raw = array.data
    a, b, c, d = array.axes
    a159 = a.i[1, 5, 9]
    d124 = d.i[0, 1, 3]

    # LGroup at "correct" place
    subset = array[a159]
    assert subset.axes[1:] == (b, c, d)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'a'))
    assert_nparray_equal(subset.data, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_nparray_equal(array[d124].data, raw[..., [0, 1, 3]])

    # multiple LGroup key (in "incorrect" order)
    assert_nparray_equal(array[d124, a159].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # mixed LGroup/positional key
    assert_nparray_equal(array[[1, 5, 9], d124].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # single None slice
    assert_nparray_equal(array[:].data, raw)

    # only Ellipsis
    assert_nparray_equal(array[...].data, raw)

    # Ellipsis and LGroup
    assert_nparray_equal(array[..., d124].data,
                         raw[..., [0, 1, 3]])

    # key with duplicate axes
    with must_raise(ValueError, "key has several values for axis: a\nkey: (a.i[1, 2], a.i[3, 4])"):
        _ = array[a.i[1, 2], a.i[3, 4]]


def test_getitem_str_positional_group():
    arr = ndtest('a=l0..l2;b=l0..l2')
    res = arr['b.i[1]']
    expected = Array([1, 4, 7], 'a=l0..l2')
    assert_larray_equal(res, expected)


def test_getitem_abstract_positional(array):
    raw = array.data
    a, b, c, d = array.axes
    a159 = X.a.i[1, 5, 9]
    d124 = X.d.i[0, 1, 3]

    # LGroup at "correct" place
    subset = array[a159]
    assert subset.axes[1:] == (b, c, d)
    assert subset.axes[0].equals(Axis([1, 5, 9], 'a'))
    assert_nparray_equal(subset.data, raw[[1, 5, 9]])

    # LGroup at "incorrect" place
    assert_nparray_equal(array[d124].data, raw[..., [0, 1, 3]])

    # multiple LGroup key (in "incorrect" order)
    assert_nparray_equal(array[d124, a159].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # mixed LGroup/positional key
    assert_nparray_equal(array[[1, 5, 9], d124].data,
                         raw[[1, 5, 9]][..., [0, 1, 3]])

    # single None slice
    assert_nparray_equal(array[:].data, raw)

    # only Ellipsis
    assert_nparray_equal(array[...].data, raw)

    # Ellipsis and LGroup
    assert_nparray_equal(array[..., d124].data,
                         raw[..., [0, 1, 3]])

    # key with duplicate axes
    with must_raise(ValueError, "key has several values for axis: a\nkey: (X.a.i[2, 3], X.a.i[1, 5])"):
        _ = array[X.a.i[2, 3], X.a.i[1, 5]]


def test_getitem_bool_larray_key_arr_whout_bool_axis():
    arr = ndtest((3, 2, 4))
    raw = arr.data

    # all dimensions
    res = arr[arr < 5]
    assert isinstance(res, Array)
    assert res.ndim == 1
    assert_nparray_equal(res.data, raw[raw < 5])

    # missing dimension
    filter_ = arr['b1'] % 5 == 0
    res = arr[filter_]
    assert isinstance(res, Array)
    assert res.ndim == 2
    assert res.shape == (3, 2)
    raw_key = raw[:, 1, :] % 5 == 0
    raw_d1, raw_d3 = raw_key.nonzero()
    assert_nparray_equal(res.data, raw[raw_d1, :, raw_d3])

    # using an Axis object
    arr = ndtest('a=a0,a1;b=0..3')
    raw = arr.data
    res = arr[arr.b < 2]
    assert_nparray_equal(res.data, raw[:, :2])

    # using an AxisReference (ExprNode)
    res = arr[X.b < 2]
    assert_nparray_equal(res.data, raw[:, :2])


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
    with must_raise(ValueError, "boolean subset key contains more axes ({id}) than array ({gender})"):
        _ = arr[key]


def test_getitem_bool_larray_and_group_key():
    arr = ndtest((3, 6, 4)).set_labels('b', '0..5')

    # using axis
    res = arr['a0,a2', arr.b < 3, 'c0:c3']
    assert isinstance(res, Array)
    assert res.ndim == 3
    expected = arr['a0,a2', '0:2', 'c0:c3']
    assert_larray_equal(res, expected)

    # using axis reference
    res = arr['a0,a2', X.b < 3, 'c0:c3']
    assert isinstance(res, Array)
    assert res.ndim == 3
    assert_larray_equal(res, expected)


def test_getitem_bool_ndarray_key_arr_whout_bool_axis(array):
    raw = array.data
    res = array[raw < 5]
    assert isinstance(res, Array)
    assert res.ndim == 1
    assert_nparray_equal(res.data, raw[raw < 5])


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
    with must_raise(ValueError, msg="boolean key with a different shape ((4,)) than array ((2,))"):
        _ = arr[key]


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

    assert_larray_equal(arr['0[a0, a2]'], arr[a['a0', 'a2']])
    assert_larray_equal(arr['0[a0:a2]'], arr[a['a0:a2']])
    msg = "1['a0', 'a2'] is not a valid label for the 'b' axis with labels: 'b0', 'b1', 'b2', 'b3', 'b4'"
    with must_raise(ValueError, msg=msg):
        _ = arr['1[a0, a2]']

    assert_larray_equal(arr['0.i[0, 2]'], arr[a.i[0, 2]])
    assert_larray_equal(arr['0.i[0:2]'], arr[a.i[0:2]])
    with must_raise(ValueError, msg='Cannot evaluate a positional group without axis'):
        _ = arr['3.i[0, 2]']


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
    assert_larray_equal(arr[a['a1']:a['a2']], expected)

    # a.2) LGroup.axis not from array.axes
    assert_larray_equal((arr[alt_a['a1']:alt_a['a2']]), expected)

    # b) slice with igroup
    # b.1) IGroup.axis from array.axes
    assert_larray_equal((arr[a.i[1]:a.i[2]]), expected)

    # b.2) IGroup.axis not from array.axes
    assert_larray_equal((arr[alt_a.i[0]:alt_a.i[1]]), expected)

    # c) list with LGroup
    # c.1) LGroup.axis from array.axes
    assert_larray_equal((arr[[a['a1'], a['a2']]]), expected)

    # c.2) LGroup.axis not from array.axes
    assert_larray_equal((arr[[alt_a['a1'], alt_a['a2']]]), expected)

    # d) list with IGroup
    # d.1) IGroup.axis from array.axes
    assert_larray_equal((arr[[a.i[1], a.i[2]]]), expected)

    # d.2) IGroup.axis not from array.axes
    assert_larray_equal((arr[[alt_a.i[0], alt_a.i[1]]]), expected)


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
    assert_larray_equal(arr[key], expected)

    # 2) key with the target axis (the one being replaced)
    key = Array(['b1', 'b0', 'b2'], b)
    # axis stays the same but data should be flipped/shuffled
    expected = from_string(r"""
a\b  b0  b1  b2
 a0   1   0   2
 a1   4   3   5""")
    assert_larray_equal(arr[key], expected)

    # 2bis) key with part of the target axis (the one being replaced)
    key = Array(['b2', 'b1'], 'b=b0,b1')
    expected = from_string(r"""
a\b  b0  b1
 a0   2   1
 a1   5   4""")
    assert_larray_equal(arr[key], expected)

    # 3) key with another existing axis (not the target axis)
    key = Array(['a0', 'a1', 'a0'], b)
    expected = from_string("""
b  b0  b1  b2
\t  0   4   2""")
    assert_larray_equal(arr[key], expected)

    # TODO: this does not work yet but should be much easier to implement with "align" in make_np_broadcastable
    # 3bis) key with *part* of another existing axis (not the target axis)
    # key = Array(['a1', 'a0'], 'b=b0,b1')
    # expected = from_string("""
    # b  b0  b1
    # \t  3   1""")
    # assert_larray_equal(arr[key], expected)

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
    assert_larray_equal(arr[key], expected)

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
    assert_larray_equal(arr[key], expected)

    # 6) key has both another existing axis (not target) and an extra axis
    key = from_string(r"""
a\c  c0  c1  c2  c3
 a0  b0  b1  b0  b1
 a1  b2  b1  b2  b1""").astype(str)
    expected = from_string(r"""
a\c  c0  c1  c2  c3
 a0   0   1   0   1
 a1   5   4   5   4""")
    assert_larray_equal(arr[key], expected)

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
    assert_larray_equal(arr[key], expected)


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
    assert_larray_equal(arr[k1, k2], expected)

    # 2) keys with a common existing axis
    k1 = from_string(""" b  b0  b1  b2
                        \t  a1  a0  a1""")
    k2 = from_string(""" b  b0  b1  b2
                        \t  b1  b2  b0""")
    expected = from_string(""" b  b0  b1  b2
                              \t   4   2   3""")
    assert_larray_equal(arr[k1, k2], expected)

    # 3) keys with each a different extra axis
    k1 = from_string(""" c  c0  c1
                        \t  a1  a0""")
    k2 = from_string(""" d  d0  d1  d2
                        \t  b1  b2  b0""")
    expected = from_string(r"""c\d  d0  d1  d2
                                c0   4   5   3
                                c1   1   2   0""")
    assert_larray_equal(arr[k1, k2], expected)

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
    assert_larray_equal(arr[k1, k2], expected)


def test_getitem_ndarray_key_guess(array):
    raw = array.data
    keys = ['d4', 'd1', 'd3', 'd2']
    key = np.array(keys)
    res = array[key]
    assert isinstance(res, Array)
    assert res.axes == array.axes.replace(X.d, Axis(keys, 'd'))
    assert_nparray_equal(res.data, raw[:, :, :, [3, 0, 2, 1]])


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

    assert_larray_equal(arr[a], arr)
    assert_larray_equal(arr[b], arr)

    b2 = Axis('b=b0,b2')

    assert_larray_equal(arr[b2], from_string(r"""a\b  b0  b2
                                                 a0   0   2
                                                 a1   3   5"""))


def test_getitem_empty_tuple():
    # an empty tuple should return a view on the original array
    arr = ndtest((2, 3))
    res = arr[()]
    assert_larray_equal(res, arr)
    assert res is not arr

    z = Array(0)
    res = z[()]
    assert res == z
    assert res is not z


def test_positional_indexer_getitem(array):
    raw = array.data
    # scalar result
    assert array.i[(0, 5, 1, 2)] == raw[0, 5, 1, 2]

    # normal indexing (supported by numpy)
    for key in [0,
                (slice(None), 5, 1),
                (0, 5),
                [1, 0],
                ([1, 0], 5)]:
        assert_nparray_equal(array.i[key].data, raw[key])

    # cross product indexing
    assert_nparray_equal(array.i[[1, 0], [5, 4]].data,
                         raw[np.ix_([1, 0], [5, 4])])

    # too many indices
    with must_raise(IndexError, "key is too long (5) for array with 4 dimensions"):
        _ = array.i[0, 0, 0, 0, 0]


def test_positional_indexer_setitem(array):
    for key in [0, (0, 2, 1, 2), (slice(None), 2, 1), (0, 2), [1, 0], ([1, 0], 2)]:
        arr = array.copy()
        raw = array.data.copy()
        arr.i[key] = 42
        raw[key] = 42
        assert_nparray_equal(arr.data, raw)

    raw = array.data
    array.i[[1, 0], [5, 4]] = 42
    raw[np.ix_([1, 0], [5, 4])] = 42
    assert_nparray_equal(array.data, raw)


def test_points_indexer_getitem():
    arr = ndtest((2, 3, 3))
    raw = arr.data

    # scalar result
    assert arr.points['a0', 'b1', 'c2'], raw[0, 1, 2]

    # array result
    keys = [
        ('a0',
            0),
        (('a0', 'c2'),
            (0, slice(None), 2)),
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
        assert_nparray_equal(arr.points[label_key].data, raw[index_key])

    # XXX: we might want to raise KeyError or IndexError instead?
    with must_raise(ValueError, msg="""'d0' is not a valid label for any axis:
 a [2]: 'a0' 'a1'
 b [3]: 'b0' 'b1' 'b2'
 c [3]: 'c0' 'c1' 'c2'"""):
        _ = arr.points['a0', 'b1', 'c2', 'd0']


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
        assert_nparray_equal(arr.data, raw)

    arr = ndtest(2)
    # XXX: we might want to raise KeyError or IndexError instead?
    with must_raise(ValueError, "'b1' is not a valid label for any axis:\n a [2]: 'a0' 'a1'"):
        arr.points['a0', 'b1'] = 42

    # test when broadcasting is involved
    arr = ndtest((2, 3, 4))
    raw = arr.data.copy()
    raw_value = raw[:, 0, 0].reshape(2, 1)
    raw[:, [0, 1, 2], [0, 1, 2]] = raw_value
    arr.points['b0,b1,b2', 'c0,c1,c2'] = arr['b0', 'c0']
    assert_nparray_equal(arr.data, raw)


def test_ipoints_indexer_getitem():
    arr = ndtest((2, 3, 3))
    raw = arr.data

    # scalar result
    assert arr.ipoints[0, 1, 2], raw[0, 1, 2]

    # array result
    keys = [
        0,
        (0, slice(None), 2),
        # key in the "correct" order
        ([1, 0, 1, 0], 1, [1, 0, 1, 0]),
        # advanced key with a missing dimension
        ([1, 0, 1, 0], slice(None), [1, 0, 1, 0]),
    ]
    for index_key in keys:
        assert_nparray_equal(arr.ipoints[index_key].data, raw[index_key])

    with must_raise(IndexError, "key is too long (4) for array with 3 dimensions"):
        _ = arr.ipoints[0, 1, 2, 0]


def test_ipoints_indexer_setitem():
    keys = [
        0,
        (0, slice(None), 2),
        (0, 1, 2),
        # key in the "correct" order
        ([1, 0, 1, 0], 1, [1, 0, 1, 0]),
        # advanced key with a missing dimension
        ([1, 0, 1, 0], slice(None), [1, 0, 1, 0]),
    ]
    for index_key in keys:
        arr = ndtest((2, 3, 3))
        raw = arr.data.copy()
        arr.ipoints[index_key] = 42
        raw[index_key] = 42
        assert_nparray_equal(arr.data, raw)

    arr = ndtest(2)
    with must_raise(IndexError, "key is too long (2) for array with 1 dimensions"):
        arr.ipoints[0, 1] = 42

    # test when broadcasting is involved
    arr = ndtest((2, 3, 4))
    raw = arr.data.copy()
    raw_value = raw[:, 0, 0].reshape(2, 1)
    raw[:, [0, 1, 2], [0, 1, 2]] = raw_value
    arr.ipoints[:, [0, 1, 2], [0, 1, 2]] = arr['b0', 'c0']
    assert_nparray_equal(arr.data, raw)


def test_setitem_larray(array, small_array):
    """
    Test Array.__setitem__(key, value) where value is an Array.
    """
    a, b, c, d = array.axes

    # 1) using a LGroup key
    as1_5_9 = a[[1, 5, 9]]

    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()

    arr[as1_5_9] = arr[as1_5_9] + 25.0
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
    assert_nparray_equal(arr.data, raw)

    # b) value has exactly the same shape but LGroup at a "wrong" positions
    arr = array.copy()
    arr[b[:], as1_5_9] = arr[as1_5_9] + 25.0
    # same raw as previous test
    assert_nparray_equal(arr.data, raw)

    # c) value has an extra length-1 axis
    arr = array.copy()
    raw = array.data.copy()

    raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
    fake_axis = Axis(['label'], 'fake')
    a_axis = arr[as1_5_9].axes.a
    value = Array(raw_value, axes=(a_axis, fake_axis, b, c, d))
    arr[as1_5_9] = value
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 26.0
    assert_nparray_equal(arr.data, raw)

    # d) value is an Array with a length-1 axis but the target region is a single cell
    # these two cases raise a deprecation warning with Numpy 1.25+ (and will stop working
    # in a future version), so we do not support that anymore (see issue #1070)
    # res = ndtest((2, 3))
    # res['a0', 'b1'] = Array([42])
    # res['a0', 'b1'] = Array([42], 'dummy=d0')
    # assert_larray_equal(res, from_string(r"""a\b b0  b1 b2
    #                                           a0  0  42  2
    #                                           a1  3   4  5"""))

    # e) value has the same axes than target but one has length 1
    # arr = array.copy()
    # raw = array.data.copy()
    # raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    # arr[as1_5_9] = arr[as1_5_9].sum(b=(b.all(),))
    # assert_nparray_equal(arr.data, raw)

    # f) value has a missing dimension
    arr = array.copy()
    raw = array.data.copy()
    arr[as1_5_9] = arr[as1_5_9].sum(b)
    raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    assert_nparray_equal(arr.data, raw)

    # 2) using a LGroup and scalar key (triggers advanced indexing/cross)

    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()

    # using 1, 5, 8 and not 9 so that the list is not collapsed to slice
    value = arr[a[1, 5, 8], c['c0']] + 25.0
    arr[a[1, 5, 8], c['c0']] = value
    raw[[1, 5, 8], :, 0] = raw[[1, 5, 8], :, 0] + 25.0
    assert_nparray_equal(arr.data, raw)

    # 3) using a string key
    arr = array.copy()
    raw = array.data.copy()
    arr['1, 5, 9'] = arr['1, 5, 9'] + 27.0
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 27.0
    assert_nparray_equal(arr.data, raw)

    # 4) using ellipsis keys
    # only Ellipsis
    arr = array.copy()
    arr[...] = 0
    assert_nparray_equal(arr.data, np.zeros_like(raw))

    # Ellipsis and LGroup
    arr = array.copy()
    raw = array.data.copy()
    arr[..., d['d1,d2,d4']] = 0
    raw[..., [0, 1, 3]] = 0
    assert_nparray_equal(arr.data, raw)

    # 5) using a single slice(None) key
    arr = array.copy()
    arr[:] = 0
    assert_nparray_equal(arr.data, np.zeros_like(raw))

    # 6) incompatible axes
    arr = small_array.copy()
    subset_axes = arr['d1'].axes
    value = small_array.copy()
    expected_msg = f"Value {value.axes - subset_axes!s} axis is not present in target subset {subset_axes!s}. " \
                   f"A value can only have the same axes or fewer axes than the subset being targeted"
    with must_raise(ValueError, expected_msg):
        arr['d1'] = value

    value = arr.rename('c', 'c_bis')['d1']
    expected_msg = f"Value {value.axes - subset_axes!s} axis is not present in target subset {subset_axes!s}. " \
                   f"A value can only have the same axes or fewer axes than the subset being targeted"
    with must_raise(ValueError, expected_msg):
        arr['d1'] = value

    # 7) incompatible labels
    c_bis = Axis('c=c1,c0')
    arr2 = Array(small_array.data, axes=(c_bis, d))
    with must_raise(ValueError, """incompatible axes:
Axis(['c0', 'c1'], 'c')
vs
Axis(['c1', 'c0'], 'c')"""):
        arr[:] = arr2

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
    assert_larray_equal(array, expected)

    # b) with wildcard combined_axis
    wild_combined_axis = combined_axis.ignore_labels()
    wild_a_key = Array([0, 0, 0, 1, 1, 1], wild_combined_axis)
    wild_b_key = Array([0, 1, 2, 0, 1, 2], wild_combined_axis)
    wild_key = (a.i[wild_a_key], b.i[wild_b_key])
    array = empty((a, b))
    array[wild_key] = value
    assert_larray_equal(array, expected)

    # c) with a wildcard value
    wild_value = value.ignore_labels()
    array = empty((a, b))
    array[key] = wild_value
    assert_larray_equal(array, expected)

    # d) with a wildcard combined axis and wildcard value
    array = empty((a, b))
    array[wild_key] = wild_value
    assert_larray_equal(array, expected)


def test_setitem_ndarray(array):
    """
    Test Array.__setitem__(key, value) where value is a raw ndarray.
    In that case, value.shape is more restricted as we rely on numpy broadcasting.
    """
    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()
    value = raw[[1, 5, 9]] + 25.0
    arr[[1, 5, 9]] = value
    raw[[1, 5, 9]] = value
    assert_nparray_equal(arr.data, raw)

    # b) value has the same axes than target but one has length 1
    arr = array.copy()
    raw = array.data.copy()
    value = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    arr[[1, 5, 9]] = value
    raw[[1, 5, 9]] = value
    assert_nparray_equal(arr.data, raw)


def test_setitem_scalar(array):
    """
    Test Array.__setitem__(key, value) where value is a scalar.
    """
    # a) list key (one dimension)
    arr = array.copy()
    raw = array.data.copy()
    arr[[1, 5, 9]] = 42
    raw[[1, 5, 9]] = 42
    assert_nparray_equal(arr.data, raw)

    # b) full scalar key (ie set one cell)
    arr = array.copy()
    raw = array.data.copy()
    arr[0, 'd2', 'b1', 'c0'] = 42
    raw[0, 1, 0, 1] = 42
    assert_nparray_equal(arr.data, raw)


def test_setitem_bool_array_key(array):
    # XXX: this test is awfully slow (more than 1s)
    a, b, c, d = array.axes

    # Array key
    # a1) same shape, same order
    arr = array.copy()
    raw = array.data.copy()
    arr[arr < 5] = 0
    raw[raw < 5] = 0
    assert_nparray_equal(arr.data, raw)

    # a2) same shape, different order
    arr = array.copy()
    raw = array.data.copy()
    key = (arr < 5).T
    arr[key] = 0
    raw[raw < 5] = 0
    assert_nparray_equal(arr.data, raw)

    # b) numpy-broadcastable shape
    # arr = array.copy()
    # raw = array.data.copy()
    # key = arr[c['c1,']] < 5
    # assert key.ndim == 4
    # arr[key] = 0
    # raw[raw[:, :, [1]] < 5] = 0
    # assert_nparray_equal(arr.data, raw)

    # c) Array-broadcastable shape (missing axis)
    arr = array.copy()
    raw = array.data.copy()
    key = arr[c['c0']] < 5
    assert key.ndim == 3
    arr[key] = 0

    raw_key = raw[:, :, 0, :] < 5
    raw_d1, raw_d2, raw_d4 = raw_key.nonzero()
    raw[raw_d1, raw_d2, :, raw_d4] = 0
    assert_nparray_equal(arr.data, raw)

    # ndarray key
    arr = array.copy()
    raw = array.data.copy()
    arr[raw < 5] = 0
    raw[raw < 5] = 0
    assert_nparray_equal(arr.data, raw)

    # d) Array with extra axes
    arr = array.copy()
    key = (arr < 5).expand([Axis(2, 'extra')])
    assert key.ndim == 5
    # Note that we could make this work by expanding the data array behind the scene but this does not seem like a
    # good idea.
    msg = "boolean subset key contains more axes ({a, b, c, d, extra*}) than array ({a, b, c, d})"
    with must_raise(ValueError, msg=msg):
        arr[key] = 0


def test_set(array):
    a, b, c, d = array.axes

    # 1) using a LGroup key
    as1_5_9 = a[[1, 5, 9]]

    # a) value has exactly the same shape as the target slice
    arr = array.copy()
    raw = array.data.copy()

    arr.set(arr[as1_5_9] + 25.0, a=as1_5_9)
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 25.0
    assert_nparray_equal(arr.data, raw)

    # b) same size but a different shape (extra length-1 axis)
    arr = array.copy()
    raw = array.data.copy()

    raw_value = raw[[1, 5, 9], np.newaxis] + 26.0
    fake_axis = Axis(['label'], 'fake')
    a_axis = arr[as1_5_9].axes.a
    value = Array(raw_value, axes=(a_axis, fake_axis, b, c, d))
    arr.set(value, a=as1_5_9)
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 26.0
    assert_nparray_equal(arr.data, raw)

    # dimension of length 1
    # arr = array.copy()
    # raw = array.data.copy()
    # raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    # arr.set(arr[as1_5_9].sum(b=(b.all(),)), a=as1_5_9)
    # assert_nparray_equal(arr.data, raw)

    # c) missing dimension
    arr = array.copy()
    raw = array.data.copy()
    arr.set(arr[as1_5_9].sum(b), a=as1_5_9)
    raw[[1, 5, 9]] = np.sum(raw[[1, 5, 9]], axis=1, keepdims=True)
    assert_nparray_equal(arr.data, raw)

    # 2) using a raw key
    arr = array.copy()
    raw = array.data.copy()
    arr.set(arr[[1, 5, 9]] + 27.0, a=[1, 5, 9])
    raw[[1, 5, 9]] = raw[[1, 5, 9]] + 27.0
    assert_nparray_equal(arr.data, raw)


def test_filter(array):
    a, b, c, d = array.axes

    as1_5_9 = a[(1, 5, 9)]
    as11 = a[11]

    # with LGroup
    assert array.filter(a=as1_5_9).shape == (3, 12, 2, 6)

    # FIXME: this should raise a comprehensible error!
    # assert array.filter(a=[as1_5_9]).shape == (3, 12, 2, 6)

    # LGroup with 1 value => collapse
    assert array.filter(a=as11).shape == (12, 2, 6)

    # LGroup with a list of 1 value => do not collapse
    assert array.filter(a=a[[11]]).shape == (1, 12, 2, 6)

    # LGroup with a list of 1 value defined as a string => do not collapse
    assert array.filter(d=d['d1,']).shape == (19, 12, 2, 1)

    # LGroup with 1 value
    # XXX: this does not work. Do we want to make this work?
    # filtered = array.filter(a=(as11,))
    # assert filtered.shape == (1, 12, 2, 6)

    # list
    assert array.filter(a=[1, 5, 9]).shape == (3, 12, 2, 6)

    # string
    assert array.filter(d='d1,d2').shape == (19, 12, 2, 2)

    # multiple axes at once
    assert array.filter(a=[1, 5, 9], d='d1,d2').shape == (3, 12, 2, 2)

    # multiple axes one after the other
    assert array.filter(a=[1, 5, 9]).filter(d='d1,d2').shape == (3, 12, 2, 2)

    # a single value for one dimension => collapse the dimension
    assert array.filter(c='c0').shape == (19, 12, 6)

    # but a list with a single value for one dimension => do not collapse
    assert array.filter(c=['c0']).shape == (19, 12, 1, 6)

    assert array.filter(c='c0,').shape == (19, 12, 1, 6)

    # with duplicate keys
    # XXX: do we want to support this? I don't see any value in that but I might be short-sighted.
    # filtered = array.filter(d='d1,d2,d1')

    # XXX: we could abuse python to allow naming groups via Axis.__getitem__
    # (but I doubt it is a good idea).
    # child = a[':17', 'child']

    # slices
    # ------

    # LGroup slice
    assert array.filter(a=a[:17]).shape == (18, 12, 2, 6)
    # string slice
    assert array.filter(d=':d3').shape == (19, 12, 2, 3)
    # raw slice
    assert array.filter(a=slice(17)).shape == (18, 12, 2, 6)

    # filter chain with a slice
    assert array.filter(a=slice(17)).filter(b='b1,b2').shape == (18, 2, 2, 6)


def test_filter_multiple_axes(array):
    # multiple values in each group
    assert array.filter(a=[1, 5, 9], d='d1,d2').shape == (3, 12, 2, 2)
    # with a group of one value
    assert array.filter(a=[1, 5, 9], c='c0,').shape == (3, 12, 1, 6)

    # with a discarded axis (there is a scalar in the key)
    assert array.filter(a=[1, 5, 9], c='c0').shape == (3, 12, 6)

    # with a discarded axis that is not adjacent to the ix_array axis ie with a sliced axis between the scalar axis
    # and the ix_array axis since our array has a axes: a, b, c, d, any of the following should be tested:
    # a+c / a+d / b+d
    # additionally, if the ix_array axis was first (ie ix_array on a), it worked even before the issue was fixed,
    # since the "indexing" subspace is tacked-on to the beginning (as the first dimension)
    assert array.filter(a=11, c='c0,c1').shape == (12, 2, 6)
    assert array.filter(a=11, d='d1,d4').shape == (12, 2, 2)
    assert array.filter(b='b10', d='d1,d4').shape == (19, 2, 2)


def test_nonzero():
    arr = ndtest((2, 3))
    a, b = arr.axes
    cond = arr > 1
    assert_larray_equal(cond, from_string(r"""a\b     b0     b1    b2
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
    assert_larray_equal(arr[a_group, b_group], expected)
    assert_larray_equal(arr.points[a_group, b_group], expected)
    assert_larray_equal(arr[cond], expected)


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
    assert slice('b0', 'b2') not in arr


def test_sum_full_axes(array):
    a, b, c, d = array.axes

    # everything
    assert array.sum() == np.asarray(array).sum()

    # using axes numbers
    assert array.sum(axis=2).shape == (19, 12, 6)
    assert array.sum(axis=(0, 2)).shape == (12, 6)

    # using Axis objects
    assert array.sum(a).shape == (12, 2, 6)
    assert array.sum(a, c).shape == (12, 6)

    # using axes names
    assert array.sum('a', 'c').shape == (12, 6)

    # chained sum
    assert array.sum(a, c).sum(b).shape == (6,)
    assert array.sum(a, c).sum(d, b) == array.sum()

    # getitem on aggregated
    aggregated = array.sum(a, c)
    assert aggregated[b_group1].shape == (7, 6)

    # filter on aggregated
    assert aggregated.filter(b=b_group1).shape == (7, 6)


def test_sum_full_axes_with_nan(array):
    array['c0', 'd2', 'b1', 0] = nan
    raw = array.data

    # everything
    assert array.sum() == np.nansum(raw)
    assert isnan(array.sum(skipna=False))

    # using Axis objects
    assert_nparray_nan_equal(array.sum(X.a).data, np.nansum(raw, 0))
    assert_nparray_nan_equal(array.sum(X.a, skipna=False).data, raw.sum(0))

    assert_nparray_nan_equal(array.sum(X.a, X.c).data, np.nansum(raw, (0, 2)))
    assert_nparray_nan_equal(array.sum(X.a, X.c, skipna=False).data, raw.sum((0, 2)))


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
    assert_nparray_nan_equal(array.mean(X.a).data, np.mean(raw, 0))
    assert_nparray_nan_equal(array.mean(X.a, X.c).data, np.mean(raw, (0, 2)))


def test_mean_groups(array):
    # using int type to test that we get a float in return
    arr = array.astype(int)
    raw = array.data
    res = arr.mean(X.b['b0', 'b2', 'b5', 'b7'])
    assert_nparray_nan_equal(res.data, np.mean(raw[:, [0, 2, 4, 5]], 1))


def test_median_full_axes(array):
    raw = array.data

    assert array.median() == np.median(raw)
    assert_nparray_nan_equal(array.median(X.a).data, np.median(raw, 0))
    assert_nparray_nan_equal(array.median(X.a, X.c).data, np.median(raw, (0, 2)))


def test_median_groups(array):
    raw = array.data
    res = array.median(X.b['b0', 'b2', 'b5'])
    assert res.shape == (19, 2, 6)
    assert_nparray_nan_equal(res.data, np.median(raw[:, [0, 2, 4]], 1))


def test_percentile_full_axes():
    arr = ndtest((2, 3, 4))
    raw = arr.data
    assert arr.percentile(10) == np.percentile(raw, 10)
    assert_nparray_nan_equal(arr.percentile(10, X.a).data, np.percentile(raw, 10, 0))
    assert_nparray_nan_equal(arr.percentile(10, X.c, X.a).data, np.percentile(raw, 10, (2, 0)))


def test_percentile_groups():
    arr = ndtest((2, 5, 3))
    raw = arr.data

    res = arr.percentile(10, X.b['b0', 'b2', 'b4'])
    assert_nparray_nan_equal(res.data, np.percentile(raw[:, [0, 2, 4]], 10, 1))


def test_cumsum(array):
    raw = array.data

    # using Axis objects
    assert_nparray_equal(array.cumsum(X.a).data, raw.cumsum(0))
    assert_nparray_equal(array.cumsum(X.d).data, raw.cumsum(3))

    # using axes numbers
    assert_nparray_equal(array.cumsum(1).data, raw.cumsum(1))

    # using axes names
    assert_nparray_equal(array.cumsum('c').data, raw.cumsum(2))


def test_group_agg_kwargs(array):
    a, b, c, d = array.axes

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    assert array.sum(c='c0').shape == (19, 12, 6)
    assert array.sum(c='c0,c1').shape == (19, 12, 6)
    assert array.sum(c=c['c0']).shape == (19, 12, 6)

    assert array.sum(b='b0,b3,b6').shape == (19, 2, 6)
    assert array.sum(b=['b0', 'b3', 'b6']).shape == (19, 2, 6)
    assert array.sum(b=b['b0,b3,b6']).shape == (19, 2, 6)

    assert array.sum(b=':').shape == (19, 2, 6)
    assert array.sum(b=b[:]).shape == (19, 2, 6)
    assert array.sum(b=b[':']).shape == (19, 2, 6)
    # Include everything between two labels. Since b0 is the first label
    # and b3 is the last one, this should be equivalent to the previous
    # tests.
    assert array.sum(b='b0:b3').shape == (19, 2, 6)
    assert_larray_equal(array.sum(b='b0:b3'), array.sum(b=':'))
    assert_larray_equal(array.sum(b=b['b0:b3']), array.sum(b=':'))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum(b=(b[:],)).shape == (19, 1, 2, 6)

    # a.3) several groups
    # string groups
    assert array.sum(b=b_groups).shape == (19, 3, 2, 6)
    # with one label in several groups
    assert array.sum(c=(['c0'], ['c0', 'c1'])).shape == (19, 12, 2, 6)
    assert array.sum(c=('c0', 'c0,c1')).shape == (19, 12, 2, 6)
    assert array.sum(c='c0;c0,c1').shape == (19, 12, 2, 6)

    res = array.sum(b=b_groups_all)
    assert res.shape == (19, 4, 2, 6)

    # a.4) several dimensions at the same time
    res = array.sum(d='d1,d3;d2,d4;:', b=b_groups_all)
    assert res.shape == (19, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    # Note that you must list "full axes" aggregates first (Python does not allow non-kwargs after kwargs.
    res = array.sum(a, c, b=b_groups_all)
    assert res.shape == (4, 6)

    # c) chain group aggregate after axis aggregate
    res = array.sum(a, c).sum(b=b_groups_all)
    assert res.shape == (4, 6)


def test_group_agg_guess_axis(array):
    raw = array.data.copy()
    a, b, c, d = array.axes

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    assert array.sum('c0').shape == (19, 12, 6)
    assert array.sum('c0,').shape == (19, 12, 6)
    assert array.sum('c0,c1').shape == (19, 12, 6)

    assert array.sum('b0,b3,b6').shape == (19, 2, 6)
    # with a name
    assert array.sum('b0,b3,b6 >> g1').shape == (19, 2, 6)
    assert array.sum(['b0', 'b3', 'b6']).shape == (19, 2, 6)

    # Include everything between two labels. Since b0 is the first label
    # and b3 is the last one, this should be equivalent to taking the
    # full axis.
    assert array.sum('b0:b3').shape == (19, 2, 6)
    assert_larray_equal(array.sum('b0:b3'), array.sum(b=':'))
    assert_larray_equal(array.sum('b0:b3'), array.sum(b))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum((b[:],)).shape == (19, 1, 2, 6)

    # a.3) several groups
    # string groups
    assert array.sum(b_groups).shape == (19, 3, 2, 6)

    # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
    # axes into play. For now the error message is unclear because it first aggregates on "g1", then tries to
    # aggregate on "g2", but there is no "b" dimension anymore.
    # assert array.sum(g1, g2, g3).shape == (19, 3, 2, 6)

    # with one label in several groups
    assert array.sum((['c0'], ['c0', 'c1'])).shape == (19, 12, 2, 6)
    assert array.sum(('c0', 'c0,c1')).shape == (19, 12, 2, 6)
    assert array.sum('c0;c0,c1').shape == (19, 12, 2, 6)
    # with group names
    res = array.sum('c0 >> first;c0,c1 >> all')
    assert res.shape == (19, 12, 2, 6)
    assert 'c' in res.axes
    assert list(res.axes.c.labels) == ['first', 'all']
    assert_nparray_equal(res['first'].data, raw[:, :, 0, :])
    assert_nparray_equal(res['all'].data, raw.sum(2))

    res = array.sum(('c0 >> first', 'c0,c1 >> all'))
    assert res.shape == (19, 12, 2, 6)
    assert 'c' in res.axes
    assert list(res.axes.c.labels), ['first', 'all']
    assert_nparray_equal(res['first'].data, raw[:, :, 0, :])
    assert_nparray_equal(res['all'].data, raw.sum(2))

    res = array.sum(b_groups_all)
    assert res.shape == (19, 4, 2, 6)

    # a.4) several dimensions at the same time
    res = array.sum('d1,d3;d2,d5;d1:', b_groups_all)
    assert res.shape == (19, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(a, c, b_groups_all)
    assert res.shape == (4, 6)

    # c) chain group aggregate after axis aggregate
    res = array.sum(a, c).sum(b_groups_all)
    assert res.shape == (4, 6)

    # issue #868: labels in reverse order with a "step" between them > index of last label
    arr = ndtest(4)
    assert arr.sum('a3,a1') == 4

    # ambiguous label and anonymous axes
    arr = ndtest([Axis("b1,b2"), Axis("b0..b2")])
    msg = """'b1' is ambiguous, it is valid in the following axes:
 {0} [2]: 'b1' 'b2'
 {1} [3]: 'b0' 'b1' 'b2'"""
    with must_raise(ValueError, msg=msg):
        arr.sum('b1;b0,b1')


def test_group_agg_label_group(array):
    a, b, c, d = array.axes
    g1, g2, g3 = b[b_group1], b[b_group2], b[b_group3]
    g_all = b[all_b]

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    c0 = c.i[[0]]
    assert array.sum(c0).shape == (19, 12, 6)
    assert array.sum(c['c0']).shape == (19, 12, 6)
    assert array.sum(c['c0,']).shape == (19, 12, 6)
    assert array.sum(c['c0,c1']).shape == (19, 12, 6)

    assert array.sum(b['b0,b3,b6']).shape == (19, 2, 6)
    assert array.sum(b[['b0', 'b3', 'b6']]).shape == (19, 2, 6)
    assert array.sum(b['b0', 'b3', 'b6']).shape == (19, 2, 6)
    assert array.sum(b['b0,b3,b6']).shape == (19, 2, 6)

    assert array.sum(b[:]).shape == (19, 2, 6)
    assert array.sum(b[':']).shape == (19, 2, 6)
    assert array.sum(b[:]).shape == (19, 2, 6)

    # Include everything between two labels. Since b0 is the first label and b3 is the last one, this should be
    # equivalent to the previous tests.
    assert array.sum(b['b0:b3']).shape == (19, 2, 6)
    assert_larray_equal(array.sum(b['b0:b3']), array.sum(b))
    assert_larray_equal(array.sum(b['b0':'b3']), array.sum(b))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum((b[:],)).shape == (19, 1, 2, 6)

    # a.3) several groups
    # string groups
    assert array.sum((g1, g2, g3)).shape == (19, 3, 2, 6)

    # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
    # axes into play. For now the error message is unclear because it first aggregates on "g1", then tries to
    # aggregate on "g2", but there is no "b" dimension anymore.
    # assert array.sum(g1, g2, g3).shape == (19, 3, 2, 6)

    # with one label in several groups
    assert array.sum((c['c0'], c[['c0', 'c1']])).shape == (19, 12, 2, 6)
    assert array.sum((c['c0'], c['c0', 'c1'])).shape == (19, 12, 2, 6)
    assert array.sum((c['c0'], c['c0,c1'])).shape == (19, 12, 2, 6)
    # XXX: do we want to support this?
    # assert array.sum(c['c0;H,c1']).shape == (19, 12, 2, 6)

    res = array.sum((g1, g2, g3, g_all))
    assert res.shape == (19, 4, 2, 6)

    # a.4) several dimensions at the same time
    # res = array.sum(d['d1,d3;d2,d5;d1:'], (g1, g2, g3, g_all))
    # assert res.shape == (19, 4, 2, 3)
    res = array.sum((d['d1,d3'], d['d2,d5'], d[:]), (g1, g2, g3, g_all))
    assert res.shape == (19, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(a, c, (g1, g2, g3, g_all))
    assert res.shape == (4, 6)

    # c) chain group aggregate after axis aggregate
    res = array.sum(a, c).sum((g1, g2, g3, g_all))
    assert res.shape == (4, 6)


def test_group_agg_label_group_no_axis(array):
    a, b, c, d = array.axes
    g1, g2, g3 = LGroup(b_group1), LGroup(b_group2), LGroup(b_group3)
    g_all = LGroup(all_b)

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    assert array.sum(LGroup('c0')).shape == (19, 12, 6)
    assert array.sum(LGroup('c0,')).shape == (19, 12, 6)
    assert array.sum(LGroup('c0,c1')).shape == (19, 12, 6)

    assert array.sum(LGroup('b0,b3,b6')).shape == (19, 2, 6)
    assert array.sum(LGroup(['b0', 'b3', 'b6'])).shape == (19, 2, 6)

    # Include everything between two labels. Since b0 is the first label
    # and b3 is the last one, this should be equivalent to the full axis.
    assert array.sum(LGroup('b0:b3')).shape == (19, 2, 6)
    assert_larray_equal(array.sum(LGroup('b0:b3')), array.sum(b))
    assert_larray_equal(array.sum(LGroup(slice('b0', 'b3'))), array.sum(b))

    # a.3) several groups
    # string groups
    assert array.sum((g1, g2, g3)).shape == (19, 3, 2, 6)

    # XXX: do we also want to support this? I do not really like it because it gets tricky when we have some other
    # axes into play. For now the error message is unclear because it first aggregates on "g1", then tries to
    # aggregate on "g2", but there is no "b" dimension anymore.
    # assert array.sum(g1, g2, g3).shape == (19, 3, 2, 6)

    # with one label in several groups
    assert array.sum((LGroup('c0'), LGroup(['c0', 'c1']))).shape == (19, 12, 2, 6)
    assert array.sum((LGroup('c0'), LGroup('c0,c1'))).shape == (19, 12, 2, 6)
    # XXX: do we want to support this?
    # assert array.sum(c['c0;c0,c1']).shape == (19, 12, 2, 6)

    res = array.sum((g1, g2, g3, g_all))
    assert res.shape == (19, 4, 2, 6)

    # a.4) several dimensions at the same time
    # res = array.sum(d['d1,d3;d2,d5;d1:'], (g1, g2, g3, g_all))
    # assert res.shape == (19, 4, 2, 3)
    res = array.sum((LGroup('d1,d3'), LGroup('d2,d5')), (g1, g2, g3, g_all))
    assert res.shape == (19, 4, 2, 2)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(a, c, (g1, g2, g3, g_all))
    assert res.shape == (4, 6)

    # c) chain group aggregate after axis aggregate
    res = array.sum(a, c).sum((g1, g2, g3, g_all))
    assert res.shape == (4, 6)


def test_group_agg_axis_ref_label_group(array):
    a, b, c, d = X.a, X.b, X.c, X.d
    g1, g2, g3 = b[b_group1], b[b_group2], b[b_group3]
    g_all = b[all_b]

    # a) group aggregate on a fresh array

    # a.1) one group => collapse dimension
    # not sure I should support groups with a single item in an aggregate
    men = c.i[[0]]
    assert array.sum(men).shape == (19, 12, 6)
    assert array.sum(c['c0']).shape == (19, 12, 6)
    assert array.sum(c['c0,']).shape == (19, 12, 6)
    assert array.sum(c['c0,c1']).shape == (19, 12, 6)

    assert array.sum(b['b0,b3,b6']).shape == (19, 2, 6)
    assert array.sum(b[['b0', 'b3', 'b6']]).shape == (19, 2, 6)
    assert array.sum(b['b0', 'b3', 'b6']).shape == (19, 2, 6)
    assert array.sum(b['b0,b3,b6']).shape == (19, 2, 6)

    assert array.sum(b[:]).shape == (19, 2, 6)
    assert array.sum(b[':']).shape == (19, 2, 6)
    assert array.sum(b[:]).shape == (19, 2, 6)

    # Include everything between two labels. Since b0 is the first label
    # and b3 is the last one, this should be equivalent to the previous
    # tests.
    assert array.sum(b['b0:b3']).shape == (19, 2, 6)
    assert_larray_equal(array.sum(b['b0:b3']), array.sum(b))
    assert_larray_equal(array.sum(b['b0':'b3']), array.sum(b))

    # a.2) a tuple of one group => do not collapse dimension
    assert array.sum((b[:],)).shape == (19, 1, 2, 6)

    # a.3) several groups
    # string groups
    assert array.sum((g1, g2, g3)).shape == (19, 3, 2, 6)

    # XXX: do we also want to support this? I do not really like it because
    # it gets tricky when we have some other axes into play. For now the
    # error message is unclear because it first aggregates on "g1", then
    # tries to aggregate on "g2", but there is no "b" dimension anymore.
    # assert array.sum(g1, g2, g3).shape == (19, 3, 2, 6)

    # with one label in several groups
    assert array.sum((c['c0'], c[['c0', 'c1']])).shape == (19, 12, 2, 6)
    assert array.sum((c['c0'], c['c0', 'c1'])).shape == (19, 12, 2, 6)
    assert array.sum((c['c0'], c['c0,c1'])).shape == (19, 12, 2, 6)
    # XXX: do we want to support this?
    # assert array.sum(c['c0;c0,c1']).shape == (19, 12, 2, 6)

    res = array.sum((g1, g2, g3, g_all))
    assert res.shape == (19, 4, 2, 6)

    # a.4) several dimensions at the same time
    # res = array.sum(d['d1,d3;d2,d5;d1:'], (g1, g2, g3, g_all))
    # assert res.shape == (19, 4, 2, 3)
    res = array.sum((d['d1,d3'], d['d2,d5'], d[:]), (g1, g2, g3, g_all))
    assert res.shape == (19, 4, 2, 3)

    # b) both axis aggregate and group aggregate at the same time
    res = array.sum(a, c, (g1, g2, g3, g_all))
    assert res.shape == (4, 6)

    # c) chain group aggregate after axis aggregate
    res = array.sum(a, c).sum((g1, g2, g3, g_all))
    assert res.shape == (4, 6)


def test_group_agg_one_axis():
    a = Axis(range(3), 'a')
    arr = ndtest(a)
    raw = arr.data

    assert arr.sum(a[0, 2]) == raw[[0, 2]].sum()


def test_group_agg_anonymous_axis():
    arr = ndtest([Axis(2), Axis(3)])
    a1, a2 = arr.axes
    raw = arr.data
    assert_nparray_equal(arr.sum(a2[0, 2]).data, raw[:, [0, 2]].sum(1))


def test_group_agg_zero_padded_label():
    arr = ndtest("a=01,02,03,10,11; b=b0..b2")
    expected = Array([36, 30, 39], "a=01_03,10,11")
    assert_larray_equal(arr.sum("01,02,03 >> 01_03; 10; 11", "b"), expected)


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
    assert_larray_equal(b.sum('b1:'), expected)


# TODO: fix this (and add other tests for references (X.) to anonymous axes
# def test_group_agg_anonymous_axis_ref():
#     arr = ndtest([Axis(2), Axis(3)])
#     raw = arr.data
#     # this does not work because x[1] refers to an axis with name 1,
#     # which does not exist. We might want to change this.
#     assert_nparray_equal(arr.sum(x[1][0, 2]).data, raw[:, [0, 2]].sum(1))


# group aggregates on a group-aggregated array
def test_group_agg_on_group_agg(array):
    a, b, c, d = array.axes
    g1, g2, g3 = b_group1, b_group2, b_group3

    agg_arr = array.sum(a, c).sum(b=(g1, g2, g3, all_b))

    # 1) one group => collapse dimension
    assert agg_arr.sum(d='d1,d2').shape == (4,)

    # 2) a tuple of one group => do not collapse dimension
    assert agg_arr.sum(d=('d1,d2',)).shape == (4, 1)

    # 3) several groups
    assert agg_arr.sum(d='d1;d2;:').shape == (4, 3)

    # this is INVALID
    # TODO: raise a nice exception
    # agg_sum = agg_arr.sum(d='d1,d2,:')

    # this is currently allowed even though it can be confusing:
    # d1 and d2 are both groups with one element each.
    assert agg_arr.sum(d=('d1', 'd2', ':')).shape == (4, 3)
    assert agg_arr.sum(d=('d1', 'd2', d[:])).shape == (4, 3)

    # explicit groups are better
    assert agg_arr.sum(d=('d1,', 'd2,', ':')).shape == (4, 3)
    assert agg_arr.sum(d=(['d1'], ['d2'], ':')).shape == (4, 3)

    # 4) groups on the aggregated dimension

    # assert agg_arr.sum(b=([g1, g3], [g2, g3])).shape == (2, 3)
    # g1, g2, g3


# group aggregates on a group-aggregated array
def test_group_agg_on_group_agg_nokw(array):
    a, b, c, d = array.axes
    g1, g2, g3 = b_group1, b_group2, b_group3

    agg_arr = array.sum(a, c).sum((g1, g2, g3, all_b))
    # XXX: should this be supported too? (it currently fails)
    # agg_arr = array.sum(a, c).sum(g1, g2, g3, all_b)

    # 1) one group => collapse dimension
    assert agg_arr.sum('d1,d2').shape == (4,)

    # 2) a tuple of one group => do not collapse dimension
    assert agg_arr.sum(('d1,d2',)).shape == (4, 1)

    # 3) several groups
    # : is ambiguous
    # assert agg_arr.sum('d1;d2;:').shape == (4, 3)
    assert agg_arr.sum('d1;d2;d1:').shape == (4, 3)

    # this is INVALID
    # TODO: raise a nice exception
    # agg_sum = agg_arr.sum(d='d1,d2,:')

    # this is currently allowed even though it can be confusing:
    # d1 and d2 are both groups with one element each.
    assert agg_arr.sum(('d1', 'd2', 'd1:')).shape == (4, 3)
    assert agg_arr.sum(('d1', 'd2', d[:])).shape == (4, 3)

    # explicit groups are better
    assert agg_arr.sum(('d1,', 'd2,', 'd1:')).shape == (4, 3)
    assert agg_arr.sum((['d1'], ['d2'], 'd1:')).shape == (4, 3)

    # 4) groups on the aggregated dimension

    # assert agg_arr.sum(b=([g1, g3], [g2, g3])).shape == (2, 3)
    # g1, g2, g3


def test_getitem_on_group_agg(array):
    a, b, c, d = array.axes

    # using a string (b_group1 is a string key)
    agg_arr = array.sum(a, c).sum(b=b_groups_all)

    # the following are all equivalent
    assert agg_arr[b_group1].shape == (6,)
    assert agg_arr[(b_group1,)].shape == (6,)
    assert agg_arr[(b_group1, slice(None))].shape == (6,)
    assert agg_arr[b_group1, slice(None)].shape == (6,)
    assert agg_arr[b_group1, :].shape == (6,)

    # one more level...
    assert agg_arr[b_group1]['d3'] == 355642.0

    # using an anonymous LGroup
    lg1 = b[b_group1]
    agg_arr = array.sum(a, c).sum(b=(lg1, b_group2, b_group2, all_b))
    with must_warn(FutureWarning, msg=group_as_aggregated_label_msg(lg1), num_expected=5):
        # the following should all be equivalent
        assert agg_arr[lg1].shape == (6,)
        assert agg_arr[(lg1,)].shape == (6,)
        # these last three are only syntactic sugar differences
        # (__getitem__ receives the *exact* same key)
        assert agg_arr[(lg1, slice(None))].shape == (6,)
        assert agg_arr[lg1, slice(None)].shape == (6,)
        assert agg_arr[lg1, :].shape == (6,)

    # using a named LGroup
    lg1 = b[b_group1] >> 'g1'
    agg_arr = array.sum(a, c).sum(b=(lg1, b_group2, b_group2, all_b))
    with must_warn(FutureWarning, msg=group_as_aggregated_label_msg(lg1), num_expected=5):
        # the following are all equivalent
        assert agg_arr[lg1].shape == (6,)
        assert agg_arr[(lg1,)].shape == (6,)
        assert agg_arr[(lg1, slice(None))].shape == (6,)
        assert agg_arr[lg1, slice(None)].shape == (6,)
        assert agg_arr[lg1, :].shape == (6,)


def test_getitem_on_group_agg_nokw(array):
    a, b, c, d = array.axes

    # using a string
    agg_arr = array.sum(a, c).sum((b_group1, b_group2, b_group3, all_b))
    # the following are all equivalent
    # b_group1 is a string key
    assert agg_arr[b_group1].shape == (6,)
    assert agg_arr[(b_group1,)].shape == (6,)
    assert agg_arr[(b_group1, slice(None))].shape == (6,)
    assert agg_arr[b_group1, slice(None)].shape == (6,)
    assert agg_arr[b_group1, :].shape == (6,)

    # one more level...
    assert agg_arr[b_group1]['d3'] == 355642.0

    # using an anonymous LGroup
    lg1 = b[b_group1]
    agg_arr = array.sum(a, c).sum((lg1, b_group2, b_group3, all_b))
    with must_warn(FutureWarning, msg=group_as_aggregated_label_msg(lg1), num_expected=5):
        # the following are all equivalent
        assert agg_arr[lg1].shape == (6,)
        assert agg_arr[(lg1,)].shape == (6,)
        assert agg_arr[(lg1, slice(None))].shape == (6,)
        assert agg_arr[lg1, slice(None)].shape == (6,)
        assert agg_arr[lg1, :].shape == (6,)

    # using a named LGroup
    lg1 = b[b_group1] >> 'g1'
    agg_arr = array.sum(a, c).sum((lg1, b_group2, b_group3, all_b))
    with must_warn(FutureWarning, msg=group_as_aggregated_label_msg(lg1), num_expected=5):
        # the following are all equivalent
        assert agg_arr[lg1].shape == (6,)
        assert agg_arr[(lg1,)].shape == (6,)
        assert agg_arr[(lg1, slice(None))].shape == (6,)
        assert agg_arr[lg1, slice(None)].shape == (6,)
        assert agg_arr[lg1, :].shape == (6,)


def test_filter_on_group_agg(array):
    a, b, c, d = array.axes

    # using a string
    # g1 = b_group1
    # agg_arr = array.sum(a, c).sum(b=(g1, b_group2, b_group3, all_b))
    # assert agg_arr.filter(b=g1).shape == (6,)

    # using a named LGroup
    g1 = b[b_group1] >> 'g1'
    agg_arr = array.sum(a, c).sum(b=(g1, b_group2, b_group3, all_b))
    with must_warn(FutureWarning, msg=group_as_aggregated_label_msg(g1)):
        assert agg_arr.filter(b=g1).shape == (6,)

    # Note that agg_arr.filter(b=(g1,)) does NOT work. It might be a
    # little confusing for users, because agg_arr[(g1,)] works but it is
    # normal because agg_arr.filter(b=(g1,)) is equivalent to:
    # agg_arr[((g1,),)] or agg_arr[(g1,), :]

    # mixed LGroup/string slices
    a0to5 = a[:5]
    a0to5_named = a[:5] >> 'a0to5'
    a6to13 = a[6:13]
    a14plus = a[14:]

    bya = array.sum(a=(a0to5, 5, a6to13, a14plus))
    assert bya.shape == (4, 12, 2, 6)

    bya = array.sum(a=(a0to5, slice(5, 10), a6to13, a14plus))
    assert bya.shape == (4, 12, 2, 6)

    # filter on an aggregated larray created with mixed groups
    # assert bya.filter(a=':17').shape == (12, 2, 6)

    bya = array.sum(a=(a0to5_named, 5, a6to13, a14plus))
    with must_warn(FutureWarning, msg=group_as_aggregated_label_msg(a0to5_named)):
        assert bya.filter(a=a0to5_named).shape == (12, 2, 6)


def test_sum_several_lg_groups(array):
    # 1) aggregated array created using LGroups
    # -----------------------------------------
    lg1 = b[b_group1] >> 'lg1'
    lg2 = b[b_group2] >> 'lg2'
    lg3 = b[b_group3] >> 'lg3'

    agg_arr = array.sum(b=(lg1, lg2, lg3))
    assert agg_arr.shape == (19, 3, 2, 6)

    # the result is indexable
    # 1.a) by LGroup
    msg1 = group_as_aggregated_label_msg(lg1)
    with must_warn(FutureWarning, msg=msg1):
        assert agg_arr.filter(b=lg1).shape == (19, 2, 6)
    msg2 = group_as_aggregated_label_msg(lg2)
    with must_warn(FutureWarning, msg=(msg1, msg2), check_file=False):
        assert agg_arr.filter(b=(lg1, lg2)).shape == (19, 2, 2, 6)

    # 1.b) by string (name of groups)
    assert agg_arr.filter(b='lg1').shape == (19, 2, 6)
    assert agg_arr.filter(b='lg1,lg2').shape == (19, 2, 2, 6)

    # 2) aggregated array created using string groups
    # -----------------------------------------------
    agg_arr = array.sum(b=(b_group1, b_group2, b_group3))
    assert agg_arr.shape == (19, 3, 2, 6)

    # the result is indexable
    # 2.a) by string (def)
    # assert agg_arr.filter(b=b_group1).shape == (19, 2, 6)
    assert agg_arr.filter(b=(b_group1, b_group2)).shape == (19, 2, 2, 6)

    # 2.b) by LGroup
    # assert agg_arr.filter(b=lg1).shape == (19, 2, 6)
    # assert agg_arr.filter(b=(lg1, lg2)).shape == (19, 2, 2, 6)


def test_sum_with_groups_from_other_axis(small_array):
    # use a group from another *compatible* axis
    d2 = Axis('d=d1..d6')
    assert small_array.sum(d=d2['d1,d3']).shape == (2,)

    # use (compatible) group from another *incompatible* axis
    # XXX: I am unsure whether this should be allowed. Maybe we
    # should simply check that the group is valid in axis, but that
    # will trigger a pretty meaningful error anyway
    d3 = Axis('d=d1,d3,d5')
    assert small_array.sum(d3['d1,d3']).shape == (2,)

    # use a group (from another axis) which is incompatible with the axis of
    # the same name in the array
    d4 = Axis('d=d1,d3,d7')
    codes = "'d1', 'd2', 'd3', 'd4', 'd5', 'd6'"
    with must_raise(ValueError, f"d['d1', 'd7'] is not a valid label for the 'd' axis with labels: {codes}"):
        small_array.sum(d4['d1,d7'])


def test_agg_kwargs(array):
    raw = array.data

    # dtype
    assert array.sum(dtype=int) == raw.sum(dtype=int)

    # ddof
    assert array.std(ddof=0) == raw.std(ddof=0)

    # out
    res = array.std(X.c)
    out = zeros_like(res)
    array.std(X.c, out=out)
    assert_larray_equal(res, out)


def test_agg_by(array):
    a, b, c, d = array.axes
    g1, g2, g3 = b_group1, b_group2, b_group3

    # no group or axis
    assert array.sum_by().shape == ()
    assert array.sum_by() == array.sum()

    # all axes
    assert array.sum_by(b, a, d, c).equals(array)
    assert array.sum_by(a, b, c, d).equals(array)

    # a) group aggregate on a fresh array

    # a.1) one group
    res = array.sum_by(b='b0,b3,b6')
    assert res.shape == ()
    assert res == array.sum(b='b0,b3,b6').sum()

    # a.2) a tuple of one group
    res = array.sum_by(b=(b[:],))
    assert res.shape == (1,)
    assert_larray_equal(res, array.sum(a, c, d, b=(b[:],)))

    # a.3) several groups
    # string groups
    res = array.sum_by(b=(g1, g2, g3))
    assert res.shape == (3,)
    assert_larray_equal(res, array.sum(a, c, d, b=(g1, g2, g3)))

    # with one label in several groups
    assert array.sum_by(c=(['c0'], ['c0', 'c1'])).shape == (2,)
    assert array.sum_by(c=('c0', 'c0,c1')).shape == (2,)

    res = array.sum_by(c='c0;c0,c1')
    assert res.shape == (2,)
    assert_larray_equal(res, array.sum(a, b, d, c='c0;c0,c1'))

    # a.4) several dimensions at the same time
    res = array.sum_by(b=(g1, g2, g3, all_b), d='d1,d3;d2,d5;:')
    assert res.shape == (4, 3)
    assert_larray_equal(res, array.sum(a, c, b=(g1, g2, g3, all_b), d='d1,d3;d2,d5;:'))

    # b) both axis aggregate and group aggregate at the same time
    # Note that you must list "full axes" aggregates first (Python does not allow non-kwargs after kwargs.
    res = array.sum_by(c, b=(g1, g2, g3, all_b))
    assert res.shape == (4, 2)
    assert_larray_equal(res, array.sum(a, d, b=(g1, g2, g3, all_b)))

    # c) chain group aggregate after axis aggregate
    res = array.sum_by(b, c)
    assert res.shape == (12, 2)
    assert_larray_equal(res, array.sum(a, d))

    res2 = res.sum_by(b=(g1, g2, g3, all_b))
    assert res2.shape == (4,)
    assert_larray_equal(res2, res.sum(c, b=(g1, g2, g3, all_b)))


def test_agg_igroup():
    arr = ndtest(3)
    res = arr.sum((X.a.i[:2], X.a.i[1:]))
    assert list(res.a.labels) == [':a1', 'a1:']


def test_ratio():
    arr = ndtest((3, 4))
    a, b = arr.axes

    expected = arr / arr.sum()
    res = arr.ratio()
    assert np.isclose(res.sum(), 1.0)

    assert_larray_equal(res, expected)
    assert_larray_equal(arr.ratio(a, b), expected)
    assert_larray_equal(arr.ratio('a', 'b'), expected)
    assert_larray_equal(arr.ratio(X.a, X.b), expected)

    expected = arr / arr.sum(a)
    assert_larray_equal(arr.ratio(a), expected)
    assert_larray_equal(arr.ratio('a'), expected)
    assert_larray_equal(arr.ratio(X.a), expected)

    expected = arr / arr.sum(b)
    assert_larray_equal(arr.ratio(b), expected)
    assert_larray_equal(arr.ratio('b'), expected)
    assert_larray_equal(arr.ratio(X.b), expected)


def test_percent():
    arr = ndtest((3, 4))
    a, b = arr.axes

    expected = arr * 100.0 / arr.sum()
    res = arr.percent()
    assert np.isclose(res.sum(), 100.0)

    assert_larray_equal(res, expected)
    assert_larray_equal(arr.percent(a, b), expected)
    assert_larray_equal(arr.percent('a', 'b'), expected)
    assert_larray_equal(arr.percent(X.a, X.b), expected)

    expected = arr * 100.0 / arr.sum(a)
    assert_larray_equal(arr.percent(a), expected)
    assert_larray_equal(arr.percent('a'), expected)
    assert_larray_equal(arr.percent(X.a), expected)

    expected = arr * 100.0 / arr.sum(b)
    assert_larray_equal(arr.percent(b), expected)
    assert_larray_equal(arr.percent('b'), expected)
    assert_larray_equal(arr.percent(X.b), expected)


def test_total(array):
    a, b, c, d = array.axes
    # array = small_array
    # c, d = array.axes

    assert array.with_total().shape == (20, 13, 3, 7)
    assert array.with_total(c).shape == (19, 12, 3, 6)
    assert array.with_total(d).shape == (19, 12, 2, 7)
    assert array.with_total(c, d).shape == (19, 12, 3, 7)

    g1 = b[b_group1] >> 'g1'
    g2 = b[b_group2] >> 'g2'
    g3 = b[b_group3] >> 'g3'
    g_all = b[:] >> 'Belgium'

    assert array.with_total(b=(g1, g2, g3), op=mean).shape == (19, 15, 2, 6)
    assert array.with_total((g1, g2, g3), op=mean).shape == (19, 15, 2, 6)
    # works but "wrong" for X.b (double what is expected because it includes g1 g2 & g3)
    # TODO: we probably want to display a warning (or even an error?) in that case.
    # If we really want that behavior, we can still split the operation:
    # .with_total((g1, g2, g3)).with_total(X.b)
    # OR we might want to only sum the axis as it was before the op (but that does not play well when working with
    #    multiple axes).
    arr1 = array.with_total(X.c, (g1, g2, g3), X.b, X.d)
    assert arr1.shape == (19, 16, 3, 7)

    # correct total but the order is not very nice
    arr2 = array.with_total(X.c, X.b, (g1, g2, g3), X.d)
    assert arr2.shape == (19, 16, 3, 7)

    # the correct way to do it
    arr3 = array.with_total(X.c, (g1, g2, g3, g_all), X.d)
    assert arr3.shape == (19, 16, 3, 7)

    # adding two groups at once
    # arr4 = array.with_total((d[':d5'], d['d5:']), op=mean)
    arr4 = array.with_total((':d4', 'd4:'), op=mean)
    assert arr4.shape == (19, 12, 2, 8)


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
    # assert reordered.shape == (2, 4, 3)

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

    assert_nparray_equal((small_array + small_array).data, raw + raw)
    assert_nparray_equal((small_array + 1).data, raw + 1)
    assert_nparray_equal((1 + small_array).data, 1 + raw)

    assert_nparray_equal((small_array - small_array).data, raw - raw)
    assert_nparray_equal((small_array - 1).data, raw - 1)
    assert_nparray_equal((1 - small_array).data, 1 - raw)

    assert_nparray_equal((small_array * small_array).data, raw * raw)
    assert_nparray_equal((small_array * 2).data, raw * 2)
    assert_nparray_equal((2 * small_array).data, 2 * raw)

    with np.errstate(invalid='ignore'):
        raw_res = raw / raw

    warn_msg = "invalid value (NaN) encountered during operation (this is typically caused by a 0 / 0)"
    with must_warn(RuntimeWarning, msg=warn_msg):
        res = small_array / small_array
    assert_nparray_nan_equal(res.data, raw_res)

    assert_nparray_equal((small_array / 2).data, raw / 2)

    with np.errstate(divide='ignore'):
        raw_res = 30 / raw
    with must_warn(RuntimeWarning, msg="divide by zero encountered during operation"):
        res = 30 / small_array
    assert_nparray_equal(res.data, raw_res)

    assert_nparray_equal((30 / (small_array + 1)).data, 30 / (raw + 1))

    raw_int = raw.astype(int)
    la_int = Array(raw_int, axes=(c, d))
    assert_nparray_equal((la_int / 2).data, raw_int / 2)
    assert_nparray_equal((la_int // 2).data, raw_int // 2)

    # test adding two larrays with different axes order
    assert_nparray_equal((small_array + small_array.transpose()).data, raw * 2)

    # mixed operations
    raw2 = raw / 2
    la_raw2 = small_array - raw2
    assert la_raw2.axes == small_array.axes
    assert_nparray_equal(la_raw2.data, raw - raw2)
    raw2_la = raw2 - small_array
    assert raw2_la.axes == small_array.axes
    assert_nparray_equal(raw2_la.data, raw2 - raw)

    la_ge_raw2 = small_array >= raw2
    assert la_ge_raw2.axes == small_array.axes
    assert_nparray_equal(la_ge_raw2.data, raw >= raw2)

    raw2_ge_la = raw2 >= small_array
    assert raw2_ge_la.axes == small_array.axes
    assert_nparray_equal(raw2_ge_la.data, raw2 >= raw)

    # arrays filled with None
    arr = full(small_array.axes, fill_value=None)
    res = arr == None                                                       # noqa: E711
    assert_larray_equal(res, ones(small_array.axes, dtype=bool))

    # Array + Axis
    arr = ndtest('a=0..10')
    res = arr + arr.a
    assert_larray_equal(res, arr + asarray(arr.a))

    # Array + <unsupported type>
    with must_raise(TypeError, "unsupported operand type(s) for +: 'Array' and 'object'"):
        res = arr + object()

    # Array + <unsupported type which implements the reverse op>
    class Test:
        def __radd__(self, other):
            return "success"

    res = arr + Test()
    assert res == 'success'


def test_binary_ops_no_name_axes(small_array):
    raw = small_array.data
    raw2 = small_array.data + 1
    arr = ndtest([Axis(length) for length in small_array.shape])
    arr2 = ndtest([Axis(length) for length in small_array.shape]) + 1

    assert_nparray_equal((arr + arr2).data, raw + raw2)
    assert_nparray_equal((arr + 1).data, raw + 1)
    assert_nparray_equal((1 + arr).data, 1 + raw)

    assert_nparray_equal((arr - arr2).data, raw - raw2)
    assert_nparray_equal((arr - 1).data, raw - 1)
    assert_nparray_equal((1 - arr).data, 1 - raw)

    assert_nparray_equal((arr * arr2).data, raw * raw2)
    assert_nparray_equal((arr * 2).data, raw * 2)
    assert_nparray_equal((2 * arr).data, 2 * raw)

    assert_nparray_nan_equal((arr / arr2).data, raw / raw2)
    assert_nparray_equal((arr / 2).data, raw / 2)

    with np.errstate(divide='ignore'):
        raw_res = 30 / raw
    with must_warn(RuntimeWarning, msg="divide by zero encountered during operation"):
        res = 30 / arr
    assert_nparray_equal(res.data, raw_res)

    assert_nparray_equal((30 / (arr + 1)).data, 30 / (raw + 1))

    raw_int = raw.astype(int)
    la_int = Array(raw_int)
    assert_nparray_equal((la_int / 2).data, raw_int / 2)
    assert_nparray_equal((la_int // 2).data, raw_int // 2)

    # adding two larrays with different axes order cannot work with unnamed axes
    # assert_nparray_equal((arr + arr.transpose()).data, raw * 2)

    # mixed operations
    raw2 = raw / 2
    la_raw2 = arr - raw2
    assert la_raw2.axes == arr.axes
    assert_nparray_equal(la_raw2.data, raw - raw2)
    raw2_la = raw2 - arr
    assert raw2_la.axes == arr.axes
    assert_nparray_equal(raw2_la.data, raw2 - raw)

    la_ge_raw2 = arr >= raw2
    assert la_ge_raw2.axes == arr.axes
    assert_nparray_equal(la_ge_raw2.data, raw >= raw2)

    raw2_ge_la = raw2 >= arr
    assert raw2_ge_la.axes == arr.axes
    assert_nparray_equal(raw2_ge_la.data, raw2 >= raw)


def test_broadcasting_no_name():
    a = ndtest([Axis(2), Axis(3)])
    b = ndtest(Axis(3))
    c = ndtest(Axis(2))

    # FIXME: this error message is awful
    with must_raise(ValueError, "axis {?}* is not compatible with {?}*"):
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
    assert np.array_equal(d, [[0, 1,  4],   # noqa: E241
                              [0, 4, 10]])

    with must_raise(ValueError, "operands could not be broadcast together with shapes (2,3) (2,) "):
        np.asarray(a) * np.asarray(c)


def test_binary_ops_with_scalar_group():
    time = Axis('time=206..2019')
    arr = ndtest(3)
    expected = arr + 206
    assert_larray_equal(time.i[0] + arr, expected)
    assert_larray_equal(arr + time.i[0], expected)


def test_comparison_ops():
    # simple array equality (identity)
    a = Axis('a=a0,a1,a2')
    arr = ndtest(a)
    res = arr == arr
    expected = ones(a)
    assert_larray_equal(res, expected)

    # simple array equality
    arr = ndtest(a)
    res = arr == zeros(a)
    expected = Array([True, False, False], a)
    assert_larray_equal(res, expected)

    # invalid types
    # a) eq
    arr = ndtest(3)
    res = arr == object()
    assert res is False

    # b) ne
    res = arr != object()
    assert res is True

    # c) others
    with must_raise(TypeError, "'>' not supported between instances of 'Array' and 'object'"):
        res = arr > object()

    # d) other type implementing the reverse comparison
    class Test:
        def __lt__(self, other):
            return "success"

    res = arr > Test()
    assert res == 'success'


def test_unary_ops(small_array):
    raw = small_array.data

    # using numpy functions
    assert_nparray_equal(np.abs(small_array - 10).data, np.abs(raw - 10))
    assert_nparray_equal(np.negative(small_array).data, np.negative(raw))
    assert_nparray_equal(np.invert(small_array).data, np.invert(raw))

    # using python builtin ops
    assert_nparray_equal(abs(small_array - 10).data, abs(raw - 10))
    assert_nparray_equal((-small_array).data, -raw)
    assert_nparray_equal((+small_array).data, +raw)
    assert_nparray_equal((~small_array).data, ~raw)


def test_binary_ops_expressions():
    arr = ndtest("age=0..5")

    expected = arr.copy()
    expected[3] = 42

    res = arr * (X.age != 3) + 42 * (X.age == 3)
    assert_larray_equal(res, expected)
    res = (X.age != 3) * arr + 42 * (X.age == 3)
    assert_larray_equal(res, expected)
    res = 42 * (X.age == 3) + arr * (X.age != 3)
    assert_larray_equal(res, expected)
    res = (X.age == 3) * 42 + arr * (X.age != 3)
    assert_larray_equal(res, expected)

    with must_raise(ValueError, "Cannot evaluate the truth value of an expression using X.axis_name"):
        res = 0 if X.age == 3 else 1


def test_mean(small_array):
    raw = small_array.data
    c, d = small_array.axes
    assert_nparray_equal((small_array.mean(d)).data, raw.mean(1))


def test_sequence():
    # int initial, (default) int inc
    res = sequence('a=a0..a2', initial=1)
    assert_larray_equal(1 + ndtest(3), res)

    # int initial, int inc, axes defined but str axis
    res = sequence('a=a0..a2', initial=1, inc=2, axes='a=a0..a2;b=b0,b1')
    expected = from_string(r"""
    a\b  b0  b1
     a0   1   1
     a1   3   3
     a2   5   5""")
    assert_larray_equal(res, expected)

    # int initial, float inc
    res = sequence('a=a0..a2', initial=1, inc=1.5)
    assert_larray_equal(res, 1 + ndtest(3) * 1.5)

    # array initial, (default) int inc
    res = sequence('b=b0..b2', initial=ndtest(2) * 3)
    assert_larray_equal(res, ndtest((2, 3)))

    # array initial, float inc
    res = sequence('b=b0..b2', initial=ndtest(2), inc=1.5)
    expected = from_string(r"""
    a\b   b0   b1   b2
     a0  0.0  1.5  3.0
     a1  1.0  2.5  4.0""")
    assert_larray_equal(res, expected)

    # array initial, float mult
    res = sequence('b=b0..b2', initial=1 + ndtest(2), mult=1.5)
    expected = from_string(r"""
    a\b   b0   b1    b2
     a0  1.0  1.5  2.25
     a1  2.0  3.0   4.5""")
    assert_larray_equal(res, expected)

    # array initial, int array mult
    initial = from_string("""
    a  a0  a1
    \t  1   2""")
    mult = from_string(r"""
    a\b b1 b2
     a0  1  2
     a1  2  1""")
    res = sequence('b=b0..b2', initial=initial, mult=mult)
    expected = from_string(r"""
    a\b b0 b1 b2
     a0  1  1  2
     a1  2  4  4""")
    assert_larray_equal(res, expected)

    # array initial, float array mult
    initial = from_string("""
    a  a0  a1
    \t  1   2""")
    mult = from_string(r"""
    a\b   b1   b2
     a0  1.0  2.0
     a1  2.0  1.0""")
    res = sequence('b=b0..b2', initial=initial, mult=mult)
    expected = from_string(r"""
    a\b   b0   b1   b2
     a0  1.0  1.0  2.0
     a1  2.0  4.0  4.0""")
    assert_larray_equal(res, expected)


def test_sort_values():
    # 1D arrays
    arr = Array([0, 1, 6, 3, -1], "a=a0..a4")
    res = arr.sort_values()
    expected = Array([-1, 0, 1, 3, 6], "a=a4,a0,a1,a3,a2")
    assert_larray_equal(res, expected)
    # ascending arg
    res = arr.sort_values(ascending=False)
    expected = Array([6, 3, 1, 0, -1], "a=a2,a3,a1,a0,a4")
    assert_larray_equal(res, expected)

    # 3D arrays
    arr = Array([[[10, 2, 4], [3, 7, 1]], [[5, 1, 6], [2, 8, 9]]],
                'a=a0,a1; b=b0,b1; c=c0..c2')
    res = arr.sort_values(axis='c')
    expected = Array([[[2, 4, 10], [1, 3, 7]], [[1, 5, 6], [2, 8, 9]]],
                     [Axis('a=a0,a1'), Axis('b=b0,b1'), Axis(3, 'c')])
    assert_larray_equal(res, expected)


def test_set_labels(small_array):
    small_array.set_labels(X.c, ['Man', 'Woman'], inplace=True)
    expected = small_array.set_labels(X.c, ['Man', 'Woman'])
    assert_larray_equal(small_array, expected)


def test_set_axes(small_array):
    d2 = Axis([label.replace('P', 'Q') for label in d.labels], 'd2')
    c2 = Axis(['Man', 'Woman'], 'c2')

    arr = Array(small_array.data, axes=(c, d2))
    # replace one axis
    arr2 = small_array.set_axes(X.d, d2)
    assert_larray_equal(arr, arr2)

    arr = Array(small_array.data, axes=(c2, d2))
    # all at once
    arr2 = small_array.set_axes([c2, d2])
    assert_larray_equal(arr, arr2)
    # using keywrods args
    arr2 = small_array.set_axes(c=c2, d=d2)
    assert_larray_equal(arr, arr2)
    # using dict
    arr2 = small_array.set_axes({X.c: c2, X.d: d2})
    assert_larray_equal(arr, arr2)
    # using list of pairs (axis_to_replace, new_axis)
    arr2 = small_array.set_axes([(X.c, c2), (X.d, d2)])
    assert_larray_equal(arr, arr2)


def test_reindex():
    # simple case (1D array, one axis reindexed)
    arr = ndtest(2)
    a = arr.a  # == Axis('a=a0,a1')
    new_a = Axis('a=a0,a1,a2')

    expected = from_string(""" a   a0   a1   a2
                              \t  0.0  1.0  nan""")

    # using the axis name
    res = arr.reindex('a', new_a)
    assert_larray_nan_equal(res, expected)

    # using an axis reference
    res = arr.reindex(X.a, new_a)
    assert_larray_nan_equal(res, expected)

    # using an axis position
    res = arr.reindex(0, new_a)
    assert_larray_nan_equal(res, expected)

    # using the actual original axis as axes_to_replace (issue #1088)
    res = arr.reindex(a, new_a)
    assert_larray_nan_equal(res, expected)

    # using another axis equal to the original axis (issue #1088)
    res = arr.reindex(a.copy(), new_a)
    assert_larray_nan_equal(res, expected)

    # using another axis with the same name but different labels
    # (unsure we should support this -- but unsure either it is worth adding extra
    #  code to explicitly raise in that case)
    res = arr.reindex(Axis('a=hello'), new_a)
    assert_larray_nan_equal(res, expected)

    # using a single axis to determine both axis_to_replace and new_axis
    res = arr.reindex(new_a)
    assert_larray_nan_equal(res, expected)

    # using an axis definition for its labels
    res = arr.reindex('a', 'a=a0,a1,a2')
    assert_larray_nan_equal(res, expected)

    # using an axis with a different name as new labels
    # TODO: unsure we should support this as the functionality seems weird
    #       to me. I think we should either raise an error if the axis name
    #       is different (force using other_axis.labels instead
    #       of other_axis) OR do not do use the old name
    #       (and make sure this effectively does a rename)
    res = arr.reindex('a', Axis('a0,a1,a2', 'b'))
    assert_larray_nan_equal(res, expected)

    # using an axis definition with a different name for its labels
    # (the axis name should be ignored - unsure we should support this, see above)
    res = arr.reindex('a', 'b=a0,a1,a2')
    assert_larray_nan_equal(res, expected)

    # using a list as the new labels
    res = arr.reindex('a', ['a0', 'a1', 'a2'])
    assert_larray_nan_equal(res, expected)

    # using the dict syntax
    res = arr.reindex({'a': new_a})
    assert_larray_nan_equal(res, expected)

    # using the dict syntax with a list of labels (issue #1068)
    res = arr.reindex({'a': ['a0', 'a1', 'a2']})
    assert_larray_nan_equal(res, expected)

    # using the dict syntax with a labels def string
    res = arr.reindex({'a': 'a0,a1,a2'})
    assert_larray_nan_equal(res, expected)

    # test error conditions
    msg = ("In Array.reindex, when using an axis reference ('axis name', X.axis_name or "
           "axis_integer_position) as axes_to_reindex, you must provide a value for `new_axis`.")
    with must_raise(TypeError, msg):
        res = arr.reindex('a')

    with must_raise(TypeError, msg):
        res = arr.reindex(X.a)

    with must_raise(TypeError, msg):
        res = arr.reindex(0)

    msg_tmpl = ("In Array.reindex, when `new_axis` is used, `axes_to_reindex`"
                " must be an Axis object or an axis reference ('axis name', "
                "X.axis_name or axis_integer_position) but got {obj_str} "
                "(which is of type {obj_type}) instead.")

    msg = msg_tmpl.format(obj_str="[Axis(['a0', 'a1'], 'a')]", obj_type='list')
    with must_raise(TypeError, msg):
        res = arr.reindex([a], new_a)

    msg = msg_tmpl.format(obj_str='{a}', obj_type='AxisCollection')
    with must_raise(TypeError, msg):
        res = arr.reindex(AxisCollection([a]), new_a)

    msg = msg_tmpl.format(
        obj_str="{Axis(['a0', 'a1'], 'a'): Axis(['a0', 'a1', 'a2'], 'a')}",
        obj_type='dict'
    )
    with must_raise(TypeError, msg):
        res = arr.reindex({a: new_a}, new_a)

    # 2d array, one axis reindexed
    arr = ndtest((2, 2))
    res = arr.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=-1)
    assert_larray_equal(res, from_string(r"""a\b  b1  b2  b0
                                              a0   1  -1   0
                                              a1   3  -1   2"""))

    arr2 = ndtest((2, 2))
    arr2.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=-1, inplace=True)
    assert_larray_equal(arr2, from_string(r"""a\b  b1  b2  b0
                                               a0   1  -1   0
                                               a1   3  -1   2"""))

    # Array fill value
    filler = ndtest(arr.a)
    res = arr.reindex(X.b, ['b1', 'b2', 'b0'], fill_value=filler)
    assert_larray_equal(res, from_string(r"""a\b  b1  b2  b0
                                              a0   1   0   0
                                              a1   3   1   2"""))

    # using labels from another axis
    arr = ndtest('a=v0..v2;b=v0,v2,v1,v3')
    res = arr.reindex('a', arr.b.labels, fill_value=-1)
    assert_larray_equal(res, from_string(r"""a\b  v0  v2  v1  v3
                                              v0   0   1   2   3
                                              v2   8   9  10  11
                                              v1   4   5   6   7
                                              v3  -1  -1  -1  -1"""))
    # using another axis for its labels (unsure we should support this, see above)
    res = arr.reindex('a', arr.b, fill_value=-1)
    assert_larray_equal(res, from_string(r"""a\b  v0  v2  v1  v3
                                              v0   0   1   2   3
                                              v2   8   9  10  11
                                              v1   4   5   6   7
                                              v3  -1  -1  -1  -1"""))

    # using an axis definition for its labels (unsure we should support this, see above)
    res = arr.reindex('a', 'b=v0,v2,v1,v3', fill_value=-1)
    assert_larray_equal(res, from_string(r"""a\b  v0  v2  v1  v3
                                              v0   0   1   2   3
                                              v2   8   9  10  11
                                              v1   4   5   6   7
                                              v3  -1  -1  -1  -1"""))

    # passing a list of Axis
    arr = ndtest((2, 2))
    res = arr.reindex([Axis("a=a0,a1"), Axis("c=c0"), Axis("b=b1,b2")], fill_value=-1)
    assert_larray_equal(res, from_string(r""" a  b\c  c0
                                             a0   b1   1
                                             a0   b2  -1
                                             a1   b1   3
                                             a1   b2  -1"""))


def test_expand():
    country = Axis("country=BE,FR,DE")
    arr = ndtest(country)

    out1 = empty((c, country))
    arr.expand(out=out1)

    out2 = empty((c, country))
    out2[:] = arr

    assert_larray_equal(out1, out2)


def test_append():
    arr = ndtest((2, 3))

    # append a scalar
    res = arr.append('b', 6, label='b3')
    expected = from_string(r"""
    a\b  b0  b1  b2  b3
     a0   0   1   2   6
     a1   3   4   5   6""")
    assert_larray_equal(res, expected)

    # append an array without the axis
    value = stack({'a0': 6, 'a1': 7}, 'a')
    res = arr.append('b', value, label='b3')
    expected = from_string(r"""
    a\b  b0  b1  b2  b3
     a0   0   1   2   6
     a1   3   4   5   7""")
    assert_larray_equal(res, expected)

    # when the value has not the axis and label already exists on another axis in array
    value = stack({'a0': 6, 'a1': 7}, 'a')
    res = arr.append('b', value, label='a1')
    expected = from_string(r"""
    a\b  b0  b1  b2  a1
     a0   0   1   2   6
     a1   3   4   5   7""")
    assert_larray_equal(res, expected)

    # when the value already has the axis
    value = stack({'b3': 6, 'b4': 7}, 'b')
    res = arr.append('b', value)
    expected = from_string(r"""
    a\b  b0  b1  b2  b3  b4
     a0   0   1   2   6   7
     a1   3   4   5   6   7""")
    assert_larray_equal(res, expected)

    # when the value already has the axis but one of the appended labels is ambiguous on the value
    value = from_string(r"""
    a\b  b3  a1
     a0   6   7
     a1   8   9""")
    res = arr.append('b', value)
    expected = from_string(r"""
    a\b  b0  b1  b2  b3  a1
     a0   0   1   2   6   7
     a1   3   4   5   8   9""")
    assert_larray_equal(res, expected)



def test_extend(small_array):
    c, d = small_array.axes

    all_d = d[:]
    tail = small_array.sum(d=(all_d,))
    with must_warn(FutureWarning, "extend() is deprecated. Use append() instead."):
        small_array = small_array.extend(d, tail)
    assert small_array.shape == (2, 7)
    # test with a string axis
    value = small_array.sum(c=(c[:],))
    with must_warn(FutureWarning, "extend() is deprecated. Use append() instead."):
        small_array = small_array.extend('c', value)
    assert small_array.shape == (3, 7)


def test_insert():
    # simple tests are in the docstring
    arr1 = ndtest((2, 3))

    # Insert value without label
    res = arr1.insert(42, before='b1')
    expected = from_string(r"""
    a\b  b0  None  b1  b2
     a0   0    42   1   2
     a1   3    42   4   5""")
    assert_larray_equal(res, expected)

    # Specify label if not in the value
    res = arr1.insert(42, before='b1', label='new')
    expected = from_string(r"""
    a\b  b0  new  b1  b2
     a0   0   42   1   2
     a1   3   42   4   5""")
    assert_larray_equal(res, expected)

    # insert at multiple places at once

    # cannot use from_string in these tests because it de-duplicates ambiguous (column) labels automatically
    res = arr1.insert([42, 43], before='b1', label='new')
    expected = from_lists([[r'a\b', 'b0', 'new', 'new', 'b1', 'b2'],
                           [  'a0',    0,    42,    43,    1,    2],   # noqa: E201,E241
                           [  'a1',    3,    42,    43,    4,    5]])  # noqa: E201,E241
    assert_larray_equal(res, expected)

    res = arr1.insert(42, before=['b1', 'b2'], label='new')
    expected = from_lists([[r'a\b', 'b0', 'new', 'b1', 'new', 'b2'],
                           [  'a0',    0,    42,    1,    42,    2],   # noqa: E201,E241
                           [  'a1',    3,    42,    4,    42,    5]])  # noqa: E201,E241
    assert_larray_equal(res, expected)

    res = arr1.insert(42, before='b1', label=['b0.1', 'b0.2'])
    expected = from_string(r"""
    a\b  b0  b0.1  b0.2  b1  b2
     a0   0    42    42   1   2
     a1   3    42    42   4   5""")
    assert_larray_equal(res, expected)

    res = arr1.insert(42, before=['b1', 'b2'], label=['b0.5', 'b1.5'])
    expected = from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    42   2
     a1   3    42   4    42   5""")
    assert_larray_equal(res, expected)

    res = arr1.insert([42, 43], before='b1', label=['b0.1', 'b0.2'])
    expected = from_string(r"""
    a\b  b0  b0.1  b0.2  b1  b2
     a0   0    42    43   1   2
     a1   3    42    43   4   5""")
    assert_larray_equal(res, expected)

    res = arr1.insert([42, 43], before=['b1', 'b2'], label='new')
    expected = from_lists([[r'a\b', 'b0', 'new', 'b1', 'new', 'b2'],
                           [  'a0',    0,    42,    1,    43,    2],   # noqa: E201,E241
                           [  'a1',    3,    42,    4,    43,    5]])  # noqa: E201,E241
    assert_larray_equal(res, expected)

    res = arr1.insert([42, 43], before=['b1', 'b2'], label=['b0.5', 'b1.5'])
    expected = from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    43   2
     a1   3    42   4    43   5""")
    assert_larray_equal(res, expected)

    res = arr1.insert([42, 43], before='b1,b2', label=['b0.5', 'b1.5'])
    expected = from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    43   2
     a1   3    42   4    43   5""")
    assert_larray_equal(res, expected)

    arr2 = ndtest(2)
    res = arr1.insert([arr2 + 42, arr2 + 43], before=['b1', 'b2'], label=['b0.5', 'b1.5'])
    expected = from_string(r"""
    a\b  b0  b0.5  b1  b1.5  b2
     a0   0    42   1    43   2
     a1   3    43   4    44   5""")
    assert_larray_equal(res, expected)

    arr3 = ndtest('a=a0,a1;b=b0.1,b0.2') + 42
    res = arr1.insert(arr3, before='b1,b2')
    expected = from_string(r"""
    a\b  b0  b0.1  b1  b0.2  b2
     a0   0    42   1    43   2
     a1   3    44   4    45   5""")
    assert_larray_equal(res, expected)

    # Override array label
    res = arr1.insert(arr3, before='b1', label='new')
    # cannot use from_string in these tests because it de-duplicates ambiguous (column) labels automatically
    expected = from_lists([[r'a\b', 'b0', 'new', 'new', 'b1', 'b2'],
                           [  'a0',    0,    42,    43,    1,    2],   # noqa: E201,E241
                           [  'a1',    3,    44,    45,    4,    5]])  # noqa: E201,E241
    assert_larray_equal(res, expected)

    # with ambiguous labels on the value
    value = from_string(r"""
    a\b  b3  a1
     a0   6   7
     a1   8   9""")
    res = arr1.append('b', value)
    expected = from_string(r"""
    a\b  b0  b1  b2  b3  a1
     a0   0   1   2   6   7
     a1   3   4   5   8   9""")
    assert_larray_equal(res, expected)

    # with ambiguous labels in the array
    arr4 = ndtest('a=v0,v1;b=v0,v1')
    res = arr4.insert(42, before='b[v1]', label='v0.5')
    expected = from_string(r"""
    a\b  v0  v0.5  v1
     v0   0    42   1
     v1   2    42   3""")
    assert_larray_equal(res, expected)

    res = arr4.insert(42, before=X.b['v1'], label='v0.5')
    assert_larray_equal(res, expected)

    res = arr4.insert(42, before=arr4.b['v1'], label='v0.5')
    assert_larray_equal(res, expected)


def test_drop():
    arr1 = ndtest(3)
    expected = Array([0, 2], 'a=a0,a2')

    # indices
    res = arr1.drop('a.i[1]')
    assert_larray_equal(res, expected)

    res = arr1.drop(X.a.i[1])
    assert_larray_equal(res, expected)

    # labels
    res = arr1.drop(X.a['a1'])
    assert_larray_equal(res, expected)

    res = arr1.drop('a[a1]')
    assert_larray_equal(res, expected)

    # 2D array
    arr2 = ndtest((2, 4))
    expected = from_string(r"""
    a\b  b0  b2
     a0   0   2
     a1   4   6""")
    res = arr2.drop(['b1', 'b3'])
    assert_larray_equal(res, expected)

    res = arr2.drop(X.b['b1', 'b3'])
    assert_larray_equal(res, expected)

    res = arr2.drop('b.i[1, 3]')
    assert_larray_equal(res, expected)

    res = arr2.drop(X.b.i[1, 3])
    assert_larray_equal(res, expected)

    a = Axis('a=label0..label2')
    b = Axis('b=label0..label2')
    arr3 = ndtest((a, b))

    res = arr3.drop('a[label1]')
    assert_larray_equal(res, from_string(r"""
       a\b  label0  label1  label2
    label0       0       1       2
    label2       6       7       8"""))

    # XXX: implement the following (#671)?
    # res = arr3.drop('0[label1]')
    res = arr3.drop(X[0]['label1'])
    assert_larray_equal(res, from_string(r"""
       a\b  label0  label1  label2
    label0       0       1       2
    label2       6       7       8"""))

    res = arr3.drop(a['label1'])
    assert_larray_equal(res, from_string(r"""
       a\b  label0  label1  label2
    label0       0       1       2
    label2       6       7       8"""))


# the aim of this test is to drop the last *value* of an axis, but instead
# of dropping the last axis *label*, drop the first one.
def test_shift_axis(small_array):
    c, d = small_array.axes

    expected = from_string(r"""
    c\d  d2  d3  d4  d5  d6
     c0   0   1   2   3   4
     c1   6   7   8   9  10""")

    res = Array(small_array[:, :'d5'], axes=[c, Axis(d.labels[1:], 'd')])
    assert_larray_equal(res, expected)

    res = Array(small_array[:, :'d5'], axes=[c, d.subaxis(slice(1, None))])
    assert_larray_equal(res, expected)

    # We can also modify the axis in-place (dangerous!)
    # d.labels = np.append(d.labels[1:], d.labels[0])
    res = small_array[:, :'d5']
    res.axes.d.labels = d.labels[1:]
    assert_larray_equal(res, expected)


def test_unique():
    arr = Array([[[0, 2, 0, 0],
                  [1, 1, 1, 0]],
                 [[0, 2, 0, 0],
                  [2, 1, 2, 0]]], 'a=a0,a1;b=b0,b1;c=c0..c3')
    assert_larray_equal(arr.unique('a'), arr)
    assert_larray_equal(arr.unique('b'), arr)
    assert_larray_equal(arr.unique('c'), arr['c0,c1,c3'])
    expected = from_string(r"""
    a_b\c  c0  c1  c2  c3
    a0_b0   0   2   0   0
    a0_b1   1   1   1   0
    a1_b1   2   1   2   0""")
    assert_larray_equal(arr.unique(('a', 'b')), expected)


@needs_pytables
def test_hdf_roundtrip(tmp_path, meta):
    import tables

    fpath = tmp_path / 'test.h5'

    arr = ndtest((2, 3), meta=meta)
    arr.to_hdf(fpath, 'a')
    res = read_hdf(fpath, 'a')

    assert_larray_equal(res, arr)
    assert res.meta == arr.meta

    # issue 72: int-like strings should not be parsed (should round-trip correctly)
    fpath = tmp_path / 'issue72.h5'
    arr = from_lists([['axis', '10', '20'],
                      [    '',    0,    1]])  # noqa: E201,E241
    arr.to_hdf(fpath, 'arr')
    res = read_hdf(fpath, 'arr')
    assert res.ndim == 1
    axis = res.axes[0]
    assert axis.name == 'axis'
    assert list(axis.labels) == ['10', '20']

    # passing group as key to to_hdf
    a3 = ndtest((4, 3, 4))
    fpath = tmp_path / 'test.h5'
    os.remove(fpath)

    # opening read-only file
    arr.to_hdf(fpath, 'arr')
    from stat import S_IRUSR, S_IRGRP, S_IROTH, S_IWUSR, S_IWGRP, S_IWOTH
    os.chmod(fpath, S_IRUSR | S_IRGRP | S_IROTH)
    res = read_hdf(fpath, 'arr')
    os.chmod(fpath, S_IWUSR | S_IWGRP | S_IWOTH)
    os.remove(fpath)

    # single element group
    for label in a3.a:
        a3[label].to_hdf(fpath, label)

    # unnamed group
    group = a3.c['c0,c2']
    msg_template = "object name is not a valid Python identifier: {key!r}; it does not match the " \
                   "pattern ``^[a-zA-Z_][a-zA-Z0-9_]*$``; you will not be able to use natural " \
                   "naming to access this object; using ``getattr()`` will still work, though"
    with must_warn(tables.NaturalNameWarning, msg=msg_template.format(key='c0,c2'), check_file=False):
        a3[group].to_hdf(fpath, group)

    # unnamed group + slice
    group = a3.c['c0::2']
    with must_warn(tables.NaturalNameWarning, msg=msg_template.format(key='c0::2'), check_file=False):
        a3[group].to_hdf(fpath, group)

    # named group
    group = a3.c['c0,c2'] >> 'even'
    a3[group].to_hdf(fpath, group)

    # group with name containing special characters (replaced by _)
    group = a3.c['c0,c2'] >> r':name?with*special/\[characters]'
    with must_warn(tables.NaturalNameWarning, msg=msg_template.format(key=':name?with*special__[characters]'),
                   check_file=False):
        a3[group].to_hdf(fpath, group)

    # passing group as key to read_hdf
    for label in a3.a:
        subset = read_hdf(fpath, label)
        assert_larray_equal(subset, a3[label])

    # load Session
    from larray.core.session import Session
    s = Session(fpath)
    assert s.names == sorted(['a0', 'a1', 'a2', 'a3', 'c0,c2', 'c0::2', 'even', ':name?with*special__[characters]'])


def test_from_string():
    expected = ndtest("c=c0,c1")

    res = from_string(""" c  c0  c1
                         \t   0   1""")
    assert_larray_equal(res, expected)

    res = from_string(r"""  c  c0  c1
                          nan   0   1""")
    assert_larray_equal(res, expected)

    res = from_string(r"""  c  c0  c1
                          NaN   0   1""")
    assert_larray_equal(res, expected)


def test_read_csv():
    res = read_csv(inputpath('test1d.csv'))
    assert_larray_equal(res, io_1d)

    res = read_csv(inputpath('test2d.csv'))
    assert_larray_equal(res, io_2d)

    res = read_csv(inputpath('test3d.csv'))
    assert_larray_equal(res, io_3d)

    res = read_csv(inputpath('testint_labels.csv'))
    assert_larray_equal(res, io_int_labels)

    res = read_csv(inputpath('test2d_classic.csv'))
    assert_larray_equal(res, ndtest("a=a0..a2; b0..b2"))

    arr = read_csv(inputpath('test1d_liam2.csv'), dialect='liam2')
    assert arr.ndim == 1
    assert arr.shape == (3,)
    assert arr.axes.names == ['time']
    assert list(arr.data) == [3722, 3395, 3347]

    arr = read_csv(inputpath('test5d_liam2.csv'), dialect='liam2')
    assert arr.ndim == 5
    assert arr.shape == (2, 5, 2, 2, 3)
    assert arr.axes.names == ['arr', 'age', 'sex', 'nat', 'time']
    assert list(arr[X.arr[1], 0, 'F', X.nat[1], :].data) == [3722, 3395, 3347]

    # missing values
    res = read_csv(inputpath('testmissing_values.csv'))
    assert_larray_nan_equal(res, io_missing_values)

    # test StringIO
    res = read_csv(StringIO('a,1,2\n,0,1\n'))
    assert_larray_equal(res, ndtest('a=1,2'))

    # sort_columns=True
    res = read_csv(StringIO('a,a2,a0,a1\n,2,0,1\n'), sort_columns=True)
    assert_larray_equal(res, ndtest(3))

    #################
    # narrow format #
    #################
    res = read_csv(inputpath('test1d_narrow.csv'), wide=False)
    assert_larray_equal(res, io_1d)

    res = read_csv(inputpath('test2d_narrow.csv'), wide=False)
    assert_larray_equal(res, io_2d)

    res = read_csv(inputpath('test3d_narrow.csv'), wide=False)
    assert_larray_equal(res, io_3d)

    # missing values
    res = read_csv(inputpath('testmissing_values_narrow.csv'), wide=False)
    assert_larray_nan_equal(res, io_narrow_missing_values)

    # unsorted values
    res = read_csv(inputpath('testunsorted_narrow.csv'), wide=False)
    assert_larray_equal(res, io_unsorted)


def test_read_eurostat():
    arr = read_eurostat(inputpath('test5d_eurostat.csv'))
    assert arr.ndim == 5
    assert arr.shape == (2, 5, 2, 2, 3)
    assert arr.axes.names == ['arr', 'age', 'sex', 'nat', 'time']
    # FIXME: integer labels should be parsed as such
    assert list(arr[X.arr['1'], '0', 'F', X.nat['1'], :].data) == [3722, 3395, 3347]


@needs_xlwings
def test_read_excel_xlwings():
    arr = read_excel(inputpath('test.xlsx'), '1d')
    assert_larray_equal(arr, io_1d)

    arr = read_excel(inputpath('test.xlsx'), '2d')
    assert_larray_equal(arr, io_2d)

    arr = read_excel(inputpath('test.xlsx'), '2d_classic')
    assert_larray_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '2d_classic', nb_axes=2)
    assert_larray_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '3d')
    assert_larray_equal(arr, io_3d)

    # for > 2d, specifying nb_axes is required if there is no name for the horizontal axis
    arr = read_excel(inputpath('test.xlsx'), '3d_classic', nb_axes=3)
    assert_larray_equal(arr, ndtest("a=1..3; b=b0,b1; c0..c2"))

    arr = read_excel(inputpath('test.xlsx'), 'int_labels')
    assert_larray_equal(arr, io_int_labels)

    # passing a Group as sheet arg
    axis = Axis('dim=1d,2d,3d,5d')

    arr = read_excel(inputpath('test.xlsx'), axis['1d'])
    assert_larray_equal(arr, io_1d)

    # missing rows, default fill_value
    arr = read_excel(inputpath('test.xlsx'), 'missing_values')
    expected = ndtest("a=1..3; b=b0,b1; c=c0..c2", dtype=float)
    expected[2, 'b0'] = nan
    expected[3, 'b1'] = nan
    assert_larray_nan_equal(arr, expected)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test.xlsx'), 'missing_values', fill_value=42)
    expected = ndtest("a=1..3; b=b0,b1; c=c0..c2", dtype=float)
    expected[2, 'b0'] = 42
    expected[3, 'b1'] = 42
    assert_larray_equal(arr, expected)

    # range
    arr = read_excel(inputpath('test.xlsx'), 'position', range='D3:H9')
    assert_larray_equal(arr, io_3d)

    #################
    # narrow format #
    #################
    arr = read_excel(inputpath('test_narrow.xlsx'), '1d', wide=False)
    assert_larray_equal(arr, io_1d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '2d', wide=False)
    assert_larray_equal(arr, io_2d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '3d', wide=False)
    assert_larray_equal(arr, io_3d)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test_narrow.xlsx'), 'missing_values', fill_value=42, wide=False)
    expected = io_narrow_missing_values.copy()
    expected[isnan(expected)] = 42
    assert_larray_equal(arr, expected)

    # unsorted values
    arr = read_excel(inputpath('test_narrow.xlsx'), 'unsorted', wide=False)
    assert_larray_equal(arr, io_unsorted)

    # range
    arr = read_excel(inputpath('test_narrow.xlsx'), 'position', range='D3:G21', wide=False)
    assert_larray_equal(arr, io_3d)

    ##############################
    #  invalid keyword argument  #
    ##############################

    with must_raise(TypeError, "'dtype' is an invalid keyword argument for this function "
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
    assert_larray_equal(bad1, good)
    assert_larray_equal(bad2, good)
    # with additional empty column in the middle of the array to read
    good2 = ndtest('a=a0,a1;b=2003..2006').astype(object)
    good2[2005] = None
    good2 = good2.set_axes('b', Axis([2003, 2004, None, 2006], 'b'))
    bad3 = read_excel(fpath, 'middleblankcol')
    bad4 = read_excel(fpath, '16384col')
    assert_larray_equal(bad3, good2)
    assert_larray_equal(bad4, good2)


@needs_openpyxl
def test_read_excel_pandas():
    arr = read_excel(inputpath('test.xlsx'), '1d', engine='openpyxl')
    assert_larray_equal(arr, io_1d)

    arr = read_excel(inputpath('test.xlsx'), '2d', engine='openpyxl')
    assert_larray_equal(arr, io_2d)

    arr = read_excel(inputpath('test.xlsx'), '2d', nb_axes=2, engine='openpyxl')
    assert_larray_equal(arr, io_2d)

    arr = read_excel(inputpath('test.xlsx'), '2d_classic', engine='openpyxl')
    assert_larray_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '2d_classic', nb_axes=2, engine='openpyxl')
    assert_larray_equal(arr, ndtest("a=a0..a2; b0..b2"))

    arr = read_excel(inputpath('test.xlsx'), '3d', index_col=[0, 1], engine='openpyxl')
    assert_larray_equal(arr, io_3d)

    arr = read_excel(inputpath('test.xlsx'), '3d', engine='openpyxl')
    assert_larray_equal(arr, io_3d)

    # for > 2d, specifying nb_axes is required if there is no name for the horizontal axis
    arr = read_excel(inputpath('test.xlsx'), '3d_classic', nb_axes=3, engine='openpyxl')
    assert_larray_equal(arr, ndtest("a=1..3; b=b0,b1; c0..c2"))

    arr = read_excel(inputpath('test.xlsx'), 'int_labels', engine='openpyxl')
    assert_larray_equal(arr, io_int_labels)

    # passing a Group as sheet arg
    axis = Axis('dim=1d,2d,3d,5d')

    arr = read_excel(inputpath('test.xlsx'), axis['1d'], engine='openpyxl')
    assert_larray_equal(arr, io_1d)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test.xlsx'), 'missing_values', fill_value=42, engine='openpyxl')
    expected = io_missing_values.copy()
    expected[isnan(expected)] = 42
    assert_larray_equal(arr, expected)

    #################
    # narrow format #
    #################
    arr = read_excel(inputpath('test_narrow.xlsx'), '1d', wide=False, engine='openpyxl')
    assert_larray_equal(arr, io_1d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '2d', wide=False, engine='openpyxl')
    assert_larray_equal(arr, io_2d)

    arr = read_excel(inputpath('test_narrow.xlsx'), '3d', wide=False, engine='openpyxl')
    assert_larray_equal(arr, io_3d)

    # missing rows + fill_value argument
    arr = read_excel(inputpath('test_narrow.xlsx'), 'missing_values',
                     fill_value=42, wide=False, engine='openpyxl')
    expected = io_narrow_missing_values.copy()
    expected[isnan(expected)] = 42
    assert_larray_equal(arr, expected)

    # unsorted values
    arr = read_excel(inputpath('test_narrow.xlsx'), 'unsorted', wide=False, engine='openpyxl')
    assert_larray_equal(arr, io_unsorted)


def test_from_lists():
    expected = ndtest((2, 2, 3))

    # simple
    res = from_lists([[ 'a', r'b\c', 'c0', 'c1', 'c2'],   # noqa: E201
                      ['a0',   'b0',    0,    1,    2],   # noqa: E241
                      ['a0',   'b1',    3,    4,    5],   # noqa: E241
                      ['a1',   'b0',    6,    7,    8],   # noqa: E241
                      ['a1',   'b1',    9,   10,   11]])  # noqa: E241
    assert_larray_equal(res, expected)

    # simple (using dump). This should be the same test as above.
    # We just make sure dump() and from_lists() round-trip correctly.
    arr_list = expected.dump()
    res = from_lists(arr_list)
    assert_larray_equal(res, expected)

    # with anonymous axes
    arr_anon = expected.rename({0: None, 1: None, 2: None})
    arr_list = arr_anon.dump()
    assert arr_list == [[None, None, 'c0', 'c1', 'c2'],
                        ['a0', 'b0',    0,    1,    2],              # noqa: E241
                        ['a0', 'b1',    3,    4,    5],              # noqa: E241
                        ['a1', 'b0',    6,    7,    8],              # noqa: E241
                        ['a1', 'b1',    9,   10,   11]]              # noqa: E241
    res = from_lists(arr_list, nb_axes=3)
    assert_larray_equal(res, arr_anon)

    # with empty ('') axes names
    arr_empty_names = expected.rename({0: '', 1: '', 2: ''})
    arr_list = arr_empty_names.dump()
    assert arr_list == [[  '',   '', 'c0', 'c1', 'c2'],              # noqa: E201,E241
                        ['a0', 'b0',    0,    1,    2],              # noqa: E241
                        ['a0', 'b1',    3,    4,    5],              # noqa: E241
                        ['a1', 'b0',    6,    7,    8],              # noqa: E241
                        ['a1', 'b1',    9,   10,   11]]              # noqa: E241
    res = from_lists(arr_list, nb_axes=3)
    # this is purposefully NOT arr_empty_names because from_lists (via df_asarray) transforms '' axes to None
    assert_larray_equal(res, arr_anon)

    # sort_rows
    expected = from_lists([['c', r'nat\year', 1991, 1992, 1993],
                           ['c0', 'BE', 0, 0, 1],
                           ['c0', 'FO', 0, 0, 2],
                           ['c1', 'BE', 1, 0, 0],
                           ['c1', 'FO', 2, 0, 0]])
    sorted_arr = from_lists([['c', r'nat\year', 1991, 1992, 1993],
                             ['c1', 'BE', 1, 0, 0],
                             ['c1', 'FO', 2, 0, 0],
                             ['c0', 'BE', 0, 0, 1],
                             ['c0', 'FO', 0, 0, 2]], sort_rows=True)
    assert_larray_equal(sorted_arr, expected)

    # sort_columns
    expected = from_lists([['c', r'nat\year', 1991, 1992, 1993],
                           ['c0', 'BE', 1, 0, 0],
                           ['c0', 'FO', 2, 0, 0],
                           ['c1', 'BE', 0, 0, 1],
                           ['c1', 'FO', 0, 0, 2]])
    sorted_arr = from_lists([['c', r'nat\year', 1992, 1991, 1993],
                             ['c0', 'BE', 0, 1, 0],
                             ['c0', 'FO', 0, 2, 0],
                             ['c1', 'BE', 0, 0, 1],
                             ['c1', 'FO', 0, 0, 2]], sort_columns=True)
    assert_larray_equal(sorted_arr, expected)


def test_to_series():
    # simple test
    arr = ndtest(3, dtype=np.int32)
    res = arr.to_series()
    expected = pd.Series([0, 1, 2], dtype="int32", index=pd.Index(['a0', 'a1', 'a2'], name='a'))
    assert res.equals(expected)

    # test for issue #1061 (object dtype labels array produce warning with Pandas1.4+)
    # We use an explicit int64 type because for some reason, under Linux, summing an int32 array
    # results in an int64 value, so it is easier to just use a int64 array in the first place so
    # that the test works on both Windows and Linux
    arr = ndtest("a=1..3", dtype=np.int64).with_total()[:3]
    res = arr.to_series()
    index = pd.Index([1, 2, 3], dtype=object, name='a')
    expected = pd.Series([0, 1, 2], dtype="int64", index=index)
    assert res.equals(expected)


def test_from_series():
    # Series with Index as index
    expected = ndtest(3)
    s = pd.Series([0, 1, 2], index=pd.Index(['a0', 'a1', 'a2'], name='a'))
    assert_larray_equal(from_series(s), expected)

    s = pd.Series([2, 0, 1], index=pd.Index(['a2', 'a0', 'a1'], name='a'))
    assert_larray_equal(from_series(s, sort_rows=True), expected)

    expected = ndtest(3)[['a2', 'a0', 'a1']]
    assert_larray_equal(from_series(s), expected)

    # Series with MultiIndex as index
    a = Axis('a=0..3')
    gender = Axis('gender=M,F')
    time = Axis('time=2015..2017')
    expected = ndtest((a, gender, time))

    index = pd.MultiIndex.from_product(expected.axes.labels, names=expected.axes.names)
    data = expected.data.flatten()
    s = pd.Series(data, index)

    res = from_series(s)
    assert_larray_equal(res, expected)

    res = from_series(s, sort_rows=True)
    assert_larray_equal(res, expected.sort_labels())

    expected[0, 'F'] = -1
    s = s.reset_index().drop([3, 4, 5]).set_index(['a', 'gender', 'time'])[0]
    res = from_series(s, fill_value=-1)
    assert_larray_equal(res, expected)


def test_to_frame():
    # these tests are for issue #1061
    arr = ndtest("a=0..2").with_total()[:2]
    df = arr.to_frame()
    assert df.columns.name == 'a'
    assert df.columns.to_list() == [0, 1, 2]

    arr = ndtest("a=0..2;b=b0").with_total('a')[:2]
    df = arr.to_frame()
    assert df.columns.name == 'b'
    assert df.columns.to_list() == ['b0']
    assert df.index.name == 'a'
    assert df.index.to_list() == [0, 1, 2]

    arr = ndtest("a=0..2;b=b0;c=c0").with_total('a')[:2]
    df = arr.to_frame()
    assert df.columns.name == 'c'
    assert df.columns.to_list() == ['c0']
    assert df.index.names == ['a', 'b']


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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, 1)
    assert arr.axes.names == [None, None]
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, 1)), [axis_index, axis_columns])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, 1)
    assert arr.axes.names == ['index', None]
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, 1)), [axis_index.rename('index'), axis_columns])
    assert_larray_equal(arr, expected)

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
    assert_larray_equal(res, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, 1)
    assert arr.axes.names == [None, 'columns']
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, 1)), [axis_index, axis_columns.rename('columns')])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, 1)
    assert arr.axes.names == ['index', 'columns']
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, 1)), [axis_index.rename('index'), axis_columns.rename('columns')])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, size)
    assert arr.axes.names == [None, None]
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, size)), [axis_index, axis_columns])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, size)
    assert arr.axes.names == ['index', None]
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, size)), [axis_index.rename('index'), axis_columns])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, size)
    assert arr.axes.names == [None, 'columns']
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, size)), [axis_index, axis_columns.rename('columns')])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (1, size)
    assert arr.axes.names == ['index', 'columns']
    assert list(arr.axes.labels[0]) == index
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data.reshape((1, size)), [axis_index.rename('index'), axis_columns.rename('columns')])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (size, 1)
    assert arr.axes.names == [None, None]
    assert list(arr.axes.labels[0]) == indexes
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data, [axis_index, axis_columns])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (size, 1)
    assert arr.axes.names == ['index', None]
    assert list(arr.axes.labels[0]) == indexes
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data, [axis_index.rename('index'), axis_columns])
    assert_larray_equal(arr, expected)

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
    arr = from_frame(df)
    assert arr.ndim == 2
    assert arr.shape == (size, 1)
    assert arr.axes.names == [None, 'columns']
    assert list(arr.axes.labels[0]) == indexes
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data, [axis_index, axis_columns.rename('columns')])
    assert_larray_equal(arr, expected)

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
    assert arr.ndim == 2
    assert arr.shape == (size, 1)
    assert arr.axes.names == [None, 'columns']
    assert list(arr.axes.labels[0]) == indexes
    assert list(arr.axes.labels[1]) == columns
    expected = Array(data, [axis_index, axis_columns.rename('columns')])
    assert_larray_equal(arr, expected)

    # 3) 3D array
    # ===========

    # 3A) Dataframe with 2 index columns
    # ==================================
    dt = [('a', int), ('c', 'U2'),
          ('2007', int), ('2010', int), ('2013', int)]
    data = np.array([
        (0, 'c1', 3722, 3395, 3347),
        (0, 'c0', 338, 316, 323),
        (1, 'c1', 2878, 2791, 2822),
        (1, 'c0', 1121, 1037, 976),
        (2, 'c1', 4073, 4161, 4429),
        (2, 'c0', 661, 1463, 1467),
        (3, 'c1', 3507, 3741, 3366),
        (3, 'c0', 2052, 2052, 2118),
    ], dtype=dt)
    df = pd.DataFrame(data)
    df.set_index(['a', 'c'], inplace=True)
    df.columns.name = 'time'

    arr = from_frame(df)
    assert arr.ndim == 3
    assert arr.shape == (4, 2, 3)
    assert arr.axes.names == ['a', 'c', 'time']
    assert list(arr[0, 'c1', :].data) == [3722, 3395, 3347]

    # 3B) Dataframe with columns.name containing \
    # ============================================
    dt = [('a', int), (r'c\time', 'U2'),
          ('2007', int), ('2010', int), ('2013', int)]
    data = np.array([
        (0, 'c1', 3722, 3395, 3347),
        (0, 'c0', 338, 316, 323),
        (1, 'c1', 2878, 2791, 2822),
        (1, 'c0', 1121, 1037, 976),
        (2, 'c1', 4073, 4161, 4429),
        (2, 'c0', 661, 1463, 1467),
        (3, 'c1', 3507, 3741, 3366),
        (3, 'c0', 2052, 2052, 2118),
    ], dtype=dt)
    df = pd.DataFrame(data)
    df.set_index(['a', r'c\time'], inplace=True)

    arr = from_frame(df, unfold_last_axis_name=True)
    assert arr.ndim == 3
    assert arr.shape == (4, 2, 3)
    assert arr.axes.names == ['a', 'c', 'time']
    assert_nparray_equal(arr[0, 'c1', :].data, np.array([3722, 3395, 3347]))

    # 3C) Dataframe with no axe names (names are None)
    # ===============================
    arr_no_names = ndtest("a0,a1;b0..b2;c0..c3")
    df_no_names = arr_no_names.df
    res = from_frame(df_no_names)
    assert_larray_equal(res, arr_no_names)

    # 3D) Dataframe with empty axe names (names are '')
    # ==================================
    arr_empty_names = ndtest("=a0,a1;=b0..b2;=c0..c3")
    assert arr_empty_names.axes.names == ['', '', '']
    df_empty_names = arr_empty_names.df
    res = from_frame(df_empty_names)
    assert_larray_equal(res, arr_empty_names)

    # 4) test sort_rows and sort_columns arguments
    # ============================================
    a = Axis('a=2,0,1,3')
    gender = Axis('gender=M,F')
    time = Axis('time=2016,206,2017')
    columns = pd.Index(time.labels, name=time.name)

    # df.index is an Index instance
    expected = ndtest((gender, time))
    index = pd.Index(gender.labels, name=gender.name)
    data = expected.data
    df = pd.DataFrame(data, index=index, columns=columns)

    expected = expected.sort_labels()
    res = from_frame(df, sort_rows=True, sort_columns=True)
    assert_larray_equal(res, expected)

    # df.index is a MultiIndex instance
    expected = ndtest((a, gender, time))
    index = pd.MultiIndex.from_product(expected.axes[:-1].labels, names=expected.axes[:-1].names)
    data = expected.data.reshape(len(a) * len(gender), len(time))
    df = pd.DataFrame(data, index=index, columns=columns)

    res = from_frame(df, sort_rows=True, sort_columns=True)
    assert_larray_equal(res, expected.sort_labels())

    # 5) test fill_value
    # ==================
    expected[0, 'F'] = -1
    df = df.reset_index().drop([3]).set_index(['a', 'gender'])
    res = from_frame(df, fill_value=-1)
    assert_larray_equal(res, expected)


def test_to_csv(tmp_path):
    io_3d.to_csv(tmp_path / 'out3d.csv')
    assert (tmp_path / 'out3d.csv').read_text() == """\
a,b\\c,c0,c1,c2
1,b0,0,1,2
1,b1,3,4,5
2,b0,6,7,8
2,b1,9,10,11
3,b0,12,13,14
3,b1,15,16,17
"""

    # stacked data (one column containing all the values and another column listing the context of the value)
    io_3d.to_csv(tmp_path / 'out3d_narrow.csv', wide=False)
    assert (tmp_path / 'out3d_narrow.csv').read_text() == """\
a,b,c,value
1,b0,c0,0
1,b0,c1,1
1,b0,c2,2
1,b1,c0,3
1,b1,c1,4
1,b1,c2,5
2,b0,c0,6
2,b0,c1,7
2,b0,c2,8
2,b1,c0,9
2,b1,c1,10
2,b1,c2,11
3,b0,c0,12
3,b0,c1,13
3,b0,c2,14
3,b1,c0,15
3,b1,c1,16
3,b1,c2,17
"""

    io_1d.to_csv(tmp_path / 'out1d.csv')
    assert (tmp_path / 'out1d.csv').read_text() == """\
a,a0,a1,a2
,0,1,2
"""


@needs_xlsxwriter
@needs_openpyxl
def test_to_excel_xlsxwriter(tmp_path):
    fpath = tmp_path / 'test_to_excel_xlsxwriter.xlsx'

    # 1D
    a1 = ndtest(3)

    # fpath/Sheet1/A1
    a1.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # fpath/Sheet1/A1(transposed)
    a1.to_excel(fpath, transpose=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # fpath/Sheet1/A1
    # stacked data (one column containing all the values and another column listing the context of the value)
    a1.to_excel(fpath, wide=False, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    stacked_a1 = a1.reshape([a1.a, Axis(['value'])])
    assert_larray_equal(res, stacked_a1)

    # 2D
    a2 = ndtest((2, 3))

    # fpath/Sheet1/A1
    a2.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a2)

    # fpath/Sheet1/A10
    # TODO: this is currently not supported (though we would only need to translate A10 to startrow=0 and startcol=0
    # a2.to_excel('fpath', 'Sheet1', 'A10', engine='xlsxwriter')
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=9)
    # assert_larray_equal(res, a2)

    # fpath/other/A1
    a2.to_excel(fpath, 'other', engine='xlsxwriter')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_larray_equal(res, a2)

    # 3D
    a3 = ndtest((2, 3, 4))

    # fpath/Sheet1/A1
    # FIXME: merge_cells=False should be the default (until Pandas is fixed to read its format)
    a3.to_excel(fpath, overwrite_file=True, engine='xlsxwriter', merge_cells=False)
    # a3.to_excel('fpath', overwrite_file=True, engine='openpyxl')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a3)

    # fpath/Sheet1/A20
    # TODO: implement position (see above)
    # a3.to_excel('fpath', 'Sheet1', 'A20', engine='xlsxwriter', merge_cells=False)
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=19)
    # assert_larray_equal(res, a3)

    # fpath/other/A1
    a3.to_excel(fpath, 'other', engine='xlsxwriter', merge_cells=False)
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_larray_equal(res, a3)

    # 1D
    a1 = ndtest(3)

    # fpath/Sheet1/A1
    a1.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # fpath/Sheet1/A1(transposed)
    a1.to_excel(fpath, transpose=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # fpath/Sheet1/A1
    # stacked data (one column containing all the values and another column listing the context of the value)
    a1.to_excel(fpath, wide=False, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    stacked_a1 = a1.reshape([a1.a, Axis(['value'])])
    assert_larray_equal(res, stacked_a1)

    # 2D
    a2 = ndtest((2, 3))

    # fpath/Sheet1/A1
    a2.to_excel(fpath, overwrite_file=True, engine='xlsxwriter')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a2)

    # fpath/Sheet1/A10
    # TODO: this is currently not supported (though we would only need to translate A10 to startrow=0 and startcol=0
    # a2.to_excel(fpath, 'Sheet1', 'A10', engine='xlsxwriter')
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=9)
    # assert_larray_equal(res, a2)

    # fpath/other/A1
    a2.to_excel(fpath, 'other', engine='xlsxwriter')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_larray_equal(res, a2)

    # 3D
    a3 = ndtest((2, 3, 4))

    # fpath/Sheet1/A1
    # FIXME: merge_cells=False should be the default (until Pandas is fixed to read its format)
    a3.to_excel(fpath, overwrite_file=True, engine='xlsxwriter', merge_cells=False)
    # a3.to_excel('fpath', overwrite_file=True, engine='openpyxl')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a3)

    # fpath/Sheet1/A20
    # TODO: implement position (see above)
    # a3.to_excel('fpath', 'Sheet1', 'A20', engine='xlsxwriter', merge_cells=False)
    # res = read_excel('fpath', 'Sheet1', engine='openpyxl', skiprows=19)
    # assert_larray_equal(res, a3)

    # fpath/other/A1
    a3.to_excel(fpath, 'other', engine='xlsxwriter', merge_cells=False)
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_larray_equal(res, a3)

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
def test_to_excel_xlwings(tmp_path):
    fpath = tmp_path / 'test_to_excel_xlwings.xlsx'

    # 1D
    a1 = ndtest(3)

    # live book/Sheet1/A1
    # a1.to_excel()

    # fpath/Sheet1/A1 (create a new file if does not exist)
    if fpath.is_file():
        os.remove(fpath)
    a1.to_excel(fpath, engine='xlwings')
    # we use openpyxl to read back instead of xlwings even if that should work, to make the test faster
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # fpath/Sheet1/A1(transposed)
    a1.to_excel(fpath, transpose=True, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # fpath/Sheet1/A1
    # stacked data (one column containing all the values and another column listing the context of the value)
    a1.to_excel(fpath, wide=False, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a1)

    # 2D
    a2 = ndtest((2, 3))

    # fpath/Sheet1/A1
    a2.to_excel(fpath, overwrite_file=True, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a2)

    # fpath/Sheet1/A10
    a2.to_excel(fpath, 'Sheet1', 'A10', engine='xlwings')
    res = read_excel(fpath, 'Sheet1', engine='openpyxl', skiprows=9)
    assert_larray_equal(res, a2)

    # fpath/other/A1
    a2.to_excel(fpath, 'other', engine='xlwings')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_larray_equal(res, a2)

    # transpose
    a2.to_excel(fpath, 'transpose', transpose=True, engine='xlwings')
    res = read_excel(fpath, 'transpose', engine='openpyxl')
    assert_larray_equal(res, a2.T)

    # 3D
    a3 = ndtest((2, 3, 4))

    # fpath/Sheet1/A1
    a3.to_excel(fpath, overwrite_file=True, engine='xlwings')
    res = read_excel(fpath, engine='openpyxl')
    assert_larray_equal(res, a3)

    # fpath/Sheet1/A20
    a3.to_excel(fpath, 'Sheet1', 'A20', engine='xlwings')
    res = read_excel(fpath, 'Sheet1', engine='openpyxl', skiprows=19)
    assert_larray_equal(res, a3)

    # fpath/other/A1
    a3.to_excel(fpath, 'other', engine='xlwings')
    res = read_excel(fpath, 'other', engine='openpyxl')
    assert_larray_equal(res, a3)

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
    with must_raise(ValueError, "Sheet names cannot exceed 31 characters"):
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
    assert res == [[r'\b', 0, 1],
                   ['a0', 0, 1],
                   ['a1', 2, 3]]
    res = arr.dump(_axes_display_names=True)
    assert res == [[r'{0}\b*', 0, 1],
                   ['a0', 0, 1],
                   ['a1', 2, 3]]


@needs_xlwings
def test_open_excel(tmp_path):
    # 1) Create new file
    # ==================
    fpath = inputpath('should_not_exist.xlsx')
    # overwrite_file must be set to True to create a new file
    msg = f"File {fpath} does not exist. Please give the path to an existing file " \
          f"or set overwrite_file argument to True"
    with must_raise(ValueError, msg=msg):
        open_excel(fpath)

    # 2) with headers
    # ===============
    with open_excel(visible=False) as wb:
        # 1D
        a1 = ndtest(3)

        # Sheet1/A1
        wb['Sheet1'] = a1.dump()
        res = wb['Sheet1'].load()
        assert_larray_equal(res, a1)

        wb[0] = a1.dump()
        res = wb[0].load()
        assert_larray_equal(res, a1)

        # Sheet1/A1(transposed)
        # TODO: implement .options on Sheet so that one can write:
        # wb[0].options(transpose=True).value = a1.dump()
        wb[0]['A1'].options(transpose=True).value = a1.dump()
        # TODO: implement .options on Range so that you can write:
        # res = wb[0]['A1:B4'].options(transpose=True).load()
        # res = from_lists(wb[0]['A1:B4'].options(transpose=True).value)
        # assert_larray_equal(res, a1)

        # 2D
        a2 = ndtest((2, 3))

        # Sheet1/A1
        wb[0] = a2.dump()
        res = wb[0].load()
        assert_larray_equal(res, a2)

        # Sheet1/A10
        wb[0]['A10'] = a2.dump()
        res = wb[0]['A10:D12'].load()
        assert_larray_equal(res, a2)

        # other/A1
        wb['other'] = a2.dump()
        res = wb['other'].load()
        assert_larray_equal(res, a2)

        # new/A10
        # we need to create the sheet first
        wb['new'] = ''
        wb['new']['A10'] = a2.dump()
        res = wb['new']['A10:D12'].load()
        assert_larray_equal(res, a2)

        # new2/A10
        # cannot store the return value of "add" because that's a raw xlwings Sheet
        wb.sheets.add('new2')
        wb['new2']['A10'] = a2.dump()
        res = wb['new2']['A10:D12'].load()
        assert_larray_equal(res, a2)

        # 3D
        a3 = ndtest((2, 3, 4))

        # 3D/A1
        wb['3D'] = a3.dump()
        res = wb['3D'].load()
        assert_larray_equal(res, a3)

        # 3D/A20
        wb['3D']['A20'] = a3.dump()
        res = wb['3D']['A20:F26'].load()
        assert_larray_equal(res, a3)

        # 3D/A20 without name for columns
        wb['3D']['A20'] = a3.dump()
        # assume we have no name for the columns axis (ie change b\c to b)
        wb['3D']['B20'] = 'b'
        res = wb['3D']['A20:F26'].load(nb_axes=3)
        assert_nparray_equal(res.data, a3.data)
        # the two first axes should be the same
        assert res.axes[:2] == a3.axes[:2]
        # the third axis should have the same labels (but not the same name obviously)
        assert_nparray_equal(res.axes[2].labels, a3.axes[2].labels)

    with open_excel(inputpath('test.xlsx')) as wb:
        expected = ndtest("a=a0..a2; b0..b2")
        res = wb['2d_classic'].load()
        assert_larray_equal(res, expected)

    # 3) without headers
    # ==================
    with open_excel(visible=False) as wb:
        # 1D
        a1 = ndtest(3)

        # Sheet1/A1
        wb['Sheet1'] = a1
        res = wb['Sheet1'].load(header=False)
        assert_nparray_equal(res.data, a1.data)

        wb[0] = a1
        res = wb[0].load(header=False)
        assert_nparray_equal(res.data, a1.data)

        # Sheet1/A1(transposed)
        # FIXME: we need to .dump(header=False) explicitly because otherwise we go via ArrayConverter which
        #        includes labels. for consistency's sake we should either change ArrayConverter to not include
        #        labels, or change wb[0] = a1 to include them (and use wb[0] = a1.data to avoid them?) but that
        #        would be heavily backward incompatible and how would I load them back?
        # wb[0]['A1'].options(transpose=True).value = a1
        wb[0]['A1'].options(transpose=True).value = a1.dump(header=False)
        res = wb[0]['A1:A3'].load(header=False)
        assert_nparray_equal(res.data, a1.data)

        # 2D
        a2 = ndtest((2, 3))

        # Sheet1/A1
        wb[0] = a2
        res = wb[0].load(header=False)
        assert_nparray_equal(res.data, a2.data)

        # Sheet1/A10
        wb[0]['A10'] = a2
        res = wb[0]['A10:C11'].load(header=False)
        assert_nparray_equal(res.data, a2.data)

        # other/A1
        wb['other'] = a2
        res = wb['other'].load(header=False)
        assert_nparray_equal(res.data, a2.data)

        # new/A10
        # we need to create the sheet first
        wb['new'] = ''
        wb['new']['A10'] = a2
        res = wb['new']['A10:C11'].load(header=False)
        assert_nparray_equal(res.data, a2.data)

        # 3D
        a3 = ndtest((2, 3, 4))

        # 3D/A1
        wb['3D'] = a3
        res = wb['3D'].load(header=False)
        assert_nparray_equal(res.data, a3.data.reshape((6, 4)))

        # 3D/A20
        wb['3D']['A20'] = a3
        res = wb['3D']['A20:D25'].load(header=False)
        assert_nparray_equal(res.data, a3.data.reshape((6, 4)))

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
    assert_larray_equal(bad1, good)
    assert_larray_equal(bad2, good)
    assert_larray_equal(bad3, good2)
    assert_larray_equal(bad4, good2)

    # 5) anonymous and wilcard axes
    # =============================
    arr = ndtest((Axis('a0,a1'), Axis(2, 'b')))
    fpath = tmp_path / 'anonymous_and_wildcard_axes.xlsx'
    with open_excel(fpath, overwrite_file=True) as wb:
        wb[0] = arr.dump()
        res = wb[0].load()
        # the result should be identical to the original array except we lost the information about
        # the wildcard axis being a wildcard axis
        expected = arr.set_axes('b', Axis([0, 1], 'b'))
        assert_larray_equal(res, expected)

    # 6) crash test
    # =============
    arr = ndtest((2, 2))
    fpath = tmp_path / 'temporary_test_file.xlsx'
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
        assert_larray_equal(wb['arr'].load(), arr)
    # remove file
    if fpath.exists():
        os.remove(fpath)


def test_ufuncs(small_array):
    raw = small_array.data

    # simple one-argument ufunc
    assert_nparray_equal(exp(small_array).data, np.exp(raw))

    # with out=
    la_out = zeros(small_array.axes)
    raw_out = np.zeros(raw.shape)

    la_out2 = exp(small_array, la_out)
    raw_out2 = np.exp(raw, raw_out)

    # FIXME: this is not the case currently
    # assert la_out2 is la_out
    # FIXME: this is not the case currently (la_out2 axes are wrong: wildcard & anonymous instead of actual axes)
    # assert_larray_equal(la_out2, la_out)
    assert_nparray_equal(la_out2.data, la_out.data)
    assert raw_out2 is raw_out

    assert_nparray_equal(la_out.data, raw_out)

    # with out= and broadcasting
    # we need to put the 'a' axis first because array numpy only supports that
    la_out = zeros([Axis([0, 1, 2], 'a')] + list(small_array.axes))
    raw_out = np.zeros((3,) + raw.shape)

    la_out2 = exp(small_array, la_out)
    raw_out2 = np.exp(raw, raw_out)

    # assert la_out2 == la_out
    # XXX: why is la_out2 transposed?
    assert_larray_equal(la_out2.transpose(X.a), la_out)
    assert raw_out2 is raw_out

    assert_nparray_equal(la_out.data, raw_out)

    c, d = small_array.axes

    low = small_array.sum(c) // 4 + 3
    raw_low = raw.sum(0) // 4 + 3
    high = small_array.sum(c) // 4 + 13
    raw_high = raw.sum(0) // 4 + 13

    # LA + scalars
    assert_nparray_equal(small_array.clip(0, 10).data, raw.clip(0, 10))
    assert_nparray_equal(clip(small_array, 0, 10).data, np.clip(raw, 0, 10))

    # LA + LA (no broadcasting)
    assert_nparray_equal(clip(small_array, 21 - small_array, 9 + small_array // 2).data,
                         np.clip(raw, 21 - raw, 9 + raw // 2))

    # LA + LA (with broadcasting)
    assert_nparray_equal(clip(small_array, low, high).data,
                         np.clip(raw, raw_low, raw_high))

    # round
    small_float = small_array + 0.6
    rounded = round(small_float)
    assert_nparray_equal(rounded.data, np.round(raw + 0.6))


def test_where():
    arr = ndtest((2, 3))
    # a\b  b0  b1  b2
    #  a0   0   1   2
    #  a1   3   4   5

    expected = from_string(r"""a\b  b0  b1  b2
                                a0  -1  -1  -1
                                a1  -1   4   5""")

    # where (no broadcasting)
    res = where(arr < 4, -1, arr)
    assert_larray_equal(res, expected)

    # where (transposed no broadcasting)
    res = where(arr < 4, -1, arr.T)
    assert_larray_equal(res, expected)

    # where (with broadcasting)
    res = where(arr['b1'] < 4, -1, arr)
    assert_larray_equal(res, from_string(r"""a\b  b0  b1  b2
                                              a0  -1  -1  -1
                                              a1   3   4   5"""))

    # with expressions (issue #1083)
    arr = ndtest("age=0..5")
    res = where(X.age == 3, 42, arr)
    assert_larray_equal(res, from_string("""age  0  1  2   3  4  5
                                             \t  0  1  2  42  4  5"""))

    res = where(X.age == 3, arr, 42)
    assert_larray_equal(res, from_string("""age   0   1   2  3   4   5
                                             \t  42  42  42  3  42  42"""))


def test_eye():
    a = Axis('a=0..2')
    c = Axis('c=c0,c1')

    # using one Axis object
    res = eye(c)
    expected = from_string(r"""
    c\c   c0   c1
     c0  1.0  0.0
     c1  0.0  1.0""")
    assert_larray_equal(res, expected)

    # using an AxisCollection
    res = eye(AxisCollection([a, c]))
    expected = from_string(r"""
    a\c   c0   c1
      0  1.0  0.0
      1  0.0  1.0
      2  0.0  0.0""")
    assert_larray_equal(res, expected)

    # using a tuple of axes
    res = eye((a, c))
    expected = from_string(r"""
    a\c   c0   c1
      0  1.0  0.0
      1  0.0  1.0
      2  0.0  0.0""")
    assert_larray_equal(res, expected)


def test_diag():
    # 2D -> 1D
    arr = ndtest((3, 3))
    diag_arr = diag(arr)
    assert diag_arr.ndim == 1
    assert diag_arr.i[0] == arr.i[0, 0]
    assert diag_arr.i[1] == arr.i[1, 1]
    assert diag_arr.i[2] == arr.i[2, 2]

    # 1D -> 2D
    arr2 = diag(diag_arr)
    assert arr2.ndim == 2
    assert arr2.i[0, 0] == arr.i[0, 0]
    assert arr2.i[1, 1] == arr.i[1, 1]
    assert arr2.i[2, 2] == arr.i[2, 2]

    # 3D -> 2D
    arr = ndtest((3, 3, 3))
    diag_arr = diag(arr)
    assert diag_arr.ndim == 2
    assert diag_arr.i[0, 0] == arr.i[0, 0, 0]
    assert diag_arr.i[1, 1] == arr.i[1, 1, 1]
    assert diag_arr.i[2, 2] == arr.i[2, 2, 2]

    # 3D -> 1D
    diag_arr = diag(arr, axes=(0, 1, 2))
    assert diag_arr.ndim == 1
    assert diag_arr.i[0] == arr.i[0, 0, 0]
    assert diag_arr.i[1] == arr.i[1, 1, 1]
    assert diag_arr.i[2] == arr.i[2, 2, 2]

    # 1D (anon) -> 2D
    diag_arr_anon = diag_arr.rename(0, None).ignore_labels()
    arr2 = diag(diag_arr_anon)
    assert arr2.ndim == 2

    # 1D (anon) -> 3D
    arr3 = diag(diag_arr_anon, ndim=3)
    assert arr3.ndim == 3
    assert arr3.i[0, 0, 0] == arr.i[0, 0, 0]
    assert arr3.i[1, 1, 1] == arr.i[1, 1, 1]
    assert arr3.i[2, 2, 2] == arr.i[2, 2, 2]

    # using Axis object
    c = Axis('c=c0,c1')
    arr = eye(c)
    diag_arr = diag(arr)
    assert diag_arr.ndim == 1
    diag_axis = diag_arr.axes[0]
    assert diag_axis.name == 'c_c'
    assert list(diag_axis.labels) == ['c0_c0', 'c1_c1']
    assert diag_arr.i[0] == 1.0
    assert diag_arr.i[1] == 1.0


def test_matmul():
    # 2D / anonymous axes
    a1 = ndtest([Axis(3), Axis(3)])
    a2 = eye(3, 3) * 2

    # Array value
    assert_larray_equal(a1 @ a2, ndtest([Axis(3), Axis(3)]) * 2)

    # ndarray value
    assert_larray_equal(a1 @ a2.data, ndtest([Axis(3), Axis(3)]) * 2)

    # non anonymous axes (N <= 2)
    arr1d = ndtest(3)
    arr2d = ndtest((3, 3))

    # 1D @ 1D
    res = arr1d @ arr1d
    assert isinstance(res, np.integer)
    assert res == 5

    # 1D @ 2D
    assert_larray_equal(arr1d @ arr2d,
                        Array([15, 18, 21], 'b=b0..b2'))

    # 2D @ 1D
    assert_larray_equal(arr2d @ arr1d,
                        Array([5, 14, 23], 'a=a0..a2'))

    # 2D(a,b) @ 2D(a,b) -> 2D(a,b)
    res = from_lists([[r'a\b', 'b0', 'b1', 'b2'],
                      ['a0', 15, 18, 21],
                      ['a1', 42, 54, 66],
                      ['a2', 69, 90, 111]])
    assert_larray_equal(arr2d @ arr2d, res)

    # 2D(a,b) @ 2D(b,a) -> 2D(a,a)
    res = from_lists([[r'a\a', 'a0', 'a1', 'a2'],
                      ['a0', 5, 14, 23],
                      ['a1', 14, 50, 86],
                      ['a2', 23, 86, 149]])
    assert_larray_equal(arr2d @ arr2d.T, res)

    # ndarray value
    assert_larray_equal(arr1d @ arr2d.data, Array([15, 18, 21]))
    assert_nparray_equal(arr2d.data @ arr2d.T.data, res.data)

    # different axes
    a1 = ndtest('a=a0..a1;b=b0..b2')
    a2 = ndtest('b=b0..b2;c=c0..c3')
    res = from_lists([[r'a\c', 'c0', 'c1', 'c2', 'c3'],
                      ['a0', 20, 23, 26, 29],
                      ['a1', 56, 68, 80, 92]])
    assert_larray_equal(a1 @ a2, res)

    # non anonymous axes (N >= 2)
    arr2d = ndtest((2, 2))
    arr3d = ndtest((2, 2, 2))
    arr4d = ndtest((2, 2, 2, 2))
    a, b, c, d = arr4d.axes
    e = Axis('e=e0,e1')
    f = Axis('f=f0,f1')

    # 4D(a, b, c, d) @ 3D(e, d, f) -> 5D(a, b, e, c, f)
    arr3d = arr3d.set_axes([e, d, f])
    res = from_lists([['a', 'b', 'e', r'c\f', 'f0', 'f1'],
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
    assert_larray_equal(arr4d @ arr3d, res)

    # 3D(e, d, f) @ 4D(a, b, c, d) -> 5D(e, a, b, d, d)
    res = from_lists([['e', 'a', 'b', r'd\d', 'd0', 'd1'],
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
    assert_larray_equal(arr3d @ arr4d, res)

    # 4D(a, b, c, d) @ 3D(b, d, f) -> 4D(a, b, c, f)
    arr3d = arr3d.set_axes([b, d, f])
    res = from_lists([['a', 'b', r'c\f', 'f0', 'f1'],
                      ['a0', 'b0', 'c0', 2, 3],
                      ['a0', 'b0', 'c1', 6, 11],
                      ['a0', 'b1', 'c0', 46, 55],
                      ['a0', 'b1', 'c1', 66, 79],
                      ['a1', 'b0', 'c0', 18, 35],
                      ['a1', 'b0', 'c1', 22, 43],
                      ['a1', 'b1', 'c0', 126, 151],
                      ['a1', 'b1', 'c1', 146, 175]])
    assert_larray_equal(arr4d @ arr3d, res)

    # 3D(b, d, f) @ 4D(a, b, c, d) -> 4D(b, a, d, d)
    res = from_lists([['b', 'a', r'd\d', 'd0', 'd1'],
                      ['b0', 'a0', 'd0', 2, 3],
                      ['b0', 'a0', 'd1', 6, 11],
                      ['b0', 'a1', 'd0', 10, 11],
                      ['b0', 'a1', 'd1', 46, 51],
                      ['b1', 'a0', 'd0', 46, 55],
                      ['b1', 'a0', 'd1', 66, 79],
                      ['b1', 'a1', 'd0', 118, 127],
                      ['b1', 'a1', 'd1', 170, 183]])
    assert_larray_equal(arr3d @ arr4d, res)

    # 4D(a, b, c, d) @ 2D(d, f) -> 5D(a, b, c, f)
    arr2d = arr2d.set_axes([d, f])
    res = from_lists([['a', 'b', r'c\f', 'f0', 'f1'],
                      ['a0', 'b0', 'c0', 2, 3],
                      ['a0', 'b0', 'c1', 6, 11],
                      ['a0', 'b1', 'c0', 10, 19],
                      ['a0', 'b1', 'c1', 14, 27],
                      ['a1', 'b0', 'c0', 18, 35],
                      ['a1', 'b0', 'c1', 22, 43],
                      ['a1', 'b1', 'c0', 26, 51],
                      ['a1', 'b1', 'c1', 30, 59]])
    assert_larray_equal(arr4d @ arr2d, res)

    # 2D(d, f) @ 4D(a, b, c, d) -> 5D(a, b, d, d)
    res = from_lists([['a', 'b', r'd\d', 'd0', 'd1'],
                      ['a0', 'b0', 'd0', 2, 3],
                      ['a0', 'b0', 'd1', 6, 11],
                      ['a0', 'b1', 'd0', 6, 7],
                      ['a0', 'b1', 'd1', 26, 31],
                      ['a1', 'b0', 'd0', 10, 11],
                      ['a1', 'b0', 'd1', 46, 51],
                      ['a1', 'b1', 'd0', 14, 15],
                      ['a1', 'b1', 'd1', 66, 71]])
    assert_larray_equal(arr2d @ arr4d, res)


def test_rmatmul():
    a1 = eye(3) * 2
    a2 = ndtest([Axis(3), Axis(3)])

    # equivalent to a1.data @ a2
    res = a2.__rmatmul__(a1.data)
    assert isinstance(res, Array)
    assert_larray_equal(res, ndtest([Axis(3), Axis(3)]) * 2)


def test_broadcast_with():
    a1 = ndtest((3, 2))
    a2 = ndtest(3)
    b = a2.broadcast_with(a1)
    assert b.ndim == a1.ndim
    assert b.shape == (3, 1)
    assert_larray_equal(b.i[:, 0], a2)

    # anonymous axes
    a1 = ndtest([Axis(3), Axis(2)])
    a2 = ndtest(Axis(3))
    b = a2.broadcast_with(a1)
    assert b.ndim == a1.ndim
    assert b.shape == (3, 1)
    assert_larray_equal(b.i[:, 0], a2)

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
    # small_h = small['c0']
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
    assert list(res.axes.a_b.labels[:2]) == ['a0_b0', 'a0_b1']
    assert_larray_equal(res['a1_b0'], arr['a1', 'b0'])

    res = arr.combine_axes((X.a, X.c))
    assert res.axes.names == ['a_c', 'b', 'd']
    assert res.size == arr.size
    assert res.shape == (2 * 4, 3, 5)
    assert list(res.axes.a_c.labels[:2]) == ['a0_c0', 'a0_c1']
    assert_larray_equal(res['a1_c0'], arr['a1', 'c0'])

    res = arr.combine_axes((X.b, X.d))
    assert res.axes.names == ['a', 'b_d', 'c']
    assert res.size == arr.size
    assert res.shape == (2, 3 * 5, 4)
    assert list(res.axes.b_d.labels[:2]) == ['b0_d0', 'b0_d1']
    assert_larray_equal(res['b1_d0'], arr['b1', 'd0'])

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
    assert_larray_equal(res['a0_c2', 'b1_e2_f1'], arr['a0', 'b1', 'c2', 'e2', 'f1'])

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
    assert_nparray_equal(res.axes[0].labels, np.arange(6))


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
    assert_larray_equal(res, arr.transpose('a', 'b', 'd', 'c'))

    # with specified names
    res = combined.rename(b_d='bd').split_axes('bd', names=('b', 'd'))
    assert res.axes.names == ['a', 'b', 'd', 'c']
    assert res.shape == (2, 3, 5, 4)
    assert_larray_equal(res, arr.transpose('a', 'b', 'd', 'c'))

    # regex
    res = combined.split_axes('b_d', names=['b', 'd'], regex=r'(\w+)_(\w+)')
    assert res.axes.names == ['a', 'b', 'd', 'c']
    assert res.shape == (2, 3, 5, 4)
    assert_larray_equal(res, arr.transpose('a', 'b', 'd', 'c'))

    # custom sep
    combined = ndtest('a|b=a0|b0,a0|b1')
    res = combined.split_axes(sep='|')
    assert_larray_equal(res, ndtest('a=a0;b=b0,b1'))

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
    assert_larray_equal(arr.split_axes(), res)

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
    assert res.a.labels.dtype.kind == 'U'
    assert res.b.labels.dtype.kind == 'U'
    assert res.c.labels.dtype.kind == 'O'
    assert_larray_equal(res, ndtest((2, 2, 2)))

    # not sorted by first part then second part (issue #364)
    arr = ndtest((2, 3))
    combined = arr.combine_axes()['a0_b0, a1_b0, a0_b1, a1_b1, a0_b2, a1_b2']
    assert_larray_equal(combined.split_axes('a_b'), arr)

    # another weirdly sorted test
    combined = arr.combine_axes()['a0_b1, a0_b0, a0_b2, a1_b1, a1_b0, a1_b2']
    assert_larray_equal(combined.split_axes('a_b'), arr['b1,b0,b2'])

    # combined does not contain all combinations of labels (issue #369)
    combined_partial = combined[['a0_b0', 'a0_b1', 'a1_b1', 'a0_b2', 'a1_b2']]
    expected = arr.astype(float)
    expected['a1', 'b0'] = nan
    assert_larray_nan_equal(combined_partial.split_axes('a_b'), expected)

    # split labels are ambiguous (issue #485)
    combined = ndtest('a_b=a0_b0..a1_b1;c_d=a0_b0..a1_b1')
    expected = ndtest('a=a0,a1;b=b0,b1;c=a0,a1;d=b0,b1')
    assert_larray_equal(combined.split_axes(('a_b', 'c_d')), expected)

    # anonymous axes
    combined = ndtest('a0_b0,a0_b1,a0_b2,a1_b0,a1_b1,a1_b2')
    expected = ndtest('a0,a1;b0,b1,b2')
    assert_larray_equal(combined.split_axes(0), expected)

    # when no axis is specified and no axis contains the sep, split_axes is a no-op.
    assert_larray_equal(combined.split_axes(), combined)

    # with varying sep characters in labels (issue #1089)
    arr = ndtest("a_b=a0_b0,a0_b1_1,a0_b1_2")
    # a_b  a0_b0  a0_b1_1  a0_b1_2
    #          0        1        2
    with must_raise(ValueError, "not all labels have the same number of separators"):
        arr.split_axes()

    # with different number of sep characters in labels than in axis name
    arr = ndtest("a_b=a0_b0_1,a0_b1_1,a0_b1_2")
    # a_b  a0_b0_1  a0_b1_1  a0_b1_2
    #            0        1        2
    with must_raise(ValueError, "number of resulting axes (3) differs from number of resulting axes names (2)"):
        arr.split_axes()


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
                      [1,  0],                                       # noqa: E241
                      [2,  1]], [a, b])                              # noqa: E241
    assert_larray_equal(res, expected)

    # same but using a group as the stacking axis
    larger_b = Axis('b=b0..b3')
    res = stack((arr0, arr1), larger_b[:'b1'])
    assert_larray_equal(res, expected)

    # simple with anonymous axis
    axis0 = Axis(3)
    arr0 = ndtest(axis0)
    arr1 = ndtest(axis0, start=-1)
    res = stack((arr0, arr1), b)
    expected = Array([[0, -1],
                      [1,  0],                                       # noqa: E241
                      [2,  1]], [axis0, b])                          # noqa: E241
    assert_larray_equal(res, expected)

    # using res_axes
    res = stack({'b0': 0, 'b1': 1}, axes=b, res_axes=(a, b))
    expected = Array([[0, 1],
                      [0, 1],
                      [0, 1]], [a, b])
    assert_larray_equal(res, expected)

    # giving elements as an Array containing Arrays
    c = Axis('c=c0,c1')
    # not using the same length for nat and type, otherwise numpy gets confused :(
    arr1 = ones('nat=BE, FO')
    arr2 = zeros('type=1..3')
    array_of_arrays = Array([arr1, arr2], c, dtype=object)
    res = stack(array_of_arrays, c)
    expected = from_string(r"""nat  type\c   c0   c1
                                BE       1  1.0  0.0
                                BE       2  1.0  0.0
                                BE       3  1.0  0.0
                                FO       1  1.0  0.0
                                FO       2  1.0  0.0
                                FO       3  1.0  0.0""")
    assert_larray_equal(res, expected)

    # non scalar/non Array
    res = stack(([1, 2, 3], [4, 5, 6]))
    expected = Array([[1, 4],
                      [2, 5],
                      [3, 6]])
    assert_larray_equal(res, expected)

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
    assert_larray_equal(res, expected)

    # b) keys not given in axes iteration order
    res = stack({('a0', 'b0'): 0,
                 ('a1', 'b0'): 2,
                 ('a2', 'b0'): 4,
                 ('a0', 'b1'): 1,
                 ('a1', 'b1'): 3,
                 ('a2', 'b1'): 5},
                (a, b))
    expected = ndtest((a, b))
    assert_larray_equal(res, expected)

    # c) key parts not given in the order of axes (ie key part for b before key part for a)
    res = stack({('a0', 'b0'): 0,
                 ('a1', 'b0'): 1,
                 ('a2', 'b0'): 2,
                 ('a0', 'b1'): 3,
                 ('a1', 'b1'): 4,
                 ('a2', 'b1'): 5},
                (b, a))
    expected = ndtest((b, a))
    assert_larray_equal(res, expected)

    # d) same as c) but with a key-value sequence
    res = stack([(('a0', 'b0'), 0),
                 (('a1', 'b0'), 1),
                 (('a2', 'b0'), 2),
                 (('a0', 'b1'), 3),
                 (('a1', 'b1'), 4),
                 (('a2', 'b1'), 5)],
                (b, a))
    expected = ndtest((b, a))
    assert_larray_equal(res, expected)


def test_stack_kwargs_no_axis_labels():
    # these tests rely on kwargs ordering, hence python 3.6+

    # 1) using scalars
    # ----------------
    # a) with an axis name
    res = stack(a0=0, a1=1, axes='a')
    expected = Array([0, 1], 'a=a0,a1')
    assert_larray_equal(res, expected)

    # b) without an axis name
    res = stack(a0=0, a1=1)
    expected = Array([0, 1], 'a0,a1')
    assert_larray_equal(res, expected)

    # 2) dict of arrays
    # -----------------
    a = Axis('a=a0,a1,a2')
    arr0 = ndtest(a)
    arr1 = ndtest(a, start=-1)

    # a) with an axis name
    res = stack(b0=arr0, b1=arr1, axes='b')
    expected = Array([[0, -1],
                      [1,  0],                                       # noqa: E241
                      [2,  1]], [a, 'b=b0,b1'])                      # noqa: E241
    assert_larray_equal(res, expected)

    # b) without an axis name
    res = stack(b0=arr0, b1=arr1)
    expected = Array([[0, -1],
                      [1,  0],                                       # noqa: E241
                      [2,  1]], [a, 'b0,b1'])                        # noqa: E241
    assert_larray_equal(res, expected)


def test_stack_dict_no_axis_labels():
    # these tests rely on dict ordering (hence require python 3.7+)

    # 1) dict of scalars
    # ------------------
    # a) with an axis name
    res = stack({'a0': 0, 'a1': 1}, 'a')
    expected = Array([0, 1], 'a=a0,a1')
    assert_larray_equal(res, expected)

    # b) without an axis name
    res = stack({'a0': 0, 'a1': 1})
    expected = Array([0, 1], 'a0,a1')
    assert_larray_equal(res, expected)

    # 2) dict of arrays
    # -----------------
    a = Axis('a=a0,a1,a2')
    arr0 = ndtest(a)
    arr1 = ndtest(a, start=-1)

    # a) with an axis name
    res = stack({'b0': arr0, 'b1': arr1}, 'b')
    expected = Array([[0, -1],
                      [1,  0],                                       # noqa: E241
                      [2,  1]], [a, 'b=b0,b1'])                      # noqa: E241
    assert_larray_equal(res, expected)

    # b) without an axis name
    res = stack({'b0': arr0, 'b1': arr1})
    expected = Array([[0, -1],
                      [1,  0],                                       # noqa: E241
                      [2,  1]], [a, 'b0,b1'])                        # noqa: E241
    assert_larray_equal(res, expected)


def test_0darray_convert():
    int_arr = Array(1)
    assert int(int_arr) == 1
    assert float(int_arr) == 1.0
    assert int_arr.__index__() == 1

    float_arr = Array(1.0)
    assert int(float_arr) == 1
    assert float(float_arr) == 1.0
    with must_raise(TypeError, match='.*') as e_info:
        float_arr.__index__()

    msg = e_info.value.args[0]
    expected_np11 = "only integer arrays with one element can be converted to an index"
    expected_np12 = "only integer scalar arrays can be converted to a scalar index"
    assert msg in {expected_np11, expected_np12}


def test_deprecated_methods():
    with must_raise(TypeError, msg="with_axes() is deprecated. Use set_axes() instead."):
        ndtest((2, 2)).with_axes('a', 'd=d0,d1')

    with must_raise(TypeError, msg="split_axis() is deprecated. Use split_axes() instead."):
        ndtest((2, 2)).combine_axes().split_axis()


def test_eq():
    a = ndtest((2, 3, 4))
    ao = a.astype(object)
    assert_larray_equal(ao.eq(ao['c0'], nans_equal=True), a == a['c0'])


def test_zip_array_values():
    arr1 = ndtest((2, 3))
    # b axis intentionally not the same on both arrays
    arr2 = ndtest((2, 2, 2))

    # 1) no axes => return input arrays themselves
    res = list(zip_array_values((arr1, arr2), ()))
    assert len(res) == 1 and len(res[0]) == 2
    r0_arr1, r0_arr2 = res[0]
    assert_larray_equal(r0_arr1, arr1)
    assert_larray_equal(r0_arr2, arr2)

    # 2) iterate on an axis not present on one of the arrays => the other array is repeated
    res = list(zip_array_values((arr1, arr2), arr2.c))
    assert len(res) == 2 and all(len(r) == 2 for r in res)
    r0_arr1, r0_arr2 = res[0]
    r1_arr1, r1_arr2 = res[1]
    assert_larray_equal(r0_arr1, arr1)
    assert_larray_equal(r0_arr2, arr2['c0'])
    assert_larray_equal(r1_arr1, arr1)
    assert_larray_equal(r1_arr2, arr2['c1'])


def test_zip_array_items():
    arr1 = ndtest('a=a0,a1;b=b0,b1')
    arr2 = ndtest('a=a0,a1;c=c0,c1')
    res = list(zip_array_items((arr1, arr2), axes=()))
    assert len(res) == 1 and len(res[0]) == 2 and len(res[0][1]) == 2
    r0_k, (r0_arr1, r0_arr2) = res[0]
    assert r0_k == ()
    assert_larray_equal(r0_arr1, arr1)
    assert_larray_equal(r0_arr2, arr2)


def test_growth_rate():
    arr = Array([1, 2, 0, 0, 0, 4, 5], axes='time=2014..2020')
    with must_warn(RuntimeWarning, "divide by zero encountered during operation"):
        res = arr.growth_rate('time')
    expected_res = Array([1.0, -1.0, 0.0, 0.0, inf, 0.25], axes='time=2015..2020')
    assert_larray_equal(res, expected_res)


if __name__ == "__main__":
    # import doctest
    # import unittest
    # from larray import core
    # doctest.testmod(core)
    # unittest.main()
    pytest.main()
