import pytest
import numpy as np

from larray.tests.common import assert_array_equal, assert_nparray_equal, needs_pytables, must_raise
from larray import Axis, LGroup, IGroup, read_hdf, X, ndtest
from larray.core.axis import AxisReference


def test_init():
    axis = Axis(3)
    assert len(axis) == 3
    assert list(axis.labels) == list(range(3))
    axis = Axis(np.int32(3))
    assert len(axis) == 3
    assert list(axis.labels) == list(range(3))

    axis = Axis([0, 1], name='test')
    assert axis.name == 'test'
    assert_nparray_equal(axis.labels, np.array([0, 1]))

    axis = Axis([0, 1], name=np.str_('test'))
    assert axis.name == 'test'
    assert type(axis.name) is not np.str_
    assert_nparray_equal(axis.labels, np.array([0, 1]))

    sex_tuple = ('M', 'F')
    sex_list = ['M', 'F']
    sex_array = np.array(sex_list)
    assert_nparray_equal(Axis(sex_tuple, 'sex').labels, sex_array)
    assert_nparray_equal(Axis(sex_list, 'sex').labels, sex_array)
    assert_nparray_equal(Axis(sex_array, 'sex').labels, sex_array)
    assert_nparray_equal(Axis('sex=M,F').labels, sex_array)

    assert_nparray_equal(Axis(range(116), 'age').labels, np.arange(116))
    axis = Axis('0..115', 'age')
    assert_nparray_equal(axis.labels, np.arange(116))
    assert list(Axis('01..12', 'zero_padding').labels) == [str(i).zfill(2) for i in range(1, 13)]
    assert list(Axis('01,02,03,10,11,12', 'zero_padding').labels) == ['01', '02', '03', '10', '11', '12']
    group = axis[:10]
    group_axis = Axis(group)
    assert_nparray_equal(group_axis.labels, np.arange(11))
    assert group_axis.name == 'age'
    other = Axis('other=0..10')
    axis = Axis(other, 'age')
    assert_nparray_equal(axis.labels, other.labels)
    assert axis.name == 'age'


def test_equals():
    assert Axis('sex=M,F').equals(Axis('sex=M,F'))
    assert Axis('sex=M,F').equals(Axis(['M', 'F'], 'sex'))
    assert not Axis('sex=M,W').equals(Axis('sex=M,F'))
    assert not Axis('sex1=M,F').equals(Axis('sex2=M,F'))
    assert not Axis('sex1=M,W').equals(Axis('sex2=M,F'))


def test_getitem():
    code = Axis('code=a1..a3,b1..b3')

    # tuple key
    res = code['a1', 'a3', 'b1']
    assert res.key == ['a1', 'a3', 'b1']
    assert res.name is None
    assert res.axis is code

    # list key
    res = code[['a1', 'a3', 'b1']]
    assert res.key == ['a1', 'a3', 'b1']
    assert res.name is None
    assert res.axis is code

    # list key with name
    res = code[['a1', 'a3', 'b1']] >> 'test'
    assert res.key == ['a1', 'a3', 'b1']
    assert res.name == 'test'
    assert res.axis is code

    # formatted string key (list)
    res = code['a1, a3, b1']
    assert list(res.key) == ['a1', 'a3', 'b1']
    assert res.name is None
    assert res.axis is code

    # formatted string key (slice)
    res = code['a2:b2']
    assert res.key == slice('a2', 'b2')
    assert res.name is None
    assert res.axis is code

    # slice with bounds
    res = code['a2':'b2']
    assert res.key == slice('a2', 'b2')
    assert res.name is None
    assert res.axis is code

    # slice with bounds with name
    res = code['a2':'b2'] >> 'a2_to_b2'
    assert res.key == slice('a2', 'b2')
    assert res.name == "a2_to_b2"
    assert res.axis is code

    # slice without bounds
    res = code[:]
    assert res.key == slice(None)
    assert res.name is None
    assert res.axis is code

    # slice without bounds with name
    res = code[:] >> 'all'
    assert res.key == slice(None)
    assert res.name == "all"
    assert res.axis is code

    # axis key
    code2 = Axis('code=a1,a3,b1')
    res = code[code2]
    assert list(res.key) == list(code2.labels)


def test_index():
    # an axis with labels having the object dtype
    a = Axis(np.array(["a0", "a1"], dtype=object), 'a')
    assert a.index('a1') == 1
    assert a.index('a1 >> A1') == 1

    time = Axis([2007, 2009], 'time')
    res = time.index(time.i[1])
    assert res == 1

    res = time.index(X.time.i[1])
    assert res == 1

    res = time.index('time.i[1]')
    assert res == 1


def test_astype():
    arr = ndtest(Axis('time=2015..2020,total')).drop('total')
    time = arr.time
    assert time.dtype.kind == 'U'
    time = time.astype(int)
    assert time.dtype.kind == 'i'


# TODO: merge this with next function
def test_getitem_lgroup_keys():
    def group_equal(g1, g2):
        return g1.key == g2.key and g1.name == g2.name and g1.axis is g2.axis

    age = Axis(range(100), 'age')
    ages = [1, 5, 9]
    val_only = LGroup(ages)
    assert group_equal(age[val_only], LGroup(ages, axis=age))
    assert group_equal(age[val_only] >> 'a_name', LGroup(ages, 'a_name', axis=age))
    val_name = LGroup(ages, 'val_name')
    assert group_equal(age[val_name], LGroup(ages, 'val_name', age))
    assert group_equal(age[val_name] >> 'a_name', LGroup(ages, 'a_name', age))
    val_axis = LGroup(ages, axis=age)
    assert group_equal(age[val_axis], LGroup(ages, axis=age))
    assert group_equal(age[val_axis] >> 'a_name', LGroup(ages, 'a_name', axis=age))
    val_axis_name = LGroup(ages, 'val_axis_name', age)
    assert group_equal(age[val_axis_name], LGroup(ages, 'val_axis_name', age))
    assert group_equal(age[val_axis_name] >> 'a_name', LGroup(ages, 'a_name', age))


def test_getitem_group_keys():
    a = Axis('a=a0..a2')
    alt_a = Axis('a=a1..a3')

    key = a['a1']

    res = a[key]
    assert res.key == 'a1'
    assert res.axis is a

    res = alt_a[key]
    assert res.key == 'a1'
    assert res.axis is alt_a

    key = a['a1':'a2']

    res = a[key]
    assert res.key == slice('a1', 'a2')
    assert res.axis is a

    res = alt_a[key]
    assert res.key == slice('a1', 'a2')
    assert res.axis is alt_a

    key = a[['a1', 'a2']]

    res = a[key]
    assert res.key == ['a1', 'a2']
    assert res.axis is a

    res = alt_a[key]
    assert res.key == ['a1', 'a2']
    assert res.axis is alt_a

    key = a.i[1]

    res = a[key]
    assert isinstance(res, LGroup)
    assert res.key == 'a1'
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert res.key == 'a1'
    assert res.axis is alt_a

    key = a.i[1:3]

    res = a[key]
    assert isinstance(res, LGroup)
    assert res.key == slice('a1', 'a2')
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert res.key == slice('a1', 'a2')
    assert res.axis is alt_a

    key = a.i[[1, 2]]

    res = a[key]
    assert isinstance(res, LGroup)
    assert list(res.key) == ['a1', 'a2']
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert list(res.key) == ['a1', 'a2']
    assert res.axis is alt_a

    lg_a1 = a['a1']
    lg_a2 = a['a2']

    res = a[lg_a1:lg_a2]
    assert isinstance(res, LGroup)
    assert res.key == slice('a1', 'a2')
    assert res.axis is a

    res = alt_a[lg_a1:lg_a2]
    assert isinstance(res, LGroup)
    assert res.key == slice('a1', 'a2')
    assert res.axis is alt_a

    pg_a1 = a.i[1]
    pg_a2 = a.i[2]

    res = a[pg_a1:pg_a2]
    assert isinstance(res, LGroup)
    assert res.key == slice('a1', 'a2')
    assert res.axis is a

    res = alt_a[pg_a1:pg_a2]
    assert isinstance(res, LGroup)
    assert res.key == slice('a1', 'a2')
    assert res.axis is alt_a

    key = [a['a1'], a['a2']]

    res = a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is alt_a

    key = [a.i[1], a.i[2]]

    res = a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is alt_a

    key = [a['a1', 'a2'], a['a2', 'a1']]

    res = a[key]
    assert isinstance(res, list)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert res[0].key == ['a1', 'a2']
    assert res[1].key == ['a2', 'a1']
    assert res[0].axis is a
    assert res[1].axis is a

    res = alt_a[key]
    assert isinstance(res, list)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert res[0].key == ['a1', 'a2']
    assert res[1].key == ['a2', 'a1']
    assert res[0].axis is alt_a
    assert res[1].axis is alt_a

    key = (a.i[1, 2], a.i[2, 1])

    res = a[key]
    assert isinstance(res, tuple)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert list(res[0].key) == ['a1', 'a2']
    assert list(res[1].key) == ['a2', 'a1']
    assert res[0].axis is a
    assert res[1].axis is a

    res = alt_a[key]
    assert isinstance(res, tuple)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert list(res[0].key) == ['a1', 'a2']
    assert list(res[1].key) == ['a2', 'a1']
    assert res[0].axis is alt_a
    assert res[1].axis is alt_a

    key = (a['a1'], a['a2'])

    res = a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is alt_a

    key = (a.i[1], a.i[2])

    res = a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is a

    res = alt_a[key]
    assert isinstance(res, LGroup)
    assert res.key == ['a1', 'a2']
    assert res.axis is alt_a

    key = (a['a1', 'a2'], a['a2', 'a1'])

    res = a[key]
    assert isinstance(res, tuple)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert res[0].key == ['a1', 'a2']
    assert res[1].key == ['a2', 'a1']
    assert res[0].axis is a
    assert res[1].axis is a

    res = alt_a[key]
    assert isinstance(res, tuple)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert res[0].key == ['a1', 'a2']
    assert res[1].key == ['a2', 'a1']
    assert res[0].axis is alt_a
    assert res[1].axis is alt_a

    key = (a.i[1, 2], a.i[2, 1])

    res = a[key]
    assert isinstance(res, tuple)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert list(res[0].key) == ['a1', 'a2']
    assert list(res[1].key) == ['a2', 'a1']
    assert res[0].axis is a
    assert res[1].axis is a

    res = alt_a[key]
    assert isinstance(res, tuple)
    assert isinstance(res[0], LGroup)
    assert isinstance(res[1], LGroup)
    assert list(res[0].key) == ['a1', 'a2']
    assert list(res[1].key) == ['a2', 'a1']
    assert res[0].axis is alt_a
    assert res[1].axis is alt_a


def test_axis_ref_getitem_group_keys():
    # test that we can retarget a key to another axis using an axis ref

    # a) when the name of the axis is different
    axis1 = Axis('axis1=a0..a2')

    g = X.axis2[axis1['a1']]
    assert isinstance(g.key, str) and g.key == 'a1'
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis2'

    g = X.axis2[axis1['a1'], axis1['a2']]
    assert isinstance(g.key, list) and g.key == ['a1', 'a2']
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis2'

    g = X.axis2[[axis1['a1'], axis1['a2']]]
    assert isinstance(g.key, list) and g.key == ['a1', 'a2']
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis2'

    g = X.axis2[axis1['a1':'a2']]
    assert isinstance(g.key, slice) and g.key == slice('a1', 'a2')
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis2'

    # b) when the name of the axis is the same (i.e. when the retarget is useless)
    #    this is what issue #787 was all about
    g = X.axis1[axis1['a1']]
    assert isinstance(g.key, str) and g.key == 'a1'
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis1'

    g = X.axis1[axis1['a1'], axis1['a2']]
    assert isinstance(g.key, list) and g.key == ['a1', 'a2']
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis1'

    g = X.axis1[[axis1['a1'], axis1['a2']]]
    assert isinstance(g.key, list) and g.key == ['a1', 'a2']
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis1'

    g = X.axis1[axis1['a1':'a2']]
    assert isinstance(g.key, slice) and g.key == slice('a1', 'a2')
    assert isinstance(g.axis, AxisReference)
    assert g.axis.name == 'axis1'


def test_init_from_group():
    code = Axis('code=C01..C03')
    code_group = code[:'C02']
    subset_axis = Axis(code_group, 'code_subset')
    assert_array_equal(subset_axis.labels, ['C01', 'C02'])


def test_matching():
    code = Axis(['A1', 'A101', 'A2', 'A201'], 'code')
    assert code.matching(regex='^..$') == LGroup(['A1', 'A2'])
    assert code.startingwith('A1') == LGroup(['A1', 'A101'])
    assert code.endingwith('01') == LGroup(['A101', 'A201'])


def test_difference():
    # 1) no duplicate label
    a = Axis('a=a0..a2')

    # remove a single label (a1)
    expected = Axis('a=a0,a2')
    assert a.difference('a1').equals(expected)
    assert a.difference(a['a1']).equals(expected)
    assert a.difference(a[['a1']]).equals(expected)
    assert a.difference([a['a1']]).equals(expected)
    # Ideally, this should raise an Exception because this is not a sequence
    # of labels (it is a sequence of sequences of labels) instead of silently
    # giving the wrong result, but I am unsure the check is worth the
    # performance loss
    # assert a.difference([a[['a1']]]).equals(expected)
    assert a.difference(a.i[1]).equals(expected)
    assert a.difference(a.i[[1]]).equals(expected)
    # this silently fails and should probably work?
    # assert a.difference([a.i[1]]).equals(expected)
    # this should raise (not a sequence of labels)
    # assert a.difference([a.i[[1]]]).equals(expected)

    # 2) with duplicate labels
    a = Axis('a=a0,a1,a0,a2')

    # 2.a) remove a non-duplicated label (a1)
    expected = Axis('a=a0,a0,a2')
    assert a.difference('a1').equals(expected)
    assert a.difference(a['a1']).equals(expected)
    assert a.difference(a[['a1']]).equals(expected)
    assert a.difference([a['a1']]).equals(expected)
    assert a.difference(a.i[1]).equals(expected)
    assert a.difference(a.i[[1]]).equals(expected)
    # assert a.difference([a.i[1]]).equals(expected)

    # 2.b) remove all instance of the duplicated label (a0)
    expected = Axis('a=a1,a2')
    assert a.difference('a0').equals(expected)
    assert a.difference(a['a0']).equals(expected)
    assert a.difference(a[['a0']]).equals(expected)
    assert a.difference([a['a0']]).equals(expected)

    # TODO: it should be possible to remove a particular instance of
    #       a duplicate label, but this is a larger issue (see #438)
    # 2.c) remove a particular (the second) instance of the duplicated label (a0)

    # 2.c.1) the first one
    # expected = Axis('a=a1,a0,a2')
    # assert a.difference(a.i[0]).equals(expected)
    # assert a.difference(a.i[[0]]).equals(expected)

    # 2.c.2) the second one
    # expected = Axis('a=a0,a1,a2')
    # assert a.difference(a.i[2]).equals(expected)
    # assert a.difference(a.i[[2]]).equals(expected)

    # 3) integer labels
    a = Axis('a=1,2,3')
    assert a.difference(2).equals(Axis('a=1,3'))
    assert a.difference('2').equals(Axis('a=1,2,3'))
    assert a.difference('..2').equals(Axis('a=3'))


def test_intersection():
    # 1) axis without duplicate label
    a = Axis('a=a0..a2')

    # intersection with a single label (a1)
    expected = Axis('a=a1')
    assert a.intersection('a1').equals(expected)
    assert a.intersection(a['a1,']).equals(expected)
    assert a.intersection(a[['a1']]).equals(expected)
    assert a.intersection([a['a1']]).equals(expected)
    # Ideally, this should raise an Exception because this is not a sequence
    # of labels (it is a sequence of sequences of labels) instead of silently
    # giving the wrong result, but I am unsure the check is worth the
    # performance loss
    # assert a.difference([a[['a1']]]).equals(expected)
    assert a.intersection(a.i[1]).equals(expected)
    assert a.intersection(a.i[[1]]).equals(expected)
    # this silently fails and should probably work?
    # assert a.difference([a.i[1]]).equals(expected)
    # this should raise (not a sequence of labels)
    # assert a.difference([a.i[[1]]]).equals(expected)

    # 2) axis with duplicate labels
    a = Axis('a=a0,a1,a0,a2')

    # 2.a) intersection with a non-duplicated label (a1)
    expected = Axis('a=a1')
    assert a.intersection('a1').equals(expected)
    assert a.intersection(a['a1']).equals(expected)
    assert a.intersection(a[['a1']]).equals(expected)
    assert a.intersection([a['a1']]).equals(expected)
    assert a.intersection(a.i[1]).equals(expected)
    assert a.intersection(a.i[[1]]).equals(expected)
    # assert a.intersection([a.i[1]]).equals(expected)

    # 2.b) intersection with an unspecified duplicated label (a0)
    expected = Axis('a=a0,a0')
    assert a.intersection('a0').equals(expected)
    assert a.intersection(a['a0']).equals(expected)
    assert a.intersection(a[['a0']]).equals(expected)
    assert a.intersection([a['a0']]).equals(expected)

    # TODO: it should be possible to intersect with particular instance(s) of
    #       a duplicate label, but this is a larger issue (see #438)
    # 2.c) intersection with one particular instance of a duplicated label (a0)

    # 3) integer labels
    a = Axis('a=1,2,3')
    assert a.intersection(2).equals(Axis('a=2'))
    # single int strings are not parsed as integers
    assert a.intersection('2').equals(Axis([], 'a'))
    assert a.intersection('..2').equals(Axis('a=1,2'))


def test_union():
    # 1) axis without duplicate label
    a = Axis('a=a0..a2')
    a2 = Axis('a2=a0..a5')

    # intersection with a single label (a1)
    expected = Axis('a=a0..a3')
    assert a.union('a3').equals(expected)
    assert a.union(a2['a3,']).equals(expected)
    assert a.union(a2[['a3']]).equals(expected)
    assert a.union([a2['a3']]).equals(expected)
    # Ideally, this should raise an Exception because this is not a sequence
    # of labels (it is a sequence of sequences of labels) instead of silently
    # giving the wrong result, but I am unsure the check is worth the
    # performance loss
    # assert a.union([a2[['a3']]]).equals(expected)
    assert a.union(a2.i[3]).equals(expected)
    assert a.union(a2.i[[3]]).equals(expected)
    # this silently fails and should probably work?
    # assert a.union([a2.i[3]]).equals(expected)
    # this should raise (not a sequence of labels)
    # assert a.union([a2.i[[3]]]).equals(expected)

    # 2) axis with duplicate labels
    a = Axis('a=a0,a1,a0,a2')

    # union drops duplicates
    expected = Axis('a=a0,a1,a2,a3')
    assert a.union('a3').equals(expected)
    assert a.union(a2['a3']).equals(expected)
    assert a.union(a2[['a3']]).equals(expected)
    assert a.union([a2['a3']]).equals(expected)
    assert a.union(a2.i[3]).equals(expected)
    assert a.union(a2.i[[3]]).equals(expected)

    # 3) integer labels
    a = Axis('a=1,2,3')
    assert a.union(2).equals(a)
    assert a.union(4).equals(Axis('a=1..4'))
    # single int strings are not parsed as integers
    assert a.union('2').equals(Axis([1, 2, 3, '2'], 'a'))
    assert a.union('4').equals(Axis([1, 2, 3, '4'], 'a'))
    assert a.union('..2').equals(Axis([1, 2, 3, 0], 'a'))


def test_iter():
    sex = Axis('sex=M,F')
    assert list(sex) == [IGroup(0, axis=sex), IGroup(1, axis=sex)]


def test_positional():
    age = Axis('age=0..115')
    key = age.i[:-1]
    assert key.key == slice(None, -1)
    assert key.axis is age


def test_contains():
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
    age247 = age['2,4,7']
    age247bis = age[['2', '4', '7']]
    age359 = age[['3', '5', '9']]
    age468 = age['4,6,8'] >> 'even'
    assert 5 in age
    assert '5' not in age
    assert age2 in age
    assert age2bis not in age
    assert age2ter not in age
    assert age2qua not in age
    assert age20 not in age
    assert age20bis not in age
    assert age20ter not in age
    assert age20qua not in age
    assert ['3', '5', '9'] not in age
    assert '3,5,9' not in age
    assert '3:9' not in age
    assert age247 not in age
    assert age247bis not in age
    assert age359 not in age
    assert age468 not in age
    agg = Axis((age2, age247, age359, age468, '2,6', ['3', '5', '7'], ('6', '7', '9')), "agg")
    assert age2bis not in agg
    assert age2ter not in agg
    assert age2qua not in age
    assert age247 in agg
    assert age247bis in agg
    assert '2,4,7' in agg
    assert ['2', '4', '7'] in agg
    assert age359 in agg
    assert '3,5,9' in agg
    assert ['3', '5', '9'] in agg
    assert age468 in agg
    assert 'even' in agg
    assert '2,6' in agg
    assert ['2', '6'] in agg
    assert age['2,6'] in agg
    assert age[['2', '6']] in agg
    assert '3,5,7' in agg
    assert ['3', '5', '7'] in agg
    assert age['3,5,7'] in agg
    assert age[['3', '5', '7']] in agg
    assert '6,7,9' in agg
    assert ['6', '7', '9'] in agg
    assert age['6,7,9'] in agg
    assert age[['6', '7', '9']] in agg
    assert 5 not in agg
    assert '5' not in agg
    assert age20 not in agg
    assert age20bis not in agg
    assert age20ter not in agg
    assert age20qua not in agg
    assert '2,7' not in agg
    assert ['2', '7'] not in agg
    assert age['2,7'] not in agg
    assert age[['2', '7']] not in agg


@needs_pytables
def test_h5_io(tmp_path):
    age = Axis('age=0..10')
    lipro = Axis('lipro=P01..P05')
    anonymous = Axis(range(3))
    wildcard = Axis(3, 'wildcard')
    string_axis = Axis(['@!àéè&%µ$~', '/*-+_§()><', 'another label'], 'string_axis')
    fpath = tmp_path / 'axes.h5'

    # ---- default behavior ----
    # int axis
    age.to_hdf(fpath)
    age2 = read_hdf(fpath, key=age.name)
    assert age.equals(age2)
    # string axis
    lipro.to_hdf(fpath)
    lipro2 = read_hdf(fpath, key=lipro.name)
    assert lipro.equals(lipro2)
    # anonymous axis
    with must_raise(ValueError, msg="Argument key must be provided explicitly in case of anonymous axis"):
        anonymous.to_hdf(fpath)
    # wildcard axis
    wildcard.to_hdf(fpath)
    wildcard2 = read_hdf(fpath, key=wildcard.name)
    assert wildcard2.iswildcard
    assert wildcard.equals(wildcard2)
    # string axis
    string_axis.to_hdf(fpath)
    string_axis2 = read_hdf(fpath, string_axis.name)
    assert string_axis.equals(string_axis2)

    # ---- specific key ----
    # int axis
    key = 'axis_age'
    age.to_hdf(fpath, key)
    age2 = read_hdf(fpath, key=key)
    assert age.equals(age2)
    # string axis
    key = 'axis_lipro'
    lipro.to_hdf(fpath, key)
    lipro2 = read_hdf(fpath, key=key)
    assert lipro.equals(lipro2)
    # anonymous axis
    key = 'axis_anonymous'
    anonymous.to_hdf(fpath, key)
    anonymous2 = read_hdf(fpath, key=key)
    assert anonymous2.name is None
    assert_array_equal(anonymous.labels, anonymous2.labels)
    # wildcard axis
    key = 'axis_wildcard'
    wildcard.to_hdf(fpath, key)
    wildcard2 = read_hdf(fpath, key=key)
    assert wildcard2.iswildcard
    assert wildcard.equals(wildcard2)

    # ---- specific hdf group + key ----
    hdf_group = 'my_axes'
    # int axis
    key = hdf_group + '/axis_age'
    age.to_hdf(fpath, key)
    age2 = read_hdf(fpath, key=key)
    assert age.equals(age2)
    # string axis
    key = hdf_group + '/axis_lipro'
    lipro.to_hdf(fpath, key)
    lipro2 = read_hdf(fpath, key=key)
    assert lipro.equals(lipro2)
    # anonymous axis
    key = hdf_group + '/axis_anonymous'
    anonymous.to_hdf(fpath, key)
    anonymous2 = read_hdf(fpath, key=key)
    assert anonymous2.name is None
    assert_array_equal(anonymous.labels, anonymous2.labels)
    # wildcard axis
    key = hdf_group + '/axis_wildcard'
    wildcard.to_hdf(fpath, key)
    wildcard2 = read_hdf(fpath, key=key)
    assert wildcard2.iswildcard
    assert wildcard.equals(wildcard2)


def test_split():
    # test splitting an anonymous axis
    a_b = Axis('a0_b0,a0_b1,a0_b2,a1_b0,a1_b1,a1_b2')
    a, b = a_b.split()
    assert a.equals(Axis(['a0', 'a1']))
    assert b.equals(Axis(['b0', 'b1', 'b2']))


if __name__ == "__main__":
    pytest.main()
