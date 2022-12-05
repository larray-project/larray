import pytest
import numpy as np

from larray.tests.common import assert_array_equal, needs_pytables, must_raise
from larray import Axis, IGroup, LGroup, LSet, ndtest, read_hdf
from larray.util.oset import OrderedSet

age = Axis('age=0..10')
lipro = Axis('lipro=P01..P05')
anonymous = Axis(range(3))
age_wildcard = Axis(10, 'wildcard')


def test_equals():
    a = Axis('a=a0..a2')
    # keys are single value
    assert a['a0'].equals(a['a0'])
    # keys are arrays
    assert a[np.array(['a0', 'a2'])].equals(a[np.array(['a0', 'a2'])])
    # axis names
    anonymous_axis = Axis('a0..a2')
    assert anonymous_axis['a0:a2'].equals(anonymous_axis['a0:a2'])
    assert not a['a0:a2'].equals(anonymous_axis['a0:a2'])
    assert not a['a0:a2'].equals(a.rename('other_name')['a0:a2'])
    # list of labels
    a02 = a['a0,a1,a2'] >> 'group'
    assert a02.equals(a02)
    a02_unamed = a['a0,a1,a2']
    assert not a02.equals(a02_unamed)
    a13 = a['a1,a2,a3'] >> 'group'
    assert not a02.equals(a13)
    a02_other_axis_name = a.rename('other_name')['a0,a1,a2'] >> 'group'
    assert not a02.equals(a02_other_axis_name)


# ################## #
#       LGroup       #
# ################## #

@pytest.fixture
def lgroups():
    class TestLGroup:
        def __init__(self):
            self.slice_both_named_wh_named_axis = LGroup('1:5', "full", age)
            self.slice_both_named = LGroup('1:5', "named")
            self.slice_both = LGroup('1:5')
            self.slice_start = LGroup('1:')
            self.slice_stop = LGroup(':5')
            self.slice_none_no_axis = LGroup(':')
            self.slice_none_wh_named_axis = LGroup(':', axis=lipro)
            self.slice_none_wh_anonymous_axis = LGroup(':', axis=anonymous)
            self.single_value = LGroup('P03')
            self.list = LGroup('P01,P03,P04')
            self.list_named = LGroup('P01,P03,P04', "P134")
    return TestLGroup()


def test_init_lgroup(lgroups):
    assert lgroups.slice_both_named_wh_named_axis.name == "full"
    assert lgroups.slice_both_named_wh_named_axis.key == slice(1, 5, None)
    assert lgroups.slice_both_named_wh_named_axis.axis is age
    assert lgroups.slice_both_named.name == "named"
    assert lgroups.slice_both_named.key == slice(1, 5, None)
    assert lgroups.slice_both.key == slice(1, 5, None)
    assert lgroups.slice_start.key == slice(1, None, None)
    assert lgroups.slice_stop.key == slice(None, 5, None)
    assert lgroups.slice_none_no_axis.key == slice(None, None, None)
    assert lgroups.slice_none_wh_named_axis.axis is lipro
    assert lgroups.slice_none_wh_anonymous_axis.axis is anonymous
    assert lgroups.single_value.key == 'P03'
    assert lgroups.list.key == ['P01', 'P03', 'P04']
    group = LGroup('1:5', age, age)
    assert group.name == age.name
    group = age['1:5'] >> age
    assert group.name == age.name
    group2 = LGroup('1', axis=age)
    group = LGroup('1', group2, axis=age)
    assert group.name == '1'
    group = age['1'] >> group2
    assert group.name == '1'
    group2 = LGroup('1:5', 'age', age)
    group = LGroup('1:5', group2, axis=age)
    assert group.name == group2.name
    group = age['1:5'] >> group2
    assert group.name == group2.name
    axis = Axis('axis=a,a0..a3,b,b0..b3,c,c0..c3')
    for code in axis.matching(regex='^.$'):
        group = axis.startingwith(code) >> code
        assert group.equals(axis.startingwith(code) >> str(code))


def test_eq_lgroup(lgroups):
    # with axis vs no axis do not compare equal
    # lgroups.slice_both == lgroups.slice_both_named_wh_named_axis
    assert lgroups.slice_both == lgroups.slice_both_named
    res = lgroups.slice_both_named_wh_named_axis == age[1:5]
    assert isinstance(res, np.ndarray)
    assert res.shape == (5,)
    assert res.all()
    assert lgroups.slice_both == LGroup(slice(1, 5))
    assert lgroups.slice_start == LGroup(slice(1, None))
    assert lgroups.slice_stop == LGroup(slice(5))
    assert lgroups.slice_none_no_axis == LGroup(slice(None))
    assert lgroups.single_value == LGroup('P03')
    assert lgroups.list == LGroup(['P01', 'P03', 'P04'])
    assert lgroups.list_named == LGroup(['P01', 'P03', 'P04'])
    assert lgroups.slice_both == slice(1, 5)
    assert lgroups.slice_start == slice(1, None)
    assert lgroups.slice_stop == slice(5)
    assert lgroups.slice_none_no_axis == slice(None)
    assert lgroups.single_value == 'P03'
    assert lgroups.list == ['P01', 'P03', 'P04']
    assert lgroups.list_named == ['P01', 'P03', 'P04']


def test_getitem_lgroup():
    axis = Axis("a=a0,a1")
    assert axis['a0'][0] == 'a'
    assert axis['a0'][1] == '0'
    assert axis['a0':'a1'][1] == 'a1'
    assert axis[:][1] == 'a1'
    assert list(axis[:][0:2]) == ['a0', 'a1']
    assert list((axis[:][[1, 0]])) == ['a1', 'a0']
    assert axis[['a0', 'a1', 'a0']][2] == 'a0'
    assert axis[('a0', 'a1', 'a0')][2] == 'a0'
    assert axis[ndtest("a=a0,a1,a0")][2] == 2


def test_sorted_lgroup():
    assert sorted(LGroup(['c', 'd', 'a', 'b'])) == [LGroup('a'), LGroup('b'), LGroup('c'), LGroup('d')]


def test_asarray_lgroup(lgroups):
    assert_array_equal(np.asarray(lgroups.slice_both_named_wh_named_axis), np.array([1, 2, 3, 4, 5]))
    assert_array_equal(np.asarray(lgroups.slice_none_wh_named_axis), np.array(['P01', 'P02', 'P03', 'P04', 'P05']))


def test_hash_lgroup(lgroups):
    # this test is a lot less important than what it used to, because we cannot have Group ticks on an axis anymore
    d = {lgroups.slice_both: 1, lgroups.single_value: 2, lgroups.list_named: 3}
    assert d.get(lgroups.slice_both) == 1
    assert d.get(lgroups.single_value) == 2
    assert d.get(lgroups.list) == 3
    assert d.get(lgroups.list_named) == 3


def test_repr_lgroup(lgroups):
    assert repr(lgroups.slice_both_named_wh_named_axis) == "age[1:5] >> 'full'"
    assert repr(lgroups.slice_both_named) == "LGroup(slice(1, 5, None)) >> 'named'"
    assert repr(lgroups.slice_both) == "LGroup(slice(1, 5, None))"
    assert repr(lgroups.list) == "LGroup(['P01', 'P03', 'P04'])"
    assert repr(lgroups.slice_none_no_axis) == "LGroup(slice(None, None, None))"
    assert repr(lgroups.slice_none_wh_named_axis) == "lipro[:]"
    assert repr(lgroups.slice_none_wh_anonymous_axis) == "LGroup(slice(None, None, None), axis=Axis([0, 1, 2], None))"


def test_to_int_lgroup():
    a = Axis(['42'], 'a')
    assert int(a['42']) == 42


def test_to_float_lgroup():
    a = Axis(['42'], 'a')
    assert float(a['42']) == 42.0


def test_to_lset_lgroup():
    alpha = Axis('alpha=a,b,c,d')

    # list key
    lg = LGroup(['a', 'a', 'c'], axis=alpha)
    res = lg.set()
    assert isinstance(res, LSet)
    assert res.axis is alpha
    assert res.key == OrderedSet(['a', 'c'])

    # scalar key
    lg = LGroup('c', axis=alpha)
    res = lg.set()
    assert isinstance(res, LSet)
    assert res.axis is alpha
    assert res.key == OrderedSet(['c'])


@needs_pytables
def test_h5_io_lgroup(tmp_path):
    fpath = tmp_path / 'lgroups.h5'
    age.to_hdf(fpath)

    named = age[':5'] >> 'age_05'
    named_axis_not_in_file = lipro['P01,P03,P05'] >> 'P_odd'
    anonymous = age[':5']
    wildcard = age_wildcard[':5'] >> 'age_w_05'
    string_group = Axis(['@!àéè&%µ$~', '/*-+_§()><', 'another label'], 'string_axis')[:] >> 'string_group'

    # ---- default behavior ----
    # named group
    named.to_hdf(fpath)
    named2 = read_hdf(fpath, key=named.name)
    assert all(named == named2)
    # anonymous group
    with must_raise(ValueError, msg="Argument key must be provided explicitly in case of anonymous group"):
        anonymous.to_hdf(fpath)
    # wildcard group
    wildcard.to_hdf(fpath)
    wildcard2 = read_hdf(fpath, key=wildcard.name)
    assert all(wildcard == wildcard2)
    # associated axis not saved yet
    named_axis_not_in_file.to_hdf(fpath)
    named2 = read_hdf(fpath, key=named_axis_not_in_file.name)
    assert all(named_axis_not_in_file == named2)
    # string group
    string_group.to_hdf(fpath)
    string_group2 = read_hdf(fpath, key=string_group.name)
    assert all(string_group == string_group2)

    # ---- specific hdf group + key ----
    hdf_group = 'my_groups'
    # named group
    key = hdf_group + '/named_group'
    named.to_hdf(fpath, key)
    named2 = read_hdf(fpath, key=key)
    assert all(named == named2)
    # anonymous group
    key = hdf_group + '/anonymous_group'
    anonymous.to_hdf(fpath, key)
    anonymous2 = read_hdf(fpath, key=key)
    assert anonymous2.name is None
    assert all(anonymous == anonymous2)
    # wildcard group
    key = hdf_group + '/wildcard_group'
    wildcard.to_hdf(fpath, key)
    wildcard2 = read_hdf(fpath, key=key)
    assert all(wildcard == wildcard2)
    # associated axis not saved yet
    key = hdf_group + '/named_group_axis_not_in_file'
    named_axis_not_in_file.to_hdf(fpath, key=key)
    named2 = read_hdf(fpath, key=key)
    assert all(named_axis_not_in_file == named2)

    # ---- specific axis_key ----
    axis_key = 'axes/associated_axis_0'
    # named group
    named.to_hdf(fpath, axis_key=axis_key)
    named2 = read_hdf(fpath, key=named.name)
    assert all(named == named2)
    # anonymous group
    key = 'anonymous'
    anonymous.to_hdf(fpath, key=key, axis_key=axis_key)
    anonymous2 = read_hdf(fpath, key=key)
    assert anonymous2.name is None
    assert all(anonymous == anonymous2)
    # wildcard group
    wildcard.to_hdf(fpath, axis_key=axis_key)
    wildcard2 = read_hdf(fpath, key=wildcard.name)
    assert all(wildcard == wildcard2)
    # associated axis not saved yet
    axis_key = 'axes/associated_axis_1'
    named_axis_not_in_file.to_hdf(fpath, axis_key=axis_key)
    named2 = read_hdf(fpath, key=named_axis_not_in_file.name)
    assert all(named_axis_not_in_file == named2)


# ################## #
#        LSet        #
# ################## #

def test_or_lset():
    # without axis
    assert LSet(['a', 'b']) | LSet(['c', 'd']) == LSet(['a', 'b', 'c', 'd'])
    assert LSet(['a', 'b', 'c']) | LSet(['c', 'd']) == LSet(['a', 'b', 'c', 'd'])
    alpha = Axis('alpha=a,b,c,d')
    res = alpha['a', 'b'].set() | alpha['c', 'd'].set()
    assert res.axis is alpha
    assert res == alpha['a', 'b', 'c', 'd'].set()
    assert alpha['a', 'b', 'c'].set() | alpha['c', 'd'].set() == alpha['a', 'b', 'c', 'd'].set()
    alpha = Axis('alpha=a,b,c,d')
    res = alpha['a', 'b'].set() | alpha['c', 'd']
    assert res.axis is alpha
    assert res == alpha['a', 'b', 'c', 'd'].set()
    assert alpha['a', 'b', 'c'].set() | alpha['c', 'd'] == alpha['a', 'b', 'c', 'd'].set()
    alpha = Axis('alpha=a,b,c,d')
    res = alpha['a', 'b'].set().named('ab') | alpha['c', 'd'].set().named('cd')
    assert res.axis is alpha
    assert res.name == 'ab | cd'
    assert res == alpha['a', 'b', 'c', 'd'].set()
    assert alpha['a', 'b', 'c'].set() | alpha['c', 'd'] == alpha['a', 'b', 'c', 'd'].set()
    num = Axis(range(10), 'num')
    assert num[1, 5, 3].set() | 4 == num[1, 5, 3, 4].set()
    assert num[1, 5, 3].set() | num[4] == num[1, 5, 3, 4].set()
    assert num[4].set() | num[1, 5, 3] == num[4, 1, 5, 3].set()
    assert num[:2].set() | num[8:] == num[0, 1, 2, 8, 9].set()
    assert num[:2].set() | num[5] == num[0, 1, 2, 5].set()


def test_and_lset():
    # without axis
    assert LSet(['a', 'b', 'c']) & LSet(['c', 'd']) == LSet(['c'])
    alpha = Axis('alpha=a,b,c,d')
    res = alpha['a', 'b', 'c'].named('abc').set() & alpha['c', 'd'].named('cd')
    assert res.axis is alpha
    assert res.name == 'abc & cd'
    assert res == alpha[['c']].set()


def test_sub_lset():
    assert LSet(['a', 'b', 'c']) - LSet(['c', 'd']) == LSet(['a', 'b'])
    assert LSet(['a', 'b', 'c']) - ['c', 'd'] == LSet(['a', 'b'])
    assert LSet(['a', 'b', 'c']) - 'b' == LSet(['a', 'c'])
    assert LSet([1, 2, 3]) - 4 == LSet([1, 2, 3])
    assert LSet([1, 2, 3]) - 2 == LSet([1, 3])


# ################## #
#       IGroup       #
# ################## #

@pytest.fixture
def igroups():
    class TestIGroup:
        def __init__(self):
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
    return TestIGroup()


def _assert_array_equal_is_true_array(a, b):
    res = a == b
    assert isinstance(res, np.ndarray)
    assert res.shape == np.asarray(b).shape
    assert res.all()


def test_asarray_igroup(igroups):
    assert_array_equal(np.asarray(igroups.slice_both), np.array(['a1', 'a2', 'a3']))


def test_eq_igroup(igroups):
    _assert_array_equal_is_true_array(igroups.slice_both, ['a1', 'a2', 'a3'])
    _assert_array_equal_is_true_array(igroups.slice_both_named, ['a1', 'a2', 'a3'])
    _assert_array_equal_is_true_array(igroups.slice_both, igroups.slice_both_named)
    _assert_array_equal_is_true_array(igroups.slice_both_named, igroups.slice_both)
    _assert_array_equal_is_true_array(igroups.slice_start, ['a1', 'a2', 'a3', 'a4'])
    _assert_array_equal_is_true_array(igroups.slice_stop, ['a0', 'a1', 'a2', 'a3'])
    _assert_array_equal_is_true_array(igroups.slice_none, ['a0', 'a1', 'a2', 'a3', 'a4'])
    assert igroups.first_value == 'a0'
    assert igroups.last_value == 'a4'
    _assert_array_equal_is_true_array(igroups.list, ['a0', 'a1', 'a3', 'a4'])
    _assert_array_equal_is_true_array(igroups.tuple, ['a0', 'a1', 'a3', 'a4'])


def test_getitem_igroup():
    axis = Axis("a=a0,a1")
    assert axis.i[0][0] == 'a'
    assert axis.i[0][1] == '0'
    assert axis.i[0:1][1] == 'a1'
    assert axis.i[:][1] == 'a1'
    assert list(axis.i[:][0:2]) == ['a0', 'a1']
    assert list((axis.i[:][[1, 0]])) == ['a1', 'a0']
    assert axis.i[[0, 1, 0]][2] == 'a0'
    assert axis.i[(0, 1, 0)][2] == 'a0'


def test_getattr_igroup():
    agg = Axis(['a1:a2', ':a2', 'a1:'], 'agg')
    assert agg.i[0].split(':') == ['a1', 'a2']
    assert agg.i[1].split(':') == ['', 'a2']
    assert agg.i[2].split(':') == ['a1', '']


def test_dir_igroup():
    agg = Axis(['a', 1], 'agg')
    assert 'split' in dir(agg.i[0])
    assert 'strip' in dir(agg.i[0])
    assert 'strip' in dir(agg.i[0])


def test_repr_igroup(igroups):
    assert repr(igroups.slice_both_named) == "code.i[1:4] >> 'a123'"
    assert repr(igroups.slice_both) == "code.i[1:4]"
    assert repr(igroups.slice_start) == "code.i[1:]"
    assert repr(igroups.slice_stop) == "code.i[:4]"
    assert repr(igroups.slice_none) == "code.i[:]"
    assert repr(igroups.first_value) == "code.i[0]"
    assert repr(igroups.last_value) == "code.i[-1]"
    assert repr(igroups.list) == "code.i[0, 1, -2, -1]"
    assert repr(igroups.tuple) == "code.i[0, 1, -2, -1]"


def test_to_int_igroup():
    a = Axis(['42'], 'a')
    assert int(a.i[0]) == 42


def test_to_float_igroup():
    a = Axis(['42'], 'a')
    assert float(a.i[0]) == 42.0


def test_to_lset_igroup():
    alpha = Axis('alpha=a,b,c,d')

    # list key
    ig = IGroup([0, 0, 2], axis=alpha)
    res = ig.set()
    assert isinstance(res, LSet)
    assert res.axis is alpha
    assert res.key == OrderedSet(['a', 'c'])

    # scalar key
    ig = IGroup(2, axis=alpha)
    res = ig.set()
    assert isinstance(res, LSet)
    assert res.axis is alpha
    assert res.key == OrderedSet(['c'])


@needs_pytables
def test_h5_io_igroup(tmp_path):
    fpath = tmp_path / 'igroups.h5'
    age.to_hdf(fpath)

    named = age.i[:6] >> 'age_05'
    named_axis_not_in_file = lipro.i[1::2] >> 'P_odd'
    anonymous = age.i[:6]
    wildcard = age_wildcard.i[:6] >> 'age_w_05'

    # ---- default behavior ----
    # named group
    named.to_hdf(fpath)
    named2 = read_hdf(fpath, key=named.name)
    assert all(named == named2)
    # anonymous group
    with must_raise(ValueError, msg="Argument key must be provided explicitly in case of anonymous group"):
        anonymous.to_hdf(fpath)
    # wildcard group
    wildcard.to_hdf(fpath)
    wildcard2 = read_hdf(fpath, key=wildcard.name)
    assert all(wildcard == wildcard2)
    # associated axis not saved yet
    named_axis_not_in_file.to_hdf(fpath)
    named2 = read_hdf(fpath, key=named_axis_not_in_file.name)
    assert all(named_axis_not_in_file == named2)

    # ---- specific hdf group + key ----
    hdf_group = 'my_groups'
    # named group
    key = hdf_group + '/named_group'
    named.to_hdf(fpath, key)
    named2 = read_hdf(fpath, key=key)
    assert all(named == named2)
    # anonymous group
    key = hdf_group + '/anonymous_group'
    anonymous.to_hdf(fpath, key)
    anonymous2 = read_hdf(fpath, key=key)
    assert anonymous2.name is None
    assert all(anonymous == anonymous2)
    # wildcard group
    key = hdf_group + '/wildcard_group'
    wildcard.to_hdf(fpath, key)
    wildcard2 = read_hdf(fpath, key=key)
    assert all(wildcard == wildcard2)
    # associated axis not saved yet
    key = hdf_group + '/named_group_axis_not_in_file'
    named_axis_not_in_file.to_hdf(fpath, key=key)
    named2 = read_hdf(fpath, key=key)
    assert all(named_axis_not_in_file == named2)

    # ---- specific axis_key ----
    axis_key = 'axes/associated_axis_0'
    # named group
    named.to_hdf(fpath, axis_key=axis_key)
    named2 = read_hdf(fpath, key=named.name)
    assert all(named == named2)
    # anonymous group
    key = 'anonymous'
    anonymous.to_hdf(fpath, key=key, axis_key=axis_key)
    anonymous2 = read_hdf(fpath, key=key)
    assert anonymous2.name is None
    assert all(anonymous == anonymous2)
    # wildcard group
    wildcard.to_hdf(fpath, axis_key=axis_key)
    wildcard2 = read_hdf(fpath, key=wildcard.name)
    assert all(wildcard == wildcard2)
    # associated axis not saved yet
    axis_key = 'axes/associated_axis_1'
    named_axis_not_in_file.to_hdf(fpath, axis_key=axis_key)
    named2 = read_hdf(fpath, key=named_axis_not_in_file.name)
    assert all(named_axis_not_in_file == named2)


if __name__ == "__main__":
    pytest.main()
