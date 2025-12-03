import numpy as np
import pytest

from larray.tests.common import assert_array_equal, assert_axis_eq, must_raise
from larray import Axis, AxisCollection


lipro = Axis('lipro=P01..P04')
sex = Axis('sex=M,F')
sex2 = Axis('sex=F,M')
age = Axis('age=0..7')
geo = Axis('geo=A11,A12,A13')
value = Axis('value=0..10')


@pytest.fixture
def col():
    return AxisCollection((lipro, sex, age))


def test_init_from_group():
    lipro_subset = lipro[:'P03']
    col2 = AxisCollection((lipro_subset, sex))
    assert col2.names == ['lipro', 'sex']
    assert_array_equal(col2.lipro.labels, ['P01', 'P02', 'P03'])
    assert_array_equal(col2.sex.labels, ['M', 'F'])


def test_init_from_string():
    col = AxisCollection('age=10;sex=M,F;year=2000..2017')
    assert col.names == ['age', 'sex', 'year']
    assert list(col.age.labels) == [10]
    assert list(col.sex.labels) == ['M', 'F']
    assert list(col.year.labels) == [y for y in range(2000, 2018)]


def test_eq(col):
    assert col == col
    assert col == AxisCollection((lipro, sex, age))
    assert col == (lipro, sex, age)
    assert col != (lipro, age, sex)


def test_getitem_name(col):
    assert_axis_eq(col['lipro'], lipro)
    assert_axis_eq(col['sex'], sex)
    assert_axis_eq(col['age'], age)


def test_getitem_int(col):
    assert_axis_eq(col[0], lipro)
    assert_axis_eq(col[-3], lipro)
    assert_axis_eq(col[1], sex)
    assert_axis_eq(col[-2], sex)
    assert_axis_eq(col[2], age)
    assert_axis_eq(col[-1], age)

    # test using a numpy int
    assert_axis_eq(col[np.int32(0)], lipro)


def test_getitem_slice(col):
    col = col[:2]
    assert len(col) == 2
    assert_axis_eq(col[0], lipro)
    assert_axis_eq(col[1], sex)


def test_setitem_name(col):
    col2 = col[:]
    col2['lipro'] = geo
    assert len(col2) == 3
    assert col2 == [geo, sex, age]
    col2['sex'] = sex2
    assert col2 == [geo, sex2, age]
    col2['geo'] = lipro
    assert col2 == [lipro, sex2, age]
    col2['age'] = geo
    assert col2 == [lipro, sex2, geo]
    col2['sex'] = sex
    col2['geo'] = age
    assert col2 == col


def test_setitem_name_axis_def(col):
    col2 = col[:]
    col2['lipro'] = 'geo=A11,A12,A13'
    assert len(col2) == 3
    assert col2 == [geo, sex, age]
    col2['sex'] = 'sex=F,M'
    assert col2 == [geo, sex2, age]
    col2['geo'] = 'lipro=P01..P04'
    assert col2 == [lipro, sex2, age]
    col2['age'] = 'geo=A11,A12,A13'
    assert col2 == [lipro, sex2, geo]
    col2['sex'] = 'sex=M,F'
    col2['geo'] = 'age=0..7'
    assert col2 == col


def test_setitem_int(col):
    col[1] = geo
    assert len(col) == 3
    assert col == [lipro, geo, age]
    col[2] = sex
    assert col == [lipro, geo, sex]
    col[-1] = age
    assert col == [lipro, geo, age]


def test_setitem_list_replace(col):
    col[['lipro', 'age']] = [geo, lipro]
    assert col == [geo, sex, lipro]


def test_setitem_slice_replace(col):
    col2 = col[:]
    col2[1:] = [geo, sex]
    assert col2 == [lipro, geo, sex]
    col2[1:] = col[1:]
    assert col2 == col


def test_setitem_slice_insert(col):
    col[1:1] = [geo]
    assert col == [lipro, geo, sex, age]


def test_setitem_slice_delete(col):
    col[1:2] = []
    assert col == [lipro, age]
    col[0:1] = []
    assert col == [age]


def test_delitem(col):
    assert len(col) == 3
    del col[0]
    assert len(col) == 2
    assert_axis_eq(col[0], sex)
    assert_axis_eq(col[1], age)
    del col['age']
    assert len(col) == 1
    assert_axis_eq(col[0], sex)
    del col[sex]
    assert len(col) == 0


def test_delitem_slice(col):
    assert len(col) == 3
    del col[0:2]
    assert len(col) == 1
    assert col == [age]
    del col[:]
    assert len(col) == 0


def test_pop(col):
    lipro, sex, age = col
    assert len(col) == 3
    assert col.pop() is age
    assert len(col) == 2
    assert col[0] is lipro
    assert col[1] is sex
    assert col.pop() is sex
    assert len(col) == 1
    assert col[0] is lipro
    assert col.pop() is lipro
    assert len(col) == 0


def test_replace(col):
    col2 = col[:]
    newcol = col2.replace('sex', geo)
    assert col2 == col
    assert len(newcol) == 3
    assert newcol.names == ['lipro', 'geo', 'age']
    assert newcol.shape == (4, 3, 8)
    newcol = newcol.replace(geo, sex)
    assert len(newcol) == 3
    assert newcol.names == ['lipro', 'sex', 'age']
    assert newcol.shape == (4, 2, 8)
    newcol = col2.replace(sex, 3)
    assert len(newcol) == 3
    assert newcol.names == ['lipro', None, 'age']
    assert newcol.shape == (4, 3, 8)
    newcol = col2.replace(sex, ['a', 'b', 'c'])
    assert len(newcol) == 3
    assert newcol.names == ['lipro', None, 'age']
    assert newcol.shape == (4, 3, 8)
    newcol = col2.replace(sex, "letters=a,b,c")
    assert len(newcol) == 3
    assert newcol.names == ['lipro', 'letters', 'age']
    assert newcol.shape == (4, 3, 8)


def test_contains(col):
    assert 'lipro' in col
    assert 'nonexisting' not in col
    assert 0 in col
    assert 1 in col
    assert 2 in col
    assert -1 in col
    assert -2 in col
    assert -3 in col
    assert 3 not in col
    assert lipro in col
    assert sex in col
    assert age in col
    assert sex2 in col
    assert geo not in col
    assert value not in col
    anon = Axis([0, 1])
    col.append(anon)
    assert anon in col
    anon2 = anon.copy()
    assert anon2 in col
    anon3 = Axis([0, 2])
    assert anon3 not in col


def test_index(col):
    assert col.index('lipro') == 0
    with must_raise(ValueError, msg="axis 'nonexisting' is not in collection"):
        col.index('nonexisting')
    assert col.index(0) == 0
    assert col.index(1) == 1
    assert col.index(2) == 2
    assert col.index(-1) == -1
    assert col.index(-2) == -2
    assert col.index(-3) == -3
    with must_raise(ValueError, msg='axis 3 is not in collection'):
        col.index(3)

    # objects actually in col
    assert col.index(lipro) == 0
    assert col.index(sex) == 1
    assert col.index(age) == 2
    assert col.index(sex2) == 1
    with must_raise(ValueError, msg="axis 'geo' is not in collection"):
        col.index(geo)
    with must_raise(ValueError, msg="axis 'value' is not in collection"):
        col.index(value)

    # test anonymous axes
    anon = Axis([0, 1])
    col.append(anon)
    assert col.index(anon) == 3
    anon2 = anon.copy()
    assert col.index(anon2) == 3
    anon3 = Axis([0, 2])
    with must_raise(ValueError, msg='Axis([0, 2], None) is not in collection'):
        col.index(anon3)


def test_get(col):
    assert_axis_eq(col.get('lipro'), lipro)
    assert col.get('nonexisting') is None
    assert col.get('nonexisting', value) is value


def test_keys(col):
    assert col.keys() == ['lipro', 'sex', 'age']


def test_getattr(col):
    assert_axis_eq(col.lipro, lipro)
    assert_axis_eq(col.sex, sex)
    assert_axis_eq(col.age, age)


def test_append(col):
    geo = Axis('geo=A11,A12,A13')
    col.append(geo)
    assert col == [lipro, sex, age, geo]


def test_extend(col):
    col.extend([geo, value])
    assert col == [lipro, sex, age, geo, value]


def test_insert(col):
    col.insert(1, geo)
    assert col == [lipro, geo, sex, age]


def test_add(col):
    col2 = col.copy()
    new = col2 + [geo, value]
    assert new == [lipro, sex, age, geo, value]
    assert col2 == col
    new = col2 + [Axis('geo=A11,A12,A13'), Axis('age=0..7')]
    assert new == [lipro, sex, age, geo]
    msg = """incompatible axes:
Axis([0, 1, 2, 3, 4, 5, 6], 'age')
vs
Axis([0, 1, 2, 3, 4, 5, 6, 7], 'age')"""
    with must_raise(ValueError, msg=msg):
        col2 + [Axis('geo=A11,A12,A13'), Axis('age=0..6')]

    # 2) other AxisCollection
    new = col2 + AxisCollection([geo, value])
    assert new == [lipro, sex, age, geo, value]


# def test_or():
#     col = AxisCollection('a0,a1')
#     res = col | Axis(2)
#     assert res == col
#
#     res = col | Axis(3)
#     assert res == AxisCollection([Axis('a0,a1'), Axis(3)])
#
#     res = col | AxisCollection([Axis(2), Axis(2)])
#     assert res == AxisCollection([Axis('a0,a1'), Axis(2)])


def test_sub():
    col = AxisCollection('a0,a1;b0,b1,b2')
    res = col - Axis(2)
    assert res == AxisCollection('b0,b1,b2')

    res = col - Axis(3)
    assert res == AxisCollection('a0,a1')

    col = AxisCollection('a0,a1;b0,b1')
    # when several axes are compatible, remove first
    res = col - Axis(2)
    assert res == AxisCollection('b0,b1')

    # when no axis is compatible, do not remove any
    res = col - Axis(3)
    assert res == col


def test_combine(col):
    res = col.combine_axes((lipro, sex))
    assert res.names == ['lipro_sex', 'age']
    assert res.size == col.size
    assert res.shape == (4 * 2, 8)
    print(res.info)
    assert_array_equal(res.lipro_sex.labels[0], 'P01_M')
    res = col.combine_axes((lipro, age))
    assert res.names == ['lipro_age', 'sex']
    assert res.size == col.size
    assert res.shape == (4 * 8, 2)
    assert_array_equal(res.lipro_age.labels[0], 'P01_0')
    res = col.combine_axes((sex, age))
    assert res.names == ['lipro', 'sex_age']
    assert res.size == col.size
    assert res.shape == (4, 2 * 8)
    assert_array_equal(res.sex_age.labels[0], 'M_0')


def test_info(col):
    expected = """\
4 x 2 x 8
 lipro [4]: 'P01' 'P02' 'P03' 'P04'
 sex [2]: 'M' 'F'
 age [8]: 0 1 2 ... 5 6 7"""
    assert col.info == expected


def test_str(col):
    assert str(col) == "{lipro, sex, age}"


def test_repr(col):
    assert repr(col) == """AxisCollection([
    Axis(['P01', 'P02', 'P03', 'P04'], 'lipro'),
    Axis(['M', 'F'], 'sex'),
    Axis([0, 1, 2, 3, 4, 5, 6, 7], 'age')
])"""


def test_setlabels():
    # test when the label is ambiguous AND the axes are anonymous
    axes = AxisCollection([Axis("b1,b2"), Axis("b0..b2")])
    with must_raise(ValueError, msg="""'b1' is ambiguous, it is valid in the following axes:
 {0} [2]: 'b1' 'b2'
 {1} [3]: 'b0' 'b1' 'b2'"""):
        axes.set_labels({'b1': 'b_one'})


if __name__ == "__main__":
    pytest.main()
