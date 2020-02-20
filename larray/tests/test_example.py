import pytest

from larray.tests.common import needs_pytables
from larray import load_example_data, Array, Session, Axis, Group


def obj_summary(obj):
    if isinstance(obj, Session):
        return {obj_name: obj_summary(child_obj) for obj_name, child_obj in obj.items()}
    elif isinstance(obj, Array):
        return [obj_summary(axis) for axis in obj.axes]
    elif isinstance(obj, Axis):
        return obj.name, len(obj)
    elif isinstance(obj, Group):
        return obj.name, obj.axis.name, len(obj)
    else:
        raise TypeError('unsupported object')


@needs_pytables
def test_load_example_data():
    demo = load_example_data('demography')
    expected_summary = {
        'hh': [('time', 26), ('geo', 3), ('hh_type', 7)],
        'pop': [('time', 26), ('geo', 3), ('age', 121), ('sex', 2), ('nat', 2)],
        'qx': [('time', 26), ('geo', 3), ('age', 121), ('sex', 2), ('nat', 2)]
    }
    assert obj_summary(demo) == expected_summary

    demo = load_example_data('demography_eurostat')
    expected_summary = {
        'births': [('country', 3), ('gender', 2), ('time', 5)],
        'citizenship': ('citizenship', 3),
        'country': ('country', 3),
        'country_benelux': ('country', 3),
        'deaths': [('country', 3), ('gender', 2), ('time', 5)],
        'even_years': ('even_years', 'time', 2),
        'gender': ('gender', 2),
        'immigration': [('country', 3),
                        ('citizenship', 3),
                        ('gender', 2),
                        ('time', 5)],
        'odd_years': ('odd_years', 'time', 3),
        'population': [('country', 3), ('gender', 2), ('time', 5)],
        'population_5_countries': [('country', 5), ('gender', 2), ('time', 5)],
        'population_benelux': [('country', 3), ('gender', 2), ('time', 5)],
        'time': ('time', 5)
    }
    assert obj_summary(demo) == expected_summary


if __name__ == "__main__":
    pytest.main()
