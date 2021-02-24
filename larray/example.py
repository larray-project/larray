import os
import larray as la


_TEST_DIR = os.path.join(os.path.dirname(__file__), 'tests')

EXAMPLE_FILES_DIR = os.path.join(_TEST_DIR, 'data')
AVAILABLE_EXAMPLE_DATA = {
    'demography': os.path.join(EXAMPLE_FILES_DIR, 'demography.h5'),
    'demography_eurostat': os.path.join(EXAMPLE_FILES_DIR, 'demography_eurostat.h5')
}
EXAMPLE_EXCEL_TEMPLATES_DIR = os.path.join(_TEST_DIR, 'excel_template')


def get_example_filepath(fname):
    r"""Return absolute path to an example file if exist.

    Parameters
    ----------
    fname : str
        Filename of an existing example file.

    Returns
    -------
    Filepath
        Absolute filepath to an example file if exists.

    Notes
    -----
    A ValueError is raised if the provided filename does not represent an existing example file.

    Examples
    --------
    >>> fpath = get_example_filepath('examples.xlsx')
    """
    fpath = os.path.abspath(os.path.join(EXAMPLE_FILES_DIR, fname))
    if not os.path.exists(fpath):
        AVAILABLE_EXAMPLE_FILES = os.listdir(EXAMPLE_FILES_DIR)
        raise ValueError(f"Example file {fname} does not exist. "
                         f"Available example files are: {AVAILABLE_EXAMPLE_FILES}")
    return fpath


# Note that we skip doctests because they require pytables, which is only an optional dependency and its hard
# to skip doctests selectively.
# CHECK: We might want to use .csv files for the example data, so that it can be loaded with any optional dependency.
def load_example_data(name):
    r"""Load arrays used in the tutorial so that all examples in it can be reproduced.

    Parameters
    ----------
    name : str
        Example data to load. Available example datasets are:

        - demography
        - demography_eurostat

    Returns
    -------
    Session
        Session containing one or several arrays.

    Examples
    --------
    >>> demo = load_example_data('demography')           # doctest: +SKIP
    >>> print(demo.summary())                            # doctest: +SKIP
    hh: time, geo, hh_type (26 x 3 x 7) [int64]
    pop: time, geo, age, sex, nat (26 x 3 x 121 x 2 x 2) [int64]
    qx: time, geo, age, sex, nat (26 x 3 x 121 x 2 x 2) [float64]
    >>> demo = load_example_data('demography_eurostat')  # doctest: +SKIP
    >>> print(demo.summary())                            # doctest: +SKIP
    Metadata:
       title: Demographic datasets for a small selection of countries in Europe
       source: demo_jpan, demo_fasec, demo_magec and migr_imm1ctz tables from Eurostat
    gender: gender ['Male' 'Female'] (2)
    country: country ['Belgium' 'France' 'Germany'] (3)
    country_benelux: country_benelux ['Belgium' 'Luxembourg' 'Netherlands'] (3)
    citizenship: citizenship ['Belgium' 'Luxembourg' 'Netherlands'] (3)
    time: time [2013 2014 2015 2016 2017] (5)
    even_years: time[2014 2016] >> even_years (2)
    odd_years: time[2013 2015 2017] >> odd_years (3)
    births: country, gender, time (3 x 2 x 5) [int32]
    deaths: country, gender, time (3 x 2 x 5) [int32]
    immigration: country, citizenship, gender, time (3 x 3 x 2 x 5) [int32]
    pop: country, gender, time (3 x 2 x 5) [int32]
    pop_benelux: country, gender, time (3 x 2 x 5) [int32]
    """
    if name is None:
        name = 'demography'
    if not isinstance(name, str):
        raise TypeError("Expected string for argument example_data")
    if name not in AVAILABLE_EXAMPLE_DATA:
        available_datasets = list(AVAILABLE_EXAMPLE_DATA.keys())
        raise ValueError(f"example_data must be chosen from list {available_datasets}")
    return la.Session(AVAILABLE_EXAMPLE_DATA[name])
