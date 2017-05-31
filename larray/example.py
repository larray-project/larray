import os
import larray as la

__all__ = ['EXAMPLE_FILES_DIR', 'load_example_data']

EXAMPLE_FILES_DIR = os.path.dirname(__file__) + '/tests/data/'
AVAILABLE_EXAMPLE_DATA = {
    'demography' : EXAMPLE_FILES_DIR + 'data.h5'
}

def load_example_data(name):
    """Load arrays used in the tutorial so that all examples in it can be reproduced.

    Parameters
    ----------
    example_data : str
        Example data to load. Available example datasets are:
        
        - demography

    Returns
    -------
    Session
        Session containing one or several arrays

    Examples
    --------
    >>> demo = load_example_data('demography')
    >>> demo.pop.info # doctest: +SKIP
    26 x 3 x 121 x 2 x 2
     time [26]: 1991 1992 1993 ... 2014 2015 2016
     geo [3]: 'BruCap' 'Fla' 'Wal'
     age [121]: 0 1 2 ... 118 119 120
     sex [2]: 'M' 'F'
     nat [2]: 'BE' 'FO'
    >>> demo.qx.info # doctest: +SKIP
    26 x 3 x 121 x 2 x 2
     time [26]: 1991 1992 1993 ... 2014 2015 2016
     geo [3]: 'BruCap' 'Fla' 'Wal'
     age [121]: 0 1 2 ... 118 119 120
     sex [2]: 'M' 'F'
     nat [2]: 'BE' 'FO'
    """
    if name is None:
        name = 'demography'
    if not isinstance(name, str):
        raise TypeError("Expected string for argument example_data")
    if name not in AVAILABLE_EXAMPLE_DATA.keys():
        raise ValueError("example_data must be chosen "
                         "from list {}".format(list(AVAILABLE_EXAMPLE_DATA.keys())))
    return la.Session(AVAILABLE_EXAMPLE_DATA[name])
