from __future__ import print_function

import os
from setuptools import setup, find_packages


def readlocal(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DISTNAME = 'larray'
VERSION = '0.19.1'
AUTHOR = 'Gaetan de Menten, Geert Bryon, Johan Duyck, Alix Damman'
AUTHOR_EMAIL = 'gdementen@gmail.com'
DESCRIPTION = "N-D labeled arrays in Python"
LONG_DESCRIPTION = readlocal("README.rst")
INSTALL_REQUIRES = ['numpy >= 1.10', 'pandas >= 0.13.1']
TESTS_REQUIRE = ['nose >= 1.0']
TEST_SUITE = 'nose.collector'

LICENSE = 'GPLv3'
PACKAGE_DATA = {'larray': ['tests/data/*']}
URL = 'https://github.com/liam2/larray'

CLASSIFIERS = [
    'Development Status :: 4 - Beta'
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]

setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    test_suite=TEST_SUITE,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
)
