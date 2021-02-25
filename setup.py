from __future__ import print_function

import os
from setuptools import setup, find_packages


def readlocal(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DISTNAME = 'larray'
VERSION = '0.33-dev'
AUTHOR = 'Gaetan de Menten, Geert Bryon, Johan Duyck, Alix Damman'
AUTHOR_EMAIL = 'gdementen@gmail.com'
DESCRIPTION = "N-D labeled arrays in Python"
LONG_DESCRIPTION = readlocal("README.rst")
SETUP_REQUIRES = []
# - pandas >= 0.20.0 is required since commit 01669f2024a7bffe47cceec0a0fd845f71b6f7cc
#   (issue 702 : fixed bug when writing metadata using HDF format)
INSTALL_REQUIRES = ['numpy >= 1.13', 'pandas >= 0.20.0']
TESTS_REQUIRE = ['pytest', 'pytest-flake8']

LICENSE = 'GPLv3'
URL = 'https://github.com/larray-project/larray'

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
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
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    packages=find_packages(),
    include_package_data=True,
)
