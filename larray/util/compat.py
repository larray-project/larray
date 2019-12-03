import sys

try:
    # the abstract base classes were moved to the abc sub-module in Python 3.3 but there is a backward compatibility
    # layer for Python up to 3.7
    from collections.abc import Iterable, Sequence
except ImportError:
    # needed for Python < 3.3 (including 2.7)
    from collections import Iterable, Sequence

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

try:
    from itertools import izip
except ImportError:
    izip = zip

if sys.version_info[0] < 3:
    basestring = basestring
    bytes = str
    unicode = unicode
    long = long
    PY2 = True
else:
    basestring = str
    bytes = bytes
    unicode = str
    long = int
    PY2 = False

if PY2:
    from StringIO import StringIO
else:
    from io import StringIO

if PY2:
    import cPickle as pickle
else:
    import pickle


def csv_open(filename, mode='r'):
    assert 'b' not in mode and 't' not in mode
    if PY2:
        return open(filename, mode + 'b')
    else:
        return open(filename, mode, newline='', encoding='utf8')


def decode(s, encoding='utf-8', errors='strict'):
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    else:
        assert s is None or isinstance(s, unicode), "unexpected " + str(type(s))
        return s
