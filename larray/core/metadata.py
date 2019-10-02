# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from larray.util.misc import PY2, basestring


if PY2:
    class AttributeDict(object):
        def __init__(self, *args, **kwargs):
            object.__setattr__(self, '__odict', OrderedDict(*args, **kwargs))

        def __getattr__(self, key):
            od = object.__getattribute__(self, '__odict')
            if hasattr(od, key):
                return getattr(od, key)
            else:
                try:
                    return od[key]
                except KeyError:
                    raise AttributeError(key)

        def __setattr__(self, key, value):
            od = object.__getattribute__(self, '__odict')
            od[key] = value

        def __delattr__(self, key):
            od = object.__getattribute__(self, '__odict')
            del od[key]

        def __reduce__(self):
            'Return state information for pickling'
            od = object.__getattribute__(self, '__odict')
            res = list(od.__reduce__())
            res[0] = self.__class__
            return tuple(res)

        def __dir__(self):
            od = object.__getattribute__(self, '__odict')
            return list(set(dir(self.__class__)) | set(self.__dict__.keys()) | set(od.keys()))

        def copy(self):
            od = object.__getattribute__(self, '__odict')
            return self.__class__(od)

        def method_factory(name):
            fullname = '__%s__' % name
            odict_method = getattr(OrderedDict, fullname)

            def method(self, *args, **kwargs):
                od = object.__getattribute__(self, '__odict')
                return odict_method(od, *args, **kwargs)
            return method

        __getitem__ = method_factory('getitem')
        __setitem__ = method_factory('setitem')
        __delitem__ = method_factory('delitem')
        __contains__ = method_factory('contains')

        __iter__ = method_factory('iter')
        __len__ = method_factory('len')

        __reversed__ = method_factory('reversed')

        __sizeof__ = method_factory('sizeof')

        def _binop(name):
            fullname = '__%s__' % name
            odict_method = getattr(OrderedDict, fullname)

            def opmethod(self, other):
                self_od = object.__getattribute__(self, '__odict')
                if not isinstance(other, AttributeDict):
                    return False
                other_od = object.__getattribute__(other, '__odict')
                return odict_method(self_od, other_od)
            opmethod.__name__ = fullname
            return opmethod

        __eq__ = _binop('eq')
        __ne__ = _binop('ne')
        __ge__ = _binop('ge')
        __gt__ = _binop('gt')
        __le__ = _binop('le')
        __lt__ = _binop('lt')

        def __repr__(self):
            return '\n'.join(['{}: {}'.format(k, v) for k, v in self.items()])

else:
    class AttributeDict(OrderedDict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            del self[key]

        def __dir__(self):
            return list(set(super(AttributeDict, self).__dir__()) | set(self.keys()))

        def __repr__(self):
            return '\n'.join(['{}: {}'.format(k, v) for k, v in self.items()])


class Metadata(AttributeDict):
    r"""
    An ordered dictionary allowing key-values accessibly using attribute notation (AttributeDict.attribute)
    instead of key notation (Dict["key"]).

    Examples
    --------
    >>> from larray import ndtest
    >>> from datetime import datetime

    Add metadata at array initialization

    >>> # Python 2 or <= 3.5
    >>> arr = ndtest((3, 3), meta=[('title', 'the title'), ('author', 'John Smith')])
    >>> # Python 3.6+
    >>> arr = ndtest((3, 3), meta=Metadata(title='the title', author='John Smith'))  # doctest: +SKIP

    Add metadata after array initialization

    >>> arr.meta.creation_date = datetime(2017, 2, 10)

    Access to metadata

    >>> arr.meta.creation_date
    datetime.datetime(2017, 2, 10, 0, 0)

    Modify metadata

    >>> arr.meta.creation_date = datetime(2017, 2, 16)

    Delete metadata

    >>> del arr.meta.creation_date
    """
    def __larray__(self):
        from larray.core.array import stack
        return stack(self.items(), axes='metadata')

    @classmethod
    def from_array(cls, array):
        from larray.core.array import aslarray
        array = aslarray(array)
        if array.ndim != 1:
            raise ValueError("Expected Array object of dimension 1. Got array of dimension {}".format(array.ndim))

        from pandas import to_numeric, to_datetime

        def _convert_value(value):
            value = to_numeric([value], errors='ignore')[0]
            if isinstance(value, basestring):
                value = to_datetime(value, errors='ignore', infer_datetime_format=True)
            return value

        return Metadata([(key, _convert_value(value)) for key, value in zip(array.axes.labels[0], array.data)])

    # ---------- IO methods ----------
    def to_hdf(self, hdfstore, key=None):
        if len(self):
            attrs = hdfstore.get_storer(key).attrs if key is not None else hdfstore.root._v_attrs
            attrs.metadata = self

    @classmethod
    def from_hdf(cls, hdfstore, key=None):
        attrs = hdfstore.get_storer(key).attrs if key is not None else hdfstore.root._v_attrs
        if 'metadata' in attrs:
            return attrs.metadata
        else:
            return None
