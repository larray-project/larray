from typing import List, Optional


class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __dir__(self) -> List[str]:
        return list(set(super().__dir__()) | set(self.keys()))

    def __repr__(self) -> str:
        return '\n'.join([f'{k}: {v}' for k, v in self.items()])


class Metadata(AttributeDict):
    r"""
    An ordered dictionary allowing key-values accessibly using attribute notation (AttributeDict.attribute)
    instead of key notation (Dict["key"]).

    Examples
    --------
    >>> from larray import ndtest
    >>> from datetime import datetime

    Add metadata at array initialization

    >>> arr = ndtest((3, 3), meta=Metadata(title='the title', author='John Smith'))

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
    def from_array(cls, array) -> 'Metadata':
        from larray.core.array import asarray
        array = asarray(array)
        if array.ndim != 1:
            raise ValueError(f"Expected Array object of dimension 1. Got array of dimension {array.ndim}")

        from pandas import to_numeric, to_datetime

        def _convert_value(value):
            # errors='ignore' => the value is unmodified if it does not parse
            value = to_numeric([value], errors='ignore')[0]
            if isinstance(value, str):
                # same here
                value = to_datetime(value, errors='ignore')
            return value

        return Metadata({key: _convert_value(value) for key, value in zip(array.axes.labels[0], array.data)})

    # ---------- IO methods ----------
    def to_hdf(self, hdfstore, key=None):
        if len(self):
            attrs = hdfstore.get_storer(key).attrs if key is not None else hdfstore.root._v_attrs
            attrs.metadata = self

    @classmethod
    def from_hdf(cls, hdfstore, key=None) -> Optional['Metadata']:
        attrs = hdfstore.get_storer(key).attrs if key is not None else hdfstore.root._v_attrs
        if 'metadata' in attrs:
            return attrs.metadata
        else:
            return None
