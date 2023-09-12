import warnings

import numpy as np
import pandas as pd
from pandas import HDFStore

from typing import Union, Dict

from larray.core.array import Array
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.group import Group, LGroup, _translate_group_key_hdf
from larray.core.metadata import Metadata
from larray.util.misc import LHDFStore
from larray.util.types import Scalar
from larray.inout.session import register_file_handler
from larray.inout.common import FileHandler, _supported_typenames, _supported_scalars_types
from larray.inout.pandas import df_asarray
from larray.example import get_example_filepath         # noqa: F401


# for backward compatibility (larray < 0.29) but any object read from an hdf file should have
# an attribute 'type'
def _get_type_from_attrs(attrs):
    return attrs.type if 'type' in attrs else 'Array'


def read_hdf(filepath_or_buffer, key, fill_value=nan, na=nan, sort_rows=False, sort_columns=False,
             name=None, **kwargs) -> Array:
    r"""Read a scalar or an axis or group or array named key from a HDF5 file in filepath (path+name).

    Parameters
    ----------
    filepath_or_buffer : str or Path or pandas.HDFStore
        Path and name where the HDF5 file is stored or a HDFStore object.
    key : str or Group
        Name of the scalar or axis or group or array.
    fill_value : scalar or Array, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether to sort the rows alphabetically.
        Must be False if the read array has been dumped with an larray version >= 0.30.
        Defaults to False.
    sort_columns : bool, optional
        Whether to sort the columns alphabetically.
        Must be False if the read array has been dumped with an larray version >= 0.30.
        Defaults to False.
    name : str, optional
        Name of the axis or group to return. If None, name is set to passed key.
        Defaults to None.

    Returns
    -------
    Array

    Examples
    --------
    >>> fname = get_example_filepath('examples.h5')

    Read array by passing its identifier (key) inside the HDF file

    >>> # The data below is derived from a subset of the demo_pjan table from Eurostat
    >>> read_hdf(fname, 'pop')                     # doctest: +SKIP
    country  gender\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male  31772665  32045129  32174258
     France       Female  33827685  34120851  34283895
    Germany         Male  39380976  39556923  39835457
    Germany       Female  41142770  41210540  41362080
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_hdf `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    key = _translate_group_key_hdf(key)
    res = None
    with LHDFStore(filepath_or_buffer, mode='r') as store:
        try:
            pd_obj = store.get(key)
        except KeyError:
            filepath = filepath_or_buffer if isinstance(filepath_or_buffer, HDFStore) else store.filename
            raise KeyError(f'No item with name {key} has been found in file {filepath}')
        attrs = store.get_storer(key).attrs
        writer = attrs.writer if 'writer' in attrs else None
        _type = _get_type_from_attrs(attrs)
        _meta = attrs.metadata if 'metadata' in attrs else None
        if _type == 'Array':
            # cartesian product is not necessary if the array was written by LArray
            cartesian_prod = writer != 'LArray'
            res = df_asarray(pd_obj, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value,
                             parse_header=False, cartesian_prod=cartesian_prod)
            if _meta is not None:
                res.meta = _meta
        elif _type == 'Axis':
            if name is None:
                name = str(pd_obj.name)
            if name == 'None':
                name = None
            labels = pd_obj.values
            if 'dtype_kind' in attrs and attrs['dtype_kind'] == 'U':
                # this check is there because there are cases where dtype_kind is 'U' but pandas returns
                # an array with object dtype containing bytes instead of a string array, and in that case
                # np.char.decode does not work
                # this is at least the case for Python2 + Pandas 0.24.2 combination
                if labels.dtype.kind == 'O':
                    labels = np.array([label.decode('utf-8') for label in labels], dtype='U')
                else:
                    labels = np.char.decode(labels, 'utf-8')
            res = Axis(labels=labels, name=name)
            res._iswildcard = attrs['wildcard']
        elif _type == 'Group':
            if name is None:
                name = str(pd_obj.name)
            if name == 'None':
                name = None
            key = pd_obj.values
            if 'dtype_kind' in attrs and attrs['dtype_kind'] == 'U':
                key = np.char.decode(key, 'utf-8')
            axis = read_hdf(filepath_or_buffer, attrs['axis_key'])
            res = LGroup(key=key, name=name, axis=axis)
        elif _type in _supported_typenames:
            res = pd_obj.values
            assert len(res) == 1
            res = res[0]
    return res


@register_file_handler('pandas_hdf', ['h5', 'hdf'])
class PandasHDFHandler(FileHandler):
    r"""
    Handler for HDF5 files using Pandas.
    """

    def _open_for_read(self):
        self.handle = HDFStore(self.fname, mode='r')

    def _open_for_write(self):
        self.handle = HDFStore(self.fname)

    def item_types(self) -> Dict[str, str]:
        handle = self.handle
        keys = [key.strip('/') for key in handle.keys()]
        types = {key: _get_type_from_attrs(handle.get_storer(key).attrs) for key in keys if '/' not in key}
        # ---- for backward compatibility (LArray < 0.33) ----
        # axes
        types.update({key.split('/')[-1]: 'Axis_Backward_Comp' for key in keys if '__axes__' in key})
        # groups
        types.update({key.split('/')[-1]: 'Group_Backward_Comp' for key in keys if '__groups__' in key})
        return types

    def _read_item(self, key, typename, *args, **kwargs) -> Union[Array, Axis, Group, Scalar]:
        if typename in _supported_typenames:
            hdf_key = '/' + key
        # ---- for backward compatibility (LArray < 0.33) ----
        elif typename == 'Axis_Backward_Comp':
            hdf_key = '__axes__/' + key
        elif typename == 'Group_Backward_Comp':
            hdf_key = '__groups__/' + key
        else:
            raise TypeError()
        return read_hdf(self.handle, hdf_key, *args, **kwargs)

    def _dump_item(self, key, value, *args, **kwargs):
        hdf_key = '/' + key
        if isinstance(value, (Array, Axis)):
            value.to_hdf(self.handle, hdf_key, *args, **kwargs)
        elif isinstance(value, Group):
            hdf_axis_key = '/' + value.axis.name
            value.to_hdf(self.handle, hdf_key, hdf_axis_key, *args, **kwargs)
        elif isinstance(value, _supported_scalars_types):
            s = pd.Series(data=value)
            self.handle.put(hdf_key, s)
            self.handle.get_storer(hdf_key).attrs.type = type(value).__name__
        else:
            raise TypeError()

    def _read_metadata(self) -> Metadata:
        metadata = Metadata.from_hdf(self.handle)
        if metadata is None:
            metadata = Metadata()
        return metadata

    def _dump_metadata(self, metadata):
        metadata.to_hdf(self.handle)

    def close(self):
        self.handle.close()
