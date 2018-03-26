from __future__ import absolute_import, print_function

import warnings

import numpy as np
from pandas import HDFStore

from larray.core.axis import Axis
from larray.core.group import Group, LGroup, _translate_group_key_hdf
from larray.core.array import LArray
from larray.util.misc import LHDFStore
from larray.inout.pandas import df_aslarray
from larray.inout.common import FileHandler


__all__ = ['read_hdf']


# TODO : add examples
def read_hdf(filepath_or_buffer, key, fill_value=np.nan, na=np.nan, sort_rows=False, sort_columns=False,
             name=None, **kwargs):
    """Reads an array named key from a HDF5 file in filepath (path+name)

    Parameters
    ----------
    filepath_or_buffer : str or pandas.HDFStore
        Path and name where the HDF5 file is stored or a HDFStore object.
    key : str or Group
        Name of the array.
    fill_value : scalar or LArray, optional
        Value used to fill cells corresponding to label combinations which are not present in the input.
        Defaults to NaN.
    sort_rows : bool, optional
        Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting). Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
        Defaults to False.
    name : str, optional
        Name of the axis or group to return. If None, name is set to passed key.
        Defaults to None.

    Returns
    -------
    LArray
    """
    if not np.isnan(na):
        fill_value = na
        warnings.warn("read_hdf `na` argument has been renamed to `fill_value`. Please use that instead.",
                      FutureWarning, stacklevel=2)

    key = _translate_group_key_hdf(key)
    res = None
    with LHDFStore(filepath_or_buffer) as store:
        pd_obj = store.get(key)
        attrs = store.get_storer(key).attrs
        # for backward compatibility but any object read from an hdf file should have an attribute 'type'
        _type = attrs.type if 'type' in dir(attrs) else 'Array'
        if _type == 'Array':
            res = df_aslarray(pd_obj, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value, parse_header=False)
        elif _type == 'Axis':
            if name is None:
                name = str(pd_obj.name)
            if name == 'None':
                name = None
            res = Axis(labels=pd_obj.values, name=name)
            res._iswildcard = attrs['wildcard']
        elif _type == 'Group':
            if name is None:
                name = str(pd_obj.name)
            if name == 'None':
                name = None
            axis = read_hdf(filepath_or_buffer, attrs['axis_key'])
            res = LGroup(key=pd_obj.values, name=name, axis=axis)
    return res


class PandasHDFHandler(FileHandler):
    """
    Handler for HDF5 files using Pandas.
    """
    def _open_for_read(self):
        self.handle = HDFStore(self.fname, mode='r')

    def _open_for_write(self):
        self.handle = HDFStore(self.fname)

    def list(self):
        return [key.strip('/') for key in self.handle.keys()]

    def _read_item(self, key, *args, **kwargs):
        if '__axes__' in key:
            session_key = key.split('/')[-1]
            kwargs['name'] = session_key
        elif '__groups__' in key:
            session_key = key.split('/')[-1]
            kwargs['name'] = session_key
        else:
            session_key = key
        key = '/' + key
        return session_key, read_hdf(self.handle, key, *args, **kwargs)

    def _dump(self, key, value, *args, **kwargs):
        if isinstance(value, Axis):
            key = '__axes__/' + key
        elif isinstance(value, Group):
            key = '__groups__/' + key
            # axis_key (see Group.to_hdf)
            args = ('__axes__/' + value.axis.name,) + args
        else:
            key = '/' + key
        value.to_hdf(self.handle, key, *args, **kwargs)

    def close(self):
        self.handle.close()