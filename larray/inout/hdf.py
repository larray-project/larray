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

    Examples
    --------
    >>> import os
    >>> from larray import EXAMPLE_FILES_DIR
    >>> fname = os.path.join(EXAMPLE_FILES_DIR, 'test.h5')

    Read array by passing its identifier (key) inside the HDF file

    >>> read_hdf(fname, '3d')
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   6   7   8
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1  15  16  17

    Missing label combinations

    >>> # by default, cells associated with missing label combinations are filled with NaN.
    >>> # In that case, an int array is converted to a float array.
    >>> read_hdf(fname, 'missing_values')
    a  b\c    c0    c1    c2
    1   b0   0.0   1.0   2.0
    1   b1   3.0   4.0   5.0
    2   b0   nan   nan   nan
    2   b1   9.0  10.0  11.0
    3   b0  12.0  13.0  14.0
    3   b1   nan   nan   nan
    >>> # using argument 'fill_value', you can choose which value to use to fill missing cells.
    >>> read_hdf(fname, 'missing_values', fill_value=0)
    a  b\c  c0  c1  c2
    1   b0   0   1   2
    1   b1   3   4   5
    2   b0   0   0   0
    2   b1   9  10  11
    3   b0  12  13  14
    3   b1   0   0   0
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
            res = df_aslarray(pd_obj, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value,
                              parse_header=False)
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