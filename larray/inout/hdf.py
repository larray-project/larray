from __future__ import absolute_import, print_function

import warnings

import numpy as np
from pandas import HDFStore

from larray.core.array import LArray
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.group import Group, LGroup, _translate_group_key_hdf
from larray.core.metadata import Metadata
from larray.util.misc import LHDFStore
from larray.inout.session import register_file_handler
from larray.inout.common import FileHandler
from larray.inout.pandas import df_aslarray
from larray.example import get_example_filepath


def read_hdf(filepath_or_buffer, key, fill_value=nan, na=nan, sort_rows=False, sort_columns=False,
             name=None, **kwargs):
    """Reads an axis or group or array named key from a HDF5 file in filepath (path+name)

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
        Whether or not to sort the rows alphabetically.
        Must be False if the read array has been dumped with an larray version >= 0.30.
        Defaults to False.
    sort_columns : bool, optional
        Whether or not to sort the columns alphabetically.
        Must be False if the read array has been dumped with an larray version >= 0.30.
        Defaults to False.
    name : str, optional
        Name of the axis or group to return. If None, name is set to passed key.
        Defaults to None.

    Returns
    -------
    LArray

    Examples
    --------
    >>> fname = get_example_filepath('examples.h5')

    Read array by passing its identifier (key) inside the HDF file

    >>> # The data below is derived from a subset of the demo_pjan table from Eurostat
    >>> read_hdf(fname, 'pop')
    country  gender\\time      2013      2014      2015
    Belgium         Male   5472856   5493792   5524068
    Belgium       Female   5665118   5687048   5713206
     France         Male  31772665  31936596  32175328
     France       Female  33827685  34005671  34280951
    Germany         Male  39380976  39556923  39835457
    Germany       Female  41142770  41210540  41362080
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
        writer = attrs.writer if 'writer' in attrs else None
        # for backward compatibility but any object read from an hdf file should have an attribute 'type'
        _type = attrs.type if 'type' in attrs else 'Array'
        _meta = attrs.metadata if 'metadata' in attrs else None
        if _type == 'Array':
            # cartesian product is not necessary if the array was written by LArray
            cartesian_prod = writer != 'LArray'
            res = df_aslarray(pd_obj, sort_rows=sort_rows, sort_columns=sort_columns, fill_value=fill_value,
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
    return res


@register_file_handler('pandas_hdf', ['h5', 'hdf'])
class PandasHDFHandler(FileHandler):
    """
    Handler for HDF5 files using Pandas.
    """
    def _open_for_read(self):
        self.handle = HDFStore(self.fname, mode='r')

    def _open_for_write(self):
        self.handle = HDFStore(self.fname)

    def list_items(self):
        keys = [key.strip('/') for key in self.handle.keys()]
        # axes
        items = [(key.split('/')[-1], 'Axis') for key in keys if '__axes__' in key]
        # groups
        items += [(key.split('/')[-1], 'Group') for key in keys if '__groups__' in key]
        # arrays
        items += [(key, 'Array') for key in keys if '/' not in key]
        return items

    def _read_item(self, key, type, *args, **kwargs):
        if type == 'Array':
            hdf_key = '/' + key
        elif type == 'Axis':
            hdf_key = '__axes__/' + key
            kwargs['name'] = key
        elif type == 'Group':
            hdf_key = '__groups__/' + key
            kwargs['name'] = key
        else:
            raise TypeError()
        return read_hdf(self.handle, hdf_key, *args, **kwargs)

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, LArray):
            hdf_key = '/' + key
            value.to_hdf(self.handle, hdf_key, *args, **kwargs)
        elif isinstance(value, Axis):
            hdf_key = '__axes__/' + key
            value.to_hdf(self.handle, hdf_key, *args, **kwargs)
        elif isinstance(value, Group):
            hdf_key = '__groups__/' + key
            hdf_axis_key = '__axes__/' + value.axis.name
            value.to_hdf(self.handle, hdf_key, hdf_axis_key, *args, **kwargs)
        else:
            raise TypeError()

    def _read_metadata(self):
        metadata = Metadata.from_hdf(self.handle)
        if metadata is None:
            metadata = Metadata()
        return metadata

    def _dump_metadata(self, metadata):
        metadata.to_hdf(self.handle)

    def close(self):
        self.handle.close()
