from __future__ import absolute_import, print_function

import os
import warnings

import numpy as np
from pandas import Series, HDFStore

from larray.core.array import LArray
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.group import Group, LGroup, _translate_group_key_hdf
from larray.core.metadata import Metadata
from larray.example import get_example_filepath
from larray.inout.common import FileHandler
from larray.inout.pandas import df_aslarray
from larray.inout.session import register_file_handler


class ClosedFileError(Exception):
    pass


class AbstractStorer(object):
    def __init__(self, filepath, mode=None, complevel=None, complib=None, fletcher32=False, **kwargs):
        pandas_hdfstore = HDFStore(filepath, mode, complevel, complib, fletcher32, **kwargs)
        self._pandas_hdfstore = pandas_hdfstore
        self._path = pandas_hdfstore._path
        self._mode = pandas_hdfstore._mode
        self._handle = pandas_hdfstore._handle
        self._complevel = pandas_hdfstore._complevel
        self._complib = pandas_hdfstore._complib
        self._fletcher32 = pandas_hdfstore._fletcher32
        self._filters = pandas_hdfstore._filters

    @property
    def root(self):
        """ return the root node """
        return self._pandas_hdfstore.root

    @property
    def attrs(self):
        return self.root._v_attrs

    @property
    def is_open(self):
        """
        return a boolean indicating whether the file is open
        """
        return self._pandas_hdfstore.is_open

    def __contains__(self, key):
        return key in self._pandas_hdfstore

    def __len__(self):
        return len(self._pandas_hdfstore)

    def get_node(self, key):
        return self._pandas_hdfstore.get_node(key)

    def close(self):
        self._pandas_hdfstore.close()

    def remove(self, key):
        """
        Remove LArray object.

        Parameters
        ----------
        key : str
            Key associated to the object to be removed.
        """
        s = self._pandas_hdfstore.get_storer(key)
        s.group._f_remove(recursive=True)

    def groups(self):
        """
        return a list of all groups containing an LArray object.
        """
        raise NotImplementedError()

    def _check_if_open(self):
        if not self.is_open:
            raise ClosedFileError("{} file is not open!".format(self._path))

    def _get(self, key, **kwargs):
        raise NotImplementedError()

    def get(self, key, **kwargs):
        key = _translate_group_key_hdf(key)
        return self._get(key, **kwargs)

    def _put(self, key, value, **kwargs):
        raise NotImplementedError()

    def put(self, key, value, **kwargs):
        key = _translate_group_key_hdf(key)
        self._put(key, value, **kwargs)


class PytablesStorer(AbstractStorer):
    """
    Read and write LArray objects into HDF5 file using pytables.
    """
    def __init__(self, filepath, mode=None, complevel=None, complib=None, fletcher32=False, **kwargs):
        AbstractStorer.__init__(self, filepath, mode, complevel, complib, fletcher32, **kwargs)

    def groups(self):
        import tables
        self._check_if_open()
        return [g for g in self._handle.walk_groups()
                if (not isinstance(g, tables.link.Link) and 'type' in g._v_attrs)]

    def _read_data(self, group, name, attrs):
        dtype = np.dtype(attrs['dtype'])
        data = group[name].read()
        if dtype.kind == 'U':
            data = np.char.decode(data, 'utf-8')
        if dtype.kind == 'O':
            data = data[0]
            data = data.astype(dtype)
        return data

    def _read_group(self, group):
        def _get_name(attrs):
            name = attrs['name']
            return name if name is None else str(name)

        attrs = group._v_attrs
        _type = attrs.type if 'type' in attrs else 'Array'
        _meta = attrs.metadata if 'metadata' in attrs else None
        res = None
        if _type == 'Array':
            axes_keys = [n._v_pathname for n in group if n._v_name.startswith('axis')]
            axes = [self._get(axis_key) for axis_key in axes_keys]
            data = self._read_data(group, 'data', attrs)
            res = LArray(data=data, axes=axes)
            if _meta is not None:
                res.meta = _meta
        elif _type == 'Axis':
            name = _get_name(attrs)
            labels = self._read_data(group, 'labels', attrs)
            res = Axis(labels=labels, name=name)
            res._iswildcard = attrs['wildcard']
        elif _type == 'Group':
            axis = self._get(attrs['axis_key'])
            name = _get_name(attrs)
            key = self._read_data(group, 'key', attrs)
            res = LGroup(key=key, name=name, axis=axis)
        return res

    def _get(self, key, **kwargs):
        group = self.get_node(key)
        if group is None:
            raise KeyError('No object named {} in the file'.format(key))
        return self._read_group(group)

    def _dump_data(self, group, name, data, attrs):
        import tables
        data = np.asarray(data)
        dtype = data.dtype
        attrs['dtype'] = dtype
        # https://www.pytables.org/MIGRATING_TO_3.x.html#unicode-all-the-strings
        # Warning: In Python 3, all strings are natively in Unicode.
        #          This introduces some difficulties, as the native HDF5 string format is not Unicode-compatible.
        #          To minimize explicit conversion troubles when writing, especially when creating data sets
        #          from existing Python objects, string objects are implicitly cast to non-Unicode byte strings
        #          for HDF5 storage by default.
        #          To avoid such problem, one way is to use the VLArray class and dump unicode string arrays
        #          as object arrays.
        if dtype.kind == 'O':
            vlarr = self._handle.create_vlarray(group, name=name, filters=self._filters, atom=tables.ObjectAtom())
            vlarr.append(data)
        else:
            if dtype.kind == 'U':
                data = np.char.encode(data, 'utf-8')
            self._handle.create_carray(group, name=name, obj=data, filters=self._filters)

    def _write_obj(self, group, value, **kwargs):
        if isinstance(value, LArray):
            attrs = group._v_attrs
            attrs['type'] = 'Array'
            # dump axes
            for axis in value.axes:
                axis_key = 'axis_{}'.format(value.axes.axis_id(axis))
                axis_group = self._handle.create_group(group, axis_key)
                self._write_obj(axis_group, axis)
            # dump data
            self._dump_data(group, name='data', data=value.data, attrs=attrs)
            # dump metadata
            self._write_obj(group, value.meta)
        elif isinstance(value, Axis):
            attrs = group._v_attrs
            attrs['type'] = 'Axis'
            attrs['name'] = value.name
            attrs['wildcard'] = value.iswildcard
            self._dump_data(group, name='labels', data=value.labels, attrs=attrs)
        elif isinstance(value, Group):
            axis_key = kwargs.pop('axis_key', None)
            if axis_key is None:
                if value.axis.name is None:
                    raise ValueError(
                        "Argument axis_key must be provided explicitly if the associated axis is anonymous")
                axis_key = value.axis.name
            if self.get_node(axis_key) is None:
                self._put(axis_key, value.axis)
            attrs = group._v_attrs
            attrs['type'] = 'Group'
            attrs['name'] = value.name
            attrs['axis_key'] = axis_key
            self._dump_data(group, name='key', data=value.eval(), attrs=attrs)
        elif isinstance(value, Metadata):
            if len(value):
                group._v_attrs['metadata'] = value
        else:
            warnings.warn('{}: Type {} is currently not supported'.format(group._v_name, type(value)))

    def _write_group(self, key, value, **kwargs):
        # remove the group if exists already
        group = self.get_node(key)
        if group is not None:
            self._handle.remove_node(group, recursive=True)
        paths = key.split('/')
        # recursively create the parent groups
        path = '/'
        for p in paths:
            if not len(p):
                continue
            new_path = path
            if not path.endswith('/'):
                new_path += '/'
            new_path += p
            group = self.get_node(new_path)
            if group is None:
                group = self._handle.create_group(path, p)
            path = new_path
        self._write_obj(group, value, **kwargs)

    def _put(self, key, value, **kwargs):
        key = _translate_group_key_hdf(key)
        self._write_group(key, value, **kwargs)


class PandasStorer(AbstractStorer):
    """
    Read and write LArray objects into HDF5 file using pandas.
    """
    def __init__(self, filepath, mode=None, complevel=None, complib=None, fletcher32=False, **kwargs):
        AbstractStorer.__init__(self, filepath, mode, complevel, complib, fletcher32, **kwargs)

    def groups(self):
        return self._pandas_hdfstore.groups()

    def _get(self, key, **kwargs):
        name = kwargs.pop('name', None)
        pd_obj = self._pandas_hdfstore.get(key)
        attrs = self._pandas_hdfstore.get_storer(key).attrs
        _writer = attrs.writer if 'writer' in attrs else None
        # for backward compatibility but any object read from an hdf file should have an attribute 'type'
        _type = attrs.type if 'type' in attrs else 'Array'
        _meta = attrs.metadata if 'metadata' in attrs else None
        res = None
        if _type == 'Array':
            sort_rows = kwargs.pop('sort_rows', False)
            sort_columns = kwargs.pop('sort_columns', False)
            fill_value = kwargs.pop('fill_value', nan)
            # cartesian product is not necessary if the array was written by LArray
            cartesian_prod = _writer != 'LArray'
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
            dtype = attrs['dtype'] if 'dtype' in attrs else None
            if dtype is not None and dtype.kind == 'U':
                labels = np.char.decode(labels, 'utf-8')
            res = Axis(labels=labels, name=name)
            res._iswildcard = attrs['wildcard']
        elif _type == 'Group':
            if name is None:
                name = str(pd_obj.name)
            if name == 'None':
                name = None
            key = pd_obj.values
            dtype = attrs['dtype'] if 'dtype' in attrs else None
            if dtype is not None and dtype.kind == 'U':
                key = np.char.decode(key, 'utf-8')
            axis = self._get(attrs['axis_key'])
            res = LGroup(key=key, name=name, axis=axis)
        return res

    def _put(self, key, value, **kwargs):
        pd_store = self._pandas_hdfstore
        if isinstance(value, LArray):
            pd_store.put(key, value.to_frame())
            attrs = pd_store.get_storer(key).attrs
            attrs.type = 'Array'
            attrs.writer = 'LArray'
            self._put(key, value.meta)
        elif isinstance(value, Axis):
            dtype = value.dtype
            labels = np.char.encode(value.labels, 'utf-8') if dtype.kind == 'U' else value.labels
            s = Series(data=labels, name=value.name)
            pd_store.put(key, s)
            attrs = pd_store.get_storer(key).attrs
            attrs.type = 'Axis'
            attrs.dtype = dtype
            attrs.wildcard = value.iswildcard
        elif isinstance(value, Group):
            axis_key = kwargs.pop('axis_key', None)
            if axis_key is None:
                if value.axis.name is None:
                    raise ValueError(
                        "Argument axis_key must be provided explicitly if the associated axis is anonymous")
                axis_key = value.axis.name
            if axis_key not in pd_store:
                self._put(axis_key, value.axis)
            data = value.eval()
            dtype = data.dtype if isinstance(data, np.ndarray) else None
            if dtype is not None and dtype.kind == 'U':
                data = np.char.encode(data, 'utf-8')
            s = Series(data=data, name=value.name)
            pd_store.put(key, s)
            attrs = pd_store.get_storer(key).attrs
            attrs.type = 'Group'
            attrs.dtype = dtype
            attrs.axis_key = axis_key
        elif isinstance(value, Metadata):
            if len(value):
                pd_store.get_storer(key).attrs.metadata = value
        else:
            warnings.warn('{}: Type {} is currently not supported'.format(key, type(value)))


_hdf_store_cls = {'pandas': PandasStorer, 'tables': PytablesStorer}


class LHDFStore(object):
    """Context manager for reading and writing LArray objects.

    Parameters
    ----------
    filepath : str or PathLike object
        File path to HDF5 file
    mode : {'a', 'w', 'r', 'r+'}, default 'a'

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
            Specifies a compression level for data.
            A value of 0 disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
            Specifies the compression library to be used.
    fletcher32 : bool, default False
            If applying compression use the fletcher32 checksum
    engine: {'auto', 'tables', 'pandas'}, optional
        Load using `engine`. Use 'pandas' to read an HDF file generated with a LArray version previous to 0.31.
        Defaults to 'auto' (use default engine if you don't know the LArray version used to produced the HDF file).

    Examples
    --------
    # TODO : write examples
    """
    def __init__(self, filepath, mode=None, complevel=None, complib=None,
                 fletcher32=False, engine='auto', **kwargs):
        try:
            import tables
        except ImportError:
            raise ImportError('LHDFStore requires PyTables to be installed')

        is_new_file = not os.path.exists(filepath)
        if is_new_file and mode in ['r', 'r+']:
            raise ValueError('The file {} has not been found.'.format(filepath))

        if engine == 'auto':
            if is_new_file:
                engine = 'tables'
            else:
                import tables
                handle = tables.open_file(filepath, mode='r')
                # for backward compatibility, we assume that the used engine is 'pandas'
                # if not found among root attributes
                engine = getattr(handle.root._v_attrs, 'engine', 'pandas')
                handle.close()
        if engine not in _hdf_store_cls.keys():
            raise ValueError("Value of the 'engine' argument must be in list: "
                             "auto" + ", ".join(_hdf_store_cls.keys()))

        storer = _hdf_store_cls[engine](filepath, mode, complevel, complib, fletcher32, **kwargs)

        if is_new_file or mode == 'w':
            storer.attrs['engine'] = engine

        if getattr(storer.attrs, 'engine', 'pandas') != engine:
            raise Exception("Cannot {action} file {file}. Passed value for 'engine' argument was {engine_arg} "
                            "while the file {file} was originally created using "
                            "{engine}".format(action="read from" if mode == 'r' else "write into", file=filepath,
                                              engine_arg=engine, engine=storer.attrs['engine']))

        self._storer = storer

    def __fspath__(self):
        return self._storer._path

    @property
    def filename(self):
        """ File path to HDF5 file """
        return self._storer._path

    @property
    def is_open(self):
        """
        Return a boolean indicating whether the file is open
        """
        return self._storer.is_open

    @property
    def meta(self):
        return getattr(self._storer.attrs, 'metadata', Metadata())

    @meta.setter
    def meta(self, meta):
        self._storer.attrs.metadata = meta

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        return self._storer.remove(key)

    # TODO: not sure about this. Should be implemented in LazySession.
    def __getattr__(self, key):
        """ allow attribute access to get stores """
        if key in self.keys():
            return self.get(key)
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, key))

    def __contains__(self, key):
        return key in self._storer

    def __len__(self):
        return len(self._storer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close the PyTables file handle
        """
        self._storer.close()

    def keys(self):
        """
        Return a (potentially unordered) list of the keys corresponding to the
        objects stored in the HDFStore. These are ABSOLUTE path-names (e.g.
        have the leading '/'
        """
        return [n._v_pathname for n in self._storer.groups()]

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        """
        Iterate on key->group
        """
        for g in self._storer.groups():
            yield g._v_pathname, g

    iteritems = items

    def summary(self):
        """
        Return a list of LArray stored in the HDF5 file.

        Examples
        --------
        TODO: write examples
        """
        if self.is_open:
            res = ""
            for name, group in self.items():
                _type = getattr(group._v_attrs, 'type', 'Unknown')
                res += "{}: {}\n".format(name, _type)
            return res
        else:
            return "File {} is CLOSED".format(self.filename)

    def get(self, key, **kwargs):
        """
        Retrieve a larray object stored in file.

        Parameters
        ----------
        key : str
            Name of the object to read.
        **kwargs

          * fill_value : scalar or LArray, optional
                Value used to fill cells corresponding to label combinations which are not present in the input.
                Defaults to NaN.
          * sort_rows : bool, optional
                Whether or not to sort the rows alphabetically (sorting is more efficient than not sorting).
                Defaults to False.
          * sort_columns : bool, optional
                Whether or not to sort the columns alphabetically (sorting is more efficient than not sorting).
                Defaults to False.
          * name : str, optional
                Name of the axis or group to return. If None, name is set to passed key.
                Defaults to None.

        Returns
        -------
        obj : same type as object stored in file.

        Examples
        --------
        TODO : write examples
        """
        return self._storer.get(key, **kwargs)

    def put(self, key, value, **kwargs):
        """
        Dump a larray object in file.

        Parameters
        ----------
        key: str
            Name of the object to dump.
        value: LArray, Axis or Group
            Object to dump.
        **kwargs

          * ???

        Examples
        --------
        TODO : write examples
        """
        self._storer.put(key, value, **kwargs)


def read_hdf(filepath_or_buffer, key, fill_value=nan, na=nan, sort_rows=False, sort_columns=False,
             name=None, **kwargs):
    r"""Reads an axis or group or array named key from a HDF5 file in filepath (path+name)

    Parameters
    ----------
    filepath_or_buffer : str or LArrayHDFStore
        Path and name where the HDF5 file is stored or a HDFStore object.
    key : str or Group
        Name of the object to read.
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
    country  gender\time      2013      2014      2015
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
    with LHDFStore(filepath_or_buffer, **kwargs) as store:
        res = store.get(key, fill_value=fill_value, sort_rows=sort_rows, sort_columns=sort_columns, name=name)
    return res


@register_file_handler('hdf', ['h5', 'hdf'])
class HDFHandler(FileHandler):
    r"""
    Handler for HDF5 files using Pandas.
    """
    def __init__(self, fname, overwrite_file=False, engine='auto'):
        super(HDFHandler, self).__init__(fname, overwrite_file)
        self.engine = engine

    def _open_for_read(self):
        self.handle = LHDFStore(self.fname, mode='r', engine=self.engine)

    def _open_for_write(self):
        self.handle = LHDFStore(self.fname, engine=self.engine)

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
        return self.handle.get(hdf_key, **kwargs)

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, LArray):
            hdf_key = '/' + key
        elif isinstance(value, Axis):
            hdf_key = '__axes__/' + key
        elif isinstance(value, Group):
            hdf_key = '__groups__/' + key
            kwargs['axis_key'] = '__axes__/' + value.axis.name
        else:
            raise TypeError()
        self.handle.put(hdf_key, value, **kwargs)

    def _read_metadata(self):
        return self.handle.meta

    def _dump_metadata(self, metadata):
        self.handle.meta = metadata

    def close(self):
        self.handle.close()
