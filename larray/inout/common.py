import os
from datetime import date, time, datetime
from collections import OrderedDict

from larray.core.axis import Axis
from larray.core.group import Group
from larray.core.array import Array


# all formats
_supported_larray_types = (Axis, Group, Array)

# only for HDF5 and pickle formats
# support list, tuple and dict?
_supported_scalars_types = (int, float, bool, bytes, str, date, time, datetime)
_supported_types = _supported_larray_types + _supported_scalars_types
_supported_typenames = {cls.__name__ for cls in _supported_types}


def _get_index_col(nb_axes=None, index_col=None, wide=True):
    if not wide:
        if nb_axes is not None or index_col is not None:
            raise ValueError("`nb_axes` or `index_col` argument cannot be used when `wide` argument is False")

    if nb_axes is not None and index_col is not None:
        raise ValueError("cannot specify both `nb_axes` and `index_col`")
    elif nb_axes is not None:
        index_col = list(range(nb_axes - 1))
    elif isinstance(index_col, int):
        index_col = [index_col]

    return index_col


class FileHandler(object):
    r"""
    Abstract class defining the methods for "file handler" subclasses.

    Parameters
    ----------
    fname : str
        Filename.

    Attributes
    ----------
    fname : str
        Filename.
    """
    def __init__(self, fname, overwrite_file=False):
        self.fname = fname
        self.original_file_name = None
        self.overwrite_file = overwrite_file

    def _open_for_read(self):
        raise NotImplementedError()

    def _open_for_write(self):
        raise NotImplementedError()

    def list_items(self):
        r"""
        Return list containing pairs (name, type) for all stored objects
        """
        raise NotImplementedError()

    def _read_item(self, key, type, *args, **kwargs):
        r"""Read item"""
        raise NotImplementedError()

    def _read_metadata(self):
        r"""Read metadata"""
        raise NotImplementedError()

    def _dump_item(self, key, value, *args, **kwargs):
        r"""Dump item. Raises an TypeError if type not taken into account by the FileHandler subclass."""
        raise NotImplementedError()

    def _dump_metadata(self, metadata):
        r"""Dump metadata"""
        raise NotImplementedError()

    def save(self):
        r"""
        Saves items in file.
        """
        pass

    def close(self):
        r"""
        Closes file.
        """
        raise NotImplementedError()

    def _get_original_file_name(self):
        if self.overwrite_file and os.path.isfile(self.fname):
            self.original_file_name = self.fname
            fname, ext = os.path.splitext(self.fname)
            self.fname = f'{fname}~{ext}'

    def _update_original_file(self):
        if self.original_file_name is not None and os.path.isfile(self.fname):
            os.remove(self.original_file_name)
            os.rename(self.fname, self.original_file_name)

    def read(self, keys, *args, **kwargs):
        r"""
        Reads file content (HDF, Excel, CSV, ...) and returns a dictionary containing loaded objects.

        Parameters
        ----------
        keys : list of str
            List of objects' names.
        *args : any
            Any other argument is passed through to the underlying read function.
        display : bool, optional
            Whether or not the function should display a message when starting and ending to load each object.
            Defaults to False.
        ignore_exceptions : bool, optional
            Whether or not an exception should stop the function or be ignored. Defaults to False.
        **kwargs : any
            Any other keyword argument is passed through to the underlying read function.

        Returns
        -------
        Metadata
            List of metadata to load.
        OrderedDict(str, Array/Axis/Group)
            Dictionary containing the loaded objects.
        """
        display = kwargs.pop('display', False)
        ignore_exceptions = kwargs.pop('ignore_exceptions', False)
        self._open_for_read()
        metadata = self._read_metadata()
        key_types = self.list_items()
        if keys is not None:
            key_types = [(key, type) for key, type in key_types if key in keys]
        res = OrderedDict()
        for key, type in key_types:
            if display:
                print("loading", type, "object", key, "...", end=' ')
            try:
                res[key] = self._read_item(key, type, *args, **kwargs)
            except Exception:
                if not ignore_exceptions:
                    raise
            if display:
                print("done")
        self.close()
        return metadata, res

    def dump(self, metadata, key_values, *args, **kwargs):
        r"""
        Dumps objects corresponding to keys in file in HDF, Excel, CSV, ... format

        Parameters
        ----------
        metadata: Metadata
            List of metadata to dump.
        key_values : list of (str, Array/Axis/Group) pairs
            Name and data of objects to dump.
        kwargs :
            * display: whether or not to display when the dump of each object is started/done.
        """
        display = kwargs.pop('display', False)
        self._get_original_file_name()
        self._open_for_write()
        if metadata is not None:
            self._dump_metadata(metadata)
        for key, value in key_values:
            if isinstance(value, Array) and value.ndim == 0:
                if display:
                    print(f'Cannot dump {key}. Dumping 0D arrays is currently not supported.')
                continue
            try:
                if display:
                    print("dumping", key, "...", end=' ')
                self._dump_item(key, value, *args, **kwargs)
                if display:
                    print("done")
            except TypeError:
                if display:
                    print(f"Cannot dump {key}. {type(value).__name__} is not a supported type")
        self.save()
        self.close()
        self._update_original_file()
