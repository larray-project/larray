import os
from datetime import date, time, datetime
from pathlib import Path

from typing import Union, List, Tuple, Dict

from larray.core.axis import Axis
from larray.core.group import Group
from larray.core.array import Array
from larray.core.metadata import Metadata


# all formats
_supported_larray_types = (Axis, Group, Array)

# only for HDF5 and pickle formats
# support list, tuple and dict?
_supported_scalars_types = (int, float, bool, bytes, str, date, time, datetime)
_supported_types = _supported_larray_types + _supported_scalars_types
_supported_typenames = {cls.__name__ for cls in _supported_types}


def _get_index_col(nb_axes=None, index_col=None, wide=True) -> List[int]:
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


class FileHandler:
    r"""
    Abstract class defining the methods for "file handler" subclasses.

    Parameters
    ----------
    fname : str or Path
        Filename.

    Attributes
    ----------
    fname : Path
        Filename.
    """

    def __init__(self, fname: Union[str, Path], overwrite_file: bool = False):
        assert fname is not None
        if isinstance(fname, str):
            fname = Path(fname)
        if not isinstance(fname, Path):
            raise TypeError(f"Expected a string or a pathlib.Path object for the 'fname' argument. "
                            f"Got an object of type {type(fname).__name__} instead.")
        self.fname = fname
        self.original_file_name = None
        self.overwrite_file = overwrite_file

    def _open_for_read(self):
        raise NotImplementedError()

    def _open_for_write(self):
        raise NotImplementedError()

    def item_types(self) -> Dict[str, str]:
        r"""
        Return dict with type of each stored object.
        """
        raise NotImplementedError()

    def _read_item(self, key, type, *args, **kwargs):
        r"""Read item."""
        raise NotImplementedError()

    def _read_metadata(self) -> Metadata:
        r"""Read metadata."""
        raise NotImplementedError()

    def _dump_item(self, key, value, *args, **kwargs):
        r"""Dump item. Raises an TypeError if type not taken into account by the FileHandler subclass."""
        raise NotImplementedError()

    def _dump_metadata(self, metadata):
        r"""Dump metadata."""
        raise NotImplementedError()

    def save(self):
        r"""
        Save items in file.
        """
        pass

    def close(self):
        r"""
        Close file.
        """
        raise NotImplementedError()

    def _get_original_file_name(self):
        if self.overwrite_file and self.fname.is_file():
            self.original_file_name = self.fname
            self.fname = self.fname.parent / (self.fname.stem + '~' + self.fname.suffix)

    def _update_original_file(self):
        if self.original_file_name is not None and self.fname.is_file():
            os.remove(self.original_file_name)
            os.rename(self.fname, self.original_file_name)

    def read(self, keys, *args, display=False, ignore_exceptions=False, **kwargs) -> Tuple[Metadata, dict]:
        r"""
        Read file content (HDF, Excel, CSV, ...) and returns a dictionary containing loaded objects.

        Parameters
        ----------
        keys : list of str
            List of objects' names.
        *args : any
            Any other argument is passed through to the underlying read function.
        display : bool, optional
            Whether the function should display a message when starting and ending to load each object.
            Defaults to False.
        ignore_exceptions : bool, optional
            Whether an exception should stop the function or be ignored. Defaults to False.
        **kwargs : any
            Any other keyword argument is passed through to the underlying read function.

        Returns
        -------
        Metadata
            List of metadata to load.
        dict(str, Array/Axis/Group)
            Dictionary containing the loaded objects.
        """
        self._open_for_read()
        metadata = self._read_metadata()
        item_types = self.item_types()
        if keys is not None:
            item_types = {key: type_ for key, type_ in item_types.items() if key in keys}
        res = {}
        for key, type_ in item_types.items():
            if display:
                print("loading", type_, "object", key, "...", end=' ')
            try:
                res[key] = self._read_item(key, type_, *args, **kwargs)
            except Exception:
                if not ignore_exceptions:
                    raise
            if display:
                print("done")
        self.close()
        return metadata, res

    def dump(self, metadata, values, *args, display=False, **kwargs):
        r"""
        Dump objects corresponding to keys in file in HDF, Excel, CSV, ... format.

        Parameters
        ----------
        metadata: Metadata
            List of metadata to dump.
        values : dict
            Objects to dump as a {name: value} dict.
        display : bool, optional
            Whether to display when the dump of each object is started/done. Defaults to False.
        """
        self._get_original_file_name()
        self._open_for_write()
        if metadata is not None:
            self._dump_metadata(metadata)
        for key, value in values.items():
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
