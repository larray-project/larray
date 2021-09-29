import pickle
from collections import OrderedDict

from typing import List, Tuple, Union

from larray.core.axis import Axis
from larray.core.group import Group
from larray.core.array import Array
from larray.core.metadata import Metadata
from larray.inout.session import register_file_handler
from larray.inout.common import FileHandler, _supported_types, _supported_typenames, _supported_scalars_types
from larray.util.types import Scalar


@register_file_handler('pickle', ['pkl', 'pickle'])
class PickleHandler(FileHandler):
    def __init__(self, fname, overwrite_file=False):
        assert fname is not None
        super(PickleHandler, self).__init__(fname, overwrite_file)

    def _open_for_read(self):
        with open(self.fname, 'rb') as f:
            self.data = OrderedDict(pickle.load(f))

    def _open_for_write(self):
        if not self.overwrite_file and self.fname.is_file():
            self._open_for_read()
        else:
            self.data = OrderedDict()

    def list_items(self) -> List[Tuple[str, str]]:
        # scalar
        items = [(key, type(value).__name__) for key, value in self.data.items()
                 if isinstance(value, _supported_scalars_types)]
        # axes
        items += [(key, 'Axis') for key, value in self.data.items() if isinstance(value, Axis)]
        # groups
        items += [(key, 'Group') for key, value in self.data.items() if isinstance(value, Group)]
        # arrays
        items += [(key, 'Array') for key, value in self.data.items() if isinstance(value, Array)]
        return items

    def _read_item(self, key, typename, *args, **kwargs) -> Union[Array, Axis, Group, Scalar]:
        if typename in _supported_typenames:
            return self.data[key]
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, _supported_types):
            self.data[key] = value
        else:
            raise TypeError()

    def _read_metadata(self) -> Metadata:
        if '__metadata__' in self.data:
            return self.data['__metadata__']
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        self.data['__metadata__'] = metadata

    def save(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)

    def close(self):
        pass
