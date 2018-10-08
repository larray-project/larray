from __future__ import absolute_import, division, print_function

import os.path
from collections import OrderedDict

from larray.core.axis import Axis
from larray.core.group import Group
from larray.core.array import LArray
from larray.core.metadata import Metadata
from larray.util.misc import pickle
from larray.inout.session import register_file_handler
from larray.inout.common import FileHandler


@register_file_handler('pickle', ['pkl', 'pickle'])
class PickleHandler(FileHandler):
    def _open_for_read(self):
        with open(self.fname, 'rb') as f:
            self.data = OrderedDict(pickle.load(f))

    def _open_for_write(self):
        if not self.overwrite_file and os.path.isfile(self.fname):
            self._open_for_read()
        else:
            self.data = OrderedDict()

    def list_items(self):
        # axes
        items = [(key, 'Axis') for key, value in self.data.items() if isinstance(value, Axis)]
        # groups
        items += [(key, 'Group') for key, value in self.data.items() if isinstance(value, Group)]
        # arrays
        items += [(key, 'Array') for key, value in self.data.items() if isinstance(value, LArray)]
        return items

    def _read_item(self, key, type, *args, **kwargs):
        if type in {'Array', 'Axis', 'Group'}:
            return self.data[key]
        else:
            raise TypeError()

    def _dump_item(self, key, value, *args, **kwargs):
        if isinstance(value, (LArray, Axis, Group)):
            self.data[key] = value
        else:
            raise TypeError()

    def _read_metadata(self):
        if '__metadata__' in self.data:
            return self.data['__metadata__']
        else:
            return Metadata()

    def _dump_metadata(self, metadata):
        self.data['__metadata__'] = metadata

    def close(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)
