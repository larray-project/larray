from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from larray.util.misc import pickle
from larray.inout.common import FileHandler


class PickleHandler(FileHandler):
    def _open_for_read(self):
        with open(self.fname, 'rb') as f:
            self.data = pickle.load(f)

    def _open_for_write(self):
        self.data = OrderedDict()

    def list(self):
        return self.data.keys()

    def _read_item(self, key):
        return key, self.data[key]

    def _dump(self, key, value):
        self.data[key] = value

    def close(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)