from __future__ import absolute_import, division, print_function


def check_pattern(k, pattern):
    return k.startswith(pattern)


class Session(object):
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, '_objects', {})
                           # self._objects = {}
        self.add(*args, **kwargs)

    def __getitem__(self, key):
        return self._objects[key]

    def __setitem__(self, key, value):
        self._objects[key] = value

    def __getattr__(self, key):
        return self._objects[key]

    def __setattr__(self, key, value):
        self._objects[key] = value

    # TODO: implement __iter__
    # def load(self, fname, names):
    #     # TODO: support path + *.csv
    #     funcs = {'.h5': read_hdf, '.xls': read_excel}
    #     func = funcs[ext]
    #     for name in names:
    #         self[name] = func(fname, name)
    #
    # def dump(self, fname, fmt='.h5'):
    #     func = LArray.to_hdf
    #     for obj in self.objects(kind=LArray):
    #         func(obj, fname, name)

    # TODO: rename to filter and return another Session
    def objects(self, pattern=None, kind=None):
        keys = self._objects.keys()
        if pattern is not None:
            keys = [k for k in keys if check_pattern(k, pattern)]
        objects = [self._objects[k] for k in sorted(keys)]
        if kind is not None:
            return [obj for obj in objects if isinstance(obj, kind)]
        else:
            return objects

    def add(self, *args, **kwargs):
        for arg in args:
            self[arg.name] = arg
        for k, v in kwargs.items():
            self[k] = v