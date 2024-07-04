import sys
import re
import fnmatch
import warnings
from pathlib import Path

import numpy as np

from typing import Any, List, Union, KeysView, ValuesView, ItemsView, Dict, Iterable

from larray.core.metadata import Metadata
from larray.core.group import Group
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.array import Array, get_axes, ndtest, zeros, zeros_like, sequence      # noqa: F401
from larray.util.misc import float_error_handler_factory, is_interactive_interpreter, renamed_to, inverseop, size2str
from larray.inout.session import ext_default_engine, get_file_handler


# XXX: inherit from dict or Array?
class Session:
    r"""
    Groups several objects together.

    Parameters
    ----------
    *args : str or dict of {str: object} or iterable of tuples (str, object)
        Path to the file containing the session to load or
        list/tuple/dictionary containing couples (name, object).
    **kwargs : dict of {str: object}

        * Objects to add written as name=object
        * meta : list of pairs or dict or Metadata
            Metadata (title, description, author, creation_date, ...) associated with the array.
            Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Warnings
    --------
    Metadata is not kept when actions or methods are applied on a session
    except for operations modifying a specific array, such as: `s['arr1'] = 0`.
    Do not add metadata to a session if you know you will apply actions or methods
    on it before dumping it.

    Examples
    --------
    >>> # scalars
    >>> i, s = 5, 'string'
    >>> # axes
    >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")
    >>> # groups
    >>> a01 = a['a0,a1'] >> 'a01'
    >>> # arrays
    >>> arr1, arr2 = ndtest((a, b)), ndtest(a)

    create a Session by passing a list of pairs (name, object)

    >>> ses = Session([('i', i), ('s', s), ('a', a), ('b', b), ('a01', a01),
    ...                ('arr1', arr1), ('arr2', arr2)])

    create a Session using keyword arguments

    >>> ses = Session(i=i, s=s, a=a, b=b, a01=a01, arr1=arr1, arr2=arr2)

    create a Session by passing a dictionary

    >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01, 'arr1': arr1, 'arr2': arr2})

    load Session from file

    >>> ses = Session('my_session.h5')  # doctest: +SKIP

    create a session with metadata

    >>> ses = Session(arr1=arr1, arr2=arr2, meta=Metadata(title='my title', author='John Smith'))
    >>> ses.meta
    title: my title
    author: John Smith
    """

    def __init__(self, *args, meta=None, **kwargs):
        object.__setattr__(self, '_objects', {})
        if meta is None:
            meta = Metadata()
        self.meta = meta

        # When the deprecation period is over, this block can be removed after we replace *args by elements=None
        if len(args) == 0:
            elements = None
        elif len(args) == 1:
            elements = args[0]
        else:
            warnings.warn("Session(obj1, ...) is deprecated, please use Session(obj1name=obj1, ...) instead",
                          FutureWarning)
            elements = {a.name: a for a in args}

        if isinstance(elements, (str, Path)):
            # assume elements is a filename
            self.load(elements)
            self.update(**kwargs)
        else:
            # iterable of tuple or dict-like
            self.update(elements, **kwargs)

    @property
    def meta(self) -> Metadata:
        r"""Return metadata of the session.

        Returns
        -------
        Metadata:
            Metadata of the session.
        """
        return self._meta

    @meta.setter
    def meta(self, meta) -> None:
        if not isinstance(meta, (list, dict, Metadata)):
            raise TypeError(f"Expected list of pairs or dict or Metadata object "
                            f"instead of {type(meta).__name__}")
        object.__setattr__(self, '_meta', meta if isinstance(meta, Metadata) else Metadata(meta))

    # XXX: behave like a dict and return keys instead?
    def __iter__(self) -> Iterable[Any]:
        return iter(self.values())

    def add(self, *args, **kwargs) -> None:
        r"""
        Deprecated. Please use Session.update instead.
        """
        warnings.warn("Session.add() is deprecated. Please use Session.update() instead.",
                      FutureWarning, stacklevel=2)
        self.update({arg.name: arg for arg in args}, **kwargs)

    def update(self, other=None, **kwargs) -> None:
        r"""
        Update the session with the key/value pairs from other or passed keyword arguments, overwriting existing keys.
        Note that the session is updated inplace and no new Session object is returned.

        Parameters
        ----------
        other: Session or dict-like object or iterable with key/value pairs
            Object containing key/value pairs to add or modify.
        **kwargs:
            If keyword arguments are specified, the session is then updated with those key/value pairs
            (e.g.: ses.update(pop=pop, births=births, deaths=deaths)).

        Examples
        --------
        >>> x, y = Axis('x=x0..x2'), Axis('y=y0..y3')
        >>> arr1 = ndtest((x, y))
        >>> arr2 = ndtest(x)
        >>> s = Session(x=x, y=y, arr1=arr1, arr2=arr2)
        >>> # print item's names in sorted order
        >>> s.names
        ['arr1', 'arr2', 'x', 'y']
        >>> s.arr2
        x  x0  x1  x2
            0   1   2

        >>> # new axis and array
        >>> z = Axis('z=z0..z2')
        >>> arr3 = ndtest((x, z))
        >>> # arr2 is modified
        >>> arr2_modified = arr2.set_axes('x', z)

        Passing another session

        >>> s2 = Session(z=z, arr2=arr2_modified, arr3=arr3)
        >>> s.names
        ['arr1', 'arr2', 'x', 'y']
        >>> s.arr2
        x  x0  x1  x2
            0   1   2
        >>> s.update(s2)
        >>> # new items have been added to the session 's'
        >>> s.names
        ['arr1', 'arr2', 'arr3', 'x', 'y', 'z']
        >>> # and array 'arr2' has been updated
        >>> s.arr2
        z  z0  z1  z2
            0   1   2

        Passing a dictionary

        >>> s = Session(x=x, y=y, arr1=arr1, arr2=arr2)
        >>> s.names
        ['arr1', 'arr2', 'x', 'y']
        >>> s.arr2
        x  x0  x1  x2
            0   1   2
        >>> d = {'z': z, 'arr2': arr2_modified, 'arr3': arr3}
        >>> s.update(d)
        >>> s.names
        ['arr1', 'arr2', 'arr3', 'x', 'y', 'z']
        >>> s.arr2
        z  z0  z1  z2
            0   1   2

        Passing an iterable with key/value pairs

        >>> s = Session(x=x, y=y, arr1=arr1, arr2=arr2)
        >>> s.names
        ['arr1', 'arr2', 'x', 'y']
        >>> s.arr2
        x  x0  x1  x2
            0   1   2
        >>> i = [('z', z), ('arr2', arr2_modified), ('arr3', arr3)]
        >>> s.update(i)
        >>> s.names
        ['arr1', 'arr2', 'arr3', 'x', 'y', 'z']
        >>> s.arr2
        z  z0  z1  z2
            0   1   2

        Passing keyword arguments

        >>> s = Session(x=x, y=y, arr1=arr1, arr2=arr2)
        >>> s.names
        ['arr1', 'arr2', 'x', 'y']
        >>> s.arr2
        x  x0  x1  x2
            0   1   2
        >>> s.update(z=z, arr2=arr2_modified, arr3=arr3)
        >>> s.names
        ['arr1', 'arr2', 'arr3', 'x', 'y', 'z']
        >>> s.arr2
        z  z0  z1  z2
            0   1   2
        """
        from collections.abc import Iterable
        if other is None:
            pass
        elif hasattr(other, 'items'):
            self._update_from_iterable(other.items())
        elif isinstance(other, Iterable):
            self._update_from_iterable(other)
        else:
            raise ValueError(f"Expected Session, dict-like or iterable object for 'other' argument. "
                             f"Got {type(other).__name__}.")
        self._update_from_iterable(kwargs.items())

    def _ipython_key_completions_(self) -> List[str]:
        return list(self.keys())

    def __getitem__(self, key) -> Union[Any, 'Session']:
        if isinstance(key, int):
            keys = list(self.keys())
            return self._objects[keys[key]]
        elif isinstance(key, Array):
            assert np.issubdtype(key.dtype, np.bool_)
            assert key.ndim == 1
            # only keep True values
            true_keys = key[key].axes[0].labels
            return Session({k: self[k] for k in true_keys})
        elif isinstance(key, (tuple, list)):
            assert all(isinstance(k, str) for k in key)
            return Session({k: self[k] for k in key})
        else:
            return self._objects[key]

    def get(self, key, default=None) -> Any:
        r"""
        Return the object corresponding to the key.
        If the key doesn't correspond to any object, a default one can be returned.

        Parameters
        ----------
        key : str
            Name of the object.
        default : object, optional
            Returned object if the key doesn't correspond to any object of the current session.

        Returns
        -------
        object
            Object corresponding to the given key or a default one if not found.

        Examples
        --------
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)
        >>> s = Session({'a': a, 'b': b, 'a01': a01, 'arr1': arr1, 'arr2': arr2})
        >>> arr = s.get('arr1')
        >>> arr
        a\b  b0  b1  b2
         a0   0   1   2
         a1   3   4   5
         a2   6   7   8
        >>> arr = s.get('arr4', zeros('a=a0,a1;b=b0,b1', dtype=int))
        >>> arr
        a\b  b0  b1
         a0   0   0
         a1   0   0
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value) -> None:
        self._objects[key] = value

    def __delitem__(self, key) -> None:
        del self._objects[key]

    def __getattr__(self, key) -> Any:
        if key in self._objects:
            return self._objects[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value) -> None:
        # the condition below is needed because, unlike __getattr__, __setattr__ is called before any property
        # see https://stackoverflow.com/a/15751159
        if key == 'meta':
            super().__setattr__(key, value)
        else:
            self._objects[key] = value

    def __delattr__(self, key) -> None:
        del self._objects[key]

    def __dir__(self) -> List[str]:
        return list(set(dir(self.__class__)) | set(self.keys()))

    # needed to make *un*pickling work (because otherwise, __getattr__ is called before ._objects exists, which leads to
    # an infinite recursion)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        # equivalent to self.__dict__ = d (we need this form because __setattr__ is overridden)
        object.__setattr__(self, '__dict__', d)

    def load(self, fname, names=None, engine='auto', display=False, **kwargs) -> None:
        r"""
        Load objects from a file, or several .csv files.
        The Excel and CSV formats can only contain objects of Array type (plus metadata).

        WARNING: never load a file using the pickle engine (.pkl or .pickle) from an untrusted source, as it can lead
        to arbitrary code execution.

        Parameters
        ----------
        fname : str or Path
            This can be either the path to a single file, a path to a directory containing .csv files or a pattern
            representing several .csv files.
        names : list of str, optional
            List of objects to load.
            If `fname` is None, list of paths to CSV files.
            Defaults to all valid objects present in the file/directory.
        engine : {'auto', 'pandas_csv', 'pandas_hdf', 'pandas_excel', 'xlwings_excel', 'pickle'}, optional
            Load using `engine`. Defaults to 'auto' (use default engine for the format guessed from the file extension).
        display : bool, optional
            Whether to display which file is being worked on. Defaults to False.

        Examples
        --------
        In one module:

        >>> # scalars
        >>> i, s = 5, 'string'                                                      # doctest: +SKIP
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")                               # doctest: +SKIP
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'                                               # doctest: +SKIP
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)                                  # doctest: +SKIP
        >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01,
        ...                'arr1': arr1, 'arr2': arr2})                             # doctest: +SKIP
        >>> # metadata
        >>> ses.meta.title = 'my title'                                             # doctest: +SKIP
        >>> ses.meta.author = 'John Smith'                                          # doctest: +SKIP
        >>> # save the session in an HDF5 file
        >>> ses.save('input.h5')                                                    # doctest: +SKIP

        In another module: load the whole session

        >>> # the load method is automatically called when passing
        >>> # the path of file to the Session constructor
        >>> ses = Session('input.h5')                                               # doctest: +SKIP
        >>> ses                                                                     # doctest: +SKIP
        Session(a, a01, arr1, arr2, b, i, s)
        >>> ses.meta                                                                # doctest: +SKIP
        title: my title
        author: John Smith

        Load only some objects

        >>> ses = Session()
        >>> ses.load('input.h5', names=['s', 'a', 'b', 'arr1', 'arr2'], display=True)   # doctest: +SKIP
        opening input.h5
        loading Axis object a ... done
        loading Array object arr1 ... done
        loading Array object arr2 ... done
        loading Axis object b ... done
        loading str object s ... done

        Using .csv files (assuming the same session as above)

        >>> ses.save('data')                                                        # doctest: +SKIP
        >>> ses = Session()                                                         # doctest: +SKIP
        >>> # load all .csv files from the 'data' directory
        >>> ses.load('data', display=True)                                          # doctest: +SKIP
        opening data
        loading Array object arr1 ... done
        loading Array object arr2 ... done
        >>> # or only arrays containing the character '1' in their names
        >>> ses.load('data/*1.csv', display=True)                                   # doctest: +SKIP
        opening data/*1.csv
        loading Array object arr1 ... done
        """
        if display:
            print("opening", fname)
        if fname is None:
            if all([Path(name).suffix == '.csv' for name in names]):
                engine = ext_default_engine['csv']
            else:
                raise ValueError(f"List of paths to only CSV files expected. Got {names}")
        elif isinstance(fname, str):
            fname = Path(fname)
        if not isinstance(fname, Path):
            raise TypeError(f"Expected a string or a Path object for the 'fname' argument. "
                            f"Got object of type '{type(fname).__name__}' instead.")
        if engine == 'auto':
            ext = fname.suffix
            ext = ext.strip('.') if ext else 'csv'
            engine = ext_default_engine[ext]
        handler_cls = get_file_handler(engine)
        if engine == 'pandas_csv' and 'sep' in kwargs:
            handler = handler_cls(fname, kwargs['sep'])
        else:
            handler = handler_cls(fname)
        metadata, objects = handler.read(names, display=display, **kwargs)
        self._update_from_iterable(objects.items())
        self.meta = metadata

    def _update_from_iterable(self, it) -> None:
        for k, v in it:
            self[k] = v

    def save(self, fname, names=None, engine='auto', overwrite=True, display=False, **kwargs) -> None:
        r"""
        Dump objects from the current session to a file, or several .csv files.
        The Excel and CSV formats only dump objects of Array type (plus metadata).

        Parameters
        ----------
        fname : str or Path
            Path of the file for the dump.
            If objects are saved in CSV files, the path corresponds to a directory.
        names : list of str or None, optional
            List of names of objects to dump.
            If `fname` is None, list of paths to CSV files.
            Defaults to all objects present in the Session.
        engine : {'auto', 'pandas_csv', 'pandas_hdf', 'pandas_excel', 'xlwings_excel', 'pickle'}, optional
            Dump using `engine`. Defaults to 'auto' (use default engine for the format guessed from the file extension).
        overwrite: bool, optional
            Whether to overwrite an existing file, if any. Ignored for CSV files and 'pandas_excel' engine.
            If False, file is updated. Defaults to True.
        display : bool, optional
            Whether to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> # scalars
        >>> i, s = 5, 'string'                                                      # doctest: +SKIP
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")                               # doctest: +SKIP
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'                                               # doctest: +SKIP
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)                                  # doctest: +SKIP
        >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01,
        ...                'arr1': arr1, 'arr2': arr2})                             # doctest: +SKIP
        >>> # metadata
        >>> ses.meta.title = 'my title'                                             # doctest: +SKIP
        >>> ses.meta.author = 'John Smith'                                          # doctest: +SKIP

        Save all objects

        >>> ses.save('output.h5', display=True)                                     # doctest: +SKIP
        dumping i ... done
        dumping s ... done
        dumping a ... done
        dumping b ... done
        dumping a01 ... done
        dumping arr1 ... done
        dumping arr2 ... done

        Save only some objects

        >>> ses.save('output.h5', names=['s', 'a', 'b', 'arr1', 'arr2'], display=True)  # doctest: +SKIP
        dumping s ... done
        dumping a ... done
        dumping b ... done
        dumping arr1 ... done
        dumping arr2 ... done

        Update file

        >>> arr1, arr4 = ndtest((3, 3)), ndtest((2, 3))                             # doctest: +SKIP
        >>> ses2 = Session({'arr1': arr1, 'arr4': arr4})                            # doctest: +SKIP
        >>> # replace arr1 and add arr4 in file output.h5
        >>> ses2.save('output.h5', overwrite=False, display=True)                   # doctest: +SKIP
        dumping arr1 ... done
        dumping arr4 ... done
        """
        if isinstance(fname, str):
            fname = Path(fname)
        if not isinstance(fname, Path):
            raise TypeError(f"Expected a string or a Path object for the 'fname' argument. "
                            f"Got object of type '{type(fname).__name__}' instead.")
        if engine == 'auto':
            ext = fname.suffix
            ext = ext.strip('.') if ext else 'csv'
            engine = ext_default_engine[ext]
        handler_cls = get_file_handler(engine)
        if engine == 'pandas_csv' and 'sep' in kwargs:
            handler = handler_cls(fname, overwrite, kwargs['sep'])
        else:
            handler = handler_cls(fname, overwrite)
        meta = self.meta if overwrite else None
        objects = self._objects
        if names is not None:
            names_set = set(names)
            objects = {k: v for k, v in objects.items() if k in names_set}
        handler.dump(meta, objects, display=display, **kwargs)

    def to_globals(self, names=None, depth=0, warn=True, inplace=False) -> None:
        r"""
        Create global variables out of objects in the session.

        Parameters
        ----------
        names : list of str or None, optional
            List of names of objects to convert to globals. Defaults to all objects present in the Session.
        depth : int, optional
            depth of call stack where to create the variables. 0 is where to_globals was called, 1 the caller of
            to_globals, etc. Defaults to 0.
        warn : bool, optional
            Whether to warn the user that this method should only be used in an interactive console (see below).
            Defaults to True.
        inplace : bool, optional
            If True, to_globals will assume all arrays already exist and have the same axes and will replace their
            content instead of creating new variables. Non array variables in the session will be ignored.
            Defaults to False.

        Notes
        -----
        This method should usually only be used in an interactive console and not in a script. Code editors are
        confused by this kind of manipulation and will likely consider as invalid the code using variables created in
        this way. Additionally, when using this method auto-completion, "show definition", "go to declaration" and
        other similar code editor features will probably not work for the variables created in this way and any
        variable derived from them.

        Examples
        --------
        >>> s = Session(arr1=ndtest(3), arr2=ndtest((2, 2)))
        >>> s.to_globals(warn=False)
        >>> arr1
        a  a0  a1  a2
            0   1   2
        >>> arr2
        a\b  b0  b1
         a0   0   1
         a1   2   3
        """
        # noinspection PyProtectedMember
        if warn and not is_interactive_interpreter():
            warnings.warn("Session.to_globals should usually only be used in interactive consoles and not in scripts. "
                          "Use warn=False to deactivate this warning.",
                          RuntimeWarning, stacklevel=2)
        d = sys._getframe(depth + 1).f_globals

        objects = self._objects
        if names is not None:
            names_set = set(names)
            objects = {k: v for k, v in objects.items() if k in names_set}
        if inplace:
            for k, v in objects.items():
                if k not in d:
                    raise ValueError(f"'{k}' not found in current namespace. Session.to_globals(inplace=True) requires "
                                     f"all arrays to already exist.")
                if not isinstance(v, Array):
                    continue
                if not d[k].axes == v.axes:
                    raise ValueError(f"Session.to_globals(inplace=True) requires the existing (destination) arrays "
                                     f"to have the same axes than those stored in the session and this is not the case "
                                     f"for '{k}'.\nexisting: {d[k].info}\nsession: {v.info}")
                d[k][:] = v
        else:
            for k, v in objects.items():
                d[k] = v

    def to_pickle(self, fname, names=None, overwrite=True, display=False, **kwargs) -> None:
        r"""
        Dump objects from the current session to a file using pickle.

        WARNING: never load a pickle file (.pkl or .pickle) from an untrusted source, as it can lead to arbitrary code
        execution.

        Parameters
        ----------
        fname : str or Path
            Path for the dump.
        names : list of str or None, optional
            Names of objects to dump.
            Defaults to all objects present in the Session.
        overwrite: bool, optional
            Whether to overwrite an existing file, if any.
            If False, file is updated. Defaults to True.
        display : bool, optional
            Whether to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> # scalars
        >>> i, s = 5, 'string'                                                      # doctest: +SKIP
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")                               # doctest: +SKIP
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'                                               # doctest: +SKIP
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)                                  # doctest: +SKIP
        >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01,
        ...                'arr1': arr1, 'arr2': arr2})                             # doctest: +SKIP
        >>> # metadata
        >>> ses.meta.title = 'my title'                                             # doctest: +SKIP
        >>> ses.meta.author = 'John Smith'                                          # doctest: +SKIP

        Save all objects

        >>> ses.to_pickle('output.pkl', display=True)                               # doctest: +SKIP
        dumping i ... done
        dumping s ... done
        dumping a ... done
        dumping b ... done
        dumping a01 ... done
        dumping arr1 ... done
        dumping arr2 ... done

        Save only some objects

        >>> ses.to_pickle('output.pkl', names=['s', 'a', 'b', 'arr1', 'arr2'], display=True)    # doctest: +SKIP
        dumping s ... done
        dumping a ... done
        dumping b ... done
        dumping arr1 ... done
        dumping arr2 ... done
        """
        self.save(fname, names, ext_default_engine['pkl'], overwrite, display, **kwargs)

    dump = renamed_to(save, 'dump', raise_error=True)

    def to_hdf(self, fname, names=None, overwrite=True, display=False, **kwargs) -> None:
        r"""
        Dump objects from the current session to an HDF file.

        Parameters
        ----------
        fname : str or Path
            Path of the file for the dump.
        names : list of str or None, optional
            Names of objects to dump.
            Defaults to all objects present in the Session.
        overwrite: bool, optional
            Whether to overwrite an existing file, if any.
            If False, file is updated. Defaults to True.
        display : bool, optional
            Whether to display which file is being worked on. Defaults to False.

        Examples
        --------
        >>> # scalars
        >>> i, s = 5, 'string'                                                      # doctest: +SKIP
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")                               # doctest: +SKIP
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'                                               # doctest: +SKIP
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)                                  # doctest: +SKIP
        >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01,
        ...                'arr1': arr1, 'arr2': arr2})                             # doctest: +SKIP
        >>> # metadata
        >>> ses.meta.title = 'my title'                                             # doctest: +SKIP
        >>> ses.meta.author = 'John Smith'                                          # doctest: +SKIP

        Save all objects

        >>> ses.to_hdf('output.h5', display=True)                                   # doctest: +SKIP
        dumping i ... done
        dumping s ... done
        dumping a ... done
        dumping b ... done
        dumping a01 ... done
        dumping arr1 ... done
        dumping arr2 ... done

        Save only some objects

        >>> ses.to_hdf('output.h5', names=['s', 'a', 'b', 'arr1', 'arr2'], display=True)    # doctest: +SKIP
        dumping s ... done
        dumping a ... done
        dumping b ... done
        dumping arr1 ... done
        dumping arr2 ... done
        """
        self.save(fname, names, ext_default_engine['hdf'], overwrite, display, **kwargs)

    dump_hdf = renamed_to(to_hdf, 'dump_hdf', raise_error=True)

    def to_excel(self, fname, names=None, overwrite=True, display=False, **kwargs) -> None:
        r"""
        Dump Array objects from the current session to an Excel file.

        Parameters
        ----------
        fname : str or Path
            Path of the file for the dump.
        names : list of str or None, optional
            Names of Array objects to dump.
            Defaults to all Array objects present in the Session.
        overwrite: bool, optional
            Whether to overwrite an existing file, if any. If False, file is updated. Defaults to True.
        display : bool, optional
            Whether to display which file is being worked on. Defaults to False.

        Notes
        -----
        - each array is saved in a separate sheet
        - all session metadata is saved in the same sheet named __metadata__

        Examples
        --------
        >>> # scalars
        >>> i, s = 5, 'string'                                                      # doctest: +SKIP
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")                               # doctest: +SKIP
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'                                               # doctest: +SKIP
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)                                  # doctest: +SKIP
        >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01,
        ...                'arr1': arr1, 'arr2': arr2})                             # doctest: +SKIP
        >>> # metadata
        >>> ses.meta.title = 'my title'                                             # doctest: +SKIP
        >>> ses.meta.author = 'John Smith'                                          # doctest: +SKIP

        Save all arrays (and arrays only)

        >>> ses.to_excel('output.xlsx', display=True)                               # doctest: +SKIP
        dumping i ... Cannot dump i. int is not a supported type
        dumping s ... Cannot dump s. str is not a supported type
        dumping a ... Cannot dump a. Axis is not a supported type
        dumping b ... Cannot dump b. Axis is not a supported type
        dumping a01 ... Cannot dump a01. LGroup is not a supported type
        dumping arr1 ... done
        dumping arr2 ... done

        Save only some arrays

        >>> ses.to_excel('output.xlsx', names=['arr1'], display=True)               # doctest: +SKIP
        dumping arr1 ... done
        """
        self.save(fname, names, ext_default_engine['xlsx'], overwrite, display, **kwargs)

    dump_excel = renamed_to(to_excel, 'dump_excel', raise_error=True)

    def to_csv(self, fname, names=None, display=False, **kwargs) -> None:
        r"""
        Dump Array objects from the current session to CSV files.

        Parameters
        ----------
        fname : str or Path
            Path for the directory that will contain CSV files.
        names : list of str or None, optional
            Names of Array objects to dump.
            Defaults to all Array objects present in the Session.
        display : bool, optional
            Whether to display which file is being worked on. Defaults to False.

        Notes
        -----
        - each array is saved in a separate file
        - all session metadata is saved in the same CSV file named __metadata__.csv

        Examples
        --------
        >>> # scalars
        >>> i, s = 5, 'string'                                                      # doctest: +SKIP
        >>> # axes
        >>> a, b = Axis("a=a0..a2"), Axis("b=b0..b2")                               # doctest: +SKIP
        >>> # groups
        >>> a01 = a['a0,a1'] >> 'a01'                                               # doctest: +SKIP
        >>> # arrays
        >>> arr1, arr2 = ndtest((a, b)), ndtest(a)                                  # doctest: +SKIP
        >>> ses = Session({'i': i, 's': s, 'a': a, 'b': b, 'a01': a01,
        ...                'arr1': arr1, 'arr2': arr2})                             # doctest: +SKIP
        >>> # metadata
        >>> ses.meta.title = 'my title'                                             # doctest: +SKIP
        >>> ses.meta.author = 'John Smith'                                          # doctest: +SKIP

        Save all arrays (and arrays only)

        >>> ses.to_csv('output', display=True)                                      # doctest: +SKIP
        dumping i ... Cannot dump i. int is not a supported type
        dumping s ... Cannot dump s. str is not a supported type
        dumping a ... Cannot dump a. Axis is not a supported type
        dumping b ... Cannot dump b. Axis is not a supported type
        dumping a01 ... Cannot dump a01. LGroup is not a supported type
        dumping arr1 ... done
        dumping arr2 ... done

        Save only some arrays

        >>> ses.to_csv('output', names=['arr1'], display=True)                      # doctest: +SKIP
        dumping arr1 ... done
        """
        self.save(fname, names, ext_default_engine['csv'], display=display, **kwargs)

    dump_csv = renamed_to(to_csv, 'dump_csv', raise_error=True)

    def filter(self, pattern=None, kind=None) -> 'Session':
        r"""
        Return a new session with objects which match some criteria.

        Parameters
        ----------
        pattern : str, optional
            Only keep arrays whose key match `pattern`.

            - `?`     matches any single character
            - `*`     matches any number of characters
            - [seq]   matches any character in seq
            - [!seq]  matches any character not in seq

        kind : (tuple of) type, optional
            Only keep objects which are instances of type(s) `kind`.

        Returns
        -------
        Session
            The filtered session.

        Examples
        --------
        >>> axis = Axis('a=a0..a2')
        >>> group = axis['a0,a1'] >> 'a01'
        >>> test1, zero1 = ndtest((2, 2)), zeros((3, 2))
        >>> s = Session({'test1': test1, 'zero1': zero1, 'axis': axis, 'group': group})

        Filter using a pattern argument

        >>> # get all items with names ending with '1'
        >>> s.filter(pattern='*1').names
        ['test1', 'zero1']

        >>> # get all items with names starting with letter in range a-k
        >>> s.filter(pattern='[a-k]*').names
        ['axis', 'group']

        Filter using kind argument

        >>> s.filter(kind=Axis).names
        ['axis']
        >>> s.filter(kind=(Axis, Group)).names
        ['axis', 'group']
        """
        objects = self._objects
        if pattern is not None:
            regex = fnmatch.translate(pattern)
            match = re.compile(regex).match
            objects = {k: v for k, v in objects.items() if match(k)}
        if kind is not None:
            objects = {k: v for k, v in objects.items() if isinstance(v, kind)}
        return Session(objects)

    @property
    def names(self) -> List[str]:
        r"""
        Return the list of names of the objects in the session.
        The list is sorted alphabetically and does not follow the internal order.

        Returns
        -------
        list of str

        See Also
        --------
        Session.keys

        Examples
        --------
        >>> axis1 = Axis("a=a0..a2")
        >>> group1 = axis1['a0,a1'] >> 'a01'
        >>> arr1, arr2 = ndtest((2, 2)), ndtest(4)
        >>> s = Session({'arr2': arr2, 'arr1': arr1, 'group1': group1, 'axis1': axis1})
        >>> # print array's names in the alphabetical order
        >>> s.names
        ['arr1', 'arr2', 'axis1', 'group1']

        >>> # keys() follows the internal order
        >>> list(s.keys())
        ['arr2', 'arr1', 'group1', 'axis1']
        """
        return sorted(self._objects.keys())

    def copy(self) -> 'Session':
        r"""Return a copy of the session.
        """
        # this actually *does* a copy of the internal mapping (the mapping is not reused-as is)
        return self.__class__(self._objects)

    def keys(self) -> KeysView[str]:
        r"""
        Return a view on the session's keys.

        Returns
        -------
        View on the session's keys.

        See Also
        --------
        Session.names

        Examples
        --------
        >>> axis1 = Axis("a=a0..a2")
        >>> group1 = axis1['a0,a1'] >> 'a01'
        >>> arr1, arr2 = ndtest((2, 2)), ndtest(4)
        >>> s = Session({'arr2': arr2, 'arr1': arr1, 'group1': group1, 'axis1': axis1})
        >>> # similar to names by follows the internal order
        >>> list(s.keys())
        ['arr2', 'arr1', 'group1', 'axis1']

        >>> # gives the names of objects in alphabetical order
        >>> s.names
        ['arr1', 'arr2', 'axis1', 'group1']
        """
        return self._objects.keys()

    def values(self) -> ValuesView[Any]:
        r"""
        Return a view on the session's values.

        Returns
        -------
        View on the session's values.

        Examples
        --------
        >>> axis1 = Axis("a=a0..a2")
        >>> group1 = axis1['a0,a1'] >> 'a01'
        >>> arr1, arr2 = ndtest((2, 2)), ndtest(4)
        >>> s = Session({'arr2': arr2, 'arr1': arr1, 'group1': group1, 'axis1': axis1})
        >>> # assuming you know the order of objects stored in the session
        >>> arr2, arr1, group1, axis1 = s.values()
        >>> # otherwise, prefer the following syntax
        >>> arr1, arr2, axis1, group1 = s['arr1', 'arr2', 'axis1', 'group1']
        >>> arr1
        a\b  b0  b1
         a0   0   1
         a1   2   3
        >>> axis1
        Axis(['a0', 'a1', 'a2'], 'a')
        """
        return self._objects.values()

    def items(self) -> ItemsView[str, Any]:
        r"""
        Return a view of the session's items ((key, value) pairs).

        Returns
        -------
        View on the session's items.

        Examples
        --------
        >>> axis1 = Axis("a=a0..a2")
        >>> group1 = axis1['a0,a1'] >> 'a01'
        >>> arr1, arr2 = ndtest((2, 2)), ndtest(4)
        >>> # make the test pass on both Windows and Linux
        >>> arr1, arr2 = arr1.astype(np.int64), arr2.astype(np.int64)
        >>> s = Session({'arr2': arr2, 'arr1': arr1, 'group1': group1, 'axis1': axis1})
        >>> for k, v in s.items():
        ...     print(f"{k}: {v.info if isinstance(v, Array) else repr(v)}")
        arr2: 4
         a [4]: 'a0' 'a1' 'a2' 'a3'
        dtype: int64
        memory used: 32 bytes
        arr1: 2 x 2
         a [2]: 'a0' 'a1'
         b [2]: 'b0' 'b1'
        dtype: int64
        memory used: 32 bytes
        group1: a['a0', 'a1'] >> 'a01'
        axis1: Axis(['a0', 'a1', 'a2'], 'a')
        """
        return self._objects.items()

    def __repr__(self) -> str:
        keys = ", ".join(self.keys())
        return f'Session({keys})'

    def __len__(self) -> int:
        return len(self._objects)

    # binary operations are dispatched element-wise to all arrays (we consider Session as an array-like)
    def _binop(opname, arrays_only=True):
        opfullname = f'__{opname}__'

        def opmethod(self, other) -> 'Session':
            self_keys = set(self.keys())
            all_keys = list(self.keys())
            if not isinstance(other, Array) and hasattr(other, 'keys'):
                all_keys += [n for n in other.keys() if n not in self_keys]
            with np.errstate(call=_session_float_error_handler):
                res = []
                for name in all_keys:
                    self_item = self.get(name, nan)
                    other_operand = other.get(name, nan) if hasattr(other, 'get') else other
                    if arrays_only and not isinstance(self_item, Array):
                        res_item = self_item
                    else:
                        try:
                            res_item = getattr(self_item, opfullname)(other_operand)
                        # TypeError for str arrays, ValueError for incompatible axes, ...
                        except Exception:
                            res_item = nan
                        # this should only ever happen when self_array is a non Array (eg. nan)
                        if res_item is NotImplemented:
                            inv_opname = f'__{inverseop(opname)}__'
                            try:
                                res_item = getattr(other_operand, inv_opname)(self_item)
                            # TypeError for str arrays, ValueError for incompatible axes, ...
                            except Exception:
                                res_item = nan
                    res.append((name, res_item))
            try:
                # XXX: print a warning?
                ses = self.__class__(res)
            except Exception:
                ses = Session(res)
            return ses
        opmethod.__name__ = opfullname
        return opmethod

    __add__ = _binop('add')
    __radd__ = _binop('radd')
    __sub__ = _binop('sub')
    __rsub__ = _binop('rsub')
    __mul__ = _binop('mul')
    __rmul__ = _binop('rmul')
    __truediv__ = _binop('truediv')
    __rtruediv__ = _binop('rtruediv')

    __eq__ = _binop('eq', arrays_only=False)
    __ne__ = _binop('ne', arrays_only=False)

    # element-wise method factory
    # unary operations are (also) dispatched element-wise to all arrays
    def _unaryop(opname):
        opfullname = f'__{opname}__'

        def opmethod(self) -> 'Session':
            with np.errstate(call=_session_float_error_handler):
                res = []
                for k, v in self.items():
                    try:
                        res_array = getattr(v, opfullname)()
                    except Exception:
                        res_array = nan
                    res.append((k, res_array))
            try:
                # XXX: print a warning?
                ses = self.__class__(res)
            except Exception:
                ses = Session(res)
            return ses
        opmethod.__name__ = opfullname
        return opmethod

    __neg__ = _unaryop('neg')
    __pos__ = _unaryop('pos')
    __abs__ = _unaryop('abs')
    __invert__ = _unaryop('invert')

    def element_equals(self, other, rtol=0, atol=0, nans_equal=False) -> Array:
        r"""Test if each element (group, axis and array) of the current session equals
        the corresponding element of another session.

        For arrays, it is equivalent to apply :py:meth:`Array.equals` with flag nans_equal=True
        to all arrays from two sessions.

        Parameters
        ----------
        other : Session
            Session to compare with.
        rtol : float or int, optional
            The relative tolerance parameter (see Notes). Defaults to 0.
        atol : float or int, optional
            The absolute tolerance parameter (see Notes). Defaults to 0.
        nans_equal : boolean, optional
            Whether to consider NaN values at the same positions in the two arrays as equal.
            By default, an array containing NaN values is never equal to another array, even if that other array
            also contains NaN values at the same positions. The reason is that a NaN value is different from
            *anything*, including itself. Defaults to True.

        Returns
        -------
        Boolean Array

        Notes
        -----
        Metadata is ignored.

        For finite values, element_equals uses the following equation to test whether two arrays are equal:

            absolute(array1 - array2) <= (atol + rtol * absolute(array2))

        See Also
        --------
        Session.equals

        Examples
        --------
        >>> a = Axis('a=a0..a2')
        >>> a01 = a['a0,a1'] >> 'a01'
        >>> s1 = Session({'a': a, 'a01': a01, 'arr1': ndtest(2), 'arr2': ndtest((2, 2))})
        >>> s2 = Session({'a': a, 'a01': a01, 'arr1': ndtest(2), 'arr2': ndtest((2, 2))})

        Identical sessions

        >>> s1.element_equals(s2)
        name     a   a01  arr1  arr2
              True  True  True  True

        Different value(s) between two arrays

        >>> s2.arr1['a1'] = 0
        >>> s1.element_equals(s2)
        name     a   a01   arr1  arr2
              True  True  False  True

        Test equality between two arrays within a given tolerance range.
        Return True if absolute(s1.arr1 - s2.arr1) <= (atol + rtol * absolute(s2.arr1)).

        >>> s1.arr1 = Array([6., 8.], "a=a0,a1")
        >>> s2.arr1 = Array([5.999, 8.001], "a=a0,a1")

        >>> s1.element_equals(s2)
        name     a   a01   arr1  arr2
              True  True  False  True
        >>> s1.element_equals(s2, atol=0.01)
        name     a   a01  arr1  arr2
              True  True  True  True
        >>> s1.element_equals(s2, rtol=0.01)
        name     a   a01  arr1  arr2
              True  True  True  True

        Different label(s)

        >>> s2.arr2 = ndtest("b=b0,b1; a=a0,a1")
        >>> s2.a = Axis('a=a0,a1')
        >>> s1.element_equals(s2)
        name      a   a01   arr1   arr2
              False  True  False  False

        Extra/missing objects

        >>> s2.arr3 = ndtest((3, 3))
        >>> del s2.a
        >>> s1.element_equals(s2)
        name      a   a01   arr1   arr2   arr3
              False  True  False  False  False
        """
        supported_objects = (Axis, Group, Array)
        self_keys = [k for k, v in self.items() if isinstance(v, supported_objects)]
        other_keys = [k for k, v in other.items() if isinstance(v, supported_objects) and k not in self_keys]
        all_keys = self_keys + other_keys

        def elem_equal(e1, e2) -> bool:
            if type(e1) is not type(e2):
                return False
            if isinstance(e1, (Group, Axis)):
                return e1.equals(e2)
            else:
                return e1.equals(e2, rtol=rtol, atol=atol, nans_equal=nans_equal)

        res = [elem_equal(self.get(key), other.get(key)) for key in all_keys]
        return Array(res, [Axis(all_keys, 'name')])

    array_equals = renamed_to(element_equals, 'array_equals', raise_error=True)

    def equals(self, other, rtol=0, atol=0, nans_equal=False) -> bool:
        r"""Test if all elements (groups, axes and arrays) of the current session are equal
        to those of another session.

        Parameters
        ----------
        other : Session
            Session to compare with.
        rtol : float or int, optional
            The relative tolerance parameter (see Notes). Defaults to 0.
        atol : float or int, optional
            The absolute tolerance parameter (see Notes). Defaults to 0.
        nans_equal : boolean, optional
            Whether to consider NaN values at the same positions in the two arrays as equal.
            By default, an array containing NaN values is never equal to another array, even if that other array
            also contains NaN values at the same positions. The reason is that a NaN value is different from
            *anything*, including itself. Defaults to True.

        Returns
        -------
        True if elements of both sessions are all equal, False otherwise.

        Notes
        -----
        Metadata is ignored.

        For finite values, equals uses the following equation to test whether two arrays are equal:

            absolute(array1 - array2) <= (atol + rtol * absolute(array2))

        See Also
        --------
        Session.element_equals

        Examples
        --------
        >>> a = Axis('a=a0..a2')
        >>> a01 = a['a0,a1'] >> 'a01'
        >>> s1 = Session({'a': a, 'a01': a01, 'arr1': ndtest(2), 'arr2': ndtest((2, 2))})
        >>> s2 = Session({'a': a, 'a01': a01, 'arr1': ndtest(2), 'arr2': ndtest((2, 2))})

        Identical sessions

        >>> s1.equals(s2)
        True

        Different value(s) between two arrays

        >>> s2.arr1['a1'] = 0
        >>> s1.equals(s2)
        False

        Test equality between two arrays within a given tolerance range.
        Return True if absolute(s1.arr1 - s2.arr1) <= (atol + rtol * absolute(s2.arr1)).

        >>> s1.arr1 = Array([6., 8.], "a=a0,a1")
        >>> s2.arr1 = Array([5.999, 8.001], "a=a0,a1")

        >>> s1.equals(s2)
        False
        >>> s1.equals(s2, atol=0.01)
        True
        >>> s1.equals(s2, rtol=0.01)
        True

        Different label(s)

        >>> s2.arr2 = ndtest("b=b0,b1; a=a0,a1")
        >>> s2.a = Axis('a=a0,a1')
        >>> s1.equals(s2)
        False

        Extra/missing axis(es), group(s), array(s)

        >>> s2.arr3 = ndtest((3, 3))
        >>> del s2.a
        >>> s1.equals(s2)
        False
        """
        return all(self.element_equals(other, rtol=rtol, atol=atol, nans_equal=nans_equal))

    def transpose(self, *args) -> 'Session':
        r"""Reorder axes of arrays in session, ignoring missing axes for each array.

        Parameters
        ----------
        *args
            Accepts either a tuple of axes specs or axes specs as `*args`. Omitted axes keep their order.
            Use ... to avoid specifying intermediate axes. Axes missing in an array are ignored.

        Returns
        -------
        Session
            Session with each array with reordered axes where appropriate.

        See Also
        --------
        Array.transpose

        Examples
        --------
        Let us create a test session and a small helper function to display sessions as a short summary.

        >>> arr1 = ndtest((2, 2, 2))
        >>> arr2 = ndtest((2, 2))
        >>> sess = Session({'arr1': arr1, 'arr2': arr2})
        >>> def print_summary(s):
        ...     print(s.summary({Array: "{key} -> {axes_names}"}))
        >>> print_summary(sess)
        arr1 -> a, b, c
        arr2 -> a, b

        Put 'b' axis in front of all arrays

        >>> print_summary(sess.transpose('b'))
        arr1 -> b, a, c
        arr2 -> b, a

        Axes missing on an array are ignored ('c' for arr2 in this case)

        >>> print_summary(sess.transpose('c', 'b'))
        arr1 -> c, b, a
        arr2 -> b, a

        Use ... to move axes to the end

        >>> print_summary(sess.transpose(..., 'a'))   # doctest: +SKIP
        arr1 -> b, c, a
        arr2 -> b, a
        """
        def lenient_transpose(v, axes):
            # filter out axes not in arr.axes
            return v.transpose([a for a in axes if a in v.axes or a is Ellipsis])
        return self.apply(lenient_transpose, args)

    def compact(self, display=False) -> 'Session':
        r"""
        Detect and remove "useless" axes (ie axes for which values are constant over the whole axis) for all array
        objects in session.

        Parameters
        ----------
        display : bool, optional
            Whether to display a message for each array that is compacted

        Returns
        -------
        Session
            A new session containing all compacted arrays

        Examples
        --------
        >>> arr1 = sequence('b=b0..b2', ndtest(3), zeros_like(ndtest(3)))
        >>> arr1
        a\b  b0  b1  b2
         a0   0   0   0
         a1   1   1   1
         a2   2   2   2
        >>> compact_ses = Session(arr1=arr1).compact(display=True)
        arr1 was constant over: b
        >>> compact_ses.arr1
        a  a0  a1  a2
            0   1   2
        """
        return Session({k: v.compact(display=display, name=k) for k, v in self._objects.items()})

    def apply(self, func, *args, kind=Array, **kwargs) -> 'Session':
        r"""
        Apply function `func` on elements of the session and return a new session.

        Parameters
        ----------
        func : function
            Function to apply to each element of the session. It should take a single `element` argument and return
            a single value.
        *args : any
            Any extra arguments are passed to the function
        kind : type or tuple of types, optional
            Type(s) of elements `func` will be applied to. Other elements will be left intact. Use `kind=object` to
            apply to all kinds of objects. Defaults to Array.
        **kwargs : any
            Any extra keyword arguments are passed to the function

        Returns
        -------
        Session
            A new session containing all processed elements

        Examples
        --------
        >>> arr1 = ndtest(2)
        >>> arr1
        a  a0  a1
            0   1
        >>> arr2 = ndtest(3)
        >>> arr2
        a  a0  a1  a2
            0   1   2
        >>> sess1 = Session({'arr1': arr1, 'arr2': arr2})
        >>> sess1
        Session(arr1, arr2)
        >>> def increment(array):
        ...     return array + 1
        >>> sess2 = sess1.apply(increment)
        >>> sess2.arr1
        a  a0  a1
            1   2
        >>> sess2.arr2
        a  a0  a1  a2
            1   2   3

        You may also pass extra arguments or keyword arguments to the function

        >>> def change(array, increment=1, multiplier=1):
        ...     return (array + increment) * multiplier
        >>> sess2 = sess1.apply(change, 2, 2)
        >>> sess2 = sess1.apply(change, 2, multiplier=2)
        >>> sess2.arr1
        a  a0  a1
            4   6
        >>> sess2.arr2
        a  a0  a1  a2
            4   6   8
        """
        return Session({k: func(v, *args, **kwargs) if isinstance(v, kind) else v for k, v in self.items()})

    def summary(self, template=None) -> str:
        r"""
        Return a summary of the content of the session.

        Parameters
        ----------
        template: dict {object type: str} or dict {object type: func}
            Template describing how items and metadata are summarized.
            For each object type, it is possible to provide either a string template or a function taking the
            the key and value of a session item as parameters and returning a string (see examples).
            A string template contains specific arguments written inside brackets {}.
            Available arguments are:

                - for groups: 'key', 'name', 'axis_name', 'labels' and 'length',
                - for axes: 'key', 'name', 'labels' and 'length',
                - for arrays: 'key', 'axes_names', 'shape', 'dtype' and 'title',
                - for session metadata: 'key', 'value',
                - for all other types: 'key', 'value'.

        Returns
        -------
        str
            Short representation of the content of the session.

        Examples
        --------
        >>> axis1 = Axis("a=a0..a2")
        >>> group1 = axis1['a0,a1'] >> 'a01'
        >>> arr1 = ndtest((2, 2), dtype=np.int64, meta={'title': 'array 1'})
        >>> arr2 = ndtest(4, dtype=np.int64, meta={'title': 'array 2'})
        >>> arr3 = ndtest((3, 2), dtype=np.int64, meta={'title': 'array 3'})
        >>> s = Session({'axis1': axis1, 'group1': group1, 'arr1': arr1, 'arr2': arr2, 'arr3': arr3})
        >>> s.meta.title = 'my title'
        >>> s.meta.author = 'John Smith'

        Default template

        >>> print(s.summary())  # doctest: +NORMALIZE_WHITESPACE
        Metadata:
            title: my title
            author: John Smith
        axis1: a ['a0' 'a1' 'a2'] (3)
        group1: a['a0', 'a1'] >> a01 (2)
        arr1: a, b (2 x 2) [int64]
        arr2: a (4) [int64]
        arr3: a, b (3 x 2) [int64]

        Using a specific template

        >>> def print_array(key, array):
        ...     axes_names = ', '.join(array.axes.display_names)
        ...     shape = ' x '.join(str(i) for i in array.shape)
        ...     return f"{key} -> {axes_names} ({shape})\n  title = {array.meta.title}\n  dtype = {array.dtype}"
        >>> template = {Axis:  "{key} -> {name} [{labels}] ({length})",
        ...             Group: "{key} -> {name}: {axis_name}{labels} ({length})",
        ...             Array: print_array,
        ...             Metadata: "\t{key} -> {value}"}
        >>> print(s.summary(template))   # doctest: +NORMALIZE_WHITESPACE
        Metadata:
            title -> my title
            author -> John Smith
        axis1 -> a ['a0' 'a1' 'a2'] (3)
        group1 -> a01: a['a0', 'a1'] (2)
        arr1 -> a, b (2 x 2)
          title = array 1
          dtype = int64
        arr2 -> a (4)
          title = array 2
          dtype = int64
        arr3 -> a, b (3 x 2)
          title = array 3
          dtype = int64
        """
        if template is None:
            template = {}
        if Axis not in template:
            template[Axis] = "{key}: {name} [{labels}] ({length})"
        if Group not in template:
            template[Group] = "{key}: {axis_name}{labels} >> {name} ({length})"
        if Array not in template:
            template[Array] = "{key}: {axes_names} ({shape}) [{dtype}]"
        if Metadata not in template:
            template[Metadata] = "\t{key}: {value}"

        def display(k, v, is_metadata=False) -> str:
            if not is_metadata:
                t = Group if isinstance(v, Group) else type(v)
                tmpl = template.get(t, "{key}: {value}")
            else:
                tmpl = template[Metadata]
            if not (isinstance(tmpl, str) or callable(tmpl)):
                raise TypeError(f"Expected a string template or a function for type {type(v)}. Got {type(tmpl)}")
            if isinstance(tmpl, str):
                if isinstance(v, Axis):
                    return tmpl.format(key=k, name=v.name, labels=v.labels_summary(), length=len(v))
                elif isinstance(v, Group):
                    return tmpl.format(key=k, name=v.name, axis_name=v.axis.name, labels=v.key, length=len(v))
                elif isinstance(v, Array):
                    return tmpl.format(key=k, axes_names=', '.join(v.axes.display_names),
                                       shape=' x '.join(str(i) for i in v.shape), dtype=v.dtype)
                else:
                    return tmpl.format(key=k, value=str(v))
            else:
                return tmpl(k, v)

        res = ''
        if len(self.meta) > 0:
            res = 'Metadata:\n'
            res += '\n'.join(display(k, v, True) for k, v in self.meta.items()) + '\n'
        res += '\n'.join(display(k, v) for k, v in self.items())
        return res

    @property
    def nbytes(self) -> str:
        r"""Return the memory in bytes consumed by the session.

        Returns
        -------
        int
            Number of bytes of memory used by the session.

        Notes
        -----
        This returns a lower bound. Some memory is not accounted for.

        Examples
        --------
        >>> # specifying explicit type so that the test has the same result on Linux
        >>> s = Session(test1=ndtest('sex=M,F', dtype=np.int32),
        ...             test2=ndtest("age=0..100", dtype=np.int32))
        >>> s.nbytes
        412
        """
        return sum(v.nbytes for v in self.values() if hasattr(v, 'nbytes'))

    @property
    def memory_used(self) -> str:
        r"""Return the memory consumed by the session in human readable form.

        Returns
        -------
        str
            Memory used by the session.

        Examples
        --------
        >>> # specifying explicit type so that the test has the same result on Linux
        >>> s = Session(test1=ndtest('sex=M,F', dtype=np.int32),
        ...             test2=ndtest("age=0..100", dtype=np.int32))
        >>> s.memory_used
        '412 bytes'
        """
        return size2str(self.nbytes)


def _exclude_private_vars(vars_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in vars_dict.items() if not k.startswith('_')}


def local_arrays(depth=0, include_private=False, meta=None) -> Session:
    r"""
    Return a session containing all local arrays sorted in alphabetical order.

    Parameters
    ----------
    depth: int
        depth of call frame to inspect. 0 is where `local_arrays` was called, 1 the caller of `local_arrays`, etc.
    include_private: boolean, optional
        Whether to include private local arrays (i.e. arrays starting with `_`). Defaults to False.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Session
    """
    # noinspection PyProtectedMember
    d = sys._getframe(depth + 1).f_locals
    if not include_private:
        d = _exclude_private_vars(d)
    return Session({k: d[k] for k in sorted(d.keys()) if isinstance(d[k], Array)}, meta=meta)


def global_arrays(depth=0, include_private=False, meta=None) -> Session:
    r"""
    Return a session containing all global arrays sorted in alphabetical order.

    Parameters
    ----------
    depth: int
        depth of call frame to inspect. 0 is where `global_arrays` was called, 1 the caller of `global_arrays`, etc.
    include_private: boolean, optional
        Whether to include private globals arrays (i.e. arrays starting with `_`). Defaults to False.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Session
    """
    # noinspection PyProtectedMember
    d = sys._getframe(depth + 1).f_globals
    if not include_private:
        d = _exclude_private_vars(d)
    return Session({k: d[k] for k in sorted(d.keys()) if isinstance(d[k], Array)}, meta=meta)


def arrays(depth=0, include_private=False, meta=None) -> Session:
    r"""
    Return a session containing all available arrays (whether they are defined in local or global variables) sorted in
    alphabetical order. Local arrays take precedence over global ones (if a name corresponds to both a local
    and a global variable, the local array will be returned).

    Parameters
    ----------
    depth: int
        depth of call frame to inspect. 0 is where `arrays` was called, 1 the caller of `arrays`, etc.
    include_private: boolean, optional
        Whether to include private arrays (i.e. arrays starting with `_`). Defaults to False.
    meta : list of pairs or dict or Metadata, optional
        Metadata (title, description, author, creation_date, ...) associated with the array.
        Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Returns
    -------
    Session
    """
    # noinspection PyProtectedMember
    caller_frame = sys._getframe(depth + 1)
    global_vars = caller_frame.f_globals
    local_vars = caller_frame.f_locals

    if not include_private:
        global_vars = _exclude_private_vars(global_vars)
        local_vars = _exclude_private_vars(local_vars)

    # We must first get all variables *then* filter by type, otherwise we could return a global array which is not
    # currently available because it is shadowed by a local non-array variable.
    all_keys = sorted(set(global_vars.keys()) | set(local_vars.keys()))
    combined_vars = [(k, local_vars[k] if k in local_vars else global_vars[k])
                     for k in all_keys]
    return Session({k: v for k, v in combined_vars if isinstance(v, Array)}, meta=meta)


_session_float_error_handler = float_error_handler_factory(4)
