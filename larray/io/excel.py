import os
import atexit

import numpy as np
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.core.axis import Axis
from larray.core.array import LArray
from larray.io.array import df_aslarray, from_lists

string_types = (str,)


if xw is not None:
    from xlwings.conversion.pandas_conv import PandasDataFrameConverter

    global_app = None


    def is_app_alive(app):
        try:
            app.books
            return True
        except Exception:
            return False


    def kill_global_app():
        global global_app

        if global_app is not None:
            if is_app_alive(global_app):
                try:
                    global_app.kill()
                except Exception:
                    pass
            del global_app
            global_app = None


    class LArrayConverter(PandasDataFrameConverter):
        writes_types = LArray

        @classmethod
        def read_value(cls, value, options):
            df = PandasDataFrameConverter.read_value(value, options)
            return df_aslarray(df)

        @classmethod
        def write_value(cls, value, options):
            df = value.to_frame(fold_last_axis_name=True)
            return PandasDataFrameConverter.write_value(df, options)

    LArrayConverter.register(LArray)


    # TODO : replace overwrite_file by mode='r'|'w'|'a' the day xlwings will support a read-only mode
    class Workbook(object):
        def __init__(self, filepath=None, overwrite_file=False, visible=None, silent=None, app=None):
            """See open_excel doc for parameters"""
            global global_app

            xw_wkb = None
            self.delayed_filepath = None
            self.new_workbook = False

            if filepath is None:
                self.new_workbook = True

            if isinstance(filepath, str):
                basename, ext = os.path.splitext(filepath)
                if ext:
                    # XXX: we might want to be more precise than .xl* because
                    #      I am unsure writing .xls (or anything other than
                    #      .xlsx and .xlsm) would work
                    if not ext.startswith('.xl'):
                        raise ValueError("'%s' is not a supported file "
                                         "extension" % ext)
                    if not os.path.isfile(filepath) and not overwrite_file:
                        raise ValueError("File {} does not exist. Please give the path to an existing file "
                                         "or set overwrite_file argument to True".format(filepath))
                    if os.path.isfile(filepath) and overwrite_file:
                        os.remove(filepath)
                    if not os.path.isfile(filepath):
                        self.new_workbook = True
                else:
                    # try to target an open but unsaved workbook. We cannot use the same code path as for other options
                    # because we do not know which Excel instance has that book
                    xw_wkb = xw.Book(filepath)
                    app = xw_wkb.app

            if app is None:
                app = "active" if filepath == -1 else "global"

            # active workbook use active app by default
            if filepath == -1:
                if app != "active":
                    raise ValueError("to connect to the active workbook, one must use the 'active' Excel instance "
                                     "(app='active' or app=None)")

            # unless explicitly set, app is set to visible for brand new or
            # active book. For unsaved_book it is left intact.
            if visible is None:
                if filepath is None or filepath == -1:
                    visible = True
                elif xw_wkb is None:
                    # filepath is not None but we don't target an unsaved book
                    visible = False

            if app == "new":
                app = xw.App(visible=visible, add_book=False)
            elif app == "active":
                app = xw.apps.active
            elif app == "global":
                if global_app is None:
                    atexit.register(kill_global_app)
                if global_app is None or not is_app_alive(global_app):
                    global_app = xw.App(visible=visible, add_book=False)
                app = global_app
            assert isinstance(app, xw.App)

            if visible:
                app.visible = visible

            if silent is None:
                silent = not visible

            update_links_backup = app.api.AskToUpdateLinks
            display_alerts_backup = app.display_alerts
            if silent:
                # try to update links silently instead of asking:
                # "Update", "Don't Update", "Help"
                app.api.AskToUpdateLinks = False

                # in case some links cannot be updated, continue instead of
                # asking: "Continue" or "Edit Links..."
                app.display_alerts = False

            if filepath is None:
                # creates a new/blank Book
                xw_wkb = app.books.add()
            elif filepath == -1:
                xw_wkb = app.books.active
            elif xw_wkb is None:
                # file already exists (and is a file)
                if os.path.isfile(filepath):
                    xw_wkb = app.books.open(filepath)
                else:
                    # let us remember the path
                    self.delayed_filepath = filepath
                    xw_wkb = app.books.add()

            if silent:
                app.api.AskToUpdateLinks = update_links_backup
                app.display_alerts = display_alerts_backup

            self.xw_wkb = xw_wkb

        def __contains__(self, key):
            if isinstance(key, int):
                length = len(self)
                return -length <= key < length
            else:
                # I would like to use:
                # return key in wb.sheets
                # but as of xlwings 0.10 wb.sheets.__contains__ does not work
                # for sheet names (it works with Sheet objects I think)
                return key in self.sheet_names()

        def _ipython_key_completions_(self):
            return list(self.sheet_names())

        def __getitem__(self, key):
            if key in self:
                return Sheet(self, key)
            else:
                raise KeyError('Workbook has no sheet named {}'.format(key))

        def __setitem__(self, key, value):
            if self.new_workbook:
                self.xw_wkb.sheets[0].name = key
                self.new_workbook = False
            key_in_self = key in self
            if isinstance(value, Sheet):
                if value.xw_sheet.book.app != self.xw_wkb.app:
                    raise ValueError("cannot copy a sheet from one instance of Excel to another")

                # xlwings index is 1-based
                # TODO: implement Workbook.index(key)
                target_idx = self[key].xw_sheet.index - 1 if key_in_self else -1
                target_sheet = self[target_idx].xw_sheet
                # add new sheet after target sheet. The new sheet will be named something like "value.name (1)" but I
                # do not think there is anything we can do about this, except rename it afterwards because Copy has no
                # name argument. See https://msdn.microsoft.com/en-us/library/office/ff837784.aspx
                value.xw_sheet.api.Copy(None, target_sheet.api)
                if key_in_self:
                    target_sheet.delete()
                # rename the new sheet
                self[target_idx].name = key
                return
            if key_in_self:
                sheet = self[key]
                sheet.clear()
            else:
                xw_sheet = self.xw_wkb.sheets.add(key, after=self[-1].xw_sheet)
                sheet = Sheet(None, None, xw_sheet=xw_sheet)
            sheet["A1"] = value

        def __delitem__(self, key):
            self[key].delete()

        def sheet_names(self):
            return [s.name for s in self]

        def save(self, path=None):
            # saved_path = self.xw_wkb.api.Path
            # was_saved = saved_path != ''
            if path is None and self.delayed_filepath is not None:
                path = self.delayed_filepath
            self.xw_wkb.save(path=path)

        def close(self):
            """
            Close the workbook in Excel. This will not quit the Excel instance, even if this was the last workbook of
            that Excel instance.
            """
            self.xw_wkb.close()

        def __iter__(self):
            return iter([Sheet(None, None, xw_sheet)
                         for xw_sheet in self.xw_wkb.sheets])

        def __len__(self):
            return len(self.xw_wkb.sheets)

        def __dir__(self):
            return list(set(dir(self.__class__)) | set(dir(self.xw_wkb)))

        def __getattr__(self, key):
            return getattr(self.xw_wkb, key)

        def __enter__(self):
            return self

        def __exit__(self, type_, value, traceback):
            self.close()

        def __repr__(self):
            cls = self.__class__
            return '<{}.{} [{}]>'.format(cls.__module__, cls.__name__, self.name)


    def _fill_slice(s, length):
        """
        replaces a slice None bounds by actual bounds.

        Parameters
        ----------
        k : slice
            slice to replace
        length : int
            length of sequence

        Returns
        -------
        slice
        """
        return slice(s.start if s.start is not None else 0, s.stop if s.stop is not None else length, s.step)


    def _concrete_key(key, shape):
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) < len(shape):
            key = key + (slice(None),) * (len(shape) - len(key))

        # We do not use slice(*k.indices(length)) because it also clips bounds which exceed the length, which we do not
        # want in this case (see issue #273).
        return [_fill_slice(k, length) if isinstance(k, slice) else k
                for k, length in zip(key, shape)]


    class Sheet(object):
        def __init__(self, workbook, key, xw_sheet=None):
            if xw_sheet is None:
                xw_sheet = workbook.xw_wkb.sheets[key]
            object.__setattr__(self, 'xw_sheet', xw_sheet)

        # TODO: we can probably scrap this for xlwings 0.9+. We need to have
        #       a unit test for this though.
        def __getitem__(self, key):
            if isinstance(key, string_types):
                return Range(self, key)

            row, col = _concrete_key(key, self.shape)
            if isinstance(row, slice) or isinstance(col, slice):
                row1, row2 = (row.start, row.stop) \
                    if isinstance(row, slice) else (row, row + 1)
                col1, col2 = (col.start, col.stop) \
                    if isinstance(col, slice) else (col, col + 1)
                return Range(self, (row1 + 1, col1 + 1), (row2, col2))
            else:
                return Range(self, (row + 1, col + 1))

        def __setitem__(self, key, value):
            if isinstance(value, LArray):
                value = value.dump(header=False)
            self[key].xw_range.value = value

        @property
        def shape(self):
            # include top-left empty rows/columns
            # XXX: is there an exposed xlwings API for this? expand maybe?
            used = self.xw_sheet.api.UsedRange
            return (used.Row + used.Rows.Count - 1,
                    used.Column + used.Columns.Count - 1)

        @property
        def ndim(self):
            return 2

        def __array__(self, dtype=None):
            return np.asarray(self[:], dtype=dtype)

        def __dir__(self):
            return list(set(dir(self.__class__)) | set(dir(self.xw_sheet)))

        def __getattr__(self, key):
            return getattr(self.xw_sheet, key)

        def __setattr__(self, key, value):
            setattr(self.xw_sheet, key, value)

        def load(self, header=True, convert_float=True, nb_index=None, index_col=None):
            return self[:].load(header=header, convert_float=convert_float, nb_index=nb_index, index_col=index_col)

        # TODO: generalize to more than 2 dimensions or scrap it
        def array(self, data, row_labels=None, column_labels=None, names=None):
            """

            Parameters
            ----------
            data : str
                range for data
            row_labels : str, optional
                range for row labels
            column_labels : str, optional
                range for column labels
            names : list of str, optional

            Returns
            -------
            LArray
            """
            if row_labels is not None:
                row_labels = np.asarray(self[row_labels])
            if column_labels is not None:
                column_labels = np.asarray(self[column_labels])
            if names is not None:
                labels = (row_labels, column_labels)
                axes = [Axis(axis_labels, name) for axis_labels, name in zip(labels, names)]
            else:
                axes = (row_labels, column_labels)
            # _converted_value is used implicitly via Range.__array__
            return LArray(np.asarray(self[data]), axes)

        def __repr__(self):
            cls = self.__class__
            xw_sheet = self.xw_sheet
            return '<{}.{} [{}]{}>'.format(cls.__module__, cls.__name__, xw_sheet.book.name, xw_sheet.name)


    class Range(object):
        def __init__(self, sheet, *args):
            xw_range = sheet.xw_sheet.range(*args)

            object.__setattr__(self, 'sheet', sheet)
            object.__setattr__(self, 'xw_range', xw_range)

        def _range_key_to_sheet_key(self, key):
            # string keys does not make sense in this case
            assert not isinstance(key, string_types)
            row_offset = self.xw_range.row1 - 1
            col_offset = self.xw_range.col1 - 1
            row, col = _concrete_key(key, self.xw_range.shape)
            row = slice(row.start + row_offset, row.stop + row_offset) \
                if isinstance(row, slice) else row + row_offset
            col = slice(col.start + col_offset, col.stop + col_offset) \
                if isinstance(col, slice) else col + col_offset
            return row, col

        # TODO: we can probably scrap this for xlwings 0.9+. We need to have
        #       a unit test for this though.
        def __getitem__(self, key):
            return self.sheet[self._range_key_to_sheet_key(key)]

        def __setitem__(self, key, value):
            self.sheet[self._range_key_to_sheet_key(key)] = value

        def _converted_value(self, convert_float=True):
            list_data = self.xw_range.value

            # As of version 0.7.2 of xlwings, there is no built-in converter for
            # this. The builtin .options(numbers=int) converter converts all
            # values to int, whether that would loose information or not, but
            # this is not what we want.
            if convert_float:
                # Excel 'numbers' are always floats
                def convert(value):
                    if isinstance(value, float):
                        int_val = int(value)
                        if int_val == value:
                            return int_val
                        return value
                    elif isinstance(value, list):
                        return [convert(v) for v in value]
                    else:
                        return value
                return convert(list_data)
            return list_data

        def __float__(self):
            # no need to use _converted_value because we will convert back to a float anyway
            return float(self.xw_range.value)

        def __int__(self):
            # no need to use _converted_value because we will convert to an int anyway
            return int(self.xw_range.value)

        def __index__(self):
            v = self._converted_value()
            if hasattr(v, '__index__'):
                return v.__index__()
            else:
                raise TypeError("only integer scalars can be converted to a scalar index")

        def __array__(self, dtype=None):
            return np.array(self._converted_value(), dtype=dtype)

        def __dir__(self):
            return list(set(dir(self.__class__)) | set(dir(self.xw_range)))

        def __getattr__(self, key):
            if hasattr(LArray, key):
                return getattr(self.__larray__(), key)
            else:
                return getattr(self.xw_range, key)

        def __setattr__(self, key, value):
            setattr(self.xw_range, key, value)

        # TODO: implement all binops
        # def __mul__(self, other):
        #     return self.__larray__() * other

        def __str__(self):
            return str(self.__larray__())
        __repr__ = __str__

        def load(self, header=True, convert_float=True, nb_index=None, index_col=None):
            if not self.ndim:
                return LArray([])

            list_data = self._converted_value(convert_float=convert_float)

            if header:
                return from_lists(list_data, nb_index=nb_index, index_col=index_col)
            else:
                return LArray(list_data)

    # XXX: remove this function?
    def open_excel(filepath=None, overwrite_file=False, visible=None, silent=None, app=None):
        return Workbook(filepath, overwrite_file, visible, silent, app)
else:
    def open_excel(filepath=None, overwrite_file=False, visible=None, silent=None, app=None):
        raise Exception("open_excel() is not available because xlwings "
                        "is not installed")

open_excel.__doc__ = \
"""
Open an Excel workbook

Parameters
----------
filepath : None, int or str, optional
    path to the Excel file. The file must exist if overwrite_file is False. 
    Use None for a new blank workbook, -1 for the last active
    workbook. Defaults to None.
overwrite_file : bool, optional
    whether or not to overwrite an existing file, if any.
    Defaults to False.
visible : None or bool, optional
    whether or not Excel should be visible. Defaults to False for
    files, True for new/active workbooks and to None ("unchanged")
    for existing unsaved workbooks.
silent : None or bool, optional
    whether or not to show dialog boxes for updating links or
    when some links cannot be updated. Defaults to False if
    visible, True otherwise.
app : None, "new", "active", "global" or xlwings.App, optional
    use "new" for opening a new Excel instance, "active" for the last active instance (including ones
    opened by the user) and "global" to (re)use the same instance for all workbooks of a program. None is
    equivalent to "active" if filepath is -1 and "global" otherwise. Defaults to None.

    The "global" instance is a specific Excel instance for all input from/output to Excel from within a
    single Python program (and should not interact with instances manually opened by the user or another
    program).
    
Returns
-------
Excel workbook.

Examples
--------
>>> from larray import *
>>> arr = ndtest((3, 3))
>>> arr
a\\b  b0  b1  b2
 a0   0   1   2
 a1   3   4   5
 a2   6   7   8

create a new Excel file and save an array

>>> # to create a new Excel file, argument overwrite_file must be set to True
>>> with open_excel('excel_file.xlsx', overwrite_file=True) as wb:   # doctest: +SKIP
...     wb['arr'] = arr.dump()
...     wb.save()

read array from an Excel file

>>> with open_excel('excel_file.xlsx') as wb:    # doctest: +SKIP
...     arr2 = wb['arr'].load()
>>> arr2    # doctest: +SKIP
a\\b  b0  b1  b2
 a0   0   1   2
 a1   3   4   5
 a2   6   7   8
"""
