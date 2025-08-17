import os
import atexit
from pathlib import Path

import numpy as np
try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.core.array import Array, ndtest             # noqa: F401
from larray.core.axis import Axis
from larray.core.constants import nan
from larray.core.group import _translate_sheet_name
from larray.inout.pandas import df_asarray
from larray.inout.misc import from_lists
from larray.util.misc import deprecate_kwarg


string_types = (str,)


if xw is not None:
    from xlwings.conversion.pandas_conv import PandasDataFrameConverter

    from xlwings.constants import FileFormat
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
                    global_app.quit()
                except Exception:
                    pass
            del global_app
            global_app = None

    class ArrayConverter(PandasDataFrameConverter):
        writes_types = Array

        @classmethod
        def read_value(cls, value, options):
            df = PandasDataFrameConverter.read_value(value, options)
            return df_asarray(df)

        @classmethod
        def write_value(cls, value, options):
            df = value.to_frame(fold_last_axis_name=True)
            return PandasDataFrameConverter.write_value(df, options)

    ArrayConverter.register(Array)

    def _disable_screen_updates(app):
        xl_app = app.api
        xl_app.ScreenUpdating = False
        xl_app.DisplayStatusBar = False
        # this makes our test suite freeze
        # app.calculation = "manual"
        # unsure we can safely do this
        # xl_app.EnableEvents = False

    class Workbook:
        # TODO: replace overwrite_file by mode='r'|'w'|'a' the day xlwings will support a read-only mode
        def __init__(self, filepath=None, overwrite_file=False, visible=None, silent=None, app=None, load_addins=None):
            global global_app

            xw_wkb = None
            self.delayed_filepath = None
            self.filepath = None
            self.new_workbook = False
            self.active_workbook = filepath == -1

            if filepath is None:
                self.new_workbook = True

            if isinstance(filepath, str):
                filepath = Path(filepath)

            if isinstance(filepath, Path):
                suffix = filepath.suffix
                if suffix:
                    # XXX: we might want to be more precise than .xl* because I am unsure writing .xls
                    #     (or anything other than .xlsx and .xlsm) would work
                    if not suffix.startswith('.xl'):
                        raise ValueError(f"'{suffix}' is not a supported file extension")
                    if not filepath.is_file() and not overwrite_file:
                        raise ValueError(f"File {filepath} does not exist. Please give the path to an existing file "
                                         f"or set overwrite_file argument to True")
                    if filepath.is_file() and overwrite_file:
                        self.filepath = filepath
                        # we create a temporary file to work on. In case of crash, the original is not destroyed.
                        # the temporary file is renamed as the original file at close.
                        filepath = filepath.parent / (filepath.stem + '~' + filepath.suffix)
                    if not filepath.is_file():
                        self.new_workbook = True
                else:
                    # try to target an open but unsaved workbook. We cannot use the same code path as for other options
                    # because we do not know which Excel instance has that book
                    xw_wkb = xw.Book(filepath)
                    app = xw_wkb.app

            # active workbook use active app by default
            if self.active_workbook and app not in {None, "active"}:
                raise ValueError("to connect to the active workbook, one must use the 'active' Excel instance "
                                 "(app='active' or app=None)")

            # unless explicitly set, app is set to visible for brand new or active book.
            # For unsaved_book it is left intact.
            if visible is None:
                if filepath is None or self.active_workbook:
                    visible = True
                elif xw_wkb is None:
                    # filepath is not None and we target a real file (not an unsaved book)
                    visible = False

            if app is None:
                if self.active_workbook:
                    app = "active"
                elif visible:
                    app = "new"
                else:
                    app = "global"

            if load_addins is None:
                load_addins = visible and app == "new"

            if app == "new":
                app = xw.App(visible=visible, add_book=False)
                if not visible:
                    _disable_screen_updates(app)
            elif app == "active":
                app = xw.apps.active
            elif app == "global":
                if global_app is None:
                    atexit.register(kill_global_app)
                if global_app is None or not is_app_alive(global_app):
                    global_app = xw.App(visible=visible, add_book=False)
                    if not visible:
                        _disable_screen_updates(global_app)
                app = global_app
            assert isinstance(app, xw.App)

            # activate XLA(M) addins, if needed
            # By default, add-ins are not activated when an Excel Workbook is opened via COM
            if load_addins:
                xl_app = app.api
                for i in range(1, xl_app.AddIns.Count + 1):
                    addin = xl_app.AddIns(i)
                    addin_path = addin.FullName
                    if addin.Installed and '.xll' not in addin_path.lower():
                        xl_app.Workbooks.Open(addin_path)

            if visible:
                app.visible = visible

            if silent is None:
                silent = not visible

            update_links_backup = app.api.AskToUpdateLinks
            display_alerts_backup = app.display_alerts
            if silent:
                # try to update links silently instead of asking: "Update", "Don't Update", "Help"
                app.api.AskToUpdateLinks = False

                # in case some links cannot be updated, continue instead of asking: "Continue" or "Edit Links..."
                app.display_alerts = False

            if filepath is None:
                # creates a new/blank Book
                xw_wkb = app.books.add()
            elif self.active_workbook:
                xw_wkb = app.books.active
            elif xw_wkb is None:
                # file already exists (and is a file)
                if filepath.is_file():
                    xw_wkb = app.books.open(filepath)
                else:
                    # let us remember the path
                    self.delayed_filepath = filepath
                    xw_wkb = app.books.add()

            if silent:
                app.api.AskToUpdateLinks = update_links_backup
                app.display_alerts = display_alerts_backup

            self.xw_wkb = xw_wkb

        @property
        def app(self):
            return self.xw_wkb.app

        def __contains__(self, key):
            if isinstance(key, int):
                length = len(self)
                return -length <= key < length
            else:
                # I would like to use: "return key in wb.sheets" but as of xlwings 0.10 wb.sheets.__contains__ does not
                # work for sheet names (it works with Sheet objects I think)
                return key in self.sheet_names()

        def _ipython_key_completions_(self):
            return list(self.sheet_names())

        def __getitem__(self, key):
            key = _translate_sheet_name(key)
            if key in self:
                return Sheet(self, key)
            else:
                raise KeyError(f'Workbook has no sheet named {key}')

        def __setitem__(self, key, value):
            key = _translate_sheet_name(key)
            if self.new_workbook:
                if isinstance(key, str):
                    self.xw_wkb.sheets[0].name = key
                self.new_workbook = False
            key_in_self = key in self
            if isinstance(value, Sheet):
                if value.xw_sheet.book.app != self.app:
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

                # rename the new sheet if necessary
                # if the key is an integer, we keep the name of the source sheet
                if isinstance(key, str):
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

        def save(self, path=None, password=None):
            r"""Save Workbook to file.

            Parameters
            ----------
            path : str or Path, optional
                Path to save the file to. Defaults to None (use the path used when opening the workbook).
            password : str, optional
                Password to protect the file. Defaults to None (no password).
            """
            # saved_path = self.xw_wkb.api.Path
            # was_saved = saved_path != ''
            if path is None and self.delayed_filepath is not None:
                path = self.delayed_filepath

            if password is not None:
                if path is None:
                    raise ValueError("saving a Workbook with a password is only supported for workbooks with an "
                                     "explicit path (given either when opening the workbook or here as the path "
                                     "argument)")
                realpath = os.path.realpath(path)
                # XXX: this is probably Windows only
                # using Password as keyword argument does not work !
                self.xw_wkb.api.SaveAs(realpath, FileFormat.xlOpenXMLWorkbook, password)
            else:
                self.xw_wkb.save(path=path)

        def close(self):
            # Close the workbook in Excel.
            # This will not quit the Excel instance, even if this was the last workbook of that Excel instance.
            if self.filepath is not None and os.path.isfile(self.xw_wkb.fullname):
                tmp_file = self.xw_wkb.fullname
                self.xw_wkb.close()
                # XXX: do we check for this case earlier and act differently depending on overwrite?
                os.remove(self.filepath)
                os.rename(tmp_file, self.filepath)
            else:
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
            # XXX: we should probably also avoid closing the workbook for visible=True???
            # XXX: we might want to disallow using open_excel as a context manager (in __enter__)
            #      when we have nothing to do in close because it is kinda misleading (this might piss off
            #      users though, so maybe a warning would be better).
            if not self.active_workbook:
                self.close()

        def __repr__(self):
            cls = self.__class__
            return f'<{cls.__module__}.{cls.__name__} [{self.name}]>'

    def _fill_slice(s, length):
        r"""
        Replace slice None bounds by actual bounds.

        Parameters
        ----------
        s : slice
            slice to replace
        length : int
            length of sequence

        Returns
        -------
        slice
        """
        return slice(s.start if s.start is not None else 0, s.stop if s.stop is not None else length, s.step)

    def _concrete_key(key, obj, ndim=2):
        r"""Expand key to ndim and replace None in slices start/stop bounds by 0 or obj.shape[corresponding_dim]
        respectively.

        Parameters
        ----------
        key : scalar, slice or tuple
            input key
        obj : object
            any object with a 'shape' attribute.
        ndim : int
            number of dimensions to expand to. We could use len(obj.shape) instead but we avoid it to not trigger
            obj.shape, which can be expensive in the case of a sheet with blank cells after the data.
        """
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) < ndim:
            key = key + (slice(None),) * (ndim - len(key))

        # only compute shape if necessary because it can be expensive in some cases
        if any(isinstance(k, slice) and k.stop is None for k in key):
            shape = obj.shape
        else:
            shape = (None, None)

        # We use _fill_slice instead of slice(*k.indices(length)) because the later also clips bounds which exceed
        # the length and we do NOT want to do that in this case (see issue #273).
        return [_fill_slice(k, length) if isinstance(k, slice) else k
                for k, length in zip(key, shape)]

    class Sheet:
        def __init__(self, workbook, key, xw_sheet=None):
            if xw_sheet is None:
                xw_sheet = workbook.xw_wkb.sheets[key]
            object.__setattr__(self, 'xw_sheet', xw_sheet)

        # TODO: we can probably scrap this for xlwings 0.9+. We need to have
        #       a unit test for this though.
        def __getitem__(self, key):
            if isinstance(key, string_types):
                return Range(self, key)

            row, col = _concrete_key(key, self)
            if isinstance(row, slice) or isinstance(col, slice):
                row1, row2 = (row.start, row.stop) if isinstance(row, slice) else (row, row + 1)
                col1, col2 = (col.start, col.stop) if isinstance(col, slice) else (col, col + 1)
                return Range(self, (row1 + 1, col1 + 1), (row2, col2))
            else:
                return Range(self, (row + 1, col + 1))

        def __setitem__(self, key, value):
            if isinstance(value, Array):
                value = value.dump(header=False)
            self[key].xw_range.value = value

        @property
        def shape(self):
            r"""
            shape of sheet including top-left empty rows/columns but excluding bottom-right ones.
            """
            from xlwings.constants import Direction as xldir

            sheet = self.xw_sheet.api
            used = sheet.UsedRange
            first_row = used.Row
            first_col = used.Column
            last_row = first_row + used.Rows.Count - 1
            last_col = first_col + used.Columns.Count - 1
            last_cell = sheet.Cells(last_row, last_col)

            # fast path for sheets with a non blank bottom-right value
            if last_cell.Value is not None:
                return last_row, last_col

            last_row_used = last_cell.End(xldir.xlToLeft).Value is not None
            last_col_used = last_cell.End(xldir.xlUp).Value is not None

            # fast path for sheets where last row and last col are not entirely blank
            if last_row_used and last_col_used:
                return last_row, last_col
            else:
                LEFT, UP = xldir.xlToLeft, xldir.xlUp

                def line_length(row, col, direction):
                    last_cell = sheet.Cells(row, col)
                    if last_cell.Value is not None:
                        return col if direction is LEFT else row
                    first_cell = last_cell.End(direction)
                    pos = first_cell.Column if direction is LEFT else first_cell.Row
                    return pos - 1 if first_cell.Value is None else pos

                if last_row < last_col:
                    if last_row_used or last_row == 1:
                        max_row = last_row
                    else:
                        for max_row in range(last_row - 1, first_row - 1, -1):
                            if line_length(max_row, last_col, LEFT) > 0:
                                break
                    if last_col_used or last_col == 1:
                        max_col = last_col
                    else:
                        max_col = max(line_length(row, last_col, LEFT) for row in range(first_row, max_row + 1))
                else:
                    if last_col_used or last_col == 1:
                        max_col = last_col
                    else:
                        for max_col in range(last_col - 1, first_col - 1, -1):
                            if line_length(last_row, max_col, UP) > 0:
                                break
                    if last_row_used or last_row == 1:
                        max_row = last_row
                    else:
                        max_row = max(line_length(last_row, col, UP) for col in range(first_col, max_col + 1))
                return max_row, max_col

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

        @deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
        def load(self, header=True, convert_float=True, nb_axes=None, index_col=None, fill_value=nan,
                 sort_rows=False, sort_columns=False, wide=True):
            return self[:].load(header=header, convert_float=convert_float, nb_axes=nb_axes, index_col=index_col,
                                fill_value=fill_value, sort_rows=sort_rows, sort_columns=sort_columns, wide=wide)

        # TODO: generalize to more than 2 dimensions or scrap it
        def array(self, data, row_labels=None, column_labels=None, names=None):
            r"""

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
            Array
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
            return Array(np.asarray(self[data]), axes)

        def __repr__(self):
            cls = self.__class__
            xw_sheet = self.xw_sheet
            return f'<{cls.__module__}.{cls.__name__} [{xw_sheet.book.name}]{xw_sheet.name}>'

    class Range:
        def __init__(self, sheet, *args):
            xw_range = sheet.xw_sheet.range(*args)

            object.__setattr__(self, 'sheet', sheet)
            object.__setattr__(self, 'xw_range', xw_range)

        def _range_key_to_sheet_key(self, key):
            # string keys does not make sense in this case
            assert not isinstance(key, string_types)
            row_offset = self.xw_range.row - 1
            col_offset = self.xw_range.column - 1
            row, col = _concrete_key(key, self.xw_range)
            row = slice(row.start + row_offset, row.stop + row_offset) if isinstance(row, slice) else row + row_offset
            col = slice(col.start + col_offset, col.stop + col_offset) if isinstance(col, slice) else col + col_offset
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

        def __larray__(self):
            return Array(self._converted_value())

        def __dir__(self):
            return list(set(dir(self.__class__)) | set(dir(self.xw_range)))

        def __getattr__(self, key):
            if hasattr(Array, key):
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

        @deprecate_kwarg('nb_index', 'nb_axes', arg_converter=lambda x: x + 1)
        def load(self, header=True, convert_float=True, nb_axes=None, index_col=None, fill_value=nan,
                 sort_rows=False, sort_columns=False, wide=True):
            if not self.ndim:
                return Array([])

            list_data = self._converted_value(convert_float=convert_float)

            if header:
                return from_lists(list_data, nb_axes=nb_axes, index_col=index_col, fill_value=fill_value,
                                  sort_rows=sort_rows, sort_columns=sort_columns, wide=wide)
            else:
                return Array(list_data)

    # XXX: deprecate this function?
    def open_excel(filepath=None, overwrite_file=False, visible=None, silent=None, app=None, load_addins=None):
        return Workbook(filepath, overwrite_file=overwrite_file, visible=visible, silent=silent, app=app,
                        load_addins=load_addins)
else:
    class Workbook:
        def __init__(self, filepath=None, overwrite_file=False, visible=None, silent=None, app=None, load_addins=None):
            raise Exception("Workbook class cannot be instantiated because xlwings is not installed")

        def app(self):
            raise Exception()

        def sheet_names(self):
            raise Exception()

        def save(self, path=None):
            raise Exception()

        def close(self):
            raise Exception()

    def open_excel(filepath=None, overwrite_file=False, visible=None, silent=None, app=None, load_addins=None):
        raise Exception("open_excel() is not available because xlwings is not installed")


# We define Workbook and open_excel documentation here since Readthedocs runs on Linux
Workbook.__doc__ = r"""
Excel Workbook.

See Also
--------
open_excel
"""

Workbook.sheet_names.__doc__ = r"""
Return the names of the Excel sheets.

Examples
--------
>>> arr, arr2, arr3 = ndtest((3, 3)), ndtest((2, 2)), ndtest(4)
>>> with open_excel('excel_file.xlsx', overwrite_file=True) as wb:   # doctest: +SKIP
...     wb['arr'] = arr.dump()
...     wb['arr2'] = arr2.dump()
...     wb['arr3'] = arr3.dump()
...     wb.save()
...
...     wb.sheet_names()
['arr', 'arr2', 'arr3']
"""

Workbook.save.__doc__ = r"""
Saves the Workbook.

If a path is being provided, this works like SaveAs() in Excel.
If no path is specified and if the file hasn't been saved previously,
it's being saved in the current working directory with the current filename.
Existing files are overwritten without prompting.

Parameters
----------
path : str or Path, optional
    Full path to the workbook. Defaults to None.

Examples
--------
>>> arr, arr2, arr3 = ndtest((3, 3)), ndtest((2, 2)), ndtest(4)
>>> with open_excel('excel_file.xlsx', overwrite_file=True) as wb:   # doctest: +SKIP
...     wb['arr'] = arr.dump()
...     wb['arr2'] = arr2.dump()
...     wb['arr3'] = arr3.dump()
...     wb.save()
"""

Workbook.close.__doc__ = r"""
Close the workbook in Excel.

Need to be called if the workbook has been opened without the `with` statement.

Examples
--------
>>> arr, arr2, arr3 = ndtest((3, 3)), ndtest((2, 2)), ndtest(4)   # doctest: +SKIP
>>> wb = open_excel('excel_file.xlsx', overwrite_file=True)       # doctest: +SKIP
>>> wb['arr'] = arr.dump()                                        # doctest: +SKIP
>>> wb['arr2'] = arr2.dump()                                      # doctest: +SKIP
>>> wb['arr3'] = arr3.dump()                                      # doctest: +SKIP
>>> wb.save()                                                     # doctest: +SKIP
>>> wb.close()                                                    # doctest: +SKIP
"""

Workbook.app.__doc__ = r"""
Return the Excel instance this workbook is attached to.
"""

open_excel.__doc__ = r"""
Open an Excel workbook

Parameters
----------
filepath : None, int, str or Path, optional
    path to the Excel file. The file must exist if overwrite_file is False. Use None for a new blank workbook,
    -1 for the currently active workbook. Defaults to None.
overwrite_file : bool, optional
    whether to overwrite an existing file, if any. Defaults to False.
visible : None or bool, optional
    whether Excel should be visible. Defaults to False for files, True for new/active workbooks and to None
    ("unchanged") for existing unsaved workbooks.
silent : None or bool, optional
    whether to show dialog boxes for updating links or when some links cannot be updated.
    Defaults to False if visible, True otherwise.
app : None, "new", "active", "global" or xlwings.App, optional
    use "new" for opening a new Excel instance, "active" for the last active instance (including ones opened by the
    user) and "global" to (re)use the same instance for all workbooks of a program. None is equivalent to "active" if
    filepath is -1, "new" if visible is True and "global" otherwise. Defaults to None.

    The "global" instance is a specific Excel instance for all input from/output to Excel from within a single Python
    program (and should not interact with instances manually opened by the user or another program).
load_addins : None or bool, optional
    whether to load Excel addins. Defaults to True if visible and app == "new", False otherwise.

Returns
-------
Excel workbook.

Examples
--------
>>> arr = ndtest((3, 3))
>>> arr
a\b  b0  b1  b2
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
a\b  b0  b1  b2
 a0   0   1   2
 a1   3   4   5
 a2   6   7   8
"""
