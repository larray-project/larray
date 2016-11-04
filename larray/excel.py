import os

import numpy as np
try:
    import xlwings as xw
except ImportError:
    xw = None

from .core import LArray, df_aslarray, Axis
from .utils import unique, basestring

string_types = (str,)


if xw is not None:
    from xlwings.conversion.pandas_conv import PandasDataFrameConverter


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


    class Workbook(object):
        def __init__(self, filepath=None, overwrite_file=False, visible=None,
                     silent=None, app=None):
            """
            open an Excel workbook

            Parameters
            ----------
            filepath : None, int or str, optional
                use None for a new blank workbook, -1 for the last active
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
            app : xlwings.App instance, optional
                an existing app instance to reuse. Defaults to creating a new
                Excel instance.
            """
            xw_wkb = None
            self.delayed_filepath = None
            if isinstance(filepath, str):
                basename, ext = os.path.splitext(filepath)
                if ext:
                    # XXX: we might want to be more precise than .xl* because
                    #      I am unsure writing .xls (or anything other than
                    #      .xlsx and .xlsm) would work
                    if not ext.startswith('.xl'):
                        raise ValueError("'%s' is not a supported file "
                                         "extension" % ext)
                    if os.path.isfile(filepath) and overwrite_file:
                        os.remove(filepath)
                else:
                    # try to target an open but unsaved workbook
                    # we cant use the same code path as for other option
                    # because we don't know which Excel instance has that book
                    xw_wkb = xw.Book(filepath)
                    app = xw_wkb.app

            # active workbook use active app by default
            if filepath == -1 and app is None:
                app = -1

            # unless explicitly set, app is set to visible for brand new or
            # active book. For unsaved_book it is left intact.
            if visible is None:
                if filepath is None or filepath == -1:
                    visible = True
                elif xw_wkb is None:
                    # filepath is not None but we don't target an unsaved book
                    visible = False

            if app is None:
                app = xw.App(visible=visible, add_book=False)
            elif app == -1:
                app = xw.apps.active

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

        def _concrete_key(self, key):
            if isinstance(key, int):
                if key < 0:
                    key += len(self)
                key += 1
            return key

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

        def __getitem__(self, key):
            key = self._concrete_key(key)
            return Sheet(self, key)

        def __setitem__(self, key, value):
            if isinstance(value, Sheet):
                if key in self:
                    xw_sheet = self[key].xw_sheet
                    # avoid having the sheet name renamed to "name (1)"
                    xw_sheet.name = '__tmp__'
                    # add new sheet before sheet to overwrite
                    value.xw_sheet.api.Copy(xw_sheet.api)
                    xw_sheet.delete()
                else:
                    xw_sheet = self[-1]
                    value.xw_sheet.api.Copy(xw_sheet.api)
                return
            if key in self:
                sheet = self[key]
                sheet.clear()
            else:
                xw_sheet = self.xw_wkb.sheets.add(key)
                sheet = Sheet(None, None, xw_sheet=xw_sheet)
            sheet["A1"] = value

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
            Close the workbook in Excel. If this was the last workbook of
            that Excel instance, it also close the Excel instance.
            """
            app = self.xw_wkb.app
            self.xw_wkb.close()
            if not app.books:
                app.quit()

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


    def _concrete_key(key, shape):
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) < len(shape):
            key = key + (slice(None),) * (len(shape) - len(key))

        return [slice(*k.indices(length)) if isinstance(k, slice) else k
                for k, length in zip(key, shape)]


    class Sheet(object):
        def __init__(self, workbook, key, xw_sheet=None):
            if xw_sheet is None:
                xw_sheet = workbook.xw_wkb.sheets[key]
            self.xw_sheet = xw_sheet

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
            # FIXME: convert value like in Range, something like:
            # return np.array(self[:]._converted_value(), dtype=dtype)
            return np.array(self[:].value, dtype=dtype)

        def __dir__(self):
            return list(set(dir(self.__class__)) | set(dir(self.xw_sheet)))

        def __getattr__(self, key):
            return getattr(self.xw_sheet, key)

        def load(self, header=True, nb_index=0, index_col=None):
            return self[:].load(header=header, nb_index=nb_index,
                                index_col=index_col)

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
            # FIXME: use _convert_value
            if row_labels is not None:
                row_labels = np.array(self[row_labels].value)
            if column_labels is not None:
                column_labels = np.array(self[column_labels].value)
            if names is not None:
                labels = (row_labels, column_labels)
                axes = [Axis(name, axis_labels)
                        for name, axis_labels in zip(names, labels)]
            else:
                axes = (row_labels, column_labels)
            return LArray(self[data], axes)


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
                if self.ndim == 1:
                    list_data = [convert(v) for v in list_data]
                elif self.ndim == 2:
                    list_data = [[convert(v) for v in line]
                                 for line in list_data]
                else:
                    raise ValueError("invalid ndim: %d" % self.ndim)
            return list_data

        def __array__(self, dtype=None):
            return np.array(self._converted_value(), dtype=dtype)

        def __larray__(self):
            # FIXME: use converted_value
            return LArray(np.array(self.xw_range.value))

        def __dir__(self):
            return list(set(dir(self.__class__)) | set(dir(self.xw_range)))

        def __getattr__(self, key):
            if hasattr(LArray, key):
                return getattr(self.__larray__(), key)
            else:
                return getattr(self.xw_range, key)

        def __setattr__(self, key, value):
            return setattr(self.xw_range, key, value)

        # TODO: implement all binops
        # def __mul__(self, other):
        #     return self.__larray__() * other

        def __str__(self):
            return str(self.__larray__())
        __repr__ = __str__

        def load(self, header=True, convert_float=True, nb_index=0,
                 index_col=None):
            if not self.ndim:
                return LArray([])
            if index_col is None and nb_index > 0:
                index_col = list(range(nb_index))
            elif isinstance(index_col, int):
                index_col = [index_col]

            list_data = self._converted_value(convert_float=convert_float)

            if header:
                # TODO: try getting values via self[1:] instead of via the
                # list so that we do not produce copies of data. Not sure which
                # would be faster
                header_line = list_data[0]
                # TODO: factor this with read_csv
                try:
                    # take the first cell which contains '\'
                    pos_last = next(i for i, v in enumerate(header_line)
                                    if isinstance(v, basestring) and '\\' in v)
                except StopIteration:
                    # if there isn't any, assume 1d array, unless
                    # "liam2 dialect"
                    pos_last = -1

                # '\' found => we have several axes names
                if pos_last >= 0:
                    axes_names = header_line[:pos_last + 1]
                    # TODO: factor this with df_aslarray
                    if isinstance(axes_names[-1], basestring) and \
                            '\\' in axes_names[-1]:
                        last_axes = [name.strip()
                                     for name in axes_names[-1].split('\\')]
                        axes_names = axes_names[:-1] + last_axes
                # no axes names but index_col provided
                elif index_col is not None:
                    # TODO: use header_line in this case too to support
                    # manually specifying nb_index when there are axes names
                    # (whether the array is 1d or not)
                    nb_axes = len(index_col) + 1
                    axes_names = [None] * nb_axes
                # assume 1d array
                else:
                    axes_names = [header_line[0]]

                # this can only happen if both nb_index=0 and index_col is None
                # TODO: nb_index should default to None instead of
                #      0 so that we can force "no index at all" (ie 1d array)
                if index_col is None:
                    nb_index = len(axes_names) - 1
                    index_col = list(range(nb_index))
                assert isinstance(index_col, list)
                # at this point index_col should be a list but it could be empty
                col_offset = (max(index_col) + 1) if index_col else 1
                # number of header lines or comment lines at the start of the
                # file
                # TODO: we need to support comments & more
                row_offset = 1
                data_no_header = list_data[row_offset:]
                data = np.array([line[col_offset:] for line in data_no_header])

                # TODO: add support for sparse data (ie make it dense) like
                #       in df_aslarray
                axes_labels = [list(unique([line[i]
                                            for line in data_no_header]))
                               for i in index_col]
                axes_labels.append(header_line[col_offset:])
                # TODO: detect anonymous axes
                axes = [Axis(name, labels)
                        for name, labels in zip(axes_names, axes_labels)]
                data = data.reshape([len(axis) for axis in axes])
                return LArray(data, axes)
            else:
                return LArray(list_data)

    # XXX: remove this function?
    def open_excel(filepath=None, **kwargs):
        return Workbook(filepath, **kwargs)
else:
    def open_excel(filepath=None, **kwargs):
        raise Exception("open_excel() is not available because xlwings "
                        "is not installed")
