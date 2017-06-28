from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import pandas as pd
import larray as la

from qtpy.QtCore import (Qt, QVariant, QModelIndex, QAbstractTableModel)
from qtpy.QtGui import (QFont, QColor)
from qtpy.QtWidgets import (QMessageBox)
from qtpy import PYQT5

PY2 = sys.version[0] == '2'


def _get_font(family, size, bold=False, italic=False):
    weight = QFont.Bold if bold else QFont.Normal
    font = QFont(family, size, weight)
    if italic:
        font.setItalic(True)
    return to_qvariant(font)

def is_float(dtype):
    """Return True if datatype dtype is a float kind"""
    return ('float' in dtype.name) or dtype.name in ['single', 'double']

def is_number(dtype):
    """Return True is datatype dtype is a number kind"""
    return is_float(dtype) or ('int' in dtype.name) or ('long' in dtype.name) or ('short' in dtype.name)

# Spyder compat
# -------------

# Note: string and unicode data types will be formatted with '%s' (see below)
SUPPORTED_FORMATS = {
    'object': '%s',
    'single': '%.2f',
    'double': '%.2f',
    'float_': '%.2f',
    'longfloat': '%.2f',
    'float32': '%.2f',
    'float64': '%.2f',
    'float96': '%.2f',
    'float128': '%.2f',
    'csingle': '%r',
    'complex_': '%r',
    'clongfloat': '%r',
    'complex64': '%r',
    'complex128': '%r',
    'complex192': '%r',
    'complex256': '%r',
    'byte': '%d',
    'short': '%d',
    'intc': '%d',
    'int_': '%d',
    'longlong': '%d',
    'intp': '%d',
    'int8': '%d',
    'int16': '%d',
    'int32': '%d',
    'int64': '%d',
    'ubyte': '%d',
    'ushort': '%d',
    'uintc': '%d',
    'uint': '%d',
    'ulonglong': '%d',
    'uintp': '%d',
    'uint8': '%d',
    'uint16': '%d',
    'uint32': '%d',
    'uint64': '%d',
    'bool_': '%r',
    'bool8': '%r',
    'bool': '%r',
}
# =======================

def get_font(section):
    return _get_font('Calibri', 11)

def to_qvariant(obj=None):
    return obj

def from_qvariant(qobj=None, pytype=None):
    # FIXME: force API level 2 instead of handling this
    if isinstance(qobj, QVariant):
        assert pytype is str
        return pytype(qobj.toString())
    return qobj

def _(text):
    return text

def to_text_string(obj, encoding=None):
    """Convert `obj` to (unicode) text string"""
    if PY2:
        # Python 2
        if encoding is None:
            return unicode(obj)
        else:
            return unicode(obj, encoding)
    else:
        # Python 3
        if encoding is None:
            return str(obj)
        elif isinstance(obj, str):
            # In case this function is not used properly, this could happen
            return obj
        else:
            return str(obj, encoding)


LARGE_SIZE = 5e5
LARGE_NROWS = 1e5
LARGE_COLS = 60

class ArrayModel(QAbstractTableModel):
    """Array Editor Table Model.

    Parameters
    ----------
    data : array-like, optional
        Input array that can be converted into a LArray 
        (Numpy ndarray, Pandas Dataframe, list or tuple, ...).
    format : str, optional
        Indicates how data are represented in cells.
        By default, they are represented as floats with 3 decimal points.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    parent : QWidget, optional
        Parent Widget.
    bg_gradient : LinearGradient, optional
        Background color gradient
    bg_value : LArray, optional
        Background color value
    minvalue : scalar
        Minimum value allowed.
    maxvalue : scalar
        Maximum value allowed.
    """

    ROWS_TO_LOAD = 500
    COLS_TO_LOAD = 40

    def __init__(self, data=None, format="%.3f", readonly=False, font=None, parent=None, bg_gradient=None,
                 bg_value=None, minvalue=None, maxvalue=None):
        QAbstractTableModel.__init__(self)

        self.dialog = parent
        self.readonly = readonly
        self._format = format

        # Backgroundcolor settings
        # TODO: use LinearGradient
        # self.bgfunc = bgfunc
        huerange = [.66, .99]  # Hue
        self.sat = .7  # Saturation
        self.val = 1.  # Value
        self.alp = .6  # Alpha-channel
        self.hue0 = huerange[0]
        self.dhue = huerange[1] - huerange[0]
        self.bgcolor_enabled = True
        # hue = self.hue0
        # color = QColor.fromHsvF(hue, self.sat, self.val, self.alp)
        # self.color = to_qvariant(color)

        if font is None:
            font = get_font("arreditor")
        self.font = font
        bold_font = get_font("arreditor")
        bold_font.setBold(True)
        self.bold_font = bold_font

        self.minvalue = minvalue
        self.maxvalue = maxvalue
        # TODO: check that data respects minvalue/maxvalue
        self.set_data(data, bg_gradient=bg_gradient, bg_value=bg_value)

    def get_format(self):
        """Return current format"""
        # Avoid accessing the private attribute _format from outside
        return self._format

    def get_data_2D(self):
        """Return data"""
        return self._data2D

    def get_data(self):
        return self.la_data

    def set_data(self, data, changes=None, current_filter=None, bg_gradient=None, bg_value=None):
        # ------------------- set changes -------------------
        if changes is None:
            changes = {}
        assert isinstance(changes, dict)
        self.changes = changes
        self._changes2D = {}
        # -------------------- set data ---------------------
        if data is None:
            data = np.empty((0, 0), dtype=np.int8)
        la_data = la.aslarray(data)
        if la_data.dtype.names is None:
            dtn = la_data.dtype.name
            if dtn not in SUPPORTED_FORMATS and not dtn.startswith('str') \
                    and not dtn.startswith('unicode'):
                QMessageBox.critical(self.dialog, "Error", "{} arrays are currently not supported".format(dtn))
                return
        # for complex numbers, shading will be based on absolute value
        # but for all other types it will be the real part
        # TODO: there are a lot more complex dtypes than this. Is there a way to get them all in one shot?
        if la_data.dtype in (np.complex64, np.complex128):
            self.color_func = np.abs
        else:
            # XXX: this is a no-op (it returns the array itself) for most types (I think all non complex types)
            #      => use an explicit nop?
            # def nop(v):
            #     return v
            # self.color_func = nop
            self.color_func = np.real
        self.la_data = la_data
        # ------------ set bg gradient and value ------------
        self.bg_gradient = bg_gradient
        self.bg_value = bg_value
        # ------ set current filter and data to display -----
        if current_filter is None:
            current_filter = {}
        assert isinstance(current_filter, dict)
        self.current_filter = current_filter
        self._set_labels_and_data_to_display()
        # ------------------- reset model -------------------
        self.reset()

    @property
    def filtered_data(self):
        return self.la_data[self.current_filter]

    def _set_labels_and_data_to_display(self):
        la_data = self.filtered_data
        if np.isscalar(la_data):
            la_data = la.aslarray(la_data)
        ndim, shape, axes = la_data.ndim, la_data.shape, la_data.axes
        # get 2D shape + xlabels + ylabels
        if ndim == 0:
            self.xlabels = [[], []]
            self.ylabels = [[]]
            shape_2D = (1, 1)
        elif ndim == 1:
            self.xlabels = [axes.display_names, axes.labels[-1]]
            self.ylabels = [[]]
            shape_2D = (1,) + shape
        else:
            self.xlabels = [axes.display_names, axes.labels[-1]]
            otherlabels = axes.labels[:-1]
            prod = Product(otherlabels)
            self.ylabels = [_LazyNone(len(prod) + 1)] + [_LazyDimLabels(prod, i) for i in range(len(otherlabels))]
            shape_2D = (np.prod(shape[:-1]), shape[-1])

        # set data (reshape to a 2D array if not)
        self._data2D = la_data.data.reshape(shape_2D)
        self.total_rows, self.total_cols = shape_2D
        size = self.total_rows * self.total_cols
        self.reset_minmax()
        # Use paging when the total size, number of rows or number of
        # columns is too large
        if size > LARGE_SIZE:
            self.rows_loaded = min(self.ROWS_TO_LOAD, self.total_rows)
            self.cols_loaded = min(self.COLS_TO_LOAD, self.total_cols)
        else:
            if self.total_rows > LARGE_NROWS:
                self.rows_loaded = self.ROWS_TO_LOAD
            else:
                self.rows_loaded = self.total_rows
            if self.total_cols > LARGE_COLS:
                self.cols_loaded = self.COLS_TO_LOAD
            else:
                self.cols_loaded = self.total_cols
        self._set_local_changes()

    def _set_local_changes(self):
        # we cannot apply the changes directly to data because it might be a view
        local_changes = {}
        for k, v in self.changes.items():
            local_key = self.map_global_to_filtered(k)
            if local_key is not None:
                local_changes[local_key] = v
        self._changes2D = local_changes

    def reset_minmax(self):
        # this will be awful to get right, because ideally, we should
        # include self.changes.values() and ignore values corresponding to
        # self.changes.keys()
        data = self.get_values()
        try:
            color_value = self.color_func(data)
            self.vmin = float(np.nanmin(color_value))
            self.vmax = float(np.nanmax(color_value))
            if self.vmax == self.vmin:
                self.vmin -= 1
            self.bgcolor_enabled = True
        # ValueError for empty arrays
        except (TypeError, ValueError):
            self.vmin = None
            self.vmax = None
            self.bgcolor_enabled = False

    def set_format(self, format):
        """Change display format"""
        self._format = format
        self.reset()

    def _index_to_position(self, index):
        """
        Cell at position (0, 0) contains the first data cell.
        Negative position represents a label
        """
        i = index.row() - len(self.xlabels) + 1
        j = index.column() - len(self.ylabels) + 1
        return i, j

    def _is_label(self, index):
        i, j = self._index_to_position(index)
        return i < 0 or j < 0

    def _position_to_labels(self, position):
        if isinstance(position, tuple) and len(position) == 2:
            ki, kj = position
            xlabel = [self.xlabels[i][kj] for i in range(1, len(self.xlabels))]
            ylabel = [self.ylabels[j][ki] for j in range(1, len(self.ylabels))]
            return tuple(ylabel + xlabel)
        else:
            QMessageBox.critical(self, "Error", "index must be a tuple of length 2")
            return tuple()

    def _position_to_dict_axes_ids_labels(self, position):
        labels = self._position_to_labels(position)
        axes_ids = list(self.filtered_data.axes.ids)
        return dict(zip(axes_ids, labels))

    def _dict_axes_ids_labels_to_position(self, dkey):
        # transform (axis:label) dict key to positional ND key
        try:
            index_key = self.filtered_data._translated_key(dkey)
        except ValueError:
            return None
        # transform positional ND key to positional 2D key
        strides = np.append(1, np.cumprod(self.filtered_data.shape[1:-1][::-1]))[::-1]
        return (index_key[:-1] * strides).sum(), index_key[-1]

    def update_global_changes(self):
        for k, v in self._changes2D.items():
            self.changes[self.map_filtered_to_global(k)] = v

    def map_filtered_to_global(self, k):
        """
        map local (filtered) 2D key to global ND key.

        Parameters
        ----------
        k: tuple
            Positional index (row, column) of the modified data cell.

        Returns
        -------
        tuple
            Labels associated with the modified element of the non-filtered array.
        """
        # transform local positional index key to (axis_ids: label) dictionary key.
        # Contains only displayed axes
        dkey = self._position_to_dict_axes_ids_labels(k)
        # add the "scalar" parts of the filter to it (ie the parts of the
        # filter which removed dimensions)
        dkey.update({k: v for k, v in self.current_filter.items() if np.isscalar(v)})
        # re-transform it to tuple (to make it hashable/to store it in .changes)
        return tuple(dkey[axis_id] for axis_id in self.la_data.axes.ids)

    def map_global_to_filtered(self, k):
        """
        map global ND key to local (filtered) 2D key

        Parameters
        ----------
        k: tuple
            Labels associated with the modified element of the non-filtered array.

        Returns
        -------
        tuple
            Positional index (row, column) of the modified data cell.
        """
        assert isinstance(k, tuple) and len(k) == self.la_data.ndim
        dkey = {axis_id: axis_key for axis_key, axis_id in zip(k, self.la_data.axes.ids)}
        # transform global dictionary key to "local" (filtered) key by removing
        # the parts of the key which are redundant with the filter
        for axis_id, axis_filter in self.current_filter.items():
            axis_key = dkey[axis_id]
            if np.isscalar(axis_filter) and axis_key == axis_filter:
                del dkey[axis_id]
            elif not np.isscalar(axis_filter) and axis_key in axis_filter:
                pass
            else:
                # that key is invalid for/outside the current filter
                return None
        # transform local dictionary key to local positional 2D key
        return self._dict_axes_ids_labels_to_position(dkey)

    def change_filter(self, axis, indices):
        # must be done before changing self.current_filter
        self.update_global_changes()
        cur_filter = self.current_filter
        axis_id = self.la_data.axes.axis_id(axis)
        if not indices or len(indices) == len(axis.labels):
            if axis_id in cur_filter:
                del cur_filter[axis_id]
        else:
            if len(indices) == 1:
                cur_filter[axis_id] = axis.labels[indices[0]]
            else:
                cur_filter[axis_id] = axis.labels[indices]
        self._set_labels_and_data_to_display()
        return self.filtered_data

    def accept_changes(self):
        """Accept changes"""
        self.update_global_changes()
        axes = self.la_data.axes
        for k, v in self.changes.items():
            self.la_data.i[axes.translate_full_key(k)] = v
        return self.la_data

    def reject_changes(self):
        """Reject changes"""
        self.changes.clear()
        # trigger view update
        self._changes2D.clear()
        self.reset_minmax()
        self.reset()

    def columnCount(self, qindex=QModelIndex()):
        """Return array column number"""
        return len(self.ylabels) - 1 + self.cols_loaded

    def rowCount(self, qindex=QModelIndex()):
        """Return array row number"""
        return len(self.xlabels) - 1 + self.rows_loaded

    def fetch_more_rows(self):
        if self.total_rows > self.rows_loaded:
            remainder = self.total_rows - self.rows_loaded
            items_to_fetch = min(remainder, self.ROWS_TO_LOAD)
            self.beginInsertRows(QModelIndex(), self.rows_loaded,
                                 self.rows_loaded + items_to_fetch - 1)
            self.rows_loaded += items_to_fetch
            self.endInsertRows()

    def fetch_more_columns(self):
        if self.total_cols > self.cols_loaded:
            remainder = self.total_cols - self.cols_loaded
            items_to_fetch = min(remainder, self.COLS_TO_LOAD)
            self.beginInsertColumns(QModelIndex(), self.cols_loaded,
                                    self.cols_loaded + items_to_fetch - 1)
            self.cols_loaded += items_to_fetch
            self.endInsertColumns()

    def bgcolor(self, state):
        """Toggle backgroundcolor"""
        self.bgcolor_enabled = state > 0
        self.reset()

    def get_labels(self, index):
        if self._is_label(index):
            return ""
        dim_names = self.xlabels[0]
        ndim = len(dim_names)
        last_dim_labels = self.xlabels[1]
        # ylabels[0] are empty
        i, j = self._index_to_position(index)
        labels = [self.ylabels[d + 1][i] for d in range(ndim - 1)] + \
                 [last_dim_labels[j]]
        return ", ".join("{}={}".format(dim_name, label)
                         for dim_name, label in zip(dim_names, labels))

    def get_value(self, index):
        i, j = self._index_to_position(index)
        if i < 0 and j < 0:
            return ""
        if i < 0:
            return str(self.xlabels[i][j])
        if j < 0:
            return str(self.ylabels[j][i])
        return self._changes2D.get((i, j), self._data2D[i, j])

    def data(self, index, role=Qt.DisplayRole):
        """Cell content"""
        if not index.isValid():
            return to_qvariant()
        # if role == Qt.DecorationRole:
        #     return ima.icon('editcopy')
        # if role == Qt.DisplayRole:
        #     return ""

        if role == Qt.TextAlignmentRole:
            if self._is_label(index):
                return to_qvariant(int(Qt.AlignCenter | Qt.AlignVCenter))
            else:
                return to_qvariant(int(Qt.AlignRight | Qt.AlignVCenter))

        elif role == Qt.FontRole:
            if self._is_label(index):
                return self.bold_font
            else:
                return self.font
        # row, column = index.row(), index.column()
        value = self.get_value(index)
        if role == Qt.DisplayRole:
            # if column == 0:
            #     return to_qvariant(value)
            if value is np.ma.masked:
                return ''
            # for headers
            elif isinstance(value, str) and not isinstance(value, np.str_):
                return value
            else:
                return to_qvariant(self._format % value)

        elif role == Qt.BackgroundColorRole:
            if self._is_label(index):
                color = QColor(Qt.lightGray)
                color.setAlphaF(.4)
                return color
            elif self.bgcolor_enabled and value is not np.ma.masked:
                if self.bg_gradient is None:
                    maxdiff = self.vmax - self.vmin
                    color_val = float(self.color_func(value))
                    hue = self.hue0 + self.dhue * (self.vmax - color_val) / maxdiff
                    color = QColor.fromHsvF(hue, self.sat, self.val, self.alp)
                    return to_qvariant(color)
                else:
                    bg_value = self.bg_value
                    x, y = self._index_to_position(index)
                    # FIXME: this is buggy on filtered data. We should change
                    # bg_value when changing the filter.
                    idx = y + x * bg_value.shape[-1]
                    value = bg_value.data.flat[idx]
                    return self.bg_gradient[value]
        elif role == Qt.ToolTipRole:
            return to_qvariant("{}\n{}".format(repr(value),self.get_labels(index)))
        return to_qvariant()

    def get_values(self, left=0, top=0, right=None, bottom=None):
        width, height = self.total_rows, self.total_cols
        if right is None:
            right = width
        if bottom is None:
            bottom = height
        values = self._data2D[left:right, top:bottom].copy()
        # both versions get the same result, but depending on inputs, the
        # speed difference can be large.
        changes2D = self._changes2D
        if values.size < len(changes2D):
            for i in range(left, right):
                for j in range(top, bottom):
                    pos = i, j
                    if pos in changes2D:
                        values[i - left, j - top] = changes2D[pos]
        else:
            for (i, j), value in changes2D.items():
                if left <= i < right and top <= j < bottom:
                    values[i - left, j - top] = value
        return values

    def convert_value(self, value):
        """
        Parameters
        ----------
        value : str
        """
        dtype = self._data2D.dtype
        if dtype.name == "bool":
            try:
                return bool(float(value))
            except ValueError:
                return value.lower() == "true"
        elif dtype.name.startswith("string"):
            return str(value)
        elif dtype.name.startswith("unicode"):
            return to_text_string(value)
        elif is_float(dtype):
            return float(value)
        elif is_number(dtype):
            return int(value)
        else:
            return complex(value)

    def convert_values(self, values):
        values = np.asarray(values)
        res = np.empty_like(values, dtype=self._data2D.dtype)
        try:
            # TODO: use array/vectorized conversion functions (but watch out
            # for bool)
            # new_data = str_array.astype(data.dtype)
            for i, v in enumerate(values.flat):
                res.flat[i] = self.convert_value(v)
        except ValueError as e:
            QMessageBox.critical(self.dialog, "Error",
                                 "Value error: %s" % str(e))
            return None
        except OverflowError as e:
            QMessageBox.critical(self.dialog, "Error",
                                 "Overflow error: %s" % e.message)
            return None
        return res

    def set_values(self, left, top, right, bottom, values):
        """
        Parameters
        ----------
        left : int
        top : int
        right : int
            exclusive
        bottom : int
            exclusive
        values : ndarray
            must not be of the correct type

        Returns
        -------
        tuple of QModelIndex or None
            actual bounds (end bound is inclusive) if update was successful,
            None otherwise
        """
        values = self.convert_values(values)
        if values is None:
            return
        values = np.atleast_2d(values)
        vshape = values.shape
        vwidth, vheight = vshape
        width, height = right - left, bottom - top
        assert vwidth == 1 or vwidth == width
        assert vheight == 1 or vheight == height

        # Add change to self.changes
        # requires numpy 1.10
        newvalues = np.broadcast_to(values, (width, height))
        oldvalues = np.empty_like(newvalues)
        for i in range(width):
            for j in range(height):
                pos = left + i, top + j
                old_value = self._changes2D.get(pos, self._data2D[pos])
                oldvalues[i, j] = old_value
                val = newvalues[i, j]
                if val != old_value:
                    self._changes2D[pos] = val

        # Update vmin/vmax if necessary
        if self.vmin is not None and self.vmax is not None:
            colorval = self.color_func(values)
            old_colorval = self.color_func(oldvalues)
            if np.any(((old_colorval == self.vmax) & (colorval < self.vmax)) |
                      ((old_colorval == self.vmin) & (colorval > self.vmin))):
                self.reset_minmax()
            if np.any(colorval > self.vmax):
                self.vmax = float(np.nanmax(colorval))
            if np.any(colorval < self.vmin):
                self.vmin = float(np.nanmin(colorval))

        xoffset = len(self.xlabels) - 1
        yoffset = len(self.ylabels) - 1
        top_left = self.index(left + xoffset, top + yoffset)
        # -1 because Qt index end bounds are inclusive
        bottom_right = self.index(right + xoffset - 1, bottom + yoffset - 1)
        self.dataChanged.emit(top_left, bottom_right)
        return top_left, bottom_right

    def setData(self, index, value, role=Qt.EditRole):
        """Cell content change"""
        if not index.isValid() or self.readonly:
            return False
        i, j = self._index_to_position(index)
        result = self.set_values(i, j, i + 1, j + 1, from_qvariant(value, str))
        return result is not None

    def flags(self, index):
        """Set editable flag"""
        if not index.isValid():
            return Qt.ItemIsEnabled
        if self._is_label(index):
            return Qt.ItemIsEnabled #QAbstractTableModel.flags(self, index)
        flags = QAbstractTableModel.flags(self, index)
        if not self.readonly:
            flags |= Qt.ItemIsEditable
        return Qt.ItemFlags(flags)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Set header data"""
        horizontal = orientation == Qt.Horizontal
        # if role == Qt.ToolTipRole:
        #     if horizontal:
        #         return to_qvariant("horiz %d" % section)
        #     else:
        #         return to_qvariant("vert %d" % section)
        if role != Qt.DisplayRole:
            # roles = {0: "display", 2: "edit",
            #          8: "background", 9: "foreground",
            #          13: "sizehint", 4: "statustip", 11: "accessibletext",
            #          1: "decoration", 6: "font", 7: "textalign",
            #          10: "checkstate"}
            # print("section", section, "ori", orientation,
            #       "role", roles.get(role, role), "result",
            #       super(ArrayModel, self).headerData(section, orientation,
            #                                          role))
            return to_qvariant()

        labels, other = self.xlabels, self.ylabels
        if not horizontal:
            labels, other = other, labels
        if labels is None:
            shape = self._data2D.shape
            # prefer a blank cell to one cell named "0"
            if not shape or shape[int(horizontal)] == 1:
                return to_qvariant()
            else:
                return to_qvariant(int(section))
        else:
            if section < len(labels[0]):
                return to_qvariant(labels[0][section])
            #     #section = section - len(other) + 1
            else:
                return to_qvariant()

            # return to_qvariant(labels[0][section])
            # if len(other) - 1 <= section < len(labels[0]):
            #     #section = section - len(other) + 1
            # else:
            #     return to_qvariant("a")

    def reset(self):
        self.beginResetModel()
        self.endResetModel()


class Product(object):
    """
    Represents the `cartesian product` of several arrays.

    Parameters
    ----------
    arrays : iterable of array
        List of arrays on which to apply the cartesian product.

    Examples
    --------
    >>> p = Product([['a', 'b', 'c'], [1, 2]])
    >>> for i in range(len(p)):
    ...     print(p[i])
    ('a', 1)
    ('a', 2)
    ('b', 1)
    ('b', 2)
    ('c', 1)
    ('c', 2)
    >>> p[1:4]
    [('a', 2), ('b', 1), ('b', 2)]
    >>> list(p)
    [('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]
    """
    def __init__(self, arrays):
        self.arrays = arrays
        assert len(arrays)
        shape = [len(a) for a in self.arrays]
        self.div_mod = [(int(np.prod(shape[i + 1:])), shape[i])
                        for i in range(len(shape))]
        self.length = np.prod(shape)

    def to_tuple(self, key):
        if key >= self.length:
            raise IndexError("index %d out of range for Product of length %d" % (key, self.length))
        return tuple(key // div % mod for div, mod in self.div_mod)

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return tuple(array[i]
                         for array, i in zip(self.arrays, self.to_tuple(key)))
        else:
            assert isinstance(key, slice), \
                "key (%s) has invalid type (%s)" % (key, type(key))
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            if step is None:
                step = 1

            return [tuple(array[i]
                          for array, i in zip(self.arrays, self.to_tuple(i)))
                    for i in range(start, stop, step)]


class _LazyLabels(object):
    def __init__(self, arrays):
        self.prod = Product(arrays)

    def __getitem__(self, key):
        return ' '.join(self.prod[key])

    def __len__(self):
        return len(self.prod)


class _LazyDimLabels(object):
    """
    Examples
    --------
    >>> p = Product([['a', 'b', 'c'], [1, 2]])
    >>> list(p)
    [('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]
    >>> l0 = _LazyDimLabels(p, 0)
    >>> l1 = _LazyDimLabels(p, 1)
    >>> for i in range(len(p)):
    ...     print(l0[i], l1[i])
    a 1
    a 2
    b 1
    b 2
    c 1
    c 2
    >>> l0[1:4]
    ['a', 'b', 'b']
    >>> l1[1:4]
    [2, 1, 2]
    >>> list(l0)
    ['a', 'a', 'b', 'b', 'c', 'c']
    >>> list(l1)
    [1, 2, 1, 2, 1, 2]
    """
    def __init__(self, prod, i):
        self.prod = prod
        self.i = i

    def __iter__(self):
        return iter(self.prod[i][self.i] for i in range(len(self.prod)))

    def __getitem__(self, key):
        key_prod = self.prod[key]
        if isinstance(key, slice):
            return [p[self.i] for p in key_prod]
        else:
            return key_prod[self.i]

    def __len__(self):
        return len(self.prod)


class _LazyRange(object):
    def __init__(self, length, offset):
        self.length = length
        self.offset = offset

    def __getitem__(self, key):
        if key >= self.offset:
            return key - self.offset
        else:
            return ''

    def __len__(self):
        return self.length + self.offset


class _LazyNone(object):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, key):
        return ' '

    def __len__(self):
        return self.length
