# -*- coding: utf-8 -*-
#
# Copyright © 2009-2012 Pierre Raybaut
# Copyright © 2015-2016 Gaëtan de Menten
# Licensed under the terms of the MIT License

# based on
# github.com/spyder-ide/spyder/blob/master/spyderlib/widgets/arrayeditor.py

"""
Array Editor Dialog based on Qt
"""

# pylint: disable=C0103
# pylint: disable=R0903
# pylint: disable=R0911
# pylint: disable=R0201

# Note that the canonical way to implement filters in a TableView would
# be to use a QSortFilterProxyModel. I would need to reimplement its
# filterAcceptsColumn and filterAcceptsRow methods, but that seems pretty
# doable, however I think it would be too slow on large arrays (because it
# suppose you have the whole array in your model) and would probably not play
# well with the partial/progressive load we have currently implemented. I have
# also read quite a few people complaining about speed issues with those.

# TODO:
# * drag & drop to reorder axes
#   http://zetcode.com/gui/pyqt4/dragdrop/
#   http://stackoverflow.com/questions/10264040/how-to-drag-and-drop-into-a-qtablewidget-pyqt
#   http://stackoverflow.com/questions/3458542/multiple-drag-and-drop-in-pyqt4
# * keep header columns & rows visible ("frozen")
#   http://doc.qt.io/qt-5/qtwidgets-itemviews-frozencolumn-example.html
# * document default icons situation (limitations)
# * document paint speed experiments
# * filter on headers. In fact this is not a good idea, because that prevents
#   selecting whole columns, which is handy. So a separate row for headers,
#   like in Excel seems better.
# * tooltip on header with current filter

# * selection change -> select headers too
# * fix vmax/vmin on edit cell with max/min
# * fix filtered edit:
#   - translate "local" changes + filter to global changes
#     -> try to be as generic as possible (DataFrame, ...)
# * nicer error on plot with more than one row/column
#   OR
# * plotting a subset should probably (to think) go via LArray/pandas objects
#   so that I have the headers info in the plots (and do not have to deal with
#   them manually)
#   > need to be generic
# * copy to clipboard possibly too
# ? automatic change digits on resize column
#   => different format per column, which is problematic UI-wise
# * keep "headers" visible
# * keyboard shortcut for filter each dim
# * tab in a filter combo, brings up next filter combo
# * view/edit DataFrames too
# * view/edit LArray over Pandas
# * resubmit editor back for inclusion in Spyder
# * custom delegates for each type (spinner for int, checkbox for bool, ...)
# ? "light" headers (do not repeat the same header several times (on the screen)

from __future__ import print_function

from itertools import chain
import math
import sys

from PyQt4.QtGui import (QApplication, QHBoxLayout, QColor, QTableView,
                         QItemDelegate, QListWidget, QSplitter,
                         QLineEdit, QCheckBox, QGridLayout,
                         QDoubleValidator, QIntValidator,
                         QDialog, QDialogButtonBox, QPushButton,
                         QMessageBox, QMenu,
                         QKeySequence, QLabel,
                         QSpinBox, QWidget, QVBoxLayout,
                         QFont, QAction, QItemSelection,
                         QItemSelectionModel, QItemSelectionRange,
                         QIcon, QStyle, QFontMetrics, QToolTip)
from PyQt4.QtCore import (Qt, QModelIndex, QAbstractTableModel, QPoint,
                          QVariant, pyqtSlot as Slot)

import numpy as np

try:
    import matplotlib
    matplotlib.use('Qt4Agg')
    del matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt4agg import FigureCanvas as FigureCanvas
    from matplotlib.backends.backend_qt4agg \
        import NavigationToolbar2QT as NavigationToolbar
    matplotlib_present = True
except ImportError:
    matplotlib_present = False

from larray.combo import FilterComboBox, FilterMenu
import larray as la


def _get_font(family, size, bold=False, italic=False):
    weight = QFont.Bold if bold else QFont.Normal
    font = QFont(family, size, weight)
    if italic:
        font.setItalic(True)
    return to_qvariant(font)

# Spyder compat
# -------------

PY2 = sys.version[0] == '2'


class IconManager(object):
    def icon(self, ref):
        # By default, only X11 will support themed icons. In order to use
        # themed icons on Mac and Windows, you will have to bundle a compliant
        # theme in one of your PySide.QtGui.QIcon.themeSearchPaths() and set the
        # appropriate PySide.QtGui.QIcon.themeName() .
        return QIcon.fromTheme(ref)
ima = IconManager()


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


def keybinding(attr):
    """Return keybinding"""
    ks = getattr(QKeySequence, attr)
    return QKeySequence.keyBindings(ks)[0]


def create_action(parent, text, icon=None, triggered=None, shortcut=None):
    """Create a QAction"""
    action = QAction(text, parent)
    if triggered is not None:
        action.triggered.connect(triggered)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(shortcut)
    action.setShortcutContext(Qt.WidgetShortcut)
    return action


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


def qapplication():
    return QApplication(sys.argv)


# =======================

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


LARGE_SIZE = 5e5
LARGE_NROWS = 1e5
LARGE_COLS = 60


def clear_layout(layout):
    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is not None:
            # widget.setParent(None)
            widget.deleteLater()
        layout.removeItem(item)


class Product(object):
    def __init__(self, arrays):
        self.arrays = arrays
        assert len(arrays)
        shape = [len(a) for a in self.arrays]
        self.div_mod = [(int(np.prod(shape[i + 1:])), shape[i])
                        for i in range(len(shape))]
        self.length = np.prod(shape)

    def to_tuple(self, key):
        return tuple(key // div % mod for div, mod in self.div_mod)

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, (int, np.int64)):
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


def is_float(dtype):
    """Return True if datatype dtype is a float kind"""
    return ('float' in dtype.name) or dtype.name in ['single', 'double']


def is_number(dtype):
    """Return True is datatype dtype is a number kind"""
    return is_float(dtype) or ('int' in dtype.name) or ('long' in dtype.name) \
           or ('short' in dtype.name)


def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows = [i.row() for i in index_list]
    cols = [i.column() for i in index_list]
    return min(rows), max(rows), min(cols), max(cols)


class ArrayModel(QAbstractTableModel):
    """Array Editor Table Model"""

    ROWS_TO_LOAD = 500
    COLS_TO_LOAD = 40

    def __init__(self, data=None, format="%.3f", xlabels=None, ylabels=None,
                 readonly=False, font=None, parent=None,
                 bg_gradient=None, bg_value=None, minvalue=None, maxvalue=None):
        QAbstractTableModel.__init__(self)

        self.dialog = parent
        self.readonly = readonly
        self._format = format

        # Backgroundcolor settings
        # TODO: use LinearGradient
        self.bg_gradient = bg_gradient
        self.bg_value = bg_value
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
        self._set_data(data, xlabels, ylabels)

    def get_format(self):
        """Return current format"""
        # Avoid accessing the private attribute _format from outside
        return self._format

    def get_data(self):
        """Return data"""
        return self._data

    def set_data(self, data, xlabels=None, ylabels=None, changes=None):
        self._set_data(data, xlabels, ylabels, changes)
        self.reset()

    def _set_data(self, data, xlabels, ylabels, changes=None):
        if changes is None:
            changes = {}
        if data is None:
            data = np.empty(0, dtype=np.int8).reshape(0, 0)
        if data.dtype.names is None:
            dtn = data.dtype.name
            if dtn not in SUPPORTED_FORMATS and not dtn.startswith('str') \
                    and not dtn.startswith('unicode'):
                msg = _("%s arrays are currently not supported")
                QMessageBox.critical(self.dialog, "Error",
                                     msg % data.dtype.name)
                return
        assert data.ndim == 2
        self.test_array = np.array([0], dtype=data.dtype)

        # for complex numbers, shading will be based on absolute value
        # but for all other types it will be the real part
        if data.dtype in (np.complex64, np.complex128):
            self.color_func = np.abs
        else:
            self.color_func = np.real
        assert isinstance(changes, dict)
        self.changes = changes
        self._data = data
        if xlabels is None:
            xlabels = [[]]
        self.xlabels = xlabels
        if ylabels is None:
            ylabels = [[]]
        self.ylabels = ylabels
        self.total_rows = self._data.shape[0]
        self.total_cols = self._data.shape[1]
        size = self.total_rows * self.total_cols
        self.reset_minmax(data)
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

    def reset_minmax(self, data):
        # this will be awful to get right, because ideally, we should
        # include self.changes.values() and ignore values corresponding to
        # self.changes.keys()
        try:
            color_value = self.color_func(data)
            self.vmin = np.nanmin(color_value)
            self.vmax = np.nanmax(color_value)
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

    def columnCount(self, qindex=QModelIndex()):
        """Array column number"""
        return len(self.ylabels) - 1 + self.cols_loaded

    def rowCount(self, qindex=QModelIndex()):
        """Array row number"""
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

    def get_value(self, index):
        i = index.row() - len(self.xlabels) + 1
        j = index.column() - len(self.ylabels) + 1
        if i < 0 and j < 0:
            return ""
        if i < 0:
            return self.xlabels[i][j]
        if j < 0:
            return self.ylabels[j][i]
        return self.changes.get((i, j), self._data[i, j])

    def data(self, index, role=Qt.DisplayRole):
        """Cell content"""
        if not index.isValid():
            return to_qvariant()
        # if role == Qt.DecorationRole:
        #     return ima.icon('editcopy')
        # if role == Qt.DisplayRole:
        #     return ""

        if role == Qt.TextAlignmentRole:
            if (index.row() < len(self.xlabels) - 1) or \
                    (index.column() < len(self.ylabels) - 1):
                return to_qvariant(int(Qt.AlignCenter | Qt.AlignVCenter))
            else:
                return to_qvariant(int(Qt.AlignRight | Qt.AlignVCenter))

        elif role == Qt.FontRole:
            if (index.row() < len(self.xlabels) - 1) or \
                    (index.column() < len(self.ylabels) - 1):
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
            if (index.row() < len(self.xlabels) - 1) or \
                    (index.column() < len(self.ylabels) - 1):
                color = QColor(Qt.lightGray)
                color.setAlphaF(.4)
                return color
            elif self.bgcolor_enabled and value is not np.ma.masked:
                if self.bg_gradient is None:
                    hue = self.hue0 + \
                          self.dhue * (self.vmax - self.color_func(value)) \
                                    / (self.vmax - self.vmin)
                    hue = float(np.abs(hue))
                    color = QColor.fromHsvF(hue, self.sat, self.val, self.alp)
                    return to_qvariant(color)
                else:
                    bg_value = self.bg_value
                    x = index.row() - len(self.xlabels) + 1
                    y = index.column() - len(self.ylabels) + 1
                    # FIXME: this is buggy on filtered data
                    idx = y + x * bg_value.shape[-1]
                    value = bg_value.data.flat[idx]
                    return self.bg_gradient[value]
        elif role == Qt.ToolTipRole:
            return to_qvariant(repr(value))
        return to_qvariant()

    def get_values(self, left, top, right, bottom):
        changes = self.changes
        values = self._data[left:right, top:bottom].copy()
        for i in range(left, right):
            for j in range(top, bottom):
                pos = i, j
                if pos in changes:
                    values[i - left, j - top] = changes[pos]
        return values

    def convert_value(self, value):
        """
        Parameters
        ----------
        value : str
        """
        dtype = self._data.dtype
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
        res = np.empty_like(values, dtype=self._data.dtype)
        try:
            # TODO: try to use array/vectorized conversion functions
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
        changes = self.changes
        newvalues = np.broadcast_to(values, (width, height))
        oldvalues = np.empty_like(newvalues)
        for i in range(width):
            for j in range(height):
                pos = left + i, top + j
                old_value = changes.get(pos, self._data[pos])
                oldvalues[i, j] = old_value
                val = newvalues[i, j]
                if val != old_value:
                    changes[pos] = val

        # Update vmin/vmax if necessary
        if self.vmin is not None and self.vmax is not None:
            colorval = self.color_func(values)
            old_colorval = self.color_func(oldvalues)
            if np.any(((old_colorval == self.vmax) & (colorval < self.vmax)) |
                      ((old_colorval == self.vmin) & (colorval > self.vmin))):
                self.reset_minmax(self._data)
            if np.any(colorval > self.vmax):
                self.vmax = np.nanmax(colorval)
            if np.any(colorval < self.vmin):
                self.vmin = np.nanmin(colorval)

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
        i = index.row() - len(self.xlabels) + 1
        j = index.column() - len(self.ylabels) + 1
        result = self.set_values(i, j, i + 1, j + 1, from_qvariant(value, str))
        return result is not None

    def flags(self, index):
        """Set editable flag"""
        if not index.isValid():
            return Qt.ItemIsEnabled
        if (index.row() < len(self.xlabels) - 1) or \
                (index.column() < len(self.ylabels) - 1):
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
            # roles = {0: "display", 2: "edit", 8: "background", 9: "foreground",
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
            shape = self._data.shape
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


class ArrayDelegate(QItemDelegate):
    """Array Editor Item Delegate"""
    def __init__(self, dtype, parent=None, font=None,
                 minvalue=None, maxvalue=None):
        QItemDelegate.__init__(self, parent)
        self.dtype = dtype
        if font is None:
            font = get_font('arrayeditor')
        self.font = font
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        # We must keep a count instead of the "current" one, because when
        # switching from one cell to the next, the new editor is created
        # before the old one is destroyed, which means it would be set to None
        # when the old one is destroyed.
        self.editor_count = 0

    def createEditor(self, parent, option, index):
        """Create editor widget"""
        model = index.model()
        value = model.get_value(index)
        if self.dtype.name == "bool":
            # toggle value
            value = not value
            model.setData(index, to_qvariant(value))
            return
        elif value is not np.ma.masked:
            minvalue, maxvalue = self.minvalue, self.maxvalue
            if minvalue is not None and maxvalue is not None:
                msg = "value must be between %s and %s" % (minvalue, maxvalue)
            elif minvalue is not None:
                msg = "value must be >= %s" % minvalue
            elif maxvalue is not None:
                msg = "value must be <= %s" % maxvalue
            else:
                msg = None

            # Not using a QSpinBox for integer inputs because I could not find
            # a way to prevent the spinbox/editor from closing if the value is
            # invalid. Using the builtin minimum/maximum of the spinbox works
            # but that provides no message so it is less clear.
            editor = QLineEdit(parent)
            if is_number(self.dtype):
                validator = QDoubleValidator(editor) if is_float(self.dtype) \
                    else QIntValidator(editor)
                if minvalue is not None:
                    validator.setBottom(minvalue)
                if maxvalue is not None:
                    validator.setTop(maxvalue)
                editor.setValidator(validator)

                def on_editor_text_edited():
                    if not editor.hasAcceptableInput():
                        QToolTip.showText(editor.mapToGlobal(QPoint()), msg)
                    else:
                        QToolTip.hideText()
                if msg is not None:
                    editor.textEdited.connect(on_editor_text_edited)

            editor.setFont(self.font)
            editor.setAlignment(Qt.AlignRight)
            editor.destroyed.connect(self.on_editor_destroyed)
            self.editor_count += 1
            return editor

    def on_editor_destroyed(self):
        self.editor_count -= 1
        assert self.editor_count >= 0

    def setEditorData(self, editor, index):
        """Set editor widget's data"""
        text = from_qvariant(index.model().data(index, Qt.DisplayRole), str)
        editor.setText(text)


class ArrayView(QTableView):
    """Array view class"""
    def __init__(self, parent, model, dtype, shape):
        QTableView.__init__(self, parent)

        self.setModel(model)
        delegate = ArrayDelegate(dtype, self,
                                 minvalue=model.minvalue,
                                 maxvalue=model.maxvalue)
        self.setItemDelegate(delegate)
        self.setSelectionMode(QTableView.ContiguousSelection)

        self.shape = shape
        self.context_menu = self.setup_context_menu()

        # make the grid a bit more compact
        self.horizontalHeader().setDefaultSectionSize(64)
        self.verticalHeader().setDefaultSectionSize(20)

        self.horizontalScrollBar().valueChanged.connect(
            self.on_horizontal_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(
            self.on_vertical_scroll_changed)
        # self.horizontalHeader().sectionClicked.connect(
        #     self.on_horizontal_header_clicked)

    def on_horizontal_header_clicked(self, section_index):
        menu = FilterMenu(self)
        header = self.horizontalHeader()
        headerpos = self.mapToGlobal(header.pos())
        posx = headerpos.x() + header.sectionPosition(section_index)
        posy = headerpos.y() + header.height()
        menu.exec_(QPoint(posx, posy))

    def on_vertical_scroll_changed(self, value):
        if value == self.verticalScrollBar().maximum():
            self.model().fetch_more_rows()

    def on_horizontal_scroll_changed(self, value):
        if value == self.horizontalScrollBar().maximum():
            self.model().fetch_more_columns()

    def setup_context_menu(self):
        """Setup context menu"""
        self.copy_action = create_action(self, _('Copy'),
                                         shortcut=keybinding('Copy'),
                                         icon=ima.icon('edit-copy'),
                                         triggered=self.copy)
        self.paste_action = create_action(self, _('Paste'),
                                          shortcut=keybinding('Paste'),
                                          icon=ima.icon('edit-paste'),
                                          triggered=self.paste)
        self.plot_action = create_action(self, _('Plot'),
                                         shortcut=keybinding('Print'),
                                         # icon=ima.icon('editcopy'),
                                         triggered=self.plot)
        menu = QMenu(self)
        menu.addActions([self.copy_action, self.plot_action, self.paste_action])
        return menu

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        if event == QKeySequence.Copy:
            self.copy()
        elif event == QKeySequence.Paste:
            self.paste()
        elif event == QKeySequence.Print:
            self.plot()
        # allow to start editing cells by pressing Enter
        elif event.key() == Qt.Key_Return:
            index = self.currentIndex()
            if self.itemDelegate(index).editor_count == 0:
                self.edit(index)
        else:
            QTableView.keyPressEvent(self, event)

    def _raw_selection_bounds(self):
        selection_model = self.selectionModel()
        assert isinstance(selection_model, QItemSelectionModel)
        selection = selection_model.selection()
        assert isinstance(selection, QItemSelection)
        assert len(selection) == 1
        srange = selection[0]
        assert isinstance(srange, QItemSelectionRange)
        return srange.top(), srange.bottom(), srange.left(), srange.right()

    def _selection_bounds(self):
        """
        Returns
        -------
        tuple
            selection bounds. end bound is exclusive
        """
        row_min, row_max, col_min, col_max = self._raw_selection_bounds()
        xlabels = self.model().xlabels
        ylabels = self.model().ylabels
        row_min -= len(xlabels) - 1
        row_min = max(row_min, 0)
        row_max -= len(xlabels) - 1
        row_max = max(row_max, 0)
        col_min -= len(ylabels) - 1
        col_min = max(col_min, 0)
        col_max -= len(ylabels) - 1
        col_max = max(col_max, 0)
        return row_min, row_max + 1, col_min, col_max + 1

    def _selection_data(self, headers=True):
        row_min, row_max, col_min, col_max = self._selection_bounds()
        raw_data = self.model().get_values(row_min, col_min, row_max, col_max)
        if headers:
            xlabels = self.model().xlabels
            ylabels = self.model().ylabels
            # FIXME: this is extremely ad-hoc. We should either use
            # model.data.ndim (orig_ndim?) or add a new concept (eg dim_names)
            # in addition to xlabels & ylabels,
            # TODO: in the future (pandas-based branch) we should use
            # to_string(data[self._selection_filter()])
            dim_names = xlabels[0]
            if len(dim_names) > 1:
                dim_names = dim_names[:-2] + [dim_names[-2] + ' \\ ' +
                                              dim_names[-1]]
            topheaders = [dim_names +
                          list(xlabels[i][col_min:col_max])
                          for i in range(1, len(xlabels))]
            if not dim_names:
                return raw_data
            elif len(dim_names) == 1:
                return chain(topheaders, [chain([''], row) for row in raw_data])
            else:
                assert len(dim_names) > 1
                return chain(topheaders,
                             [chain([ylabels[j][r + row_min]
                                     for j in range(1, len(ylabels))],
                                    row)
                              for r, row in enumerate(raw_data)])
        else:
            return raw_data

    @Slot()
    def copy(self):
        """Copy array as text to clipboard"""
        data = self._selection_data()

        # np.savetxt make things more complicated, especially on py3
        def vrepr(v):
            if isinstance(v, float):
                return repr(v)
            else:
                return str(v)
        text = '\n'.join('\t'.join(vrepr(v) for v in line) for line in data)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    @Slot()
    def paste(self):
        model = self.model()
        row_min, row_max, col_min, col_max = self._selection_bounds()
        clipboard = QApplication.clipboard()
        text = str(clipboard.text())
        list_data = [line.split('\t') for line in text.splitlines()]
        try:
            # take the first cell which contains '\'
            pos_last = next(i for i, v in enumerate(list_data[0]) if '\\' in v)
        except StopIteration:
            # if there isn't any, assume 1d array
            pos_last = 0
        if pos_last:
            # ndim > 1
            list_data = [line[pos_last + 1:] for line in list_data[1:]]
        elif len(list_data) == 2 and list_data[1][0] == '':
            # ndim == 1
            list_data = [list_data[1][1:]]
        new_data = np.array(list_data)
        if new_data.shape[0] > 1:
            row_max = row_min + new_data.shape[0]
        if new_data.shape[1] > 1:
            col_max = col_min + new_data.shape[1]

        result = model.set_values(row_min, col_min, row_max, col_max, new_data)
        if result is None:
            return

        # TODO: when pasting near bottom/right boundaries and size of
        # new_data exceeds destination size, we should either have an error
        # or clip new_data
        self.selectionModel().select(QItemSelection(*result),
                                     QItemSelectionModel.ClearAndSelect)

    def plot(self):
        if not matplotlib_present:
            raise Exception("plot() is not available because matplotlib is not "
                            "installed")
        # we use np.asarray to work around missing "newaxis" implementation
        # in LArray
        data = self._selection_data(headers=False)

        assert data.ndim == 2
        column = data.shape[0] == 1
        row = data.shape[1] == 1
        assert row or column
        data = data[0] if column else data[:, 0]

        figure = plt.figure()

        # create an axis
        ax = figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        x = np.arange(data.shape[0])
        ax.plot(x, data)

        main = PlotDialog(figure, self)
        main.show()


class PlotDialog(QDialog):
    def __init__(self, figure, parent=None):
        super(PlotDialog, self).__init__(parent)

        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self.setLayout(layout)

        canvas.draw()


class ArrayEditorWidget(QWidget):
    def __init__(self, parent, data, readonly=False,
                 xlabels=None, ylabels=None, bg_value=None,
                 bg_gradient=None, minvalue=None, maxvalue=None):
        QWidget.__init__(self, parent)
        if not isinstance(data, (np.ndarray, la.LArray)):
            data = np.array(data)
        self.model = ArrayModel(None, readonly=readonly, parent=self,
                                bg_value=bg_value, bg_gradient=bg_gradient,
                                minvalue=minvalue, maxvalue=maxvalue)
        self.view = ArrayView(self, self.model, data.dtype, data.shape)

        self.filters_layout = QHBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignLeft)

        label = QLabel("Digits")
        btn_layout.addWidget(label)
        spin = QSpinBox(self)
        spin.valueChanged.connect(self.digits_changed)
        self.digits_spinbox = spin
        btn_layout.addWidget(spin)

        scientific = QCheckBox(_('Scientific'))
        scientific.stateChanged.connect(self.scientific_changed)
        self.scientific_checkbox = scientific
        btn_layout.addWidget(scientific)

        bgcolor = QCheckBox(_('Background color'))
        bgcolor.stateChanged.connect(self.model.bgcolor)
        self.bgcolor_checkbox = bgcolor
        btn_layout.addWidget(bgcolor)

        layout = QVBoxLayout()
        layout.addLayout(self.filters_layout)
        layout.addWidget(self.view)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.set_data(data, xlabels, ylabels)

    def set_data(self, data, xlabels=None, ylabels=None):
        self.old_data_shape = None
        self.current_filter = {}
        self.global_changes = {}
        if np.isscalar(data):
            data = np.array(data)
            readonly = True
        if isinstance(data, la.LArray):
            self.la_data = data
            filters_layout = self.filters_layout
            clear_layout(filters_layout)
            filters_layout.addWidget(QLabel(_("Filters")))
            # XXX: do this in Axis.display_name?
            for i, axis in enumerate(data.axes):
                name = axis.name if axis.name is not None else 'dim %d' % i
                filters_layout.addWidget(QLabel(name))
                filters_layout.addWidget(self.create_filter_combo(axis))
            filters_layout.addStretch()
            data, xlabels, ylabels = larray_to_array_and_labels(data)
        else:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            self.la_data = None
        self.filtered_data = self.la_data
        if data.size == 0:
            QMessageBox.critical(self, _("Error"), _("Array is empty"))
        if data.ndim == 1:
            data = data.reshape(1, data.shape[0])
            ylabels = [[]]
        # FIXME: partially redundant with code above
        if len(data.shape) == 1:
            self.old_data_shape = data.shape
            data.shape = (data.shape[0], 1)
        elif len(data.shape) == 0:
            self.old_data_shape = data.shape
            data.shape = (1, 1)

        if data.ndim > 2:
            data = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])

            # if xlabels is not None and len(xlabels) != self.data.shape[1]:
            #     self.error(_("The 'xlabels' argument length do no match array "
            #                  "column number"))
            #     return False
            # if ylabels is not None and len(ylabels) != self.data.shape[0]:
            #     self.error(_("The 'ylabels' argument length do no match array row "
            #                  "number"))
            #     return False
        self._set_raw_data(data, xlabels, ylabels)

    def _set_raw_data(self, data, xlabels, ylabels, changes=None):
        # FIXME: this method should be *FAST*, as it is used for each filter
        # change
        ndecimals, use_scientific = self.choose_format(data)
        # XXX: self.ndecimals vs self.digits
        self.digits = ndecimals
        self.use_scientific = use_scientific
        self.data = data
        self.model.set_format(self.cell_format)
        if changes is None:
            changes = {}
        self.model.set_data(data, xlabels, ylabels, changes)

        self.digits_spinbox.setValue(ndecimals)
        self.digits_spinbox.setEnabled(is_float(data.dtype))

        self.scientific_checkbox.setChecked(use_scientific)
        self.scientific_checkbox.setEnabled(is_number(data.dtype))

        self.bgcolor_checkbox.setChecked(self.model.bgcolor_enabled)
        self.bgcolor_checkbox.setEnabled(self.model.bgcolor_enabled)

    def choose_scientific(self, data):
        # max_digits = self.get_max_digits()
        # default width can fit 8 chars
        # FIXME: use max_digits?
        avail_digits = 8
        if data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return False

        _, frac_zeros, int_digits, _ = self.format_helper(data)

        # if there are more integer digits than we can display or we can
        # display more information by using scientific format, do so
        # (scientific format "uses" 4 digits, so we win if have >= 4 zeros
        #  -- *including the integer one*)
        return int_digits > avail_digits or frac_zeros >= 3

    def choose_ndecimals(self, data, scientific):
        if data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return 0

        # max_digits = self.get_max_digits()
        # default width can fit 8 chars
        # FIXME: use max_digits?
        avail_digits = 8
        data_frac_digits, frac_zeros, int_digits, vmin = self.format_helper(
            data)
        if scientific:
            int_digits = 2 if vmin < 0 else 1
            exp_digits = 4
        else:
            exp_digits = 0
        # - 1 for the dot
        ndecimals = avail_digits - 1 - int_digits - exp_digits

        if ndecimals < 0:
            ndecimals = 0

        if data_frac_digits < ndecimals:
            ndecimals = data_frac_digits
        return ndecimals

    def choose_format(self, data):
        # TODO: refactor so that the expensive format_helper is not called
        # twice (or the values are cached)
        use_scientific = self.choose_scientific(data)
        return self.choose_ndecimals(data, use_scientific), use_scientific

    def format_helper(self, data):
        if not data.size:
            return 0, 0, 0, 0
        data_frac_digits = self._data_digits(data)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        absmax = max(abs(vmin), abs(vmax))
        logabsmax = math.log10(absmax) if absmax else 0
        frac_zeros = math.ceil(-logabsmax) if logabsmax < 0 else 0
        # max(1, ...) because there is at least one integer digit
        log10max = math.log10(vmax) if vmax > 0 else 0
        pos_int_digits = max(1, math.ceil(log10max))
        if vmin < 0:
            # + 1 for sign
            logvmin = math.log10(-vmin) if vmin else 0
            neg_int_digits = max(1, math.ceil(logvmin)) + 1
        else:
            neg_int_digits = 0
        int_digits = max(pos_int_digits, neg_int_digits)
        return data_frac_digits, frac_zeros, int_digits, vmin

    def get_max_digits(self, need_sign=False, need_dot=False, scientific=False):
        font = get_font("arreditor")  # QApplication.font()
        col_width = 60
        margin_width = 6  # a wild guess
        avail_width = col_width - margin_width
        metrics = QFontMetrics(font)

        def str_width(c):
            return metrics.size(Qt.TextSingleLine, c).width()

        digit_width = max(str_width(str(i)) for i in range(10))
        dot_width = metrics.size(Qt.TextSingleLine, '.').width()
        sign_width = max(str_width('+'), str_width('-'))
        if need_sign:
            avail_width -= sign_width
        if need_dot:
            avail_width -= dot_width
        if scientific:
            avail_width -= str_width('e') + sign_width + 2 * digit_width
        return avail_width // digit_width

    def _data_digits(self, data, maxdigits=6):
        if not data.size:
            return 0
        threshold = 10 ** -(maxdigits + 1)
        for ndigits in range(maxdigits):
            maxdiff = np.max(np.abs(data - np.round(data, ndigits)))
            if maxdiff < threshold:
                return ndigits
        return maxdigits

    def accept_changes(self):
        """Accept changes"""
        self.update_global_changes()
        for k, v in self.global_changes.items():
            self.la_data[k] = v
        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    def reject_changes(self):
        """Reject changes"""
        self.global_changes = {}
        self.model.changes = {}
        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    @property
    def cell_format(self):
        if self.data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return '%s'
        else:
            return '%%.%d%s' % (self.digits, 'e' if self.use_scientific else 'f')

    def scientific_changed(self, value):
        self.use_scientific = value
        self.digits = self.choose_ndecimals(self.data, value)
        self.digits_spinbox.setValue(self.digits)
        self.model.set_format(self.cell_format)

    def digits_changed(self, value):
        self.digits = value
        self.model.set_format(self.cell_format)

    def create_filter_combo(self, axis):
        def filter_changed(checked_items):
            self.change_filter(axis, checked_items)
        combo = FilterComboBox(self)
        combo.addItems([str(l) for l in axis.labels])
        combo.checkedItemsChanged.connect(filter_changed)
        return combo

    def change_filter(self, axis, indices):
        # must be done before changing self.current_filter
        self.update_global_changes()
        cur_filter = self.current_filter
        # if index == 0:
        if not indices or len(indices) == len(axis.labels):
            if axis.name in cur_filter:
                del cur_filter[axis.name]
        else:
            if len(indices) == 1:
                cur_filter[axis.name] = axis.labels[indices[0]]
            else:
                cur_filter[axis.name] = axis.labels[indices]
        filtered = self.la_data[cur_filter]
        local_changes = self.get_local_changes(filtered)
        self.filtered_data = filtered
        if np.isscalar(filtered):
            # no need to make the editor readonly as we can still propagate the
            # .changes back into the original array.
            data, xlabels, ylabels = np.array([[filtered]]), None, None
        else:
            data, xlabels, ylabels = larray_to_array_and_labels(filtered)

        self._set_raw_data(data, xlabels, ylabels, local_changes)

    def get_local_changes(self, filtered):
        # we cannot apply the changes directly to data because it might be a
        # view
        changes = {}
        for k, v in self.global_changes.items():
            local_key = self.map_global_to_filtered(k, filtered)
            if local_key is not None:
                changes[local_key] = v
        return changes

    def update_global_changes(self):
        changes = self.global_changes
        model_changes = self.model.changes
        for k, v in model_changes.items():
            changes[self.map_filtered_to_global(k)] = v

    def map_global_to_filtered(self, k, filtered):
        """
        map global ND key to local (filtered) 2D key
        """
        assert isinstance(k, tuple) and len(k) == self.la_data.ndim

        dkey = {axis.name: axis_key
                for axis_key, axis in zip(k, self.la_data.axes)}

        # transform global dictionary key to "local" (filtered) key by removing
        # the parts of the key
        for axis_name, axis_filter in self.current_filter.items():
            axis_key = dkey[axis_name]
            if axis_key == axis_filter or axis_key in axis_filter:
                del dkey[axis_name]
            else:
                # that key is invalid for/outside the current filter
                return None

        # transform local label key to local index key
        try:
            index_key = filtered.translated_key(dkey)
        except ValueError:
            return None

        # transform local index ND key to local index 2D key
        mult = np.append(1, np.cumprod(filtered.shape[1:-1][::-1]))[::-1]
        return (index_key[:-1] * mult).sum(), index_key[-1]

    def map_filtered_to_global(self, k):
        """
        map local (filtered) 2D key to global ND key
        """
        assert isinstance(k, tuple) and len(k) == 2

        # transform local index key to local label key
        model = self.model
        ki, kj = k
        xlabels = model.xlabels
        ylabels = model.ylabels
        xlabel = [xlabels[i][kj] for i in range(1, len(xlabels))]
        ylabel = [ylabels[j][ki] for j in range(1, len(ylabels))]
        label_key = tuple(ylabel + xlabel)

        # compute dictionary key out of it
        data = self.filtered_data
        axes_names = data.axes.names if isinstance(data, la.LArray) else []
        dkey = dict(zip(axes_names, label_key))

        # add the "scalar" parts of the filter to it (ie the parts of the
        # filter which removed dimensions)
        dkey.update({k: v for k, v in self.current_filter.items()
                     if np.isscalar(v)})

        # re-transform it to tuple (to make it hashable/to store it in .changes)
        return tuple(dkey[axis.name] for axis in self.la_data.axes)


def larray_to_array_and_labels(data):
    assert isinstance(data, la.LArray)

    def to_str(a):
        if a.dtype.type != np.str_:
            a = a.astype(np.str_)

        # Numpy stores Strings as np.str_ by default, not Python strings
        # convert that to array of Python strings
        return a.astype(object)

    xlabels = [data.axes.names, to_str(data.axes.labels[-1])]

    class LazyLabels(object):
        def __init__(self, arrays):
            self.prod = Product(arrays)

        def __getitem__(self, key):
            return ' '.join(self.prod[key])

        def __len__(self):
            return len(self.prod)

    class LazyDimLabels(object):
        def __init__(self, prod, i):
            self.prod = prod
            self.i = i

        def __iter__(self):
            return iter(self.prod[:][self.i])

        def __getitem__(self, key):
            return self.prod[key][self.i]

        def __len__(self):
            return len(self.prod)

    class LazyRange(object):
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

    class LazyNone(object):
        def __init__(self, length):
            self.length = length

        def __getitem__(self, key):
            return ' '

        def __len__(self):
            return self.length

    otherlabels = [to_str(axlabels) for axlabels in data.axes.labels[:-1]]
    # ylabels = LazyLabels(otherlabels)
    coldims = 1
    # ylabels = [str(i) for i in range(len(row_labels))]
    data = data.data[:]
    if data.ndim == 1:
        data = data.reshape(1, data.shape[0])
        ylabels = [[]]
    else:
        prod = Product(otherlabels)
        ylabels = [LazyNone(len(prod) + coldims)] + [
            LazyDimLabels(prod, i) for i in range(len(otherlabels))]
        # ylabels = [LazyRange(len(prod), coldims)] + [
        #     LazyDimLabels(prod, i) for i in range(len(otherlabels))]

    if data.ndim > 2:
        data = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])
    return data, xlabels, ylabels


class ArrayEditor(QDialog):
    """Array Editor Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        self.arraywidget = None

    def setup_and_check(self, data, title='', readonly=False,
                        xlabels=None, ylabels=None,
                        minvalue=None, maxvalue=None):
        """
        Setup ArrayEditor:
        return False if data is not supported, True otherwise
        """
        if isinstance(data, la.LArray):
            axes_info = ' x '.join("%s (%d)" % (axis.display_name, len(axis))
                                   for axis in data.axes)
            title = (title + ': ' + axes_info) if title else axes_info

        self.data = data
        layout = QGridLayout()
        self.setLayout(layout)

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array viewer") if readonly else _("Array editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        self.arraywidget = ArrayEditorWidget(self, data, readonly,
                                             xlabels, ylabels,
                                             minvalue=minvalue,
                                             maxvalue=maxvalue)
        layout.addWidget(self.arraywidget, 1, 0)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # not using a QDialogButtonBox with standard Ok/Cancel buttons
        # because that makes it impossible to disable the AutoDefault on them
        # (Enter always "accepts"/close the dialog) which is annoying for edit()
        ok_button = QPushButton("&OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setAutoDefault(False)
        btn_layout.addWidget(ok_button)
        if not readonly:
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            cancel_button.setAutoDefault(False)
            btn_layout.addWidget(cancel_button)
        layout.addLayout(btn_layout, 2, 0)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True

    @Slot()
    def accept(self):
        """Reimplement Qt method"""
        self.arraywidget.accept_changes()
        QDialog.accept(self)

    @Slot()
    def reject(self):
        """Reimplement Qt method"""
        self.arraywidget.reject_changes()
        QDialog.reject(self)

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data


class SessionEditor(QDialog):
    """Session Editor Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        self.arraywidget = None

    def setup_and_check(self, data, title='', readonly=False):
        """
        Setup SessionEditor:
        return False if data is not supported, True otherwise
        """
        assert isinstance(data, la.Session)
        self.data = data

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Session viewer") if readonly else _("Session editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._listwidget = QListWidget(self)
        self._listwidget.addItems(self.data.names)
        self._listwidget.currentItemChanged.connect(self.on_item_changed)

        self.arraywidget = ArrayEditorWidget(self, data[0], readonly)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._listwidget)
        splitter.addWidget(self.arraywidget)
        splitter.setSizes([5, 95])
        splitter.setCollapsible(1, False)

        layout.addWidget(splitter)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        buttons = QDialogButtonBox.Ok
        if not readonly:
            buttons |= QDialogButtonBox.Cancel
        bbox = QDialogButtonBox(buttons)
        bbox.accepted.connect(self.accept)
        if not readonly:
            bbox.rejected.connect(self.reject)
        btn_layout.addWidget(bbox)
        layout.addLayout(btn_layout)

        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True

    def on_item_changed(self, curr, prev):
        self.arraywidget.set_data(self.data[str(curr.text())])

    @Slot()
    def accept(self):
        """Reimplement Qt method"""
        self.arraywidget.accept_changes()
        QDialog.accept(self)

    @Slot()
    def reject(self):
        """Reimplement Qt method"""
        self.arraywidget.reject_changes()
        QDialog.reject(self)

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data


class LinearGradient(object):
    """
    I cannot believe I had to roll my own class for this when PyQt already
    contains QLinearGradient... but you cannot get intermediate values out of
    QLinearGradient!
    """
    def __init__(self, stop_points=None):
        if stop_points is None:
            stop_points = []
        # sort by position
        stop_points = sorted(stop_points, key=lambda x: x[0])
        positions, colors = zip(*stop_points)
        self.positions = np.array(positions)
        self.colors = np.array(colors)

    def __getitem__(self, key):
        if key != key:
            key = self.positions[0]
        pos_idx = np.searchsorted(self.positions, key, side='right') - 1
        # if we are exactly on one of the bounds
        if pos_idx > 0 and key in self.positions:
            pos_idx -= 1
        pos0, pos1 = self.positions[pos_idx:pos_idx + 2]
        col0, col1 = self.colors[pos_idx:pos_idx + 2]
        color = col0 + (col1 - col0) * (key - pos0) / (pos1 - pos0)
        return to_qvariant(QColor.fromHsvF(*color))


class ArrayComparator(QDialog):
    """Session Editor Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data1 = None
        self.data2 = None
        self.array1widget = None
        self.array2widget = None

    def setup_and_check(self, data1, data2, title=''):
        """
        Setup SessionEditor:
        return False if data is not supported, True otherwise
        """
        assert isinstance(data1, la.LArray)
        assert isinstance(data2, la.LArray)
        self.data1 = data1
        self.data2 = data2

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        diff = (data2 - data1)
        vmax = np.nanmax(diff)
        vmin = np.nanmin(diff)
        absmax = max(abs(vmax), abs(vmin))
        # scale diff to 0-1
        if absmax:
            bg_value = (diff / absmax) / 2 + 0.5
        else:
            # TODO: implement full() and full_like()
            bg_value = la.empty_like(diff)
            bg_value[:] = 0.5
        gradient = LinearGradient([(0, [.66, .85, 1., .6]),
                                   (0.5 - 1e-300, [.66, .15, 1., .6]),
                                   (0.5, [1., 0., 1., 1.]),
                                   (0.5 + 1e-300, [.99, .15, 1., .6]),
                                   (1, [.99, .85, 1., .6])])

        self.array1widget = ArrayEditorWidget(self, data1, readonly=True,
                                              bg_value=bg_value,
                                              bg_gradient=gradient)
        self.diffwidget = ArrayEditorWidget(self, diff, readonly=True,
                                            bg_value=bg_value,
                                            bg_gradient=gradient)
        self.array2widget = ArrayEditorWidget(self, data2, readonly=True,
                                              bg_value=1 - bg_value,
                                              bg_gradient=gradient)

        splitter = QHBoxLayout()
        splitter.addWidget(self.array1widget)
        splitter.addWidget(self.diffwidget)
        splitter.addWidget(self.array2widget)

        layout.addLayout(splitter)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        buttons = QDialogButtonBox.Ok
        bbox = QDialogButtonBox(buttons)
        bbox.accepted.connect(self.accept)
        btn_layout.addWidget(bbox)
        layout.addLayout(btn_layout)

        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True


def find_names(obj, depth=1):
    # noinspection PyProtectedMember
    l = sys._getframe(depth).f_locals
    return sorted(k for k, v in l.items() if v is obj)


def get_title(obj, depth=1):
    names = find_names(obj, depth=depth + 1)
    # names can be == [] if we compute an array just to view it
    # eg. view(arr['H'])
    if len(names) > 3:
        names = names[:3] + ['...']
    return ', '.join(names)


def edit(array, title='', minvalue=None, maxvalue=None):
    _app = qapplication()
    if not title:
        title = get_title(array, depth=2)
    dlg = ArrayEditor()
    if dlg.setup_and_check(array, title=title,
                           minvalue=minvalue, maxvalue=maxvalue):
        dlg.exec_()


def view(obj, title=''):
    _app = qapplication()
    if not title:
        title = get_title(obj, depth=2)

    if isinstance(obj, la.Session):
        dlg = SessionEditor()
    else:
        dlg = ArrayEditor()
    if dlg.setup_and_check(obj, title=title, readonly=True):
        dlg.exec_()


def compare(obj1, obj2, title=''):
    _app = qapplication()
    dlg = ArrayComparator()
    if dlg.setup_and_check(obj1, obj2, title=title):
        dlg.exec_()


if __name__ == "__main__":
    """Array editor test"""

    lipro = la.Axis('lipro', ['P%02d' % i for i in range(1, 16)])
    age = la.Axis('age', ':115')
    sex = la.Axis('sex', 'H,F')

    vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,' \
          'A43,A44,A45,A46,A71,A72,A73'
    wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,' \
          'A83,A84,A85,A91,A92,A93'
    bru = 'A21'
    # list of strings
    belgium = la.union(vla, wal, bru)

    geo = la.Axis('geo', belgium)

    # data1 = np.arange(30).reshape(2, 15)
    # arr1 = la.LArray(data1, axes=(sex, lipro))
    # edit(arr1)

    # data2 = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
    #           .astype(float)
    # data2 = np.random.random(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
    #           .astype(float)
    # data2 = (np.random.randint(10, size=(116, 44, 2, 15)) - 5) / 17
    # data2 = np.random.randint(10, size=(116, 44, 2, 15)) / 100 + 1567
    # data2 = np.random.normal(51000000, 10000000, size=(116, 44, 2, 15))
    data2 = np.random.normal(0, 1, size=(116, 44, 2, 15))
    arr2 = la.LArray(data2, axes=(age, geo, sex, lipro))
    # arr2 = arr2['F', 'A11', '1']

    # 8.5Gb... and still snappy, yeah!
    # dummy = la.Axis('dummy', range(7000))
    # dummy = la.Axis('dummy', range(2))
    # d2 = la.Axis('d2', range(2))
    # d3 = la.Axis('d3', range(2))
    # d4 = la.Axis('d4', range(2))
    # data3 = np.arange(116 * 44 * 2 * 2 * 2 * 2 * 2 * 15) \
    #           .reshape(116, 44, 2, 2, 2, 2, 2, 15)\
    #           .astype(float)
    # print(data3.nbytes)
    # print(np.prod(data3.shape[:-1]))
    # arr2 = la.LArray(data3, axes=(age, geo, sex, dummy, d2, d3, d4, lipro))
    # view(arr2['0', 'A11', 'F', 'P01'])
    # view(arr1)
    # view(arr2['0', 'A11'])
    # edit(arr1)
    # print(arr2['0', 'A11', :, 'P01'])
    # edit(arr2.astype(int), minvalue=-99, maxvalue=55.123456)
    # edit(arr2.astype(int), minvalue=-99)
    edit(arr2, minvalue=-99, maxvalue=25.123456)
    # print(arr2['0', 'A11', :, 'P01'])

    # data2 = np.random.normal(0, 10.0, size=(5000, 20))
    # arr2 = la.LArray(data2,
    #                  axes=(la.Axis('d0', list(range(5000))),
    #                        la.Axis('d1', list(range(20)))))
    # edit(arr2)

    # view(['a', 'bb', 5599])
    # view(np.arange(12).reshape(2, 3, 2))
    # view([])

    # data3 = np.random.normal(0, 1, size=(2, 15))
    # arr3 = la.LArray(data3, axes=(sex, lipro))
    # data4 = np.random.normal(0, 1, size=(2, 15))
    # arr4 = la.LArray(data4, axes=(sex, lipro))
    # arr4 = arr3.copy()
    # arr4['F', 'P01':] = arr3['F', 'P01':] / 2
    # compare(arr3, arr4)
