# -*- coding: utf-8 -*-
#
# Copyright © 2009-2012 Pierre Raybaut
# Copyright © 2015 Gaëtan de Menten
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


# TODO:
# * commit
# * scientific toggle -> detect ndigits
# * fix scientific toggle/digits changed wh digits > 9
# * display dimension sizes
# * document default icons situation (limitations)
# * document paint speed experiments
# * filter on headers
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
# * automatic change digits on resize column => different format per column
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
                         QItemDelegate,
                         QLineEdit, QCheckBox, QGridLayout,
                         QDoubleValidator, QDialog, QDialogButtonBox,
                         QMessageBox, QInputDialog, QMenu,
                         QApplication, QKeySequence, QLabel,
                         QSpinBox, QWidget, QVBoxLayout,
                         QAbstractItemDelegate,
                         QFont, QAction, QItemSelection,
                         QItemSelectionModel, QItemSelectionRange,
                         QIcon, QStyle, QFontMetrics)
from PyQt4.QtCore import (Qt, QModelIndex, QAbstractTableModel, QPoint,
                          pyqtSlot as Slot)

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
        if isinstance(key, int):
            return tuple(array[i]
                         for array, i in zip(self.arrays, self.to_tuple(key)))
        else:
            assert isinstance(key, slice)
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

    def __init__(self, data, format="%.3f", xlabels=None, ylabels=None,
                 column_labels=None, row_labels=None,
                 readonly=False, font=None, parent=None):
        QAbstractTableModel.__init__(self)

        assert data.ndim == 2
        self.dialog = parent
        self.readonly = readonly
        self.test_array = np.array([0], dtype=data.dtype)

        # for complex numbers, shading will be based on absolute value
        # but for all other types it will be the real part
        if data.dtype in (np.complex64, np.complex128):
            self.color_func = np.abs
        else:
            self.color_func = np.real

        self._format = format

        # Backgroundcolor settings
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

        self._set_data(data, xlabels, ylabels)

    def get_format(self):
        """Return current format"""
        # Avoid accessing the private attribute _format from outside
        return self._format

    def get_data(self):
        """Return data"""
        return self._data

    def set_data(self, data, xlabels=None, ylabels=None):
        self._set_data(data, xlabels, ylabels)
        self.reset()

    def _set_data(self, data, xlabels, ylabels):
        self.changes = {}
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
        try:
            self.vmin = np.nanmin(self.color_func(data))
            self.vmax = np.nanmax(self.color_func(data))
            if self.vmax == self.vmin:
                self.vmin -= 1
        except TypeError:
            self.vmin = None
            self.vmax = None
            self.bgcolor_enabled = False

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
                return to_qvariant(int(Qt.AlignCenter|Qt.AlignVCenter))
            else:
                return to_qvariant(int(Qt.AlignRight|Qt.AlignVCenter))
        elif role == Qt.FontRole:
            if (index.row() < len(self.xlabels) - 1) or \
                    (index.column() < len(self.ylabels) - 1):
                return self.bold_font
            else:
                return self.font
        # elif role not in (Qt.DisplayRole, Qt.BackgroundColorRole):
        #     return to_qvariant()

        # row, column = index.row(), index.column()
        # if column == 0:
        #     value = "yada"
        # else:
        #     value = self.changes.get((row, column - 1), self._data[row,
        #                                                            column - 1])
        value = self.get_value(index)
        if role == Qt.DisplayRole:
            # if column == 0:
            #     return to_qvariant(value)
            if value is np.ma.masked:
                return ''
            elif isinstance(value, str):
                return value
            else:
                return to_qvariant(self._format % value)

        elif role == Qt.BackgroundColorRole and self.bgcolor_enabled \
                and value is not np.ma.masked:
            if (index.row() < len(self.xlabels) - 1) or \
                    (index.column() < len(self.ylabels) - 1):
                color = QColor(Qt.lightGray)
                color.setAlphaF(.4)
                return color
            else:
                hue = self.hue0 + \
                      self.dhue * (self.vmax - self.color_func(value)) \
                                / (self.vmax - self.vmin)
                hue = float(np.abs(hue))
                color = QColor.fromHsvF(hue, self.sat, self.val, self.alp)
                return to_qvariant(color)
        elif role == Qt.ToolTipRole:
            return to_qvariant(repr(value))
        return to_qvariant()

    def setData(self, index, value, role=Qt.EditRole):
        """Cell content change"""
        if not index.isValid() or self.readonly:
            return False
        i = index.row() - len(self.xlabels) + 1
        j = index.column() - len(self.ylabels) + 1
        value = from_qvariant(value, str)
        if self._data.dtype.name == "bool":
            try:
                val = bool(float(value))
            except ValueError:
                val = value.lower() == "true"
        elif self._data.dtype.name.startswith("string"):
            val = str(value)
        elif self._data.dtype.name.startswith("unicode"):
            val = to_text_string(value)
        else:
            if value.lower().startswith('e') or value.lower().endswith('e'):
                return False
            try:
                val = complex(value)
                if not val.imag:
                    val = val.real
            except ValueError as e:
                QMessageBox.critical(self.dialog, "Error",
                                     "Value error: %s" % str(e))
                return False
        try:
            self.test_array[0] = val  # could raise an Exception
        except OverflowError as e:
            print(type(e.message))
            QMessageBox.critical(self.dialog, "Error",
                                 "Overflow error: %s" % e.message)
            return False

        # Add change to self.changes
        oldvalue = self.color_func(self.changes.get((i, j), self._data[i, j]))
        if (oldvalue == self.vmax and val < self.vmax) or \
                (oldvalue == self.vmin and val > self.vmin):
            # TODO: reset vmin & vmax
            pass

        self.changes[(i, j)] = val
        colorval = self.color_func(val)
        self.dataChanged.emit(index, index)
        if colorval > self.vmax:
            self.vmax = colorval
        if val < self.vmin:
            self.vmin = colorval
        return True

    def flags(self, index):
        """Set editable flag"""
        if not index.isValid():
            return Qt.ItemIsEnabled
        if (index.row() < len(self.xlabels) - 1) or \
                (index.column() < len(self.ylabels) - 1):
            return Qt.ItemIsEnabled #QAbstractTableModel.flags(self, index)
        return Qt.ItemFlags(QAbstractTableModel.flags(self, index)|
                            Qt.ItemIsEditable)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Set header data"""
        horizontal = orientation == Qt.Horizontal
        if role == Qt.ToolTipRole:
            if horizontal:
                return to_qvariant("horiz %d" % section)
            else:
                return to_qvariant("vert %d" % section)
        if role != Qt.DisplayRole:
            # roles = {0: "display", 2: "edit", 8: "background", 9: "foreground",
            #          13: "sizehint", 4: "statustip", 11: "accessibletext",
            #          1: "decoration", 6: "font", 7: "textalign"}
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
    def __init__(self, dtype, parent=None, font=None):
        QItemDelegate.__init__(self, parent)
        self.dtype = dtype
        if font is None:
            font = get_font('arrayeditor')
        self.font = font

    def createEditor(self, parent, option, index):
        """Create editor widget"""
        model = index.model()
        value = model.get_value(index)
        if model._data.dtype.name == "bool":
            # toggle value
            value = not value
            model.setData(index, to_qvariant(value))
            return
        elif value is not np.ma.masked:
            editor = QLineEdit(parent)
            editor.setFont(self.font)
            editor.setAlignment(Qt.AlignRight)
            if is_number(self.dtype):
                editor.setValidator(QDoubleValidator(editor))
            editor.returnPressed.connect(self.on_editor_return_pressed)
            return editor

    def on_editor_return_pressed(self):
        """Commit and close editor"""
        editor = self.sender()
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QAbstractItemDelegate.NoHint)

    def setEditorData(self, editor, index):
        """Set editor widget's data"""
        text = from_qvariant(index.model().data(index, Qt.DisplayRole), str)
        editor.setText(text)


#TODO: Implement "Paste" (from clipboard) feature
class ArrayView(QTableView):
    """Array view class"""
    def __init__(self, parent, model, dtype, shape):
        QTableView.__init__(self, parent)

        self.setModel(model)
        self.setItemDelegate(ArrayDelegate(dtype, self))
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
        self.plot_action = create_action(self, _('Plot'),
                                         shortcut=keybinding('Print'),
                                         # icon=ima.icon('editcopy'),
                                         triggered=self.plot)
        menu = QMenu(self)
        menu.addActions([self.copy_action, self.plot_action])
        return menu

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        if event == QKeySequence.Copy:
            self.copy()
        elif event == QKeySequence.Print:
            self.plot()
        else:
            QTableView.keyPressEvent(self, event)

    def _selection_bounds(self):
        selection_model = self.selectionModel()
        assert isinstance(selection_model, QItemSelectionModel)
        selection = selection_model.selection()
        assert isinstance(selection, QItemSelection)
        assert len(selection) == 1
        srange = selection[0]
        assert isinstance(srange, QItemSelectionRange)
        return srange.top(), srange.bottom(), srange.left(), srange.right()

    def _selection_data(self, headers=True):
        row_min, row_max, col_min, col_max = self._selection_bounds()
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
        data = self.model().get_data()
        raw_data = data[row_min:row_max + 1, col_min:col_max + 1]
        if headers:
            topheaders = [['' for i in range(1, len(ylabels))] +
                          list(xlabels[i][col_min:col_max+1])
                          for i in range(1, len(xlabels))]
            return chain(topheaders,
                         [chain([ylabels[i][r + row_min]
                                 for i in range(1, len(ylabels))],
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
        print(text)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

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
                 xlabels=None, ylabels=None):
        QWidget.__init__(self, parent)
        if np.isscalar(data):
            data = np.array(data)
            readonly = True
        self.data = data
        self.old_data_shape = None
        if len(self.data.shape) == 1:
            self.old_data_shape = self.data.shape
            self.data.shape = (self.data.shape[0], 1)
        elif len(self.data.shape) == 0:
            self.old_data_shape = self.data.shape
            self.data.shape = (1, 1)

        data_frac_digits = self._data_digits(data)

        max_digits = self.get_max_digits()

        # default width can fit 8 chars
        avail_digits = 8
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        absmax = max(abs(vmin), abs(vmax))
        logabsmax = math.log10(absmax) if absmax else 0
        frac_zeros = math.ceil(-logabsmax) if logabsmax < 0 else 0

        # max(1, ...) because there is at least one integer digit
        pos_int_digits = max(1, math.ceil(math.log10(vmax) if vmax else 0))
        if vmin < 0:
            # + 1 for sign
            logvmin = math.log10(-vmin) if vmin else 0
            neg_int_digits = max(1, math.ceil(logvmin)) + 1
        else:
            neg_int_digits = 0

        int_digits = max(pos_int_digits, neg_int_digits)

        # if there are more integer digits than we can display
        # or we can display more information by using scientific format, do so
        # (scientific format "uses" 4 digits, so we win if have >= 4 zeros --
        #  *including the integer one*)
        if int_digits > avail_digits or frac_zeros >= 3:
            use_scientific = True
            # -1.5e+01
            int_digits = 2 if vmin < 0 else 1
            exp_digits = 4
        else:
            use_scientific = False
            exp_digits = 0

        # - 1 for the dot
        ndecimals = avail_digits - 1 - int_digits - exp_digits

        if ndecimals < 0:
            ndecimals = 0

        if data_frac_digits < ndecimals:
            ndecimals = data_frac_digits

        letter = 'e' if use_scientific else 'f'
        format = '%%.%d%s' % (ndecimals, letter)
        # format = SUPPORTED_FORMATS.get(data.dtype.name, '%s')
        self.model = ArrayModel(self.data, format=format,
                                xlabels=xlabels, ylabels=ylabels,
                                readonly=readonly, parent=self)
        self.view = ArrayView(self, self.model, data.dtype, data.shape)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignLeft)

        label = QLabel("Digits")
        spin = QSpinBox(self)
        spin.setValue(ndecimals)
        spin.setEnabled(is_float(data.dtype))
        btn_layout.addWidget(label)
        btn_layout.addWidget(spin)
        spin.valueChanged.connect(self.digits_changed)

        # btn = QPushButton(_("Format"))
        # disable format button for int type
        # btn.setEnabled(is_float(data.dtype))
        # btn_layout.addWidget(btn)
        # btn.clicked.connect(self.change_format)

        scientific = QCheckBox(_('Scientific'))
        scientific.setChecked(use_scientific)
        scientific.stateChanged.connect(self.scientific_changed)
        btn_layout.addWidget(scientific)

        bgcolor = QCheckBox(_('Background color'))
        bgcolor.setChecked(self.model.bgcolor_enabled)
        bgcolor.setEnabled(self.model.bgcolor_enabled)
        bgcolor.stateChanged.connect(self.model.bgcolor)
        btn_layout.addWidget(bgcolor)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

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
        threshold = 10 ** -(maxdigits + 1)
        for ndigits in range(maxdigits):
            maxdiff = np.max(np.abs(data - np.round(data, ndigits)))
            if maxdiff < threshold:
                return ndigits
        return maxdigits

    def accept_changes(self):
        """Accept changes"""
        for (i, j), value in list(self.model.changes.items()):
            self.data[i, j] = value
        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    def reject_changes(self):
        """Reject changes"""
        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    def scientific_changed(self, value):
        cur_format = self.model.get_format()
        # FIXME: will break if digits > 9
        digits = int(cur_format[2])
        letter = 'e' if value else 'f'
        fmt = '%%.%d%s' % (digits, letter)
        self.model.set_format(fmt)

    def digits_changed(self, value):
        cur_format = self.model.get_format()
        # FIXME: will break if digits > 9
        letter = cur_format[3]
        fmt = '%%.%d%s' % (value, letter)
        self.model.set_format(fmt)

    def change_format(self):
        """Change display format"""
        fmt, valid = QInputDialog.getText(self, _('Format'),
                                          _("Float formatting"),
                                          QLineEdit.Normal,
                                          self.model.get_format())
        if valid:
            fmt = str(fmt)
            try:
                fmt % 1.1
            except:
                msg = _("Format (%s) is incorrect") % fmt
                QMessageBox.critical(self, _("Error"), msg)
                return
            self.model.set_format(fmt)


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
        self.layout = None

    def larray_to_array_and_labels(self, data):
        assert isinstance(data, la.LArray)

        def to_str(a):
            if a.dtype.type != np.str_:
                a = a.astype(np.str_)

            # Numpy stores Strings as np.str_ by default, not Python strings
            # convert that to array of Python strings
            return a.astype(object)

        xlabels = [data.axes_names, to_str(data.axes_labels[-1])]

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

        otherlabels = [to_str(axlabels) for axlabels in data.axes_labels[:-1]]
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

    def setup_and_check(self, data, title='', readonly=False,
                        xlabels=None, ylabels=None):
        """
        Setup ArrayEditor:
        return False if data is not supported, True otherwise
        """
        if isinstance(data, la.LArray):
            title = ' x '.join(data.axes_names)
            self.la_data = data
            data, xlabels, ylabels = self.larray_to_array_and_labels(data)
            self.current_filter = {}
        else:
            self.la_data = None
            self.current_filter = None

        self.data = data
        if data.size == 0:
            self.error(_("Array is empty"))
            return False
        if data.ndim > 3:
            self.error(_("Arrays with more than 3 dimensions are not supported"))
            return False
        # if xlabels is not None and len(xlabels) != self.data.shape[1]:
        #     self.error(_("The 'xlabels' argument length do no match array "
        #                  "column number"))
        #     return False
        # if ylabels is not None and len(ylabels) != self.data.shape[0]:
        #     self.error(_("The 'ylabels' argument length do no match array row "
        #                  "number"))
        #     return False

        if data.dtype.names is None:
            dtn = data.dtype.name
            if dtn not in SUPPORTED_FORMATS and not dtn.startswith('str') \
                    and not dtn.startswith('unicode'):
                self.error(_("%s arrays are currently not supported")
                           % data.dtype.name)
                return False

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)
        self.resize(800, 600)

        # Stack widget
        self.arraywidget = ArrayEditorWidget(self, data, readonly,
                                             xlabels, ylabels)
        self.layout.addWidget(self.arraywidget, 1, 0)

        # Buttons configuration
        btn_layout = QHBoxLayout()

        if self.la_data is not None:
            btn_layout.addWidget(QLabel(_("Filters")))
            for axis in self.la_data.axes:
                btn_layout.addWidget(QLabel(axis.name))
                btn_layout.addWidget(self.create_filter_combo(axis))

        # if is_record_array or is_masked_array or data.ndim == 3:
        btn_layout.addStretch()

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        btn_layout.addWidget(bbox)
        self.layout.addLayout(btn_layout, 2, 0)

        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True

    def create_filter_combo(self, axis):
        # def filter_changed(index):
        #     self.change_filter(axis, index)
        # combo = QComboBox(self)
        # combo.addItems(['--'] + [str(l) for l in axis.labels])
        # combo.currentIndexChanged.connect(filter_changed)
        def filter_changed(checked_items):
            self.change_filter(axis, checked_items)
        combo = FilterComboBox(self)
        combo.addItems([str(l) for l in axis.labels])
        combo.checkedItemsChanged.connect(filter_changed)
        return combo

    # def change_filter(self, axis, index):
    def change_filter(self, axis, indices):
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
        print("new filter", cur_filter)
        filtered = self.la_data[cur_filter]
        if np.isscalar(filtered):
            #TODO: make it readonly
            data, xlabels, ylabels = np.array([[filtered]]), None, None
        else:
            data, xlabels, ylabels = self.larray_to_array_and_labels(filtered)

        self.data = data
        #FIXME: we should get model.changes and convert them to "global changes"
        # (because set_data reset the changes dict)
        self.arraywidget.model.set_data(data, xlabels, ylabels)

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

    def error(self, message):
        """An error occured, closing the dialog box"""
        QMessageBox.critical(self, _("Array editor"), message)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.reject()


def edit(array):
    _app = qapplication()
    dlg = ArrayEditor()
    if dlg.setup_and_check(array):
        dlg.exec_()


def view(array):
    _app = qapplication()
    dlg = ArrayEditor()
    if dlg.setup_and_check(array, readonly=True):
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
    data2 = (np.random.randint(10, size=(116, 44, 2, 15)) - 5) / 17
    data2 = np.random.randint(10, size=(116, 44, 2, 15)) / 100 + 1567
    # data2 = np.random.normal(51000000, 10000000, size=(116, 44, 2, 15))
    data2 = np.random.normal(0, 1, size=(116, 44, 2, 15))
    arr2 = la.LArray(data2, axes=(age, geo, sex, lipro))

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
    # edit(arr2)
    # print(arr2['0', 'A11', :, 'P01'])

    # data2 = np.random.normal(0, 10.0, size=(5000, 20))
    # arr2 = la.LArray(data2,
    #                  axes=(la.Axis('d0', list(range(5000))),
    #                        la.Axis('d1', list(range(20)))))
    edit(arr2)
