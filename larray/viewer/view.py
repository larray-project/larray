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
# be to use a QSortFilterProxyModel. In this case, we would need to reimplement
# its filterAcceptsColumn and filterAcceptsRow methods. The problem is that
# it does seem to be really designed for very large arrays and it would
# probably be too slow on those (I have read quite a few people complaining
# about speed issues with those) possibly because it suppose you have the whole
# array in your model. It would also probably not play well with the
# partial/progressive load we have currently implemented.

# TODO:
# * drag & drop to reorder axes
#   http://zetcode.com/gui/pyqt4/dragdrop/
#   http://stackoverflow.com/questions/10264040/
#       how-to-drag-and-drop-into-a-qtablewidget-pyqt
#   http://stackoverflow.com/questions/3458542/multiple-drag-and-drop-in-pyqt4
#   http://ux.stackexchange.com/questions/34158/
#       how-to-make-it-obvious-that-you-can-drag-things-that-you-normally-cant
# * keep header columns & rows visible ("frozen")
#   http://doc.qt.io/qt-5/qtwidgets-itemviews-frozencolumn-example.html
# * document default icons situation (limitations)
# * document paint speed experiments
# * filter on headers. In fact this is not a good idea, because that prevents
#   selecting whole columns, which is handy. So a separate row for headers,
#   like in Excel seems better.
# * tooltip on header with current filter

# * selection change -> select headers too
# * nicer error on plot with more than one row/column
#   OR
# * plotting a subset should probably (to think) go via LArray/pandas objects
#   so that I have the headers info in the plots (and do not have to deal with
#   them manually)
#   > ideally, I would like to keep this generic (not LArray-specific)
# ? automatic change digits on resize column
#   => different format per column, which is problematic UI-wise
# * keyboard shortcut for filter each dim
# * tab in a filter combo, brings up next filter combo
# * view/edit DataFrames too
# * view/edit LArray over Pandas (ie sparse)
# * resubmit editor back for inclusion in Spyder
# ? custom delegates for each type (spinner for int, checkbox for bool, ...)
# ? "light" headers (do not repeat the same header several times (on the screen)
#   it would be nicer but I am not sure it is a good idea because with many
#   dimensions, you can no longer see the current label for the first
#   dimension(s) if you scroll down a bit. This is solvable if, instead
#   of only the first line ever corresponding to the label displaying it,
#   I could make it so that it is the first line displayable on the screen
#   which gets it. It would be a bit less nice because of strange artifacts
#   when scrolling, but would be more useful. The beauty problem could be
#   solved later too via fading or something like that, but probably not
#   worth it for a while.

from __future__ import print_function

import math
import os
import re
import sys
from itertools import chain

import numpy as np
from qtpy import PYQT5
from qtpy.QtCore import (Qt, QPoint, QItemSelection, QItemSelectionModel, QItemSelectionRange, QSettings,
                         QUrl, Slot)
from qtpy.QtGui import (QColor, QDoubleValidator, QIntValidator, QKeySequence, QDesktopServices,
                        QIcon, QFontMetrics, QCursor)
from qtpy.QtWidgets import (QApplication, QHBoxLayout, QTableView, QItemDelegate, QListWidget, QSplitter,
                            QListWidgetItem, QLineEdit, QCheckBox, QGridLayout, QFileDialog, QDialog,
                            QDialogButtonBox, QPushButton, QMessageBox, QMenu, QMainWindow, QLabel,
                            QSpinBox, QWidget, QVBoxLayout, QAction, QStyle, QToolTip, QShortcut)

try:
    import matplotlib
    from matplotlib.figure import Figure

    if PYQT5:
        from matplotlib.backends.backend_qt5agg import FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    else:
        from matplotlib.backends.backend_qt4agg import FigureCanvas
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

    matplotlib_present = True
except ImportError:
    matplotlib_present = False

try:
    import xlwings as xw
except ImportError:
    xw = None

try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.inprocess import QtInProcessKernelManager
    from IPython import get_ipython

    ipython_instance = get_ipython()

    # Having several instances of IPython of different types in the same
    # process are not supported. We use
    # ipykernel.inprocess.ipkernel.InProcessInteractiveShell
    # and qtconsole and notebook use
    # ipykernel.zmqshell.ZMQInteractiveShell, so this cannot work.
    # For now, we simply fallback to not using IPython if we are run
    # from IPython (whether qtconsole or notebook). The correct solution is
    # probably to run the IPython console in a different process but I do not
    # know what would be the consequences. I fear it could be slow to transfer
    # the session data to the other process.
    if ipython_instance is None:
        qtconsole_available = True
    else:
        qtconsole_available = False
except ImportError:
    qtconsole_available = False

from larray.viewer.model import ArrayModel, _, get_font, from_qvariant, to_qvariant, is_float, is_number

from larray.viewer.combo import FilterComboBox, FilterMenu
import larray as la

PY2 = sys.version[0] == '2'


# Spyder compat
# -------------

class IconManager(object):
    def icon(self, ref):
        # By default, only X11 will support themed icons. In order to use
        # themed icons on Mac and Windows, you will have to bundle a compliant
        # theme in one of your PySide.QtGui.QIcon.themeSearchPaths() and set the
        # appropriate PySide.QtGui.QIcon.themeName() .
        return QIcon.fromTheme(ref)
ima = IconManager()


def keybinding(attr):
    """Return keybinding"""
    ks = getattr(QKeySequence, attr)
    return QKeySequence.keyBindings(ks)[0]


def create_action(parent, text, icon=None, triggered=None, shortcut=None, statustip=None):
    """Create a QAction"""
    action = QAction(text, parent)
    if triggered is not None:
        action.triggered.connect(triggered)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(shortcut)
    if statustip is not None:
        action.setStatusTip(statustip)
    # action.setShortcutContext(Qt.WidgetShortcut)
    return action

def clear_layout(layout):
    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is not None:
            # widget.setParent(None)
            widget.deleteLater()
        layout.removeItem(item)

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows = [i.row() for i in index_list]
    cols = [i.column() for i in index_list]
    return min(rows), max(rows), min(cols), max(cols)


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

        # TODO: find a cleaner way to do this
        # For some reason the shortcuts in the context menu are not available if the widget does not have the focus,
        # EVEN when using action.setShortcutContext(Qt.ApplicationShortcut) (or Qt.WindowShortcut) so we redefine them
        # here. I was also unable to get the function an action.triggered is connected to, so I couldn't do this via
        # a loop on self.context_menu.actions.
        shortcuts = [
            (keybinding('Copy'), self.copy),
            (QKeySequence("Ctrl+E"), self.to_excel),
            (keybinding('Paste'), self.paste),
            (keybinding('Print'), self.plot)
        ]
        for key_seq, target in shortcuts:
            shortcut = QShortcut(key_seq, self)
            shortcut.activated.connect(target)

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
        self.excel_action = create_action(self, _('Copy to Excel'),
                                          shortcut="Ctrl+E",
                                          # icon=ima.icon('edit-copy'),
                                          triggered=self.to_excel)
        self.paste_action = create_action(self, _('Paste'),
                                          shortcut=keybinding('Paste'),
                                          icon=ima.icon('edit-paste'),
                                          triggered=self.paste)
        self.plot_action = create_action(self, _('Plot'),
                                         shortcut=keybinding('Print'),
                                         # icon=ima.icon('editcopy'),
                                         triggered=self.plot)
        menu = QMenu(self)
        menu.addActions([self.copy_action, self.excel_action, self.plot_action,
                         self.paste_action])
        return menu

    def autofit_columns(self):
        """Resize cells to contents"""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # Spyder loads more columns before resizing, but since it does not
        # load all columns anyway, I do not see the point
        # self.model().fetch_more_columns()
        self.resizeColumnsToContents()
        QApplication.restoreOverrideCursor()

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        # comparing with the keysequence and not with event directly as we
        # did before because that only seems to work for shortcut
        # defined using QKeySequence.StandardKey, which is not the case for
        # Ctrl + E
        keyseq = QKeySequence(event.modifiers() | event.key())
        if keyseq == QKeySequence.Copy:
            self.copy()
        elif keyseq == QKeySequence.Paste:
            self.paste()
        elif keyseq == QKeySequence.Print:
            self.plot()
        elif keyseq == "Ctrl+E":
            self.to_excel()
        # allow to start editing cells by pressing Enter
        elif event.key() == Qt.Key_Return and not self.model().readonly:
            index = self.currentIndex()
            if self.itemDelegate(index).editor_count == 0:
                self.edit(index)
        else:
            QTableView.keyPressEvent(self, event)

    def _selection_bounds(self, none_selects_all=True):
        """
        Returns
        -------
        tuple
            selection bounds. end bound is exclusive
        """
        model = self.model()
        selection_model = self.selectionModel()
        assert isinstance(selection_model, QItemSelectionModel)
        selection = selection_model.selection()
        assert isinstance(selection, QItemSelection)
        if not selection:
            if none_selects_all:
                return 0, model.total_rows, 0, model.total_cols
            else:
                return None
        assert len(selection) == 1
        srange = selection[0]
        assert isinstance(srange, QItemSelectionRange)
        xoffset = len(self.model().xlabels) - 1
        yoffset = len(self.model().ylabels) - 1
        row_min = max(srange.top() - xoffset, 0)
        row_max = max(srange.bottom() - xoffset, 0)
        col_min = max(srange.left() - yoffset, 0)
        col_max = max(srange.right() - yoffset, 0)
        # if not all rows/columns have been loaded
        if row_min == 0 and row_max == self.model().rows_loaded - 1:
            row_max = self.model().total_rows - 1
        if col_min == 0 and col_max == self.model().cols_loaded - 1:
            col_max = self.model().total_cols - 1
        return row_min, row_max + 1, col_min, col_max + 1

    def _selection_data(self, headers=True, none_selects_all=True):
        """
        Returns an iterator over selected labels and data
        if headers=True and a Numpy ndarray containing only
        the data otherwise.

        Parameters
        ----------
        headers : bool, optional
            Labels are also returned if True.
        none_selects_all : bool, optional
            If True (default) and selection is empty, returns all data.

        Returns
        -------
        numpy.ndarray or itertools.chain
        """
        bounds = self._selection_bounds(none_selects_all=none_selects_all)
        if bounds is None:
            return None
        row_min, row_max, col_min, col_max = bounds
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
                dim_headers = dim_names[:-2] + [dim_names[-2] + ' \\ ' +
                                                dim_names[-1]]
            else:
                dim_headers = dim_names
            topheaders = [dim_headers + list(xlabels[i][col_min:col_max])
                          for i in range(1, len(xlabels))]
            if not dim_names:
                return raw_data
            elif len(dim_names) == 1:
                # 1 dimension
                return chain(topheaders, [chain([''], row) for row in raw_data])
            else:
                # >1 dimension
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
        """Copy selection as text to clipboard"""
        data = self._selection_data()
        if data is None:
            return

        # np.savetxt make things more complicated, especially on py3
        # XXX: why don't we use repr for everything?
        def vrepr(v):
            if isinstance(v, float):
                return repr(v)
            else:
                return str(v)
        text = '\n'.join('\t'.join(vrepr(v) for v in line) for line in data)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    @Slot()
    def to_excel(self):
        """View selection in Excel"""
        if xw is None:
            raise Exception("to_excel() is not available because xlwings is "
                            "not installed")
        data = self._selection_data()
        if data is None:
            return
        # convert (row) generators to lists then array
        # TODO: the conversion to array is currently necessary even though xlwings will translate it back to a list
        #       anyway. The problem is that our lists contains numpy types and especially np.str_ crashes xlwings.
        #       unsure how we should fix this properly: in xlwings, or change _selection_data to return only standard
        #       Python types.
        xw.view(np.array([list(r) for r in data]))

    @Slot()
    def paste(self):
        model = self.model()
        bounds = self._selection_bounds()
        if bounds is None:
            return
        row_min, row_max, col_min, col_max = bounds
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
            raise Exception("plot() is not available because matplotlib is not installed")
        data = self._selection_data(headers=False)
        if data is None:
            return

        row_min, row_max, col_min, col_max = self._selection_bounds()
        dim_names = self.model().xlabels[0]
        # label for each selected column
        xlabels = self.model().xlabels[1][col_min:col_max]
        # list of selected labels for each index column
        labels_per_index_column = [col_labels[row_min:row_max] for col_labels in self.model().ylabels[1:]]
        # list of (str) label for each selected row
        ylabels = [[str(label) for label in row_labels]
                   for row_labels in zip(*labels_per_index_column)]
        # if there is only one dimension, ylabels is empty
        if not ylabels:
            ylabels = [[]]

        assert data.ndim == 2

        figure = Figure()

        # create an axis
        ax = figure.add_subplot(111)

        if data.shape[1] == 1:
            # plot one column
            xlabel = ','.join(dim_names[:-1])
            xticklabels = ['\n'.join(ylabels[row]) for row in range(row_max - row_min)]
            xdata = np.arange(row_max - row_min)
            ax.plot(xdata, data[:, 0])
            ax.set_ylabel(xlabels[0])
        else:
            # plot each row as a line
            xlabel = dim_names[-1]
            xticklabels = [str(label) for label in xlabels]
            xdata = np.arange(col_max - col_min)
            for row in range(len(data)):
                ax.plot(xdata, data[row], label=' '.join(ylabels[row]))

        # set x axis
        ax.set_xlabel(xlabel)
        ax.set_xlim((xdata[0], xdata[-1]))
        # we need to do that because matplotlib is smart enough to
        # not show all ticks but a selection. However, that selection
        # may include ticks outside the range of x axis
        xticks = [t for t in ax.get_xticks().astype(int) if t <= len(xticklabels) - 1]
        xticklabels = [xticklabels[t] for t in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        if data.shape[1] != 1 and ylabels != [[]]:
            # set legend
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.legend()

        canvas = FigureCanvas(figure)
        main = PlotDialog(canvas, self)
        main.show()


class PlotDialog(QDialog):
    def __init__(self, canvas, parent=None):
        super(PlotDialog, self).__init__(parent)

        toolbar = NavigationToolbar(canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self.setLayout(layout)
        canvas.draw()


def ndigits(value):
    """
    number of integer digits

    >>> ndigits(1)
    1
    >>> ndigits(99)
    2
    >>> ndigits(-99.1)
    3
    """
    negative = value < 0
    value = abs(value)
    log10 = math.log10(value) if value > 0 else 0
    if log10 == np.inf:
        int_digits = 308
    else:
        # max(1, ...) because there is at least one integer digit.
        # explicit conversion to int for Python2.x
        int_digits = max(1, int(math.floor(log10)) + 1)
    # one digit for sign if negative
    return int_digits + negative


class ArrayEditorWidget(QWidget):
    def __init__(self, parent, data, readonly=False, bg_value=None, bg_gradient=None, minvalue=None, maxvalue=None):
        QWidget.__init__(self, parent)
        readonly = np.isscalar(data)
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
        self.set_data(data, bg_value=bg_value, bg_gradient=bg_gradient)

    def set_data(self, data, current_filter=None, bg_gradient=None, bg_value=None):
        if current_filter is None:
            current_filter = {}
        self.current_filter = current_filter
        self.global_changes = {}
        self.la_data = la.aslarray(data)
        axes = self.la_data.axes
        display_names = axes.display_names

        filters_layout = self.filters_layout
        clear_layout(filters_layout)
        filters_layout.addWidget(QLabel(_("Filters")))
        for axis, display_name in zip(axes, display_names):
            filters_layout.addWidget(QLabel(display_name))
            filters_layout.addWidget(self.create_filter_combo(axis))
        filters_layout.addStretch()
        self.filtered_data = self.la_data

        self._update(self.la_data, bg_gradient=bg_gradient, bg_value=bg_value)

    def _update(self, la_data, changes=None, bg_gradient=None, bg_value=None):
        size = la_data.size
        # this will yield a data sample of max 199
        step = (size // 100) if size > 100 else 1
        data_sample = la_data.data.flat[::step]

        # TODO: refactor so that the expensive format_helper is not called
        # twice (or the values are cached)
        use_scientific = self.choose_scientific(data_sample)

        # XXX: self.ndecimals vs self.digits
        self.digits = self.choose_ndecimals(data_sample, use_scientific)
        self.use_scientific = use_scientific
        self.data = la_data.data
        self.model.set_format(self.cell_format)
        if changes is None:
            changes = {}
        self.model.set_data(la_data, changes, bg_gradient=bg_gradient, bg_value=bg_value)

        self.digits_spinbox.setValue(self.digits)
        self.digits_spinbox.setEnabled(is_number(la_data.dtype))

        self.scientific_checkbox.setChecked(use_scientific)
        self.scientific_checkbox.setEnabled(is_number(la_data.dtype))

        self.bgcolor_checkbox.setChecked(self.model.bgcolor_enabled)
        self.bgcolor_checkbox.setEnabled(self.model.bgcolor_enabled)

    def choose_scientific(self, data):
        # max_digits = self.get_max_digits()
        # default width can fit 8 chars
        # FIXME: use max_digits?
        avail_digits = 8
        if data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return False

        frac_zeros, int_digits, _ = self.format_helper(data)

        # if there are more integer digits than we can display or we can
        # display more information by using scientific format, do so
        # (scientific format "uses" 4 digits, so we win if have >= 4 zeros
        #  -- *including the integer one*)
        # TODO: only do so if we would actually display more information
        # 0.00001 can be displayed with 8 chars
        # 1e-05
        # would
        return int_digits > avail_digits or frac_zeros >= 4

    def choose_ndecimals(self, data, scientific):
        if data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return 0

        # max_digits = self.get_max_digits()
        # default width can fit 8 chars
        # FIXME: use max_digits?
        avail_digits = 8
        data_frac_digits = self._data_digits(data)
        _, int_digits, negative = self.format_helper(data)
        if scientific:
            int_digits = 2 if negative else 1
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

    def format_helper(self, data):
        if not data.size:
            return 0, 0, False
        data = np.where(np.isfinite(data), data, 0)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        absmax = max(abs(vmin), abs(vmax))
        logabsmax = math.log10(absmax) if absmax else 0
        # minimum number of zeros before meaningful fractional part
        frac_zeros = math.ceil(-logabsmax) - 1 if logabsmax < 0 else 0
        int_digits = max(ndigits(vmin), ndigits(vmax))
        return frac_zeros, int_digits, vmin < 0

    def get_max_digits(self, need_sign=False, need_dot=False, scientific=False):
        font = get_font("arreditor")  # QApplication.font()
        col_width = 60
        margin_width = 6  # a wild guess
        avail_width = col_width - margin_width
        metrics = QFontMetrics(font)

        def str_width(c):
            return metrics.size(Qt.TextSingleLine, c).width()

        digit_width = max(str_width(str(i)) for i in range(10))
        dot_width = str_width('.')
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

    @property
    def dirty(self):
        self.update_global_changes()
        return len(self.global_changes) > 1

    def accept_changes(self):
        """Accept changes"""
        self.update_global_changes()
        la_data = self.la_data
        for k, v in self.global_changes.items():
            la_data.i[la_data.axes.translate_full_key(k)] = v
        # update model data & reset global_changes
        self.set_data(self.la_data, current_filter=self.current_filter)

    def reject_changes(self):
        """Reject changes"""
        self.global_changes.clear()
        # trigger view update
        self.model.changes.clear()
        self.model.reset_minmax()
        self.model.reset()

    @property
    def cell_format(self):
        if self.data.dtype.type in (np.str, np.str_, np.bool_, np.bool,
                                    np.object_):
            return '%s'
        else:
            format_letter = 'e' if self.use_scientific else 'f'
            return '%%.%d%s' % (self.digits, format_letter)

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
        axis_id = self.la_data.axes.axis_id(axis)
        # if index == 0:
        if not indices or len(indices) == len(axis.labels):
            if axis_id in cur_filter:
                del cur_filter[axis_id]
        else:
            if len(indices) == 1:
                cur_filter[axis_id] = axis.labels[indices[0]]
            else:
                cur_filter[axis_id] = axis.labels[indices]
        filtered = self.la_data[cur_filter]
        local_changes = self.get_local_changes(filtered)
        self.filtered_data = filtered

        self._update(filtered, local_changes)

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
        # TODO: it would be a better idea to handle the filter in the model,
        # and only store changes as "global changes".
        for k, v in self.model.changes.items():
            self.global_changes[self.map_filtered_to_global(k)] = v

    def map_global_to_filtered(self, k, filtered):
        """
        map global ND key to local (filtered) 2D key
        """
        assert isinstance(k, tuple) and len(k) == self.la_data.ndim

        dkey = {axis_id: axis_key
                for axis_key, axis_id in zip(k, self.la_data.axes.ids)}

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

        # transform local label key to local index key
        try:
            index_key = filtered._translated_key(dkey)
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
        # XXX: why can't we store the filter as index?
        model = self.model
        ki, kj = k
        xlabels = model.xlabels
        ylabels = model.ylabels
        xlabel = [xlabels[i][kj] for i in range(1, len(xlabels))]
        ylabel = [ylabels[j][ki] for j in range(1, len(ylabels))]
        label_key = tuple(ylabel + xlabel)

        # compute dictionary key out of it
        data = self.filtered_data
        axes_ids = list(data.axes.ids) if isinstance(data, la.LArray) else []
        dkey = dict(zip(axes_ids, label_key))

        # add the "scalar" parts of the filter to it (ie the parts of the
        # filter which removed dimensions)
        dkey.update({k: v for k, v in self.current_filter.items()
                     if np.isscalar(v)})

        # re-transform it to tuple (to make it hashable/to store it in .changes)
        return tuple(dkey[axis_id] for axis_id in self.la_data.axes.ids)


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
        if np.isscalar(data):
            readonly = True
        if isinstance(data, la.LArray):
            axes_info = ' x '.join("%s (%d)" % (display_name, len(axis))
                                   for display_name, axis
                                   in zip(data.axes.display_names, data.axes))
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

        self.arraywidget = ArrayEditorWidget(self, data, readonly, xlabels, ylabels,
                                             minvalue=minvalue, maxvalue=maxvalue)
        layout.addWidget(self.arraywidget, 1, 0)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # not using a QDialogButtonBox with standard Ok/Cancel buttons
        # because that makes it impossible to disable the AutoDefault on them
        # (Enter always "accepts"/close the dialog) which is annoying for edit()
        if readonly:
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.reject)
            close_button.setAutoDefault(False)
            btn_layout.addWidget(close_button)
        else:
            ok_button = QPushButton("&OK")
            ok_button.clicked.connect(self.accept)
            ok_button.setAutoDefault(False)
            btn_layout.addWidget(ok_button)
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            cancel_button.setAutoDefault(False)
            btn_layout.addWidget(cancel_button)
        # r_button = QPushButton("resize")
        # r_button.clicked.connect(self.resize_to_contents)
        # btn_layout.addWidget(r_button)
        layout.addLayout(btn_layout, 2, 0)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True

    def autofit_columns(self):
        self.arraywidget.view.autofit_columns()

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


assignment_pattern = re.compile('[^\[\]]+[^=]=[^=].+')
setitem_pattern = re.compile('(.+)\[.+\][^=]=[^=].+')
history_vars_pattern = re.compile('_i?\d+')
# XXX: add all scalars except strings (from numpy or plain Python)?
# (long) strings are not handled correctly so should NOT be in this list
# tuple, list
DISPLAY_IN_GRID = (la.LArray, np.ndarray)


class MappingEditor(QMainWindow):
    """Session Editor Dialog"""

    MAX_RECENT_FILES = 10

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        # to handle recently opened files
        settings = QSettings()
        # XXX: use recent_file_list?
        if settings.value("recentFileList") is None:
            settings.setValue("recentFileList", [])
        self.recent_file_actions = [QAction(self) for _ in range(self.MAX_RECENT_FILES)]
        self.current_file = None
        self.current_array = None
        self.current_array_name = None

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        self.arraywidget = None
        self._listwidget = None
        self.eval_box = None
        self.expressions = {}
        self.kernel = None
        self._unsaved_modifications = False

        self.setup_menu_bar()

    def setup_and_check(self, data, title='', readonly=False, minvalue=None, maxvalue=None):
        """
        Setup MappingEditor:
        return False if data is not supported, True otherwise
        """
        if not isinstance(data, la.Session):
            data = la.Session(data)
        self.data = data

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Session viewer") if readonly else _("Session editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.title = title
        self.setWindowTitle(self.title)

        self.statusBar().showMessage("Welcome to the LArray Viewer", 4000)

        widget = QWidget()
        self.setCentralWidget(widget)

        layout = QVBoxLayout()
        widget.setLayout(layout)

        self._listwidget = QListWidget(self)
        arrays = [k for k, v in self.data.items() if self._display_in_grid(k, v)]
        self.add_list_items(arrays)
        self._listwidget.currentItemChanged.connect(self.on_item_changed)
        self._listwidget.setMinimumWidth(45)

        del_item_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self._listwidget)
        del_item_shortcut.activated.connect(self.delete_current_item)

        start_array = la.zeros(1) if arrays else la.zeros(0)
        self.arraywidget = ArrayEditorWidget(self, start_array, readonly)

        if qtconsole_available:
            # Create an in-process kernel
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel = kernel_manager.kernel

            kernel.shell.run_cell('from larray import *')
            kernel.shell.push(dict(self.data.items()))
            text_formatter = kernel.shell.display_formatter.formatters['text/plain']

            def void_formatter(array, *args, **kwargs):
                return ''

            for type_ in DISPLAY_IN_GRID:
                text_formatter.for_type(type_, void_formatter)

            self.kernel = kernel

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            ipython_widget = RichJupyterWidget()
            ipython_widget.kernel_manager = kernel_manager
            ipython_widget.kernel_client = kernel_client
            ipython_widget.executed.connect(self.ipython_cell_executed)
            ipython_widget._display_banner = False

            self.eval_box = ipython_widget
            self.eval_box.setMinimumHeight(20)

            arraywidget = self.arraywidget
            if not readonly:
                # Buttons configuration
                btn_layout = QHBoxLayout()
                btn_layout.addStretch()

                bbox = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Discard)

                apply_btn = bbox.button(QDialogButtonBox.Apply)
                apply_btn.clicked.connect(self.apply_changes)

                discard_btn = bbox.button(QDialogButtonBox.Discard)
                discard_btn.clicked.connect(self.discard_changes)

                btn_layout.addWidget(bbox)

                arraywidget_layout = QVBoxLayout()
                arraywidget_layout.addWidget(self.arraywidget)
                arraywidget_layout.addLayout(btn_layout)

                # you cant add a layout directly in a splitter, so we have to wrap it in a widget
                arraywidget = QWidget()
                arraywidget.setLayout(arraywidget_layout)

            right_panel_widget = QSplitter(Qt.Vertical)
            right_panel_widget.addWidget(arraywidget)
            right_panel_widget.addWidget(self.eval_box)
            right_panel_widget.setSizes([90, 10])
        else:
            self.eval_box = QLineEdit()
            self.eval_box.returnPressed.connect(self.line_edit_update)

            right_panel_layout = QVBoxLayout()
            right_panel_layout.addWidget(self.arraywidget)
            right_panel_layout.addWidget(self.eval_box)

            # you cant add a layout directly in a splitter, so we have to wrap
            # it in a widget
            right_panel_widget = QWidget()
            right_panel_widget.setLayout(right_panel_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._listwidget)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([10, 90])
        main_splitter.setCollapsible(1, False)

        layout.addWidget(main_splitter)

        self._listwidget.setCurrentRow(0)

        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True

    def _reset(self):
        self.data = la.Session()
        self._listwidget.clear()
        self.current_array = None
        self.current_array_name = None
        if qtconsole_available:
            self.kernel.shell.reset()
            self.kernel.shell.run_cell('from larray import *')
            self.ipython_cell_executed()
        else:
            self.eval_box.setText('None')
            self.line_edit_update()

    def setup_menu_bar(self):
        """Setup menu bar"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')

        file_menu.addAction(create_action(self, _('&New'), shortcut="Ctrl+N", triggered=self.new))
        file_menu.addAction(create_action(self, _('&Open'), shortcut="Ctrl+O", triggered=self.open,
                                          statustip=_('Load session from file')))
        file_menu.addAction(create_action(self, _('&Save'), shortcut="Ctrl+S", triggered=self.save,
                                          statustip=_('Save all arrays as a session in a file')))
        file_menu.addAction(create_action(self, _('Save &As'), triggered=self.save_as,
                                          statustip=_('Save all arrays as a session in a file')))

        recent_files_menu = file_menu.addMenu("Open &Recent")
        for action in self.recent_file_actions:
            action.setVisible(False)
            action.triggered.connect(self.open_recent_file)
            recent_files_menu.addAction(action)
        self.update_recent_file_actions()
        recent_files_menu.addSeparator()
        recent_files_menu.addAction(create_action(self, _('&Clear List'), triggered=self._clear_recent_files))

        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Quit'), shortcut="Ctrl+Q", triggered=self.close))

        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction(create_action(self, _('Online &Documentation'), shortcut="Ctrl+H",
                                          triggered=self.open_documentation))
        help_menu.addAction(create_action(self, _('Online &Tutorial'), triggered=self.open_tutorial))
        help_menu.addAction(create_action(self, _('Online Objects and Functions (API) &Reference'),
                                          triggered=self.open_api_documentation))

    def add_list_item(self, name):
        listitem = QListWidgetItem(self._listwidget)
        listitem.setText(name)
        value = self.data[name]
        if isinstance(value, la.LArray):
            listitem.setToolTip(str(value.info))

    def add_list_items(self, names):
        for name in names:
            self.add_list_item(name)

    def delete_list_item(self, to_delete):
        deleted_items = self._listwidget.findItems(to_delete, Qt.MatchExactly)
        assert len(deleted_items) == 1
        deleted_item_idx = self._listwidget.row(deleted_items[0])
        self._listwidget.takeItem(deleted_item_idx)

    def select_list_item(self, to_display):
        changed_items = self._listwidget.findItems(to_display, Qt.MatchExactly)
        assert len(changed_items) == 1
        prev_selected = self._listwidget.selectedItems()
        assert len(prev_selected) <= 1
        # if the currently selected item (value) need to be refreshed (e.g it was modified)
        if prev_selected and prev_selected[0] == changed_items[0]:
            # we need to update the array widget explicitly
            self.set_current_array(self.data[to_display], to_display)
        else:
            # for some reason, on_item_changed is not triggered when no item was selected
            if not prev_selected:
                self.set_current_array(self.data[to_display], to_display)
            self._listwidget.setCurrentItem(changed_items[0])

    def update_mapping(self, value):
        # XXX: use ordered set so that the order is non-random if the underlying container is ordered?
        keys_before = set(self.data.keys())
        keys_after = set(value.keys())
        # contains both new and updated keys (but not deleted keys)
        changed_keys = [k for k in keys_after if value[k] is not self.data.get(k)]

        # when a key is re-assigned, it can switch from being displayable to non-displayable or vice versa
        displayable_keys_before = set(k for k in keys_before if self._display_in_grid(k, self.data[k]))
        displayable_keys_after = set(k for k in keys_after if self._display_in_grid(k, value[k]))
        deleted_displayable_keys = displayable_keys_before - displayable_keys_after
        new_displayable_keys = displayable_keys_after - displayable_keys_before
        # this can contain more keys than new_displayble_keys (because of existing keys which changed value)
        changed_displayable_keys = [k for k in changed_keys if self._display_in_grid(k, value[k])]

        # 1) update session/mapping
        # a) deleted old keys
        for k in keys_before - keys_after:
            del self.data[k]
        # b) add new/modify existing keys
        for k in changed_keys:
            self.data[k] = value[k]

        # 2) update list widget
        for k in deleted_displayable_keys:
            self.delete_list_item(k)
        self.add_list_items(new_displayable_keys)

        # 3) mark session as dirty if needed
        if len(changed_displayable_keys) > 0 or deleted_displayable_keys:
            self._unsaved_modifications = True

        # 4) change displayed array in the array widget
        # only display first result if there are more than one
        to_display = changed_displayable_keys[0] if changed_displayable_keys else None
        if to_display is not None:
            self.select_list_item(to_display)
        return to_display

    def delete_current_item(self):
        current_item = self._listwidget.currentItem()
        name = str(current_item.text())
        del self.data[name]
        if qtconsole_available:
            self.kernel.shell.del_var(name)
        self._listwidget.takeItem(self._listwidget.row(current_item))

    def line_edit_update(self):
        s = self.eval_box.text()
        if assignment_pattern.match(s):
            context = self.data._objects.copy()
            exec(s, la.__dict__, context)
            varname = self.update_mapping(context)
            if varname is not None:
                self.expressions[varname] = s
        else:
            self.view_expr(eval(s, la.__dict__, self.data))

    def view_expr(self, array):
        self._listwidget.clearSelection()
        self.set_current_array(array, '<expr>')

    def _display_in_grid(self, k, v):
        return not k.startswith('__') and isinstance(v, DISPLAY_IN_GRID)

    def ipython_cell_executed(self):
        user_ns = self.kernel.shell.user_ns
        ip_keys = set(['In', 'Out', '_', '__', '___',
                       '__builtin__',
                       '_dh', '_ih', '_oh', '_sh', '_i', '_ii', '_iii',
                       'exit', 'get_ipython', 'quit'])
        # '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__',
        clean_ns_keys = set([k for k, v in user_ns.items() if not history_vars_pattern.match(k)]) - ip_keys
        clean_ns = {k: v for k, v in user_ns.items() if k in clean_ns_keys}

        # user_ns['_i'] is not updated yet (refers to the -2 item)
        # 'In' and '_ih' point to the same object (but '_ih' is supposed to be the non-overridden one)
        cur_input_num = len(user_ns['_ih']) - 1
        last_input = user_ns['_ih'][-1]
        if setitem_pattern.match(last_input):
            m = setitem_pattern.match(last_input)
            varname = m.group(1)
            # otherwise it should have failed at this point, but let us be sure
            if varname in clean_ns:
                if self._display_in_grid(varname, clean_ns[varname]):
                    self.select_list_item(varname)
        else:
            # not setitem => assume expr or normal assignment
            if last_input in clean_ns:
                # the name exists in the session (variable)
                if self._display_in_grid('', self.data[last_input]):
                    # select and display it
                    self.select_list_item(last_input)
            else:
                # any statement can contain a call to a function which updates globals
                self.update_mapping(clean_ns)

                # if the statement produced any output (probably because it is a simple expression), display it.

                # _oh and Out are supposed to be synonyms but "_ih" is supposed to be the non-overridden one.
                # It would be easier to use '_' instead but that refers to the last output, not the output of the
                # last command. Which means that if the last command did not produce any output, _ is not modified.
                cur_output = user_ns['_oh'].get(cur_input_num)
                if cur_output is not None:
                    if self._display_in_grid('_', cur_output):
                        self.view_expr(cur_output)

                    if isinstance(cur_output, matplotlib.axes.Subplot) and 'inline' not in matplotlib.get_backend():
                        canvas = FigureCanvas(cur_output.figure)
                        main = PlotDialog(canvas, self)
                        main.show()

    def on_item_changed(self, curr, prev):
        if curr is not None:
            name = str(curr.text())
            array = self.data[name]
            self.set_current_array(array, name)
            expr = self.expressions.get(name, name)
            if qtconsole_available:
                # this does not work because it updates the NEXT input, not the
                # current one (it is supposed to be called from within the console)
                # self.kernel.shell.set_next_input(expr, replace=True)
                # self.kernel_client.input(expr)
                pass
            else:
                self.eval_box.setText(expr)

    def update_title(self):
        array = self.current_array
        name = self.current_array_name
        title = []
        if isinstance(array, la.LArray):
            # current file (if not None)
            if self.current_file is not None:
                if os.path.isdir(self.current_file):
                    title = ['{}/{}.csv'.format(self.current_file, name)]
                else:
                    title = [self.current_file]
            # array info
            axes = array.axes
            axes_info = ' x '.join("%s (%d)" % (display_name, len(axis))
                                   for display_name, axis
                                   in zip(axes.display_names, axes))
            title += [(name + ': ' + axes_info) if name else axes_info]
        # name of non-LArray displayed item (if not None)
        elif name:
            title = [name]
        # extra info
        title += [self.title]
        self.setWindowTitle(' - '.join(title))

    def set_current_array(self, array, name):
        self.current_array = array
        self.current_array_name = name
        self.arraywidget.set_data(array)
        self.update_title()

    def _add_arrays(self, arrays):
        for k, v in arrays.items():
            self.data[k] = v
            self.add_list_item(k)
        if qtconsole_available:
            self.kernel.shell.push(dict(arrays))

    def _is_unsaved_modifications(self):
        if self.arraywidget.model.readonly:
            return False
        else:
            return self.arraywidget.dirty or self._unsaved_modifications

    def _ask_to_save_if_unsaved_modifications(self):
        """
        Returns
        -------
        bool
            whether or not the process should continue
        """
        if self._is_unsaved_modifications():
            ret = QMessageBox.warning(self, "Warning", "The data has been modified.\nDo you want to save your changes?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if ret == QMessageBox.Save:
                self.apply_changes()
                return self.save()
            elif ret == QMessageBox.Cancel:
                return False
            else:
                return True
        else:
            return True

    def new(self):
        if self._ask_to_save_if_unsaved_modifications():
            self._reset()
            self.arraywidget.set_data(la.zeros(0))
            self.set_current_file(None)
            self._unsaved_modifications = False
            self.statusBar().showMessage("Viewer has been reset", 4000)

    def _open_file(self, filepath):
        self._reset()
        session = la.Session()
        if '.csv' in filepath:
            filepath = [filepath]
        if isinstance(filepath, (list, tuple)):
            session.load(None, filepath)
            dirname = os.path.dirname(filepath[0])
            basenames = [os.path.basename(fpath) for fpath in filepath]
            self.set_current_file(dirname)
            self.statusBar().showMessage("CSV files {} loaded".format(' ,'.join(basenames)), 4000)
        else:
            session.load(filepath)
            self.set_current_file(filepath)
            self.statusBar().showMessage("File {} loaded".format(os.path.basename(filepath)), 4000)
        self._add_arrays(session)
        self._listwidget.setCurrentRow(0)
        self._unsaved_modifications = False

    def open(self):
        if self._ask_to_save_if_unsaved_modifications():
            filter = "All (*.xls *xlsx *.h5 *.csv);;Excel Files (*.xls *xlsx);;HDF Files (*.h5);;CSV Files (*.csv)"
            res = QFileDialog.getOpenFileNames(self, filter=filter)
            # Qt5 returns a tuple (filepaths, '') instead of a string
            filepaths = res[0] if PYQT5 else res
            if len(filepaths) >= 1:
                if all(['.csv' in filepath for filepath in filepaths]):
                    self._open_file(filepaths)
                elif len(filepaths) == 1:
                    self._open_file(filepaths[0])
                else:
                    QMessageBox.critical(self, "Error",
                                         "Only several CSV files can be loaded in the same time")

    def open_recent_file(self):
        if self._ask_to_save_if_unsaved_modifications():
            action = self.sender()
            if action:
                filepath = action.data()
                if os.path.exists(filepath):
                    self._open_file(filepath)
                else:
                    QMessageBox.warning(self, "Warning", "File {} could not be found".format(filepath))

    def _save_data(self, filepath):
        session = la.Session({k: v for k, v in self.data.items() if self._display_in_grid(k, v)})
        session.save(filepath)
        self.set_current_file(filepath)
        self._unsaved_modifications = False
        self.statusBar().showMessage("Arrays saved in file {}".format(filepath), 4000)

    def save(self):
        """
        Returns
        -------
        bool
            whether or not the data was actually saved
        """
        if self.current_file is not None:
            self._save_data(self.current_file)
            return True
        else:
            return self.save_as()

    def save_as(self):
        # TODO: use filter
        dialog = QFileDialog(self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        accepted = dialog.exec_() == QDialog.Accepted
        if accepted:
            self._save_data(dialog.selectedFiles()[0])
        return accepted

    def open_documentation(self):
        QDesktopServices.openUrl(QUrl("http://larray.readthedocs.io/en/stable/"))

    def open_tutorial(self):
        QDesktopServices.openUrl(QUrl("http://larray.readthedocs.io/en/stable/notebooks/LArray_intro.html"))

    def open_api_documentation(self):
        QDesktopServices.openUrl(QUrl("http://larray.readthedocs.io/en/stable/api.html"))

    def set_current_file(self, filepath):
        self.update_recent_files([filepath])
        self.current_file = filepath
        self.update_title()

    def update_recent_files(self, filepaths):
        settings = QSettings()
        files = settings.value("recentFileList")
        for filepath in filepaths:
            if filepath is not None:
                if filepath in files:
                    files.remove(filepath)
                files = [filepath] + files
        settings.setValue("recentFileList", files[:self.MAX_RECENT_FILES])
        self.update_recent_file_actions()

    def _clear_recent_files(self):
        settings = QSettings()
        settings.setValue("recentFileList", [])
        self.update_recent_file_actions()

    def update_recent_file_actions(self):
        settings = QSettings()
        recent_files = settings.value("recentFileList")
        if recent_files is None:
            recent_files = []

        # zip will iterate up to the shortest of the two
        for filepath, action in zip(recent_files, self.recent_file_actions):
            action.setText(os.path.basename(filepath))
            action.setStatusTip(filepath)
            action.setData(filepath)
            action.setVisible(True)
        # if we have less recent recent files than actions, hide the remaining actions
        for action in self.recent_file_actions[len(recent_files):]:
            action.setVisible(False)

    def closeEvent(self, event):
        if self._ask_to_save_if_unsaved_modifications():
            event.accept()
        else:
            event.ignore()

    def apply_changes(self):
        # update _unsaved_modifications only if 1 or more changes have been applied
        if len(self.arraywidget.model.changes) > 0:
            self._unsaved_modifications = True
        self.arraywidget.accept_changes()

    def discard_changes(self):
        self.arraywidget.reject_changes()

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
        assert len(np.unique(self.positions)) == len(self.positions)
        self.colors = np.array(colors)

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : float

        Returns
        -------
        QColor
        """
        if key != key:
            key = self.positions[0]
        pos_idx = np.searchsorted(self.positions, key, side='right') - 1
        # if we are exactly on one of the bounds
        if pos_idx > 0 and key in self.positions:
            pos_idx -= 1
        pos0, pos1 = self.positions[pos_idx:pos_idx + 2]
        # col0 and col1 are ndarrays
        col0, col1 = self.colors[pos_idx:pos_idx + 2]
        assert pos0 != pos1
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

        self.arrays = None
        self.array = None
        self.arraywidget = None

    def setup_and_check(self, arrays, names, title=''):
        """
        Setup ArrayComparator:
        return False if data is not supported, True otherwise
        """
        assert all(isinstance(a, la.LArray) for a in arrays)
        self.arrays = arrays
        self.array = la.stack(arrays, la.Axis(names, 'arrays'))

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        diff = self.array - self.array[la.x.arrays.i[0]]
        absmax = abs(diff).max()

        # max diff label
        maxdiff_layout = QHBoxLayout()
        maxdiff_layout.addWidget(QLabel('maximum absolute difference: ' +
                                        str(absmax)))
        maxdiff_layout.addStretch()
        layout.addLayout(maxdiff_layout)

        if absmax:
            # scale diff to 0-1
            bg_value = (diff / absmax) / 2 + 0.5
        else:
            # all 0.5 (white)
            bg_value = la.full_like(diff, 0.5)
        gradient = LinearGradient([(0, [.66, .85, 1., .6]),
                                   (0.5 - 1e-16, [.66, .15, 1., .6]),
                                   (0.5, [1., 0., 1., 1.]),
                                   (0.5 + 1e-16, [.99, .15, 1., .6]),
                                   (1, [.99, .85, 1., .6])])

        self.arraywidget = ArrayEditorWidget(self, self.array, readonly=True,
                                             bg_value=bg_value,
                                             bg_gradient=gradient)

        layout.addWidget(self.arraywidget)

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


# TODO: it should be possible to reuse both MappingEditor and ArrayComparator
class SessionComparator(QDialog):
    """Session Comparator Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.sessions = None
        self.names = None
        self.arraywidget = None
        self.maxdiff_label = None
        self.gradient = LinearGradient([(0, [.66, .85, 1., .6]),
                                        (0.5 - 1e-16, [.66, .15, 1., .6]),
                                        (0.5, [1., 0., 1., 1.]),
                                        (0.5 + 1e-16, [.99, .15, 1., .6]),
                                        (1, [.99, .85, 1., .6])])

    def setup_and_check(self, sessions, names, title=''):
        """
        Setup SessionComparator:
        return False if data is not supported, True otherwise
        """
        assert all(isinstance(s, la.Session) for s in sessions)
        self.sessions = sessions
        self.names = names

        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Session comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        names = sorted(set.union(*[set(s.names) for s in self.sessions]))
        self._listwidget = listwidget = QListWidget(self)
        self._listwidget.addItems(names)
        self._listwidget.currentItemChanged.connect(self.on_item_changed)

        for i, name in enumerate(names):
            arrays = [s.get(name) for s in self.sessions]
            eq = [la.larray_equal(a, arrays[0]) for a in arrays[1:]]
            if not all(eq):
                listwidget.item(i).setForeground(Qt.red)

        array, absmax, bg_value = self.get_array(names[0])

        if not array.size:
            array = la.LArray(['no data'])
        self.arraywidget = ArrayEditorWidget(self, array, readonly=True,
                                             bg_value=bg_value,
                                             bg_gradient=self.gradient)

        right_panel_layout = QVBoxLayout()

        # max diff label
        maxdiff_layout = QHBoxLayout()
        maxdiff_layout.addWidget(QLabel('maximum absolute difference:'))
        self.maxdiff_label = QLabel(str(absmax))
        maxdiff_layout.addWidget(self.maxdiff_label)
        maxdiff_layout.addStretch()
        right_panel_layout.addLayout(maxdiff_layout)

        # array_splitter.setSizePolicy(QSizePolicy.Expanding,
        #                              QSizePolicy.Expanding)
        right_panel_layout.addWidget(self.arraywidget)

        # you cant add a layout directly in a splitter, so we have to wrap it
        # in a widget
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._listwidget)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([5, 95])
        main_splitter.setCollapsible(1, False)

        layout.addWidget(main_splitter)

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

    def get_array(self, name):
        arrays = [s.get(name) for s in self.sessions]
        array = la.stack(arrays, la.Axis(self.names, 'sessions'))
        diff = array - array[la.x.sessions.i[0]]
        absmax = abs(diff).max()
        # scale diff to 0-1
        if absmax:
            bg_value = (diff / absmax) / 2 + 0.5
        else:
            bg_value = la.full_like(diff, 0.5)
        # only show rows with a difference. For some reason, this is abysmally
        # slow though.
        # row_filter = (array != array[la.x.sessions.i[0]]).any(la.x.sessions)
        # array = array[row_filter]
        # bg_value = bg_value[row_filter]
        return array, absmax, bg_value

    def on_item_changed(self, curr, prev):
        array, absmax, bg_value = self.get_array(str(curr.text()))
        self.maxdiff_label.setText(str(absmax))
        self.arraywidget.set_data(array, bg_value=bg_value,
                                  bg_gradient=self.gradient)
