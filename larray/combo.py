from qtpy import QtGui, QtCore, QtWidgets


class StandardItemModelIterator(object):
    def __init__(self, model):
        self.model = model
        self.pos = 0

    def __next__(self):
        if self.pos < self.model.rowCount():
            item = self.model.item(self.pos)
            self.pos += 1
            return item
        else:
            raise StopIteration
    next = __next__


class SequenceStandardItemModel(QtGui.QStandardItemModel):
    """
    an iterable and indexable StandardItemModel
    """
    def __iter__(self):
        return StandardItemModelIterator(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.rowCount()
            if step is None:
                step = 1
            return [self.item(i) for i in range(start, stop, step)]
        else:
            if key >= self.rowCount():
                raise IndexError("index %d is out of range" % key)
            return self.item(key)

    def __len__(self):
        return self.rowCount()


class StandardItem(QtGui.QStandardItem):
    def __init__(self, value):
        super(StandardItem, self).__init__(value)

    def get_checked(self):
        return self.checkState() == QtCore.Qt.Checked

    def set_checked(self, value):
        if isinstance(value, bool):
            qtvalue = (QtCore.Qt.Unchecked, QtCore.Qt.Checked)[value]
        else:
            qtvalue = QtCore.Qt.PartiallyChecked
        self.setCheckState(qtvalue)
    checked = property(get_checked, set_checked)


class FilterMenu(QtWidgets.QMenu):
    activate = QtCore.Signal(int)
    checkedItemsChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(QtWidgets.QMenu, self).__init__(parent)

        self._list_view = QtWidgets.QListView(parent)
        self._list_view.setFrameStyle(0)
        model = SequenceStandardItemModel()
        self._list_view.setModel(model)
        self._model = model
        self.addItem("(select all)")
        model[0].setTristate(True)

        action = QtWidgets.QWidgetAction(self)
        action.setDefaultWidget(self._list_view)
        self.addAction(action)
        self.installEventFilter(self)
        self._list_view.installEventFilter(self)
        self._list_view.window().installEventFilter(self)

        model.itemChanged.connect(self.on_model_item_changed)
        self._list_view.pressed.connect(self.on_list_view_pressed)
        self.activate.connect(self.on_activate)

    def on_list_view_pressed(self, index):
        item = self._model.itemFromIndex(index)
        # item is None when the button has not been used yet (and this is
        # triggered via enter)
        if item is not None:
            item.checked = not item.checked

    def on_activate(self, row):
        target_item = self._model[row]
        for item in self._model[1:]:
            item.checked = item is target_item

    def on_model_item_changed(self, item):
        model = self._model
        model.blockSignals(True)
        if item.index().row() == 0:
            # (un)check first => (un)check others
            for other in model[1:]:
                other.checked = item.checked

        items_checked = [item for item in model[1:] if item.checked]
        num_checked = len(items_checked)

        if num_checked == 0 or num_checked == len(model) - 1:
            model[0].checked = bool(num_checked)
        elif num_checked == 1:
            model[0].checked = 'partial'
        else:
            model[0].checked = 'partial'
        model.blockSignals(False)
        is_checked = [i for i, item in enumerate(model[1:]) if item.checked]
        self.checkedItemsChanged.emit(is_checked)

    def addItem(self, text):
        item = StandardItem(text)
        # not editable
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.checked = True
        self._model.appendRow(item)

    def addItems(self, items):
        for item in items:
            self.addItem(item)

    def eventFilter(self, obj, event):
        event_type = event.type()

        if event_type == QtCore.QEvent.KeyRelease:
            key = event.key()

            # tab key closes the popup
            if obj == self._list_view.window() and key == QtCore.Qt.Key_Tab:
                self.hide()

            # return key activates *one* item and closes the popup
            # first time the key is sent to the menu, afterwards to
            # list_view
            elif (obj == self._list_view and
                          key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return)):
                self.activate.emit(self._list_view.currentIndex().row())
                self.hide()
                return True

        return False


class FilterComboBox(QtWidgets.QToolButton):
    checkedItemsChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(FilterComboBox, self).__init__(parent)
        self.setText("(no filter)")
        # QtGui.QToolButton.InstantPopup would be slightly less work (the
        # whole button works by default, instead of only the arrow) but it is
        # uglier
        self.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)

        menu = FilterMenu(self)
        self.setMenu(menu)
        self._menu = menu
        menu.checkedItemsChanged.connect(self.on_checked_items_changed)
        self.installEventFilter(self)

    def on_checked_items_changed(self, indices_checked):
        num_checked = len(indices_checked)
        model = self._menu._model
        if num_checked == 0 or num_checked == len(model) - 1:
            self.setText("(no filter)")
        elif num_checked == 1:
            self.setText(model[indices_checked[0] + 1].text())
        else:
            self.setText("multi")
        self.checkedItemsChanged.emit(indices_checked)

    def addItem(self, text):
        self._menu.addItem(text)

    def addItems(self, items):
        self._menu.addItems(items)

    def eventFilter(self, obj, event):
        event_type = event.type()

        # this is not enabled because it causes all kind of troubles
        # if event_type == QtCore.QEvent.KeyPress:
        #     key = event.key()
        #
        #     # allow opening the popup via enter/return
        #     if (obj == self and
        #             key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)):
        #         self.showMenu()
        #         return True

        if event_type == QtCore.QEvent.KeyRelease:
            key = event.key()

            # allow opening the popup with up/down
            if (obj == self and
                    key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                            QtCore.Qt.Key_Space)):
                self.showMenu()
                return True

            # return key activates *one* item and closes the popup
            # first time the key is sent to self, afterwards to list_view
            elif (obj == self and
                    key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return)):
                self._menu.activate.emit(self._list_view.currentIndex().row())
                self._menu.hide()
                return True

        if event_type == QtCore.QEvent.MouseButtonRelease:
            # clicking anywhere (not just arrow) on the button shows the popup
            if obj == self:
                self.showMenu()

        return False


if __name__ == '__main__':
    import sys

    class TestDialog(QtWidgets.QDialog):
        def __init__(self):
            super(QtWidgets.QDialog, self).__init__()
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            combo = FilterComboBox(self)
            for i in range(20):
                combo.addItem('Item %s' % i)
            layout.addWidget(combo)

    app = QtWidgets.QApplication(sys.argv)
    dialog = TestDialog()
    dialog.resize(200, 200)
    dialog.show()
    sys.exit(app.exec_())
