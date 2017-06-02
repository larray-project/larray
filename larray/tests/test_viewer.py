import pytest
from pytestqt import qtbot

from qtpy.QtCore import Qt, QModelIndex
import numpy as np
import pandas as pd

from larray import ndtest, zeros, LArray
from larray import view, edit, compare
from larray.viewer.model import ArrayModel

@pytest.fixture(scope="module")
def data():
    return ndtest((5, 5, 5, 5))

@pytest.fixture(scope="module")
def model(data):
    return ArrayModel(data)

# see https://pytest-qt.readthedocs.io/en/latest/modeltester.html
# def test_generic(model, qtmodeltester):
#     qtmodeltester.check(model)

def test_create_model():
    # data = None --> empty array with shape (0,0) is generated
    model = ArrayModel()
    assert model.get_data().shape == (0, 0)

    # data = scalar
    model = ArrayModel(LArray(5))
    assert model.get_data().shape == (1, 1)

    # data = 1D array --> a fake axis is added
    model = ArrayModel(ndtest(5))
    assert model.get_data().shape == (1, 5)

    # data = 3D array --> reshaped to 2D with dim (axis1*axis2, axis3)
    model = ArrayModel(ndtest((5, 5, 5)))
    assert model.get_data().shape == (5*5, 5)

def test_row_column_count(data, model):
    assert model.rowCount() == np.prod(data.shape[:-1]) + 1
    assert model.columnCount() == data.ndim - 1 + len(data.axes[-1])

    data2 = zeros((ArrayModel.ROWS_TO_LOAD + 1, ArrayModel.COLS_TO_LOAD + 1))
    model2 = ArrayModel(data2)
    assert model2.rowCount() == ArrayModel.ROWS_TO_LOAD + 2
    assert model2.columnCount() == ArrayModel.COLS_TO_LOAD + 2

def test_get_labels(model):
    index = model.index(len(model.xlabels) -2, len(model.ylabels) -2)
    assert model.get_labels(index) == ""
    index = model.index(len(model.xlabels) -1, len(model.ylabels) -1)
    assert model.get_labels(index) == "a=a0, b=b0, c=c0, d=d0"

def test_get_value(data, model):
    assert model.get_value(model.index(0, 0)) == ""

    for j, label in enumerate(data.axes[-1].labels):
        assert model.get_value(model.index(0, data.ndim - 1 + j)) == label

    for i, label in enumerate(data.axes[-2].labels):
        assert model.get_value(model.index(i + 1, data.ndim - 2)) == label

    index = model.index(1, data.ndim - 1)
    assert model.get_value(index) == data.i[0, 0, 0, 0]

if __name__ == "__main__":
    pytest.main()
