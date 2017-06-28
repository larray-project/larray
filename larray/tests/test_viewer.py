import pytest
from pytestqt import qtbot

from qtpy.QtCore import Qt, QModelIndex
import numpy as np
import pandas as pd

from larray import ndtest, zeros, LArray
from larray import view, edit, compare
from larray.viewer.model import ArrayModel, Product, LARGE_NROWS, LARGE_COLS

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
    assert model.get_data_2D().shape == (0, 0)

    # data = scalar
    model = ArrayModel(LArray(5))
    assert model.get_data_2D().shape == (1, 1)

    # data = 1D array --> reshaped to 2D array with dim (1, len(data))
    model = ArrayModel(ndtest(5))
    assert model.get_data_2D().shape == (1, 5)

    # data = 3D array --> reshaped to 2D array with dim (axis1*axis2, axis3)
    model = ArrayModel(ndtest((5, 5, 5)))
    assert model.get_data_2D().shape == (5 * 5, 5)

def test_row_column_count(data, model):
    nb_labels_prod_axes = np.prod(data.shape[:-1])
    nb_labels_last_axis = len(data.axes[-1])
    # nb_rows = prod(dim_0, ..., dim_n-1) + 1 row to display labels of the last dim
    assert model.rowCount() == nb_labels_prod_axes + 1
    # nb_cols = (n-1) columns to display the product of labels of the n-1 first dimensions +
    #           m columns for the labels of the last dimension
    assert model.columnCount() == data.ndim - 1 + nb_labels_last_axis

    data2 = zeros((int(LARGE_NROWS) + 10, LARGE_COLS + 10))
    model2 = ArrayModel(data2)
    assert model2.rowCount() == ArrayModel.ROWS_TO_LOAD + 1
    assert model2.columnCount() == ArrayModel.COLS_TO_LOAD + 1

def test_length_xylabels(data, model):
    # xlabels --> xlabels[0] = axis names; xlabels[1] = labels of the last axis
    assert len(model.xlabels) == 2
    # ylabels --> ylabels[0] = empty; ylabels[i] = labels for the ith dimension (i <= n-1)
    assert len(model.ylabels) == data.ndim

def test_get_labels(data, model):
    # row and column start at 0
    first_data_row = 1               # = len(model.xlabels) -1
    first_data_col = data.ndim - 1   # = len(model.ylabels) -1

    index = model.index(first_data_row - 1, first_data_col - 1)
    assert model.get_labels(index) == ""
    index = model.index(first_data_row, first_data_col)
    assert model.get_labels(index) == "a=a0, b=b0, c=c0, d=d0"

def test_get_value(data, model):
    # row and column start at 0
    first_data_row = 1               # = len(model.xlabels) -1
    first_data_col = data.ndim - 1   # = len(model.ylabels) -1

    # first cell is empty
    assert model.get_value(model.index(0, 0)) == ""

    # testing xlabels
    labels_last_axis = data.axes[-1].labels
    for j, label in enumerate(labels_last_axis):
        assert model.get_value(model.index(0, first_data_col + j)) == label
        assert model.xlabels[1][j] == label

    # test ylabels
    for i, labels in enumerate(Product(data.axes.labels[:-1])):
        for j, label in enumerate(labels):
            assert model.get_value(model.index(first_data_row + i, j)) == label
            assert model.ylabels[j+1][i] == label

    # test first data
    index = model.index(first_data_row, first_data_col)
    assert model.get_value(index) == data.i[0, 0, 0, 0]

if __name__ == "__main__":
    pytest.main()
