from __future__ import absolute_import, division, print_function

import numpy as np

from larray.tests.common import assert_array_equal
from larray.tests.test_array import small_array
from larray import Axis, zeros, exp, clip, where, round, cos


def test_unary():
    args = [0, np.zeros(2), zeros((2, 2))]
    for a in args:
        assert_array_equal(a + 1, np.cos(a))
        assert_array_equal(a + 1, cos(a))


def test_ufuncs(small_array):
    raw = small_array.data

    # simple one-argument ufunc
    assert_array_equal(exp(small_array), np.exp(raw))

    # with out=
    la_out = zeros(small_array.axes)
    raw_out = np.zeros(raw.shape)

    la_out2 = exp(small_array, la_out)
    raw_out2 = np.exp(raw, raw_out)

    # FIXME: this is not the case currently
    # self.assertIs(la_out2, la_out)
    assert_array_equal(la_out2, la_out)
    assert raw_out2 is raw_out

    assert_array_equal(la_out, raw_out)

    # with out= and broadcasting
    # we need to put the 'a' axis first because array numpy only supports that
    la_out = zeros([Axis([0, 1, 2], 'a')] + list(small_array.axes))
    raw_out = np.zeros((3,) + raw.shape)

    la_out2 = exp(small_array, la_out)
    raw_out2 = np.exp(raw, raw_out)

    # self.assertIs(la_out2, la_out)
    # XXX: why is la_out2 transposed?
    assert_array_equal(la_out2.transpose('a'), la_out)
    assert raw_out2 is raw_out

    assert_array_equal(la_out, raw_out)

    sex, lipro = small_array.axes

    low = small_array.sum(sex) // 4 + 3
    raw_low = raw.sum(0) // 4 + 3
    high = small_array.sum(sex) // 4 + 13
    raw_high = raw.sum(0) // 4 + 13

    # LA + scalars
    assert_array_equal(small_array.clip(0, 10), raw.clip(0, 10))
    assert_array_equal(clip(small_array, 0, 10), np.clip(raw, 0, 10))

    # LA + LA (no broadcasting)
    assert_array_equal(clip(small_array, 21 - small_array, 9 + small_array // 2),
                       np.clip(raw, 21 - raw, 9 + raw // 2))

    # LA + LA (with broadcasting)
    assert_array_equal(clip(small_array, low, high),
                       np.clip(raw, raw_low, raw_high))

    # where (no broadcasting)
    assert_array_equal(where(small_array < 5, -5, small_array),
                       np.where(raw < 5, -5, raw))

    # where (transposed no broadcasting)
    assert_array_equal(where(small_array < 5, -5, small_array.T),
                       np.where(raw < 5, -5, raw))

    # where (with broadcasting)
    result = where(small_array['P01'] < 5, -5, small_array)
    assert result.axes.names == ['sex', 'lipro']
    assert_array_equal(result, np.where(raw[:, [0]] < 5, -5, raw))

    # round
    small_float = small_array + 0.6
    rounded = round(small_float)
    assert_array_equal(rounded, np.round(raw + 0.6))
