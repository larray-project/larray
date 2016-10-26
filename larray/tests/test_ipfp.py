from __future__ import absolute_import, division, print_function

from unittest import TestCase
import unittest

from larray import Axis, LArray, ndrange
from larray.tests.test_la import assert_array_equal
from larray.ipfp import ipfp


class TestIPFP(TestCase):
    def test_ipfp(self):
        a = Axis('a', 2)
        b = Axis('b', 2)
        initial = LArray([[2, 1],
                          [1, 2]], [a, b])

        # sums already correct
        # [3, 3], [3, 3]
        r = ipfp([initial.sum(a), initial.sum(b)], initial)
        assert_array_equal(r, [[2, 1],
                               [1, 2]])

        # different sums (ie the usual case)
        along_a = LArray([2, 1], b)
        along_b = LArray([1, 2], a)
        r = ipfp([along_a, along_b], initial)
        assert_array_equal(r, [[0.8, 0.2],
                               [1.0, 1.0]])

        # different sums, more precise threshold
        along_a = LArray([2, 1], b)
        along_b = LArray([1, 2], a)
        r = ipfp([along_a, along_b], initial, threshold=0.01)
        assert_array_equal(r, [[0.8450704225352113, 0.15492957746478875],
                               [1.1538461538461537, 0.8461538461538463]])

    def test_ipfp_no_values(self):
        # 6, 12, 18
        along_a = ndrange([('b', 3)], start=1) * 6
        # 6, 12, 18
        along_b = ndrange([('a', 3)], start=1) * 6
        r = ipfp([along_a, along_b])
        assert_array_equal(r, [[1.0, 2.0, 3.0],
                               [2.0, 4.0, 6.0],
                               [3.0, 6.0, 9.0]])

        along_a = LArray([2, 1], Axis('b', 2))
        along_b = LArray([1, 2], Axis('a', 2))
        r = ipfp([along_a, along_b])
        assert_array_equal(r, [[2 / 3, 1 / 3],
                               [4 / 3, 2 / 3]])

    def test_ipfp_no_values_no_name(self):
        r = ipfp([[6, 12, 18], [6, 12, 18]])
        assert_array_equal(r, [[1.0, 2.0, 3.0],
                               [2.0, 4.0, 6.0],
                               [3.0, 6.0, 9.0]])

        r = ipfp([[2, 1], [1, 2]])
        assert_array_equal(r, [[2 / 3, 1 / 3],
                               [4 / 3, 2 / 3]])

    def test_ipfp_no_name(self):
        initial = LArray([[2, 1],
                          [1, 2]])

        # sums already correct
        # [3, 3], [3, 3]
        r = ipfp([initial.sum(axis=0), initial.sum(axis=1)], initial)
        assert_array_equal(r, [[2, 1],
                               [1, 2]])

        # different sums (ie the usual case)
        along_a = LArray([2, 1])
        along_b = LArray([1, 2])
        r = ipfp([along_a, along_b], initial)
        assert_array_equal(r, [[0.8, 0.2],
                               [1.0, 1.0]])

    def test_ipfp_non_larray(self):
        initial = [[2, 1],
                   [1, 2]]

        # sums already correct
        r = ipfp([[3, 3], [3, 3]], initial)
        assert_array_equal(r, [[2, 1],
                               [1, 2]])

        # different sums (ie the usual case)
        r = ipfp([[2, 1], [1, 2]], initial)
        assert_array_equal(r, [[0.8, 0.2],
                               [1.0, 1.0]])

if __name__ == "__main__":
    unittest.main()
