from __future__ import absolute_import, division, print_function

from unittest import TestCase
import unittest

from larray import Axis, LArray, ndrange
from larray.tests.test_la import assert_array_equal
from larray.ipfp import ipfp


class TestIPFP(TestCase):
    def test_ipfp(self):
        a = Axis(2, 'a')
        b = Axis(2, 'b')
        initial = LArray([[2, 1], [1, 2]], [a, b])

        # array sums already match target sums
        # [3, 3], [3, 3]
        r = ipfp([initial.sum(a), initial.sum(b)], initial)
        assert_array_equal(r, [[2, 1], [1, 2]])

        # array sums do not match target sums (ie the usual case)
        along_a = LArray([2, 1], b)
        along_b = LArray([1, 2], a)
        r = ipfp([along_a, along_b], initial)
        assert_array_equal(r, [[0.8, 0.2], [1.0, 1.0]])

        # same as above but using a more precise threshold
        r = ipfp([along_a, along_b], initial, threshold=0.01)
        assert_array_equal(r, [[0.8450704225352113, 0.15492957746478875],
                               [1.1538461538461537, 0.8461538461538463]])

        # inverted target sums
        with self.assertRaisesRegexp(ValueError, "axes of target sum along axis 0 \(a\) do not match corresponding "
                                                 "array axes: got {a\*} but expected {b\*}. Are the target sums in the "
                                                 "correct order\?"):
            ipfp([along_b, along_a], initial, threshold=0.01)

    def test_ipfp_no_values(self):
        # 6, 12, 18
        along_a = ndrange([(3, 'b')], start=1) * 6
        # 6, 12, 18
        along_b = ndrange([(3, 'a')], start=1) * 6
        r = ipfp([along_a, along_b])
        assert_array_equal(r, [[1.0, 2.0, 3.0],
                               [2.0, 4.0, 6.0],
                               [3.0, 6.0, 9.0]])

        along_a = LArray([2, 1], Axis(2, 'b'))
        along_b = LArray([1, 2], Axis(2, 'a'))
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
        initial = LArray([[2, 1], [1, 2]])

        # sums already correct
        # [3, 3], [3, 3]
        r = ipfp([initial.sum(axis=0), initial.sum(axis=1)], initial)
        assert_array_equal(r, [[2, 1], [1, 2]])

        # different sums (ie the usual case)
        along_a = LArray([2, 1])
        along_b = LArray([1, 2])
        r = ipfp([along_a, along_b], initial)
        assert_array_equal(r, [[0.8, 0.2], [1.0, 1.0]])

    def test_ipfp_non_larray(self):
        initial = [[2, 1], [1, 2]]

        # sums already correct
        r = ipfp([[3, 3], [3, 3]], initial)
        assert_array_equal(r, [[2, 1], [1, 2]])

        # different sums (ie the usual case)
        r = ipfp([[2, 1], [1, 2]], initial)
        assert_array_equal(r, [[0.8, 0.2], [1.0, 1.0]])

if __name__ == "__main__":
    unittest.main()
