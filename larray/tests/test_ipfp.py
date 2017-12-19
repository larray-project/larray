from __future__ import absolute_import, division, print_function

from unittest import TestCase

import pytest

from larray.tests.common import assert_array_equal
from larray import Axis, LArray, ndtest, ipfp, X


class TestIPFP(TestCase):
    def test_ipfp(self):
        a = Axis('a=a0,a1')
        b = Axis('b=b0,b1')
        initial = LArray([[2, 1], [1, 2]], [a, b])

        # array sums already match target sums
        # [3, 3], [3, 3]
        r = ipfp([initial.sum(a), initial.sum(b)], initial)
        assert_array_equal(r, initial)

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
        with self.assertRaisesRegexp(ValueError, "axes of target sum along a \(axis 0\) do not match corresponding "
                                                 "array axes: got {a} but expected {b}. Are the target sums in the "
                                                 "correct order\?"):
            ipfp([along_b, along_a], initial)

        # different target sums totals
        along_a = LArray([2, 1], b)
        along_b = LArray([1, 3], a)
        with self.assertRaisesRegexp(ValueError, "target sum along b \(axis 1\) is different than target sum along "
                                                 "a \(axis 0\): 4 vs 3"):
            ipfp([along_a, along_b], initial)

        # all zero values
        initial = LArray([[0, 0], [1, 2]], [a, b])
        along_a = LArray([2, 1], b)
        along_b = LArray([1, 2], a)
        with self.assertRaisesRegexp(ValueError, "found all zero values sum along b \(axis 1\) but non zero target "
                                                 "sum:\na0: 1"):
            ipfp([along_a, along_b], initial)

        # zero target sum
        initial = LArray([[2, 1], [1, 2]], [a, b])
        along_a = LArray([0, 1], b)
        along_b = LArray([1, 0], a)
        with self.assertRaisesRegexp(ValueError, "found Non Zero Values but Zero target Sum \(nzvzs\) along a "
                                                 "\(axis 0\), use nzvzs='warn' or 'fix' to set them to zero "
                                                 "automatically:\nb0: 3"):
            ipfp([along_a, along_b], initial)

        # negative initial values
        initial = LArray([[2, -1], [1, 2]], [a, b])
        with self.assertRaisesRegexp(ValueError, "negative value\(s\) found:\na0_b1: -1"):
            ipfp([along_a, along_b], initial)

    # def test_ipfp_big(self):
    #     initial = ndtest((4000, 4000))
    #     targets = [initial.sum(axis) for axis in initial.axes]
    #     ipfp(targets, display_progress='condensed')

    def test_ipfp_3d(self):
        initial = ndtest((2, 2, 2))
        initial_axes = initial.axes

        # array sums already match target sums
        targets = [initial.sum(axis) for axis in initial.axes]
        r = ipfp(targets, initial)
        assert_array_equal(r, initial)
        self.assertEqual(r.axes, initial_axes)

        # array sums do not match target sums (ie the usual case)
        targets = [initial.sum(axis) + 1 for axis in initial.axes]
        r = ipfp(targets, initial)
        assert_array_equal(r, [[[0.0,               2.0],
                                [2.688963210702341, 3.311036789297659]],
                               [[4.551453540217585, 5.448546459782415],
                                [6.450132391879964, 7.549867608120035]]])
        self.assertEqual(r.axes, initial_axes)

        # same as above but using a more precise threshold
        r = ipfp(targets, initial, threshold=0.01)
        assert_array_equal(r, [[[0.0,               1.9999999999999998],
                                [2.994320023433978, 3.0056799765660225]],
                               [[4.990248916408187, 5.009751083591813],
                                [6.009541632308118, 7.990458367691883]]])
        self.assertEqual(r.axes, initial_axes)

    def test_ipfp_3d_with_axes(self):
        initial = ndtest((2, 2, 2))
        initial_axes = initial.axes

        # array sums already match target sums (first axes)
        axes = (X.a, X.b)
        targets = [initial.sum(axis) for axis in axes]
        r = ipfp(targets, initial, axes=axes)
        assert_array_equal(r, initial)
        self.assertEqual(r.axes, initial_axes)

        # array sums already match target sums (other axes)
        axes = (X.a, X.c)
        targets = [initial.sum(axis) for axis in axes]
        r = ipfp(targets, initial, axes=axes)
        assert_array_equal(r, initial)
        self.assertEqual(r.axes, initial_axes)

        # array sums do not match target sums (ie the usual case) (first axes)
        axes = (X.a, X.b)
        targets = [initial.sum(axis) + 1 for axis in axes]
        r = ipfp(targets, initial, axes=axes)
        assert_array_equal(r, [[[0.0,               1.3059701492537312],
                                [3.0,               3.6940298507462686]],
                               [[4.680851063829787, 5.603448275862069 ],
                                [6.319148936170213, 7.3965517241379315]]])
        self.assertEqual(r.axes, initial_axes)
        # check that the result is the same as N 2D ipfp calls
        assert_array_equal(r['c0'], ipfp([t['c0'] for t in targets], initial['c0']))
        assert_array_equal(r['c1'], ipfp([t['c1'] for t in targets], initial['c1']))

        # array sums do not match target sums (ie the usual case) (other axes)
        axes = (X.a, X.c)
        targets = [initial.sum(axis) + 1 for axis in axes]
        r = ipfp(targets, initial, axes=axes)
        assert_array_equal(r, [[[0.0,               2.0              ],
                                [2.432432432432432, 3.567567567567567]],
                               [[4.615384615384615, 5.384615384615385],
                                [6.539792387543252, 7.460207612456748]]])
        self.assertEqual(r.axes, initial_axes)
        # check that the result is the same as N 2D ipfp calls
        assert_array_equal(r['b0'], ipfp([t['b0'] for t in targets], initial['b0']))
        assert_array_equal(r['b1'], ipfp([t['b1'] for t in targets], initial['b1']))

    def test_ipfp_no_values(self):
        # 6, 12, 18
        along_a = ndtest([(3, 'b')], start=1) * 6
        # 6, 12, 18
        along_b = ndtest([(3, 'a')], start=1) * 6
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
    pytest.main()

