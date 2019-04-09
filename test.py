#!/usr/bin/env python3.6

import unittest

import torch
import numpy as np

import utils


class TestCentroid(unittest.TestCase):
    def test_center_square(self):
        t = torch.zeros(1, 1, 100, 100)
        t[0, 0, 40:60, 40:60] = 1

        res = utils.soft_centroid(t)[0, 0]
        exp = torch.Tensor([49.5, 49.5])
        self.assertTrue(torch.equal(res, exp), (res, exp))

    def test_line(self):
        t = torch.zeros(1, 1, 100, 100)
        t[0, 0, :, 20] = 1

        res = utils.soft_centroid(t)[0, 0]
        exp = torch.Tensor([49.5, 20])
        self.assertTrue(torch.equal(res, exp), (res, exp))

    def test_empty(self):
        t = torch.zeros(1, 1, 100, 100)

        res = utils.soft_centroid(t)[0, 0]
        exp = torch.Tensor([0, 0])
        self.assertTrue(torch.equal(res, exp), (res, exp))


class TestDice(unittest.TestCase):
    def test_equal(self):
        t = torch.zeros(1, 100, 100)
        t[0, 40:60, 40:60] = 1

        c = utils.class2one_hot(t, C=2)

        self.assertEqual(utils.dice_coef(c, c)[0, 0], 1)

    def test_empty(self):
        t = torch.zeros(1, 100, 100)
        t[0, 40:60, 40:60] = 1

        c = utils.class2one_hot(t, C=2)

        self.assertEqual(utils.dice_coef(c, c)[0, 0], 1)

    def test_caca(self):
        t = torch.zeros(1, 100, 100)
        t[0, 40:60, 40:60] = 1

        c = utils.class2one_hot(t, C=2)
        z = torch.zeros_like(c)
        z[0, 1, ...] = 1

        self.assertEqual(utils.dice_coef(c, z, smooth=0)[0, 0], 0)  # Annoying to deal with the almost equal thing


class TestNumpyHaussdorf(unittest.TestCase):
    def test_closure(self):
        a = np.zeros((256, 256))
        a[50:60, :] = 1

        self.assertEqual(utils.numpy_haussdorf(a, a), 0)

    def test_empty(self):
        a = np.zeros((256, 256))

        self.assertEqual(utils.numpy_haussdorf(a, a), 0)

    def test_caca(self):
        a = np.zeros((256, 256))
        a[50:60, :] = 1

        z = np.zeros_like(a)

        self.assertEqual(utils.numpy_haussdorf(z, a), 16)

    def test_symmetry(self):
        a = np.zeros((256, 256))
        a[50:60, :] = 1

        z = np.zeros_like(a)

        self.assertEqual(utils.numpy_haussdorf(z, a), utils.numpy_haussdorf(a, z))


class TestDistMap(unittest.TestCase):
    def test_closure(self):
        a = np.zeros((1, 256, 256))
        a[:, 50:60, :] = 1

        o = utils.class2one_hot(torch.Tensor(a).type(torch.float32), C=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        neg = (res <= 0) * res

        self.assertEqual(neg.sum(), (o * res).sum())

    def test_full_coverage(self):
        a = np.zeros((1, 256, 256))
        a[:, 50:60, :] = 1

        o = utils.class2one_hot(torch.Tensor(a).type(torch.float32), C=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        self.assertEqual((res[1] <= 0).sum(), a.sum())
        self.assertEqual((res[1] > 0).sum(), (1 - a).sum())

    def test_empty(self):
        a = np.zeros((1, 256, 256))

        o = utils.class2one_hot(torch.Tensor(a).type(torch.float32), C=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        self.assertEqual(res[1].sum(), 0)
        self.assertEqual((res[0] <= 0).sum(), a.size)

    def test_max_dist(self):
        """
        The max dist for a box should be at the midle of the object, +-1
        """
        a = np.zeros((1, 256, 256))
        a[:, 1:254, 1:254] = 1

        o = utils.class2one_hot(torch.Tensor(a).type(torch.float32), C=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        self.assertEqual(res[0].max(), 127)
        self.assertEqual(np.unravel_index(res[0].argmax(), (256, 256)), (127, 127))

        self.assertEqual(res[1].min(), -126)
        self.assertEqual(np.unravel_index(res[1].argmin(), (256, 256)), (127, 127))

    def test_border(self):
        """
        Make sure the border inside the object is 0 in the distance map
        """

        for l in range(3, 5):
            a = np.zeros((1, 25, 25))
            a[:, 3:3 + l, 3:3 + l] = 1

            o = utils.class2one_hot(torch.Tensor(a).type(torch.float32), C=2).numpy()
            res = utils.one_hot2dist(o[0])
            self.assertEqual(res.shape, (2, 25, 25))

            border = (res[1] == 0)

            self.assertEqual(border.sum(), 4 * (l - 1))


if __name__ == "__main__":
    unittest.main()
