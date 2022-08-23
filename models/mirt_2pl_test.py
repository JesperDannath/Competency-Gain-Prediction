import unittest
from mirt_2pl import mirt_2pl
import numpy as np


class test_mirt_2pl(unittest.TestCase):

    def setUp(self):
        A1 = np.array([[1],
                       [1],
                       [1]], dtype=np.float)
        delta1 = np.array([0, 0, 0], dtype=np.float)
        self.mirt_2pl_1d = mirt_2pl(
            item_dimension=3, latent_dimension=1, A=A1, delta=delta1)
        A2 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float)
        delta2 = np.array([1, 1, 1], dtype=np.float)
        self.mirt_2pl_2d = mirt_2pl(
            item_dimension=3, latent_dimension=2, A=A2, delta=delta2)

    def test_dimensions(self):
        self.assertEqual(self.mirt_2pl_1d.item_dimension, 3)
        self.assertEqual(self.mirt_2pl_2d.item_dimension, 3)
        self.assertEqual(self.mirt_2pl_1d.latent_dimension, 1)
        self.assertEqual(self.mirt_2pl_2d.latent_dimension, 2)

    def test_icc_result_shapes(self):
        # One Person input
        theta1 = np.array([[0]])
        res1 = self.mirt_2pl_1d.icc(theta1)
        self.assertEqual(res1.shape, (1, 3))
        theta2 = np.array([[0, 1, 2]])
        res2 = self.mirt_2pl_2d.icc(theta2)
        self.assertEqual(res2.shape, (1, 3))
        # Two Person input
        theta1 = np.array([[0],
                           [1]])
        res1 = self.mirt_2pl_1d.icc(theta1)
        self.assertEqual(res1.shape, (2, 3))
        theta2 = np.array([[0, 1, 2],
                           [1, 2, 3]])
        res2 = self.mirt_2pl_2d.icc(theta2)
        self.assertEqual(res2.shape, (2, 3))

    def test_icc_result(self):
        # Naive result
        theta1 = np.array([[0]])
        res1 = self.mirt_2pl_1d.icc(theta1)
        self.assertTrue(np.array_equal(res1, np.array([[0.5, 0.5, 0.5]])))
        # Check for correct sign in exponent
        theta2 = np.array([[1, 1, 1]])
        res2 = self.mirt_2pl_2d.icc(theta2)
        self.assertTrue(np.array_equal(np.round(res2, 2),
                                       np.array([[0.88, 0.88, 0.88]])))


if __name__ == '__main__':
    unittest.main()
