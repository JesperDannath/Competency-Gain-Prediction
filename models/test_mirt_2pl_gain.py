import unittest
from mirt_2pl_gain import mirt_2pl_gain
import numpy as np
import pandas as pd


class test_mirt_2pl_gain(unittest.TestCase):

    def setUp(self):
        A = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 1]], dtype=np.float)
        delta = np.array([1, 1, 1, 0], dtype=np.float)
        sigma = np.array([[1, 0.5, 0.5],
                          [0.5, 1, 0.5],
                          [0.5, 0.5, 1]])
        sigma_corr = np.array([[0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5]])
        self.mirt_2pl_gain = mirt_2pl_gain(
            item_dimension=4, latent_dimension=3, A=A, delta=delta, early_sigma=sigma,
            late_sigma=sigma, latent_corr=sigma_corr, mu=np.array([1.5,1.5,1.5]),
            convolution_variance=np.array([3,3,3]))

    def test_dimensions(self):
        self.assertEqual(self.mirt_2pl_gain.item_dimension, 4)
        self.assertEqual(self.mirt_2pl_gain.latent_dimension, 3)

    def test_initialization(self):
        model = mirt_2pl_gain(
            item_dimension=4, latent_dimension=3)
        parameters = model.get_parameters()
        item_parameters = parameters["item_parameters"]
        self.assertTrue((item_parameters["q_matrix"] == np.ones((4, 3))).all())
        self.assertTrue(
            (item_parameters["q_matrix"] == item_parameters["discrimination_matrix"]).all())
        self.assertTrue(
            "early_conditional_covariance" in parameters["person_parameters"].keys())

    def test_icc(self):
        theta = np.array([[0, 0, 0],
                          [1, 1, 1]])
        s = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [1, 1, 1]])
        icc_values = self.mirt_2pl_gain.icc(theta=theta, s=s,cross=True)
        self.assertTrue(icc_values.shape == (2, 3, 4))


if __name__ == '__main__':
    unittest.main()
