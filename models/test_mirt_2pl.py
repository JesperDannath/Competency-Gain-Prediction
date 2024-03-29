import unittest
from mirt_2pl import mirt_2pl
import numpy as np
import pandas as pd


class test_mirt_2pl(unittest.TestCase):

    def setUp(self):
        A1 = np.array([[1],
                       [1],
                       [1]], dtype=np.float)
        delta1 = np.array([0, 0, 0], dtype=np.float)
        sigma1 = np.array([[1]])
        self.mirt_2pl_1d = mirt_2pl(
            item_dimension=3, latent_dimension=1, A=A1, delta=delta1, sigma=sigma1)
        A2 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float)
        delta2 = np.array([1, 1, 1], dtype=np.float)
        sigma2 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        self.mirt_2pl_2d = mirt_2pl(
            item_dimension=3, latent_dimension=3, A=A2, delta=delta2, sigma=sigma2)

    def test_dimensions(self):
        self.assertEqual(self.mirt_2pl_1d.item_dimension, 3)
        self.assertEqual(self.mirt_2pl_2d.item_dimension, 3)
        self.assertEqual(self.mirt_2pl_1d.latent_dimension, 1)
        self.assertEqual(self.mirt_2pl_2d.latent_dimension, 3)

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

    def test_joint_competency_answer_density(self):
        theta2 = np.array([[1, 1, 0.5]])
        answer_vector = np.array([[1, 0, 1]])
        joint_density = self.mirt_2pl_2d.joint_competency_answer_density(
            theta=theta2, response_vector=answer_vector)
        self.assertTrue(joint_density >= 0.0)
        # Test Multiple competentcy-vectors as input
        answer_matrix = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 0]])
        theta_matrix = np.array([[1, 1, 1], [1, 0, 3]])
        joint_density = self.mirt_2pl_2d.joint_competency_answer_density(
            theta=theta_matrix, response_vector=answer_matrix)
        self.assertTrue(len(joint_density) == 2)

    def test_q_matrix_functionality(self):
        Q = np.array([[1, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype=np.float)
        A_positive = np.array([[0.4, 0],
                               [0, 0.5],
                               [0.1, 0],
                               [1, 0.5]], dtype=np.float)
        delta = np.array([1, 1, 1], dtype=np.float)
        sigma = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        mirt_2pl_model = mirt_2pl(
            item_dimension=4, latent_dimension=3, A=A_positive, Q=Q, delta=delta, sigma=sigma)
        self.assertTrue(mirt_2pl_model.check_discriminations())
        A_negative = np.array([[0.4, 1],
                               [0, 0.5],
                               [0.1, 0],
                               [1, 0.5]], dtype=np.float)
        param_dict = {"item_parameters": {
            "discrimination_matrix": A_negative}}
        with self.assertRaises(Exception) as context:
            mirt_2pl_model.set_parameters(param_dict)

    def test_corr_to_sigma(self):
        Q = np.array([[1, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype=np.float)
        A_positive = np.array([[0.4, 0],
                               [0, 0.5],
                               [0.1, 0],
                               [1, 0.5]], dtype=np.float)
        delta = np.array([1, 1, 1], dtype=np.float)
        sigma = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        mirt_2pl_model = mirt_2pl(
            item_dimension=4, latent_dimension=3, A=A_positive, Q=Q, delta=delta, sigma=sigma)
        corr = np.array([0.5, 0.1, 0.5])
        new_sigma = mirt_2pl_model.corr_to_sigma(corr)
        self.assertTrue(np.array_equal(new_sigma, np.array(
            [[1, 0.5, 0.1], [0.5, 1, 0.5], [0.1, 0.5, 1]])))

    def test_predict_competency(self):
        U = pd.DataFrame(np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [0, 0, 1]]))
        # One-Dimensional
        competency_1d = self.mirt_2pl_1d.predict_competency(U)
        self.assertTrue(competency_1d.shape == (3, 1))
        # Two-Dimensional
        competency_1d = self.mirt_2pl_2d.predict_competency(U)
        self.assertTrue(competency_1d.shape == (3, 3))


if __name__ == '__main__':
    unittest.main()
