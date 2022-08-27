from binascii import Incomplete
import unittest
import numpy as np
import os
import sys
from e_step import e_step
from e_step_mirt_2pl import e_step_ga_mml
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


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
        sigma2 = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])
        self.mirt_2pl_2d = mirt_2pl(
            item_dimension=3, latent_dimension=3, A=A2, delta=delta2, sigma=sigma2)
        self.incomplete_data = np.array([[1,1,1],
                                        [0,0,0],
                                        [1,0,0],
                                        [0,1,0],
                                        [0,0,1]])
        self.e_step_2pl = e_step_ga_mml(model=self.mirt_2pl_2d, incomplete_data=self.incomplete_data)

    def test_conditional_ability_normalizing_constant(self):
        response_pattern = np.array([1,1,0])
        normalizing_constant = self.e_step_2pl.conditional_ability_normalising_constant(
            response_pattern=response_pattern)
        self.assertTrue(normalizing_constant != 0.0)

    def test_step_ga_mirt_2pl(self):
        #Test with internal parameter-values
        result_function_dict = self.e_step_2pl.step(self.incomplete_data)
        self.assertTrue("q_0" in result_function_dict.keys())
        self.assertTrue("q_item_list" in result_function_dict.keys())

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


if __name__ == '__main__':
    unittest.main()
