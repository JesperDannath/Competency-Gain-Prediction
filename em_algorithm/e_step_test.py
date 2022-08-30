from binascii import Incomplete
import unittest
import numpy as np
import pandas as pd
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
        sigma2 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        self.mirt_2pl_2d = mirt_2pl(
            item_dimension=3, latent_dimension=3, A=A2, delta=delta2, sigma=sigma2)
        self.incomplete_data = pd.DataFrame(np.array([[1, 1, 1],
                                                      [0, 0, 0],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1]]))
        self.e_step_2pl = e_step_ga_mml(
            model=self.mirt_2pl_2d, incomplete_data=self.incomplete_data)

    def test_conditional_ability_normalizing_constant(self):
        response_pattern = np.array([1, 1, 0])
        normalizing_constant = self.e_step_2pl.conditional_ability_normalising_constant(
            response_pattern=response_pattern)
        # nquad lösung für [-10,10]: 0.1472
        # Monte Carlo Lösung: 0.1486
        self.assertTrue(normalizing_constant != 0.0)

    def test_step_ga_mirt_2pl(self):
        # Test with internal parameter-values
        result_function_dict = self.e_step_2pl.step(self.incomplete_data)
        # Test functionality of step function
        self.assertTrue("q_0" in result_function_dict.keys())
        self.assertTrue("q_item_list" in result_function_dict.keys())
        # Test functionality of output functions
        # Test q_0
        q_0 = result_function_dict["q_0"]
        sigma_input = np.array([[1, 0.5, 0.5],
                                [0.5, 1, 0.5],
                                [0.5, 0.5, 1]])
        q_0_value = q_0(sigma_input)
        self.assertTrue(q_0_value != 0.0)
        # Test q_item
        A_input = np.array([[1, 1, 1],
                            [1, 0, 0],
                            [1, 1, 0]])
        delta_input = np.array([0, 0, 0])
        for i in range(0, self.mirt_2pl_2d.item_dimension):
            q_item = result_function_dict["q_item_list"][i]
            a_item = A_input[:, i]
            delta_item = delta_input[i]
            q_item_value = q_item(a_item=a_item, delta_item=delta_item)
            self.assertTrue(q_item_value != 0.0)


if __name__ == '__main__':
    unittest.main()
