import unittest
import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import approx_fprime
from e_step import e_step
from e_step_mirt_2pl_gain import e_step_ga_mml_gain
from scipy.stats import multivariate_normal
import scipy
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./simulation_framework"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl_gain import mirt_2pl_gain
    from simulate_responses import response_simulation
    from simulate_competency import respondent_population


class test_mirt_2pl(unittest.TestCase):

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
        self.V = np.array([[1, 1, 1, 1],
                           [0, 0, 0, 0],
                           [1, 0, 1, 0],
                           [1, 1, 0, 0],
                           [0, 1, 1, 1]])
        self.e_step_2pl_gain = e_step_ga_mml_gain(
            model=self.mirt_2pl_gain, incomplete_data=self.V)

    def test_conditional_ability_normalizing_constant(self):
        theta = np.array([[1, 0.5, 2],
                          [-1, 0, 1],
                          [-2, -0.5, 0.5],
                          [1, 1, 1],
                          [1, 2, 3]])
        normalizing_constant = self.e_step_2pl_gain.conditional_ability_normalising_constant(theta=theta,
                                                                                             response_data=self.V)
        # nquad lösung für [-10,10]:
        # Monte Carlo Lösung:
        self.assertTrue((normalizing_constant != 0.0).all())

    def test_step_ga_mirt_2pl(self):
        theta = np.array([[1, 0.5, 2],
                          [-1, 0, 1],
                          [-2, -0.5, 0.5],
                          [1, 1, 1],
                          [1, 2, 3]])
        # Test with internal parameter-values
        result_function_dict = self.e_step_2pl_gain.step(
            pd.DataFrame(self.V), theta=pd.DataFrame(theta))
        # Test functionality of step function
        self.assertTrue("q_0" in result_function_dict.keys())
        self.assertTrue("q_item_list" in result_function_dict.keys())
        self.assertTrue("q_0_grad" in result_function_dict.keys())
        # Test functionality of output functions
        # Test q_0
        q_0 = result_function_dict["q_0"]
        sigma_psi = self.mirt_2pl_gain.person_parameters["covariance"]
        q_0_value = q_0(sigma_psi)
        self.assertTrue(q_0_value != 0.0)
        q_0_grad = result_function_dict["q_0_grad"]
        q_0_grad_value = q_0_grad(sigma_psi)
        self.assertTrue(q_0_grad_value.shape == (6, 6))
        # Test q_item
        A_input = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 1, 1]], dtype=np.float)
        delta_input = np.array([0, 0, 0, 1])
        for i in range(0, self.mirt_2pl_gain.item_dimension):
            q_item = result_function_dict["q_item_list"][i]
            a_item = A_input[i, :]
            delta_item = delta_input[i]
            q_item_value = q_item(a_item=a_item, delta_item=delta_item)
            self.assertTrue(q_item_value != 0.0)


if __name__ == '__main__':
    unittest.main()
