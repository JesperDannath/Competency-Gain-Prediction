import unittest
import numpy as np
import pandas as pd
import os
import sys
from m_step import m_step
from m_step_mirt_2pl_gain import m_step_ga_mml_gain
from e_step_mirt_2pl_gain import e_step_ga_mml_gain
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl_gain import mirt_2pl_gain


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
                           [0, 1, 1, 1],
                           [1, 0, 1, 0],
                           [1, 1, 1, 0]])
        self.e_step_2pl_gain = e_step_ga_mml_gain(
            model=self.mirt_2pl_gain, incomplete_data=self.V)
        theta = np.array([[1, 0.5, 2],
                          [-1, 0, 1],
                          [-2, -0.5, 0.5],
                          [1, 1, 1],
                          [1, 2, 3],
                          [-1, 0, 1],
                          [1, 1, 1]])
        # Test with internal parameter-values
        self.result_function_dict = self.e_step_2pl_gain.step(
            pd.DataFrame(self.V), theta=pd.DataFrame(theta))

    def test_step_ga_mirt_2pl(self):
        m_step_2pl_gain = m_step_ga_mml_gain(self.mirt_2pl_gain, sigma_constraint="unconstrained")
        # Test functionality of step function
        m_step_2pl_gain.step(pe_functions=self.result_function_dict)


if __name__ == '__main__':
    unittest.main()
