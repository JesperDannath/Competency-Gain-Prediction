import unittest
import numpy as np
import pandas as pd
import os
import sys
from m_step import m_step
from m_step_mirt_2pl import m_step_ga_mml
from e_step_mirt_2pl import e_step_ga_mml
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class test_mirt_2pl(unittest.TestCase):

    def setUp(self):
        A = np.array([[1, 0],
                      [0, 1],
                      [0, 1]], dtype=np.float)
        delta = np.array([1, 1, 1], dtype=np.float)
        sigma = np.array([[1, 0],
                          [0, 1]])
        self.mirt_2pl_2d = mirt_2pl(
            item_dimension=3, latent_dimension=2, A=A, delta=delta, sigma=sigma)
        self.incomplete_data = pd.DataFrame(np.array([[1, 1, 1],
                                                      [0, 0, 0],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1]]))
        # Perform e-step to obtain
        self.e_step_2pl = e_step_ga_mml(
            model=self.mirt_2pl_2d, incomplete_data=self.incomplete_data)
        self.result_function_dict = self.e_step_2pl.step(self.incomplete_data)
        # Set Up M-Step
        self.m_step_2pl = m_step_ga_mml(model=self.mirt_2pl_2d)

    def test_genetic_algorithm(self):
        print("test genetic algorithm")
        # Test with quadratic function

        def q_func(input):
            return(-1*(input[0]**2 + input[1]**2))
        x0 = np.array([10, -10])
        result = self.m_step_2pl.genetic_algorithm(
            fitness_function=q_func, x0=x0, population_size=100)
        diff = np.sqrt(np.sum(np.square(result) -
                              np.square(np.array([0.0, 0.0]))))
        self.assertTrue(diff < 0.3)
        # Test item fitness function
        fitness_function = self.result_function_dict["q_item_list"][0]

        def wrapped_fitness_function(input):
            return fitness_function(
                input[0:len(input)-1], input[len(input)-1])
        a_init = self.mirt_2pl_2d.item_parameters["discrimination_matrix"][0, :]
        delta_init = self.mirt_2pl_2d.item_parameters["intercept_vector"][0]
        x0 = np.concatenate(
            (a_init, np.expand_dims(delta_init, 0)), axis=0)
        new_parameters_dict = self.m_step_2pl.genetic_algorithm(
            fitness_function=wrapped_fitness_function, x0=x0, population_size=10)
        print(new_parameters_dict)

    def test_step_ga_mirt_2pl(self):
        # Test functionality of step function
        self.m_step_2pl.step(pe_functions=self.result_function_dict)


if __name__ == '__main__':
    unittest.main()
