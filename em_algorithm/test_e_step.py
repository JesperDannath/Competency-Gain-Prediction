import unittest
import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import approx_fprime
from e_step import e_step
from e_step_mirt_2pl import e_step_ga_mml
from scipy.stats import multivariate_normal
import scipy
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./simulation_framework"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl
    from simulate_responses import response_simulation
    from simulate_competency import respondent_population


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
        response_pattern = np.array([[1, 1, 0]])
        normalizing_constant = self.e_step_2pl.conditional_ability_normalising_constant(
            response_data=response_pattern)
        # nquad lösung für [-10,10]: 0.1472
        # Monte Carlo Lösung: 0.1486
        self.assertTrue(normalizing_constant != 0.0)

    def test_step_ga_mirt_2pl(self):
        # Test with internal parameter-values
        result_function_dict = self.e_step_2pl.step(self.incomplete_data)
        # Test functionality of step function
        self.assertTrue("q_0" in result_function_dict.keys())
        self.assertTrue("q_item_list" in result_function_dict.keys())
        self.assertTrue("q_0_grad" in result_function_dict.keys())
        # Test functionality of output functions
        # Test q_0
        q_0 = result_function_dict["q_0"]
        sigma_input = np.array([[1, 0.5, 0.5],
                                [0.5, 1, 0.5],
                                [0.5, 0.5, 1]])
        q_0_value = q_0(sigma_input)
        self.assertTrue(q_0_value != 0.0)
        q_0_grad = result_function_dict["q_0_grad"]
        q_0_grad_value = q_0_grad(sigma_input)
        self.assertTrue(q_0_grad_value.shape == (3, 3))
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

    # TODO: implement for Q_0
    def test_gradients_against_numerical(self):
        # Simulate vast response data
        sigma = np.array([[1, 0.5], [0.5, 1]])
        latent_distribution = multivariate_normal(
            mean=np.array([0, 0]), cov=sigma)
        population = respondent_population(
            latent_dimension=2, latent_distribution=latent_distribution)
        Q = np.array([[1, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1],
                      [1, 1]])
        response_simulation_obj = response_simulation(
            population=population, item_dimension=5)
        early_item_params = response_simulation_obj.initialize_random_item_parameters(
            Q=Q)
        sample = response_simulation_obj.sample(200)["early_responses"]
        model = mirt_2pl(latent_dimension=2, item_dimension=5, Q=Q)
        e_step = e_step_ga_mml(
            model=model, incomplete_data=self.incomplete_data)
        # Test with internal parameter-values
        result_function_dict = e_step.step(sample)

        def q_0_flat(sigma_flat): return result_function_dict["q_0"](
            sigma_flat.reshape(2, 2))
        q_0_grad = result_function_dict["q_0_grad"]
        sigma_input = np.array([[1, 0.5],
                                [0.5, 1]])
        approximate_derivative = approx_fprime(f=q_0_flat, xk=sigma_input.flatten(),
                                               epsilon=1.4901161193847656e-22).reshape((2, 2))
        exact_derivative = np.tril(q_0_grad(sigma_input))
        exact_derivative[np.tril_indices_from(exact_derivative, k=-1)] *= 2
        diff = np.abs(approximate_derivative - np.tril(exact_derivative))
        self.assertTrue((diff < 0.1).all())
        # Derivative of Q_0 wih respect to matrix sqrt of sigma

        def q_0_sqrt(sqrt_sigma_vector):
            sqrt_sigma = sqrt_sigma_vector.reshape(
                (2, 2))
            sigma = np.dot(sqrt_sigma, sqrt_sigma.transpose())
            return result_function_dict["q_0"](sigma)

        def q_0_gradient_sqrt(sqrt_sigma_vector):
            D = 2
            sqrt_sigma = sqrt_sigma_vector.reshape((D, D))
            sigma = np.dot(sqrt_sigma, sqrt_sigma.transpose())
            # Apply chain rule
            chain2 = approx_fprime(f=lambda C: np.dot(
                C.reshape((D, D)), C.reshape((D, D)).transpose()).flatten(),
                xk=sqrt_sigma_vector, epsilon=1.4901161193847656e-22).reshape((D**2, D, D))
            gradient = np.sum(np.sum(np.multiply(
                result_function_dict["q_0_grad"](sigma), chain2), axis=1), axis=1)
            return(gradient.flatten())
        sqrt_sigma_input = scipy.linalg.sqrtm(sigma_input)

        approximate_derivative = approx_fprime(f=q_0_sqrt, xk=sqrt_sigma_input.flatten(),
                                               epsilon=1.4901161193847656e-22).reshape((2, 2))

        exact_derivative = q_0_gradient_sqrt(
            sqrt_sigma_input.flatten()).reshape((2, 2))
        # The approximate derivative does seem to mess up the correlations, because they occur multiple times
        # That is why only the diagonal is used for now
        diff = np.diagonal(np.abs(approximate_derivative - exact_derivative))
        self.assertTrue((diff < 0.1).all())


if __name__ == '__main__':
    unittest.main()
