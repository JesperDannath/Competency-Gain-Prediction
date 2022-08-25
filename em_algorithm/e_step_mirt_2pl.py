from multiprocessing import reduction
import e_step
import numpy as np
import pandas as pd
import os
import sys
from scipy import integrate
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class e_step_ga_mml(e_step):

    def __init__(self, model: mirt_2pl) -> None:
        super().__init__()
        self.model = model

    # TODO: Ensure that model uses the current parameters
    # TODO: Make this Monte Carlo
    def conditional_ability_normalising_constant(self, response_pattern):
        def joint_prob(theta): return self.model.joint_competency_answer_density(
            theta=theta, response_vector=response_pattern)
        normalizing_constant = integrate.nquad(joint_prob)
        return(normalizing_constant)

    def step(self, response_data: pd.DataFrame, current_item_parameters: dict, current_person_parameters: dict) -> dict:
        """E-Step for a classic MIRT Model based on Bock & Aitkin (1981) as well as Zhang (2005). 

        Args:
            response_data (pd.DataFrame): Dichotomous Data for Item Responses, rows=respondents, cols=Items
            current_item_parameters (dict): Current item Parameters from last M-Step or initiation
            current_person_parameters (dict): Current person Parameters from last M-Step or initiation
        """
        # Calculate Expected Values
        M = response_data.shape[0]
        J = response_data.shape[1]
        normalising_constant_array = np.empty(shape=M)
        for i, response_pattern in enumerate(response_data.to_numpy()):
            normalising_constant_array[i] = self.conditional_ability_probability(
                response_pattern)

        def r_0(theta: np.array):
            numerator = np.array([self.model.joint_competency_answer_density(
                theta=theta, response_vector=response_pattern) for response_pattern in response_data.to_numpy()])
            denominator = normalising_constant_array
            return(np.sum(np.divide(numerator, denominator)))

        def r_item(item: int, theta: np.array):
            numerator = np.array([self.model.joint_competency_answer_density(
                theta=theta, response_vector=response_pattern) for response_pattern in response_data.to_numpy()])
            # This coefficient is different to r_0
            numerator = np.multiply(numerator, response_data.iloc[:, item])
            denominator = normalising_constant_array
            # TODO: Monte Carlo
            return(np.sum(np.divide(numerator, denominator)))

        def q_0(sigma):
            def func(theta): return np.multiply(r_0(theta), self.model.latent_density(
                theta, sigma=sigma))  # TODO: This needs to be a function of sigma
            return(integrate.quad(func))

        def q_item(item: int, a: np.array, item_delta: np.array):
            def func(theta):
                icc_value = self.model.icc(theta=theta, A=np.expand_dims(a, axis=1), delta=np.expand_dims(
                    item_delta, axis=1))  # TODO: check whether this is correct
                r_0_theta = r_0(theta)
                r_item_theta = r_item(item, theta)
                log_likelihood_item = np.multiply(np.log(
                    icc_value), r_0_theta) + np.multiply(np.subtract(r_0_theta, r_item_theta), np.log(1-icc_value))
                return(log_likelihood_item)
            return(integrate.quad(func))  # TODO: Monte Carlo

        q_function_dict = {"q_0": q_0, "q_item_list": [lambda a, item_delta: q_item(
            item=j, a=a, item_delta=item_delta) for j in range(0, J)]}
        return(q_function_dict)
