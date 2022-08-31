from e_step import e_step
import numpy as np
import pandas as pd
import os
import sys
from scipy import integrate
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class e_step_ga_mml(e_step):

    def __init__(self, model: mirt_2pl, incomplete_data) -> None:
        super().__init__(incomplete_data=incomplete_data)
        self.model = model

    # TODO: Ensure that model uses the current parameters
    # TODO: Make this more efficient using multiple response patterns as input
    def conditional_ability_normalising_constant(self, response_pattern, N=1000):
        # function
        def response_prob(theta): return self.model.response_vector_probability(
            theta=theta, response_vector=response_pattern)
        # Monte Carlo Integration
        mean = 0
        for i in range(0, N):
            theta = self.model.sample_competency()
            mean += response_prob(theta)/N
        return(mean)

    def step(self, response_data: pd.DataFrame, current_item_parameters: dict = {}, current_person_parameters: dict = {}) -> dict:
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
            normalising_constant_array[i] = self.conditional_ability_normalising_constant(
                response_pattern)

        def r_0(theta: np.array):
            numerator = np.array([self.model.response_vector_probability(
                theta=theta, response_vector=response_pattern) for response_pattern in response_data.to_numpy()])
            denominator = normalising_constant_array
            return(np.sum(np.divide(numerator, denominator)))

        def r_item(item: int, theta: np.array):
            numerator = np.array([self.model.response_vector_probability(
                theta=theta, response_vector=response_pattern) for response_pattern in response_data.to_numpy()])
            # This coefficient is different to r_0
            numerator = np.multiply(numerator, response_data.iloc[:, item])
            denominator = normalising_constant_array
            # TODO: Monte Carlo
            return(np.sum(np.divide(numerator, denominator)))

        def q_0(sigma, N=1000):
            def func(theta):
                factor = np.log(self.model.latent_density(theta, sigma=sigma))
                numerator = np.array([self.model.response_vector_probability(
                    theta=theta, response_vector=response_pattern) for response_pattern in response_data.to_numpy()])
                denominator = normalising_constant_array
                sum = np.sum(np.divide(numerator, denominator))
                return(np.multiply(factor, sum))
            mean = 0
            for i in range(0, N):
                theta = self.model.sample_competency()
                mean += func(theta)/N
            return(mean)

        def q_item(item: int, a: np.array, item_delta: np.array, N=1000):
            def func(theta):
                icc_value = self.model.icc(theta=np.expand_dims(theta, axis=0), A=np.expand_dims(
                    a, axis=0), delta=np.array([item_delta]))  # TODO: check whether this is correct
                r_0_theta = r_0(theta)
                r_item_theta = r_item(item, theta)
                log_likelihood_item = np.multiply(np.log(
                    icc_value), r_0_theta) + np.multiply(np.subtract(r_0_theta, r_item_theta), np.log(1-icc_value))
                return(log_likelihood_item.tolist()[0][0])
            # Monte Carlo integration
            mean = 0
            for i in range(0, N):
                theta = self.model.sample_competency()
                mean += func(theta)/N
            return(mean)

        q_function_dict = {"q_0": q_0, "q_item_list": [lambda a_item, delta_item: q_item(
            item=j, a=a_item, item_delta=delta_item) for j in range(0, J)]}
        return(q_function_dict)
