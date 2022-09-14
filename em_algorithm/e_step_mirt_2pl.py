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

    def __init__(self, model: mirt_2pl, incomplete_data: pd.DataFrame = pd.DataFrame()) -> None:
        super().__init__(incomplete_data=incomplete_data)
        self.model = model

    # # TODO: Make this more efficient using multiple response patterns as input
    # def conditional_ability_normalising_constant(self, response_pattern, N=1000):
    #     # function
    #     def response_prob(theta): return self.model.response_matrix_probability(
    #         theta=np.expand_dims(theta, axis=0), response_matrix=np.expand_dims(response_pattern, axis=0))
    #     # Monte Carlo Integration
    #     mean = 0
    #     for i in range(0, N):
    #         theta = self.model.sample_competency()
    #         mean += response_prob(theta)/N
    #     return(mean)

    # TODO: Make this more efficient using multiple response patterns as input
    def conditional_ability_normalising_constant(self, response_data, N=1000):
        # function
        def response_prob(theta): return self.model.response_matrix_probability(
            theta=theta, response_matrix=response_data)
        # Monte Carlo integration
        theta = self.model.sample_competency(sample_size=N)
        result_matrix = response_prob(theta)
        return(np.mean(result_matrix, axis=0))

    def step(self, response_data: pd.DataFrame, current_item_parameters: dict = {}, current_person_parameters: dict = {}, N=1000) -> dict:
        """E-Step for a classic MIRT Model based on Bock & Aitkin (1981) as well as Zhang (2005).

        Args:
            response_data (pd.DataFrame): Dichotomous Data for Item Responses, rows=respondents, cols=Items
            current_item_parameters (dict): Current item Parameters from last M-Step or initiation
            current_person_parameters (dict): Current person Parameters from last M-Step or initiation
        """
        # Calculate Expected Values
        M = response_data.shape[0]
        J = response_data.shape[1]
        normalising_constant_array = self.conditional_ability_normalising_constant(
            response_data.to_numpy())  # np.empty(shape=M)
        # for i, response_pattern in enumerate(response_data.to_numpy()):
        #     normalising_constant_array[i] = self.conditional_ability_normalising_constant(
        #         response_pattern)

        def r_0(theta: np.array):
            numerator = self.model.response_matrix_probability(
                theta=theta, response_matrix=response_data.to_numpy())
            denominator = normalising_constant_array
            return(np.sum(np.divide(numerator, denominator), axis=1))

        def r_item(item: int, theta: np.array):
            numerator = np.array(self.model.response_matrix_probability(
                theta=theta, response_matrix=response_data.to_numpy()))
            # This coefficient is different to r_0
            numerator = np.multiply(
                numerator, response_data.iloc[:, item].to_numpy().transpose())
            denominator = normalising_constant_array
            # TODO: Monte Carlo
            return(np.sum(np.divide(numerator, denominator), axis=1))

        # def q_0(sigma, N=400):
        #     def func(theta):
        #         factor = np.log(self.model.latent_density(theta, sigma=sigma))
        #         numerator = np.array(self.model.response_matrix_probability(
        #             theta=theta, response_matrix=response_data.to_numpy()))
        #         denominator = normalising_constant_array
        #         sum = np.sum(np.divide(numerator, denominator), axis=1)
        #         return(np.multiply(factor, sum))
        #     # Monte Carlo integration
        #     theta = self.model.sample_competency(sample_size=N)
        #     result_vector = func(theta)
        #     return(np.mean(result_vector))

        # def q_item(item: int, a: np.array, item_delta: np.array, N=1000):
        #     def func(theta):
        #         icc_values = self.model.icc(theta=theta, A=np.expand_dims(
        #             a, axis=0), delta=np.array([item_delta])).transpose()[0]
        #         r_0_theta = r_0(theta)
        #         r_item_theta = r_item(item, theta)
        #         log_likelihood_item = np.multiply(np.log(
        #             icc_values), r_item_theta) + np.multiply(np.log(1-icc_values), np.subtract(r_0_theta, r_item_theta))
        #         return(log_likelihood_item)
        #     # Monte Carlo integration
        #     theta = self.model.sample_competency(sample_size=N)
        #     result_vector = func(theta)
        #     return(np.mean(result_vector))

        theta_sample = self.model.sample_competency(sample_size=N)
        r_0_theta = r_0(theta_sample)
        r_item_theta_list = [r_item(item, theta_sample)
                             for item in range(0, J)]

        q_0 = self.q_0(
            theta=theta_sample, normalising_constant_array=normalising_constant_array, response_data=response_data)
        q_item_list = [self.q_item(item=j, theta=theta_sample, r_0_theta=r_0_theta,
                                   r_item_theta=r_item_theta_list[j]) for j in range(0, J)]

        q_function_dict = {"q_0": q_0, "q_item_list": q_item_list}
        return(q_function_dict)

    def q_0(self, theta, normalising_constant_array, response_data):
        numerator = np.array(self.model.response_matrix_probability(
            theta=theta, response_matrix=response_data.to_numpy()))
        denominator = normalising_constant_array
        sum = np.sum(np.divide(numerator, denominator), axis=1)

        def func(sigma):
            factor = np.log(self.model.latent_density(theta, sigma=sigma))
            product = np.multiply(factor, sum)
            return(np.mean(product))
        return(func)

    def q_item(self, item: int, theta, r_0_theta, r_item_theta):
        def func(a_item: np.array, delta_item: np.array):
            icc_values = self.model.icc(theta=theta, A=np.expand_dims(
                a_item, axis=0), delta=np.array([delta_item])).transpose()[0]
            log_likelihood_item = np.multiply(np.log(
                icc_values), r_item_theta) + np.multiply(np.log(1-icc_values), np.subtract(r_0_theta, r_item_theta))
            return(np.mean(log_likelihood_item))
        return(func)
