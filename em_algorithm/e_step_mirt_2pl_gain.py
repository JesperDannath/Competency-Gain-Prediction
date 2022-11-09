from e_step import e_step
from e_step_mirt_2pl import e_step_ga_mml
import numpy as np
import pandas as pd
import os
import sys
from scipy import integrate
from scipy.optimize import approx_fprime
from scipy.stats import ttest_ind
from hotelling.stats import hotelling_t2
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl_gain import mirt_2pl_gain


class e_step_ga_mml_gain(e_step_ga_mml):

    def __init__(self, model: mirt_2pl_gain, incomplete_data: pd.DataFrame = pd.DataFrame()) -> None:
        super().__init__(incomplete_data=incomplete_data, model=model)

    # TODO: Make this more efficient using multiple response patterns as input
    def conditional_ability_normalising_constant(self, response_data, theta, N=5000):
        # function
        def response_prob(s): return np.multiply(self.model.response_matrix_probability(
            theta=theta, s=s, response_matrix=response_data), self.model.latent_density(type="early_conditional", s=s, theta=theta))
        # Monte Carlo integration
        s = self.model.sample_gain(sample_size=N, qmc=True)
        result_matrix = response_prob(s)
        return(np.mean(result_matrix, axis=0))

    def step(self, response_data: pd.DataFrame, theta: pd.DataFrame, current_item_parameters: dict = {}, current_person_parameters: dict = {}, iter=-1) -> dict:
        """E-Step for a classic MIRT Model based on Bock & Aitkin (1981) as well as Zhang (2005).

        Args:
            response_data (pd.DataFrame): Dichotomous Data for Item Responses, rows=respondents, cols=Items
            current_item_parameters (dict): Current item Parameters from last M-Step or initiation
            current_person_parameters (dict): Current person Parameters from last M-Step or initiation
        """
        self.model.update_conditional_covariance(
        )  # TODO: ist das hier der beste Ort? In set_parameters einschließen (Intransparent!)
        normalising_constant_array = self.conditional_ability_normalising_constant(
            response_data.to_numpy(), theta.to_numpy())

        if iter == -1:
            self.N = 1000
            self.mc_variance_data = self.get_mc_variance_data(
                normalising_constant_array, response_data, theta=theta)  # TODO: Solve this unecessary calculation more elegant with ifelse
        elif iter == 1:
            #self.N = 200 + 6**self.model.latent_dimension
            self.N = 75*2**self.model.latent_dimension
            self.mc_variance_data = self.get_mc_variance_data(
                normalising_constant_array, response_data, theta=theta)
        self.last_mc_variance_data = self.mc_variance_data
        self.mc_variance_data = self.get_mc_variance_data(
            normalising_constant_array, response_data, theta=theta)
        # p_change = self.compare_qmc_data(
        #    last_qmc_data=self.last_qmc_variance_data, qmc_data=self.qmc_variance_data)
        p_change = ttest_ind(
            self.mc_variance_data, self.last_mc_variance_data, axis=0, equal_var=False).pvalue[0]
        # TODO: Try one-sided alternative (Sigma should increase). Better don't do test if likelihood decreases
        if (iter > 1) & (p_change > 0.1) & (self.N < 3000):
            self.N = int(self.N*1.08)

        print("Current Monte Carlo Sample size: {0}".format(self.N))
        s_sample = self.model.sample_gain(
            sample_size=self.N, qmc=True)
        return(self.prepare_q_functions(s_sample, response_data, theta, normalising_constant_array))

    def prepare_q_functions(self, s_sample, response_data, theta, normalising_constant_array):
        M = response_data.shape[0]
        J = response_data.shape[1]
        response_matrix_probability = self.model.response_matrix_probability(s=s_sample,
                                                                             theta=theta.to_numpy(), response_matrix=response_data.to_numpy())
        early_conditional_density = self.model.latent_density(
            type="early_conditional", s=s_sample, theta=theta)
        # Calculate repeating inner functions
        # r_0_s = self.r_0(
        #     s=s_sample, response_matrix_probability=response_matrix_probability,
        #     normalising_constant_array=normalising_constant_array,
        #     response_data=response_data, theta=theta)
        # r_item_list = [self.r_item(item=item, s=s_sample, response_matrix_probability=response_matrix_probability,
        #                           normalising_constant_array=normalising_constant_array, response_data=response_data, theta=theta)
        #               for item in range(0, J)]
        # Calculate final q-functions
        q_0 = self.q_0(s=s_sample,
                       theta=theta, response_matrix_probability=response_matrix_probability,
                       early_conditional_density=early_conditional_density,
                       normalising_constant_array=normalising_constant_array, response_data=response_data)
        q_0_grad = self.q_0_gradient(
            s=s_sample, theta=theta, response_data=response_data, response_matrix_probability=response_matrix_probability,
            early_conditional_density=early_conditional_density,
            normalising_constant_array=normalising_constant_array)
        q_item_list = [self.q_item(item=j, s=s_sample, theta=theta, response_data=response_data,
                                   response_matrix_probability=response_matrix_probability,
                                   early_conditional_density=early_conditional_density,
                                   normalising_constant_array=normalising_constant_array) for j in range(0, J)]

        q_function_dict = {"q_0": q_0, "q_0_grad": q_0_grad,
                           "q_item_list": q_item_list}
        return(q_function_dict)

    def get_mc_variance_data(self, normalising_constant_array, response_data, theta):
        sigma = self.model.person_parameters["covariance"]
        A = self.model.item_parameters["discrimination_matrix"]
        delta = self.model.item_parameters["intercept_vector"]
        J = response_data.shape[1]
        s_sample = self.model.sample_gain(
            sample_size=self.N, qmc=False)
        response_matrix_probability = self.model.response_matrix_probability(s=s_sample,
                                                                             theta=theta.to_numpy(), response_matrix=response_data.to_numpy())
        early_conditional_density = self.model.latent_density(
            type="early_conditional", s=s_sample, theta=theta)
        q_0 = self.q_0(s=s_sample,
                       theta=theta, response_matrix_probability=response_matrix_probability,
                       early_conditional_density=early_conditional_density,
                       normalising_constant_array=normalising_constant_array, response_data=response_data)
        q_values = q_0(sigma, return_sample=True)[1].reshape((self.N, 1))
        return(q_values)

    # def r_0(self, s, theta, normalising_constant_array, response_data):
    #     numerator = self.model.response_matrix_probability(s=s,
    #                                                        theta=theta.to_numpy(), response_matrix=response_data.to_numpy())
    #     denominator = normalising_constant_array
    #     return(np.sum(np.divide(numerator, denominator), axis=1))

    # def r_item(self, item: int, s, theta: np.array, normalising_constant_array, response_data):
    #     numerator = np.array(self.model.response_matrix_probability(s=s,
    #                                                                 theta=theta.to_numpy(), response_matrix=response_data.to_numpy()))
    #     # This coefficient is different to r_0
    #     numerator = np.multiply(
    #         numerator, response_data.iloc[:, item].to_numpy().transpose())
    #     denominator = normalising_constant_array
    #     return(np.sum(np.divide(numerator, denominator), axis=1))

    def q_0(self, s, theta, response_matrix_probability, early_conditional_density, normalising_constant_array, response_data):
        # numerator = np.array(self.model.response_matrix_probability(s=s,
        #                                                             theta=theta.to_numpy(), response_matrix=response_data.to_numpy()))
        numerator = np.array(response_matrix_probability)
        numerator = np.multiply(numerator, early_conditional_density)
        denominator = normalising_constant_array
        quotient = np.divide(numerator, denominator)

        def func(sigma_psi, return_sample=False):
            factor = np.log(self.model.latent_density(
                theta=theta, s=s, sigma=sigma_psi, type="full_cross", save=True))
            product = np.multiply(factor, quotient)
            sum = np.sum(product, axis=1)
            if return_sample:
                return(np.mean(sum), sum)
            return(np.mean(sum))
        return(func)

    def q_0_gradient(self, s, theta, response_matrix_probability, early_conditional_density, normalising_constant_array, response_data):
        D = self.model.latent_dimension
        N = s.shape[0]
        # numerator = np.array(self.model.response_matrix_probability(s=s,
        #                                                             theta=theta.to_numpy(), response_matrix=response_data.to_numpy()))
        numerator = np.array(response_matrix_probability)
        numerator = np.multiply(numerator, early_conditional_density)
        denominator = normalising_constant_array
        quotient = np.divide(numerator, denominator)
        s_tile, theta_tile = self.model.tile_competency_gain(s=s, theta=theta)
        competency = np.concatenate((s_tile, theta_tile), axis=2)

        def func(sigma_psi):
            #quadratic_form = quadratic_form_func(sigma)
            inv_sigma_psi = np.linalg.inv(sigma_psi)

            sum1 = np.multiply(-0.5, inv_sigma_psi)
            # sum2
            sum2 = np.dot(competency, inv_sigma_psi)
            #sum2 = -1*np.sum(np.multiply(sum2, sum2), axis=2).squeeze()
            sum2 = 0.5*np.matmul(sum2.reshape(N, theta.shape[0], 2*D, 1),
                                 sum2.reshape(N, theta.shape[0], 1, 2*D)).squeeze()
            sum2 = np.transpose(sum2, axes=[0, 1, 3, 2])
            integrant = np.multiply(quotient.reshape(
                (N, theta.shape[0], 1, 1)), np.add(sum1, sum2))
            integrant = np.sum(integrant, axis=1)
            return(np.mean(integrant, axis=0))
        return(func)

    # def q_item(self, item: int, s, theta, response_data, normalising_constant_array):
    #     numerator = np.array(self.model.response_matrix_probability(s=s,
    #                                                                 theta=theta.to_numpy(), response_matrix=response_data.to_numpy()))
    #     numerator = np.multiply(numerator, self.model.latent_density(
    #         type="early_conditional", s=s, theta=theta))
    #     # This coefficient is different to r_0
    #     numerator_item = np.multiply(
    #         numerator, response_data.iloc[:, item].to_numpy().transpose())
    #     denominator = normalising_constant_array
    #     r_0 = np.divide(numerator, denominator)
    #     r_item = np.divide(numerator_item, denominator)

    #     def func(a_item: np.array, delta_item: np.array):
    #         icc_values = self.model.icc(s=s, theta=theta, A=np.expand_dims(
    #             a_item, axis=0), delta=np.array([delta_item]), save=False).transpose()[0]
    #         inv_icc = 1 - icc_values
    #         #icc_values[icc_values == 0] = np.float64(1.7976931348623157e-320)
    #         #inv_icc[inv_icc == 0] = np.float64(1.7976931348623157e-320)
    #         log_likelihood_item = np.multiply(np.log(icc_values), r_item) + np.multiply(
    #             np.log(inv_icc), np.subtract(r_0, r_item))  # TODO: Make this save
    #         log_likelihood_item = np.sum(log_likelihood_item, axis=1)
    #         return(np.mean(log_likelihood_item))
    #     return(func)

    def q_item(self, item: int, s, theta, response_data, response_matrix_probability, early_conditional_density, normalising_constant_array):
        # numerator = np.array(self.model.response_matrix_probability(s=s,
        #                                                             theta=theta.to_numpy(), response_matrix=response_data.to_numpy()))
        numerator = np.array(response_matrix_probability)
        numerator = np.multiply(numerator, early_conditional_density)
        # This coefficient is different to r_0
        numerator_item = np.multiply(
            numerator, response_data.iloc[:, item].to_numpy().transpose())
        denominator = normalising_constant_array
        r_0 = np.divide(numerator, denominator)
        r_item = np.divide(numerator_item, denominator)
        r_diff = np.subtract(r_0, r_item)
        r = r_item.copy()
        incorrect_indices = np.where(r_item == 0)
        r[incorrect_indices] = r_diff[incorrect_indices]
        #correct_indices = np.where(r_item != 0)

        def func(a_item: np.array, delta_item: np.array):
            icc_values = self.model.icc(s=s, theta=theta, A=np.expand_dims(
                a_item, axis=0), delta=np.array([delta_item]), save=False).transpose()[0]
            icc_values[incorrect_indices] = 1 - icc_values[incorrect_indices]
            log_likelihood_item = np.multiply(
                np.log(icc_values + np.float64(1.7976931348623157e-309)), r)
            log_likelihood_item = np.sum(log_likelihood_item, axis=1)
            return(np.mean(log_likelihood_item))
        return(func)
