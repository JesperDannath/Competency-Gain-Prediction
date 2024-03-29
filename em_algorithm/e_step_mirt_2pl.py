from e_step import e_step
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
    from mirt_2pl import mirt_2pl


class e_step_ga_mml(e_step):

    def __init__(self, model: mirt_2pl, incomplete_data: pd.DataFrame = pd.DataFrame()) -> None:
        super().__init__(incomplete_data=incomplete_data)
        self.model = model



    # TODO: Make this more efficient using multiple response patterns as input
    def conditional_ability_normalising_constant(self, response_data, N=5000):
        # function
        def response_prob(theta): return self.model.response_matrix_probability(
            theta=theta, response_matrix=response_data)
        # Monte Carlo integration
        theta = self.model.sample_competency(sample_size=N, qmc=True)
        result_matrix = response_prob(theta)
        return(np.mean(result_matrix, axis=0))

    def step(self, response_data: pd.DataFrame, current_item_parameters: dict = {}, current_person_parameters: dict = {}, iter=-1) -> dict:
        """E-Step for a classic MIRT Model based on Bock & Aitkin (1981) as well as Zhang (2005).

        Args:
            response_data (pd.DataFrame): Dichotomous Data for Item Responses, rows=respondents, cols=Items
            current_item_parameters (dict): Current item Parameters from last M-Step or initiation
            current_person_parameters (dict): Current person Parameters from last M-Step or initiation
        """
        # Calculate Expected Values
        normalising_constant_array = self.conditional_ability_normalising_constant(
            response_data.to_numpy())


        if iter == -1:
            self.N = 1000
            self.mc_variance_data = self.get_mc_variance_data(
                normalising_constant_array, response_data)  # TODO: Solve this unecessary calculation more elegant with ifelse
        elif iter == 1:
            #self.N = 200 + 6**self.model.latent_dimension
            self.N = 75*2**self.model.latent_dimension
            self.mc_variance_data = self.get_mc_variance_data(
                normalising_constant_array, response_data)
        self.last_mc_variance_data = self.mc_variance_data
        self.mc_variance_data = self.get_mc_variance_data(
            normalising_constant_array, response_data)
        # p_change = self.compare_qmc_data(
        #    last_qmc_data=self.last_qmc_variance_data, qmc_data=self.qmc_variance_data)
        p_change = ttest_ind(
            self.mc_variance_data, self.last_mc_variance_data, axis=0, equal_var=False).pvalue[0]
        # TODO: Try one-sided alternative (Sigma should increase). Better don't do test if likelihood decreases
        if (iter > 1) & (p_change > 0.2) & (self.N < 3000):
            self.N = int(self.N*1.08)

        print("Current Monte Carlo Sample size: {0}".format(self.N))
        theta_sample = self.model.sample_competency(
            sample_size=self.N, qmc=True)
        return(self.prepare_q_functions(theta_sample, response_data, normalising_constant_array))

    def prepare_q_functions(self, theta_sample, response_data, normalising_constant_array):
        M = response_data.shape[0]
        J = response_data.shape[1]
        # Calculate repeating inner functions
        r_0_theta = self.r_0(
            theta_sample, normalising_constant_array, response_data)
        r_item_theta_list = [self.r_item(item, theta_sample, normalising_constant_array, response_data)
                             for item in range(0, J)]
        # Calculate final q-functions
        q_0 = self.q_0(
            theta=theta_sample, normalising_constant_array=normalising_constant_array, response_data=response_data)
        q_0_grad = self.q_0_gradient(theta=theta_sample, r_0_theta=r_0_theta)
        q_item_list = [self.q_item(item=j, theta=theta_sample, r_0_theta=r_0_theta,
                                   r_item_theta=r_item_theta_list[j]) for j in range(0, J)]

        q_function_dict = {"q_0": q_0, "q_0_grad": q_0_grad,
                           "q_item_list": q_item_list}
        return(q_function_dict)


    def get_mc_variance_data(self, normalising_constant_array, response_data):
        sigma = self.model.person_parameters["covariance"]
        A = self.model.item_parameters["discrimination_matrix"]
        delta = self.model.item_parameters["intercept_vector"]
        J = response_data.shape[1]
        theta_sample = self.model.sample_competency(
            sample_size=self.N, qmc=False)
        q_0 = self.q_0(
            theta=theta_sample, normalising_constant_array=normalising_constant_array, response_data=response_data)
        q_values = q_0(sigma, return_sample=True)[1].reshape((self.N, 1))
        mean = np.mean(q_values)
        sd = np.sqrt(np.var(q_values))
        error = (sd*1.645)/np.sqrt(self.N)
        ci = (mean - error, mean + error)
        return(q_values)

    # TODO: Maybe different test for smaller sample sizes (rank-sum)
    def get_qmc_variance_data(self, normalising_constant_array, response_data, sample_size=30):
        sigma = self.model.person_parameters["covariance"]
        A = self.model.item_parameters["discrimination_matrix"]
        delta = self.model.item_parameters["intercept_vector"]
        J = response_data.shape[1]
        q_data = np.empty(shape=(sample_size, 1))
        for i in range(0, sample_size):
            theta_sample = self.model.sample_competency(
                sample_size=self.N, qmc=True)
            q_0 = self.q_0(
                theta=theta_sample, normalising_constant_array=normalising_constant_array, response_data=response_data)
            q_values = q_0(sigma)
            q_data[i] = np.array(q_values)
        return(q_data)

    def compare_qmc_data(self, last_qmc_data, qmc_data):
        t2 = hotelling_t2(x=last_qmc_data, y=qmc_data, bessel=True)
        return(t2[2])  # indexes the p-value

    def r_0(self, theta: np.array, normalising_constant_array, response_data):
        numerator = self.model.response_matrix_probability(
            theta=theta, response_matrix=response_data.to_numpy())
        denominator = normalising_constant_array
        return(np.sum(np.divide(numerator, denominator), axis=1))

    def r_item(self, item: int, theta: np.array, normalising_constant_array, response_data):
        numerator = np.array(self.model.response_matrix_probability(
            theta=theta, response_matrix=response_data.to_numpy()))
        # This coefficient is different to r_0
        numerator = np.multiply(
            numerator, response_data.iloc[:, item].to_numpy().transpose())
        denominator = normalising_constant_array
        return(np.sum(np.divide(numerator, denominator), axis=1))

    def q_0(self, theta, normalising_constant_array, response_data):
        numerator = np.array(self.model.response_matrix_probability(
            theta=theta, response_matrix=response_data.to_numpy()))
        denominator = normalising_constant_array
        sum = np.sum(np.divide(numerator, denominator), axis=1)

        def func(sigma, return_sample=False):
            factor = np.log(self.model.latent_density(
                theta, sigma=sigma, save=True))
            product = np.multiply(factor, sum)
            if return_sample:
                return(np.mean(product), product)
            return(np.mean(product))
        return(func)


    def q_0_gradient(self, theta, r_0_theta):
        D = self.model.latent_dimension
        N = theta.shape[0]

        def func(sigma):
            #quadratic_form = quadratic_form_func(sigma)
            inv_sigma = np.linalg.inv(sigma)

            sum1 = np.multiply(-0.5, inv_sigma)
            # sum2
            sum2 = np.dot(np.expand_dims(theta, 1), inv_sigma)
            #sum2 = -1*np.sum(np.multiply(sum2, sum2), axis=2).squeeze()
            sum2 = 0.5*np.matmul(sum2.reshape(N, 1, D, 1),
                                 sum2.reshape(N, 1, 1, D)).squeeze()
            sum2 = np.transpose(sum2, axes=[0, 2, 1])
            integrant = np.multiply(r_0_theta.reshape(
                (N, 1, 1)), np.add(sum1, sum2))
            return(np.mean(integrant, axis=0))
        return(func)

    def q_item(self, item: int, theta, r_0_theta, r_item_theta):
        r_diff = np.subtract(r_0_theta, r_item_theta)

        def func(a_item: np.array, delta_item: np.array):
            icc_values = self.model.icc(theta=theta, A=np.expand_dims(
                a_item, axis=0), delta=np.array([delta_item]), save=False).transpose()[0]
            log_likelihood_item = np.multiply(np.log(
                icc_values + np.float64(1.7976931348623157e-309)), r_item_theta) + np.multiply(np.log(1-icc_values + np.float64(1.7976931348623157e-309)), r_diff)
            return(np.mean(log_likelihood_item))
        return(func)
