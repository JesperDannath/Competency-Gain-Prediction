from irt_model import irt_model
import numpy as np
import pandas as pd
import copy
import scipy
from scipy.stats import multivariate_normal
from scipy.stats.qmc import MultivariateNormalQMC
from scipy.optimize import minimize


class mirt_2pl(irt_model):

    def __init__(self, item_dimension: int, latent_dimension: int, A=np.empty(0), Q=np.empty(0), delta=np.empty(0), sigma=np.empty(0)) -> None:
        """
        Args:
            item_dimension (int): Number of items in test
            latent_dimension (int): Dimensiontality of latent competency
            A (np.array): Matrix of item discriminations with shape (item_dimension, latent_dimension)
            delta (np.array): Array of item-intercepts with shape (item_dimension)
            Q (np.array): Matrix of Constraints on A of the form a_{i,j} == 0
        """
        super().__init__(item_dimension, latent_dimension)
        self.initialize_parameters(A, Q, delta, sigma)
        self.type = "normal"

    def initialize_parameters(self, A, Q, delta, sigma):
        if Q.size == 0:
            Q = np.ones((self.item_dimension, self.latent_dimension))
        if A.size == 0:
            A = Q.copy()
        else:
            if A.shape != Q.shape:
                raise Exception(
                    "Wrong item Dimension specified or shapes of A and Q don't match")
        if delta.size == 0:
            delta = np.ones(self.item_dimension)
        if sigma.size == 0:
            sigma = np.identity(self.latent_dimension)
        self.check_discriminations(A, Q)
        self.item_parameters = {
            "discrimination_matrix": A, "intercept_vector": delta, "q_matrix": Q}
        self.person_parameters = {"covariance": sigma}

    def initialize_from_responses(self, response_data: pd.DataFrame, sigma=True):
        #A = np.ones((self.item_dimension, self.latent_dimension))
        #self.item_parameters["q_matrix"] = A
        A = self.item_parameters["q_matrix"]
        # delta is initialized with the inverse logistic function
        item_response_mean = np.mean(response_data, axis=0)
        item_response_mean[item_response_mean == 0] = 0.01
        delta = np.log(np.divide(item_response_mean, 1 -
                                 item_response_mean)).to_numpy()
        n = self.latent_dimension - 1
        if sigma:
            sigma = self.corr_to_sigma(
                corr_u=0.5*np.ones(int((n*(n+1))/2)))
            person_parameters = {"covariance": sigma}
        else:
            person_parameters = {}
        item_parameters = {
            "discrimination_matrix": A, "intercept_vector": delta}
        self.set_parameters(
            {"item_parameters": item_parameters, "person_parameters": person_parameters})

    def check_discriminations(self, A=np.empty(0), Q=np.empty(0)) -> bool:
        if A.size == 0:
            A = self.item_parameters["discrimination_matrix"]
        if Q.size == 0:
            Q = self.item_parameters["q_matrix"]
        if not np.array_equal(np.multiply(A, Q), A):
            raise Exception(
                "Discriminations (A) do not match to Constraints (Q)")
        return(True)

    def check_sigma(self, sigma: np.array = np.empty(0)):
        if sigma.size == 0:
            sigma = self.person_parameters["covariance"]
        if not sigma.shape == (self.latent_dimension, self.latent_dimension):
            raise Exception("Covariance is of wrong shape")
        if not np.array_equal(sigma.transpose(), sigma):
            print(sigma)
            raise Exception("Covariance is not symmetric")
        if not np.all(np.linalg.eigvals(sigma) >= 0):
            # print(sigma)
            raise Exception("New Covariance not positive semidefinite")
        return(True)

    def fix_sigma(self, sigma: np.array, sigma_constraint=""):
        """Fix the main Diagonal of sigma if not all entrys are one.
        Args:
            sigma (np.array): Latent Covariance matrix of shape (latent_dim, latent_dim)
        """
        sd_vector = np.sqrt(sigma.diagonal())
        inv_sd_matrix = np.linalg.inv(np.diag(sd_vector))
        correlation_matrix = np.dot(
            np.dot(inv_sd_matrix, sigma), inv_sd_matrix)
        return(correlation_matrix)

    def icc(self, theta: np.array, A=np.empty(0), delta=np.empty(0), save=False) -> np.array:
        """Item Characteristic Curve for a MIRT-2PL Model or similar logistic IRT Models
        Args:
            A (np.array): Matrix of item discriminations with shape (item_dimension, latent_dimension)
            delta (np.array): Array of item-intercepts with shape (item_dimension)
            theta (np.array): Matrix of latent competencies with shape (sample_size, latent_dimension)
        """
        if A.size == 0:
            A = self.item_parameters["discrimination_matrix"]
        if delta.size == 0:
            delta = self.item_parameters["intercept_vector"]
        linear_predictor = np.add(
            np.dot(A, np.transpose(theta)), np.expand_dims(delta, axis=1))
        p = np.transpose(
            np.divide(1, np.add(1, np.exp(np.multiply(-1, linear_predictor)))))
        if save:
            # p[p == 0] = np.min(p[p > 0])  # np.float64(1.7976931348623157e-308)
            # p[p == 1] = np.max(p[p < 1])  # np.float64(
            # 1)-np.float64(1.7976931348623157e-16)
            p[p == 0] = np.float64(1.7976931348623157e-308)
            p[p == 1] = np.float64(
                1)-np.float64(1.7976931348623157e-16)
        return(p)

    def latent_density(self, theta: np.array, sigma: np.array = np.empty(0), mu: np.array = np.empty(0), save=False):
        if sigma.size == 0:
            sigma = self.person_parameters["covariance"]
        if mu.size == 0:
            mu = np.zeros(self.latent_dimension)
        if len(mu.shape) == 2:
            pdf_values = np.zeros((mu.shape[0], theta.shape[0]))
            for i, mean in enumerate(mu):  # TODO: Make this faster
                pdf_values[i] = multivariate_normal.pdf(
                    x=theta, mean=mean, cov=sigma)
            # return(pdf)
        else:
            pdf_values = multivariate_normal.pdf(x=theta, mean=mu, cov=sigma)
        if save and (type(pdf_values) != np.float64):
            # np.min(pdf_values[pdf_values > 0])
            pdf_values[pdf_values == 0] = np.float64(1.7976931348623157e-320)
        return(pdf_values)

    def sample_competency(self, sample_size=1, qmc=False):
        if not qmc:
            sample = multivariate_normal.rvs(size=sample_size, mean=np.zeros(
                self.latent_dimension), cov=self.person_parameters["covariance"])
            # if self.latent_dimension == 1:
            #    sample = np.expand_dims(sample, axis=1)
        else:
            sample = MultivariateNormalQMC(mean=np.zeros(
                self.latent_dimension), cov=self.person_parameters["covariance"], engine=None).random(sample_size)
        # Ensure correct dimensionality if sample_siz==1 or latent_dim==1
        sample = sample.reshape((sample_size, self.latent_dimension))
        return(sample)

    def response_matrix_probability(self, theta, response_matrix: np.array, A=np.empty(0),
                                    delta=np.empty(0)) -> np.array:
        """Calculate a matrix of response-vector probabilities. One entry in the resulting matrix
        reflects the response vector probability for response-vector i for a person with competency j

        Args:
            theta (np.array): competency matrix of shape (competency_sample_size, latent_dimension)
            response_matrix (np.array): response-matrix of shape (number_response_shapes, item_dimension)
            A (_type_, optional): _description_. Defaults to np.empty(0).
            delta (_type_, optional): _description_. Defaults to np.empty(0).

        Returns:
            np.array: _description_
        """
        if A.size == 0:
            A = self.item_parameters["discrimination_matrix"]
        if delta.size == 0:
            delta = self.item_parameters["intercept_vector"]
        correct_response_probabilities = self.icc(
            theta, A, delta)
        # We want to apply each response vector to each competency-induced correct-response-probability
        correct_response_probabilities = np.expand_dims(
            correct_response_probabilities, axis=1)
        probability_vector = np.add(np.multiply(correct_response_probabilities, response_matrix),
                                    np.multiply(np.subtract(1, correct_response_probabilities), np.subtract(1, response_matrix)))
        probability = np.prod(probability_vector, axis=2)
        # TODO: Log-option, bigger storage 128 bit?
        return(probability)

    def joint_competency_answer_density(self, theta, response_vector: np.array, A=np.empty(0),
                                        delta=np.empty(0), sigma=np.empty(0)) -> np.array:
        """Calculate the joint probability-density of a certain response-vector and a latent-trait vector
        Args:
            theta (np.array): array of shape (latent_dimension) or matrix of shape (sample_size, latent_dimension)
            response_vector (np.array): array of shape (item_dimension)
            A (np.array): Matrix of item discriminations with shape (item_dimension, latent_dimension)
            delta (np.array): Array of item-intercepts with shape (item_dimension)
            sigma (np.array, optional): Covariance Matrix. Array of shape (latent_dimension, latent_dimension)
        Returns:
            np.array: _description_
        """
        if sigma.size == 0:
            sigma = self.person_parameters["covariance"]
        if A.size == 0:
            A = self.item_parameters["discrimination_matrix"]
        if delta.size == 0:
            delta = self.item_parameters["intercept_vector"]
        response_probability = self.response_matrix_probability(
            theta, response_vector, A, delta)
        marginal_competency_density = multivariate_normal.pdf(x=theta, mean=np.zeros(
            self.latent_dimension), cov=sigma)
        if len(marginal_competency_density.shape) != 0:
            marginal_competency_density = np.expand_dims(
                marginal_competency_density, axis=1)
        joint_density = np.multiply(
            response_probability, marginal_competency_density)
        return(joint_density)

    def set_parameters(self, parameters: dict):
        if "item_parameters" in parameters.keys():
            self.item_parameters.update(parameters["item_parameters"])
            self.check_discriminations()
        if "person_parameters" in parameters.keys():
            self.person_parameters.update(parameters["person_parameters"])
            self.check_sigma()

    def get_parameters(self) -> dict:
        parameter_dict = {"item_parameters": self.item_parameters,
                          "person_parameters": self.person_parameters}
        return(copy.deepcopy(parameter_dict))

    def corr_to_sigma(self, corr_u, check=True):
        """Creates a Covariance Matrix sigma from an array of Correlations
        Args:
            corr_u (np.array): array of latant trait correlations. Entrys are the upper triangular Matrix
            of sigma from left to right and top to bottom.
        """
        dim = self.person_parameters["covariance"].shape[0]
        new_sigma = np.identity(dim).astype(
            np.float64)
        # new_sigma[np.triu_indices(self.latent_dimension, k=1)] = corr
        np.place(new_sigma,
                 mask=np.triu(np.ones(dim), k=1).astype(np.bool), vals=corr_u)
        corr_l = new_sigma.transpose()[np.tril_indices_from(new_sigma, k=-1)]
        np.place(new_sigma,
                 mask=np.tril(np.ones(dim), k=-1).astype(np.bool), vals=corr_l)
        # new_sigma[np.tril_indices(self.latent_dimension, k=-1)] = corr
        if check == True:
            self.check_sigma(new_sigma)
        return(new_sigma)

    # TODO: Write unittest
    def fill_zero_discriminations(self, discriminations, item: int = -1) -> np.array:
        if item == -1:
            mask = self.item_parameters["q_matrix"].flatten().astype(np.bool)
            A = np.zeros(mask.shape)
            np.place(A, mask=mask, vals=discriminations)
            return(A.reshape((self.item_dimension, self.latent_dimension)))
        else:
            mask = self.item_parameters["q_matrix"][item].astype(np.bool)
            a_item = np.zeros(mask.shape)
            try:
                np.place(a_item, mask=mask, vals=discriminations)
            except ValueError:
                print("mask: {0}".format(mask))
                print("Discriminations: {0}".format(discriminations))
                raise Exception("Could not fill zero Discriminations")
            return(a_item)

    def marginal_response_loglikelihood(self, response_data: pd.DataFrame(), N=1000):
        theta = self.sample_competency(N)
        response_matrix_prob = self.response_matrix_probability(
            theta=theta, response_matrix=response_data.to_numpy())
        marginal_vector_probabilities = np.log(
            np.mean(response_matrix_prob, axis=0))
        response_loglikelihood = np.sum(marginal_vector_probabilities)
        return(response_loglikelihood)

    def answer_log_likelihood(self, theta, answer_vector):
        ICC_values = self.icc(theta)
        latent_density = self.latent_density(theta)
        log_likelihood = np.dot(answer_vector, np.log(ICC_values)[0]) + np.dot(
            (1-answer_vector), np.log(1-ICC_values)[0]) + np.log(latent_density)
        return(log_likelihood)

    def predict_competency(self, response_data: pd.DataFrame) -> np.array:
        """Given the estimated item-parameters for a MIRT-Model and some response_data, this function will estimate the latent ability for every respondent.

        Args:
            response_data (pd.DataFrame or np.array): Response data from Item's 
        """
        competency_matrix = np.zeros(
            shape=(response_data.shape[0], self.latent_dimension))
        for i, response_pattern in enumerate(response_data.to_numpy()):
            def nll(x): return -1*self.answer_log_likelihood(x,
                                                             response_pattern)
            x0 = self.sample_competency()
            res = minimize(nll, x0=x0, method='BFGS')
            competency_matrix[i] = res.x
        return(competency_matrix)
