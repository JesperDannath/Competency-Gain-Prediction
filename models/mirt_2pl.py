from models.irt_model import irt_model
import numpy as np
from scipy.stats import multivariate_normal


class mirt_2pl(irt_model):

    def __init__(self, item_dimension, latent_dimension, A=None, delta=None, sigma=None) -> None:
        """
        Args:
            item_dimension (int): Number of items in test
            latent_dimension (int): Dimensiontality of latent competency
            A (np.array): Matrix of item discriminations with shape (item_dimension, latent_dimension)
            delta (np.array): Array of item-intercepts with shape (item_dimension)
        """
        super().__init__(item_dimension, latent_dimension)
        self.item_parameters = {
            "discrimination_matrix": A, "intercept_vector": delta}
        self.person_parameters = {"covariance": sigma}

    def icc(self, theta, A=np.empty(0), delta=np.empty(0)) -> np.array:
        """_summary_
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
        #p = 1/(1+np.exp(-linear_predictor))
        return(p)

    def latent_density(self, theta: np.array, sigma: np.array = np.empty(0)):
        if sigma.size == 0:
            sigma = self.person_parameters["covariance"]
        return(multivariate_normal.pdf(x=theta, mean=np.zeros(
            self.latent_dimension), cov=sigma))

    def sample_competency(self):
        return(multivariate_normal.rvs(mean=np.zeros(
            self.latent_dimension), cov=self.person_parameters["covariance"]))

    def response_vector_probability(self, theta, response_vector: np.array, A=np.empty(0),
                                    delta=np.empty(0)) -> np.array:
        if A.size == 0:
            A = self.item_parameters["discrimination_matrix"]
        if delta.size == 0:
            delta = self.item_parameters["intercept_vector"]
        correct_response_probabilities = self.icc(
            np.expand_dims(theta, axis=0), A, delta)[0]
        probability_vector = np.add(np.multiply(correct_response_probabilities, response_vector),
                                    np.multiply(np.subtract(1, correct_response_probabilities), np.subtract(1, response_vector)))
        probability = np.prod(probability_vector)
        return(probability)

    def joint_competency_answer_density(self, theta, response_vector: np.array, A=np.empty(0),
                                        delta=np.empty(0), sigma=np.empty(0)) -> np.array:
        """Calculate the joint probability-density of a certain response-vector and a latent-trait vector
        Args:
            theta (np.array): array of shape (latent_dimension)
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
        response_probability = self.response_vector_probability(
            theta, response_vector, A, delta)
        marginal_competency_density = multivariate_normal.pdf(x=theta, mean=np.zeros(
            self.latent_dimension), cov=sigma)
        joint_density = np.multiply(
            response_probability, marginal_competency_density)
        return(joint_density)

    def set_parameters(self, parameters):
        self.item_parameters = parameters["item_parameters"]
        self.person_parameters = parameters["person_parameters"]

    def corr_to_sigma(self, corr):
        new_sigma = self.person_parameters["covariance"].copy()
        new_sigma[np.triu_indices(self.latent_dimension, k=1)] = corr
        new_sigma[np.triu_indices(self.latent_dimension, k=1)] = corr
        return(new_sigma)
