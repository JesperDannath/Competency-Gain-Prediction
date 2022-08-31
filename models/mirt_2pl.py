from irt_model import irt_model
import numpy as np
from scipy.stats import multivariate_normal


class mirt_2pl(irt_model):

    def __init__(self, item_dimension: int, latent_dimension: int, A=np.empty(0), Q=np.empty(0), delta=None, sigma=None) -> None:
        """
        Args:
            item_dimension (int): Number of items in test
            latent_dimension (int): Dimensiontality of latent competency
            A (np.array): Matrix of item discriminations with shape (item_dimension, latent_dimension)
            delta (np.array): Array of item-intercepts with shape (item_dimension)
            Q (np.array): Matrix of Constraints on A of the form a_{i,j} == 0
        """
        super().__init__(item_dimension, latent_dimension)
        if Q.size == 0:
            Q = np.ones((item_dimension, latent_dimension))
        self.check_discriminations(A, Q)
        self.item_parameters = {
            "discrimination_matrix": A, "intercept_vector": delta, "q_matrix": Q}
        self.person_parameters = {"covariance": sigma}

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
            raise Exception("Covariance is not symmetric")
        if not np.all(np.linalg.eigvals(sigma) >= 0):
            raise Exception("New Covariance not positive semidefinite")
        return(True)

    def icc(self, theta: np.array, A=np.empty(0), delta=np.empty(0)) -> np.array:
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

    def set_parameters(self, parameters: dict):
        if "item_parameters" in parameters.keys():
            self.item_parameters.update(parameters["item_parameters"])
            self.check_discriminations()
        if "person_parameters" in parameters.keys():
            self.person_parameters.update(parameters["person_parameters"])

    def corr_to_sigma(self, corr):
        """Creates a Covariance Matrix sigma from an array of Correlations
        Args:
            corr (np.array): array of latant trait correlations. Entrys are the upper triangular Matrix
            of sigma from left to right and top to bottom. 
        """
        new_sigma = self.person_parameters["covariance"].copy().astype(
            np.float64)
        #new_sigma[np.triu_indices(self.latent_dimension, k=1)] = corr
        np.place(new_sigma,
                 mask=np.triu(np.ones(self.latent_dimension), k=1).astype(np.bool), vals=corr)
        np.place(new_sigma,
                 mask=np.tril(np.ones(self.latent_dimension), k=-1).astype(np.bool), vals=corr)
        #new_sigma[np.tril_indices(self.latent_dimension, k=-1)] = corr
        self.check_sigma(new_sigma)
        return(new_sigma)

    def fill_zero_discriminations(self, discriminations, item: int) -> np.array:
        mask = self.item_parameters["q_matrix"][item].astype(np.bool)
        a_item = np.zeros(mask.shape)
        np.place(a_item, mask=mask, vals=discriminations)
        return(a_item)
