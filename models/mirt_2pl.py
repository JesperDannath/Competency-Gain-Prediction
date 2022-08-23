from irt_model import irt_model
import numpy as np


class mirt_2pl(irt_model):

    def __init__(self, item_dimension, latent_dimension, A, delta) -> None:
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

    def icc(self, theta, A=np.empty(0), delta=np.empty(0)) -> np.array:
        """_summary_
        Args:
            A (np.array): Matrix of item discriminations with shape (item_dimension, latent_dimension)
            delta (np.array): Array of item-intercepts with shaoe (item_dimension)
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
