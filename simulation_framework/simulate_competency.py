import numpy as np
import pandas as pd
from sklearn.datasets import make_sparse_spd_matrix
from scipy.stats import multivariate_normal

class respondent_population():

    def __init__(self, latent_dimension: int, latent_distribution=None, timesteps=1) -> None:
        self.latent_dimension = latent_dimension
        if latent_distribution==None:
            self.latent_distribution = multivariate_normal(mean=np.zeros(latent_dimension), cov=np.identity(latent_dimension))
        else:
            self.latent_distribution = latent_distribution
        self.latent_dimension = latent_dimension
        self.intervention=False

    def initialize_random_person_parameters(self, Q=np.empty(0), q_share=0.0):
        #TODO:  nochmal Q^tQ nehmen und dann den gewichteten Mittelwert bilden!
        #evtl. auch nochmal die Recovery mit unabhängigen Fähigkeiten testen
        X = np.random.rand(self.latent_dimension, self.latent_dimension) #TODO: Um Werte kleiner zu machen Wurzel ziehen
        # cov = np.dot(X, X.transpose())
        # sd_vector = np.sqrt(cov.diagonal())
        # inv_sd_matrix = np.linalg.inv(np.diag(sd_vector))
        # correlation_matrix = np.dot(
        #     np.dot(inv_sd_matrix, cov), inv_sd_matrix)
        correlation_matrix = make_sparse_spd_matrix(
             dim=self.latent_dimension, alpha=0.2, norm_diag=True, smallest_coef=0.0, largest_coef=0.9)
        self.latent_distribution = multivariate_normal(mean=np.zeros(self.latent_dimension), cov=correlation_matrix)
        correlation_matrix = np.round(np.abs(correlation_matrix), 4)
        #Calculate COV-Matrix implied by the Q-Matrix
        q_cov = np.dot(Q.transpose(), Q)
        q_sd_vector = np.sqrt(q_cov.diagonal())
        inv_q_sd_matrix = np.linalg.inv(np.diag(q_sd_vector))
        q_corr = np.dot(
             np.dot(inv_q_sd_matrix, q_cov), inv_q_sd_matrix)
        correlation_matrix = np.add((1-q_share)*correlation_matrix, q_share*q_corr)
        correlation_matrix = np.round(correlation_matrix, 4)
        # TODO: Use mean for Competency Gain model
        self.latent_distribution = multivariate_normal(mean=np.zeros(self.latent_dimension), cov=correlation_matrix)
        return({"covariance": correlation_matrix})
    
    def sample(self, sample_size:int) -> pd.DataFrame:
        """Create a random variable sample from a the defined latent_distribution
        Args:
            sample_size (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        if self.intervention:
            ... #TODO: Concatenate
        else:
            sample = self.latent_distribution.rvs(size=sample_size)
            if self.latent_dimension==1 : 
                sample = np.expand_dims(sample, axis=1)
            return(sample)

    def intervention(self, sample_size, conditional_distribution):
        """_summary_

        Args:
            sample_size (_type_): _description_
            conditional_distribution (_type_): _description_
        """
        self.latent_change = conditional_distribution(self.latent_distribution).rvs(size=sample_size)
        self.intervention = True


        