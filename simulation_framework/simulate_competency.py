import numpy as np
import pandas as pd
from sklearn.datasets import make_sparse_spd_matrix
from scipy.stats import multivariate_normal


class respondent_population():

    def __init__(self, latent_dimension: int, latent_distribution=None, intervention=True) -> None:
        self.latent_dimension = latent_dimension
        if (latent_distribution == None) and (intervention):
            self.latent_distribution = multivariate_normal(
                mean=np.zeros(latent_dimension), cov=np.identity(latent_dimension))
            self.intervention = True
        elif (latent_distribution == None) and (intervention):
            self.latent_distribution = multivariate_normal(mean=np.zeros(
                2*latent_dimension), cov=np.identity(2*latent_dimension))
            self.intervention = False
        else:
            self.latent_distribution = latent_distribution
        self.latent_dimension = latent_dimension

    def corr_from_Q(self, Q):
        q_cov = np.dot(Q.transpose(), Q)
        q_sd_vector = np.sqrt(q_cov.diagonal())
        inv_q_sd_matrix = np.linalg.inv(np.diag(q_sd_vector))
        q_corr = np.dot(
            np.dot(inv_q_sd_matrix, q_cov), inv_q_sd_matrix)
        return(q_corr)

    def initialize_random_person_parameters(self, early_Q=np.empty(0), late_Q=np.empty(0), late_mean_value=1, q_share=0.0):
        # TODO:  nochmal Q^tQ nehmen und dann den gewichteten Mittelwert bilden!
        # evtl. auch nochmal die Recovery mit unabhängigen Fähigkeiten testen
        # TODO: Um Werte kleiner zu machen Wurzel ziehen
        X = np.random.rand(self.latent_dimension, self.latent_dimension)
        # cov = np.dot(X, X.transpose())
        # sd_vector = np.sqrt(cov.diagonal())
        # inv_sd_matrix = np.linalg.inv(np.diag(sd_vector))
        # correlation_matrix = np.dot(
        #     np.dot(inv_sd_matrix, cov), inv_sd_matrix)
        D = self.latent_dimension
        if not self.intervention:
            correlation_matrix = make_sparse_spd_matrix(
                dim=self.latent_dimension, alpha=0.1, norm_diag=True, smallest_coef=0.0, largest_coef=0.9)
            mean = np.zeros(self.latent_dimension)
            self.latent_distribution = multivariate_normal(
                mean=mean, cov=correlation_matrix)
        else:
            correlation_matrix = make_sparse_spd_matrix(
                dim=2*self.latent_dimension, alpha=0.1, norm_diag=True, smallest_coef=0.0, largest_coef=0.9)
            mean = np.concatenate(
                (np.zeros(self.latent_dimension), late_mean_value*np.ones(self.latent_dimension)))
            self.latent_distribution = multivariate_normal(
                mean=mean, cov=correlation_matrix)
        # Enforce correlation to be positive
        # TODO: Must this be the case for correlation between early and late ability?
        correlation_matrix = np.round(np.abs(correlation_matrix), 4)
        # Calculate COV-Matrix implied by the Q-Matrix
        q_corr_early = self.corr_from_Q(early_Q)
        if not self.intervention:
            correlation_matrix = np.add(
                (1-q_share)*correlation_matrix, q_share*q_corr_early)
            self.latent_distribution = multivariate_normal(
                mean=mean, cov=np.round(correlation_matrix, 4))
        else:
            correlation_matrix[0:D, 0:D] = np.add(
                (1-q_share)*correlation_matrix[0:D, 0:D], q_share*q_corr_early)
            q_corr_late = self.corr_from_Q(late_Q)
            correlation_matrix[D:2*D, D:2*D] = np.add(
                (1-q_share)*correlation_matrix[D:2*D, D:2*D], q_share*q_corr_late)
            self.latent_distribution = multivariate_normal(
                mean=mean, cov=np.round(correlation_matrix, 4))
        return({"covariance": correlation_matrix})

    def sample(self, sample_size: int) -> pd.DataFrame:
        """Create a random variable sample from a the defined latent_distribution
        Args:
            sample_size (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        D = self.latent_dimension
        sample = self.latent_distribution.rvs(size=sample_size)
        if self.latent_dimension == 1:
            sample = np.expand_dims(sample, axis=1)
        if self.intervention:
            theta_sample = sample[:, 0:D]
            s_sample = sample[:, D:2*D]
            return(theta_sample, s_sample)
        else:
            return(sample)

    # def intervention(self, sample_size, conditional_distribution):
    #     """_summary_

    #     Args:
    #         sample_size (_type_): _description_
    #         conditional_distribution (_type_): _description_
    #     """
    #     self.latent_change = conditional_distribution(
    #         self.latent_distribution).rvs(size=sample_size)
    #     self.intervention = True
