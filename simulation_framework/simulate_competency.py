import numpy as np
import pandas as pd
from sklearn.datasets import make_sparse_spd_matrix
from scipy.stats import multivariate_normal
import os
import sys
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl
    from mirt_2pl_gain import mirt_2pl_gain


class respondent_population():

    def __init__(self, latent_dimension: int, latent_distribution=None, intervention=True) -> None:
        self.latent_dimension = latent_dimension
        if (latent_distribution == None) and (not intervention):
            self.latent_distribution = multivariate_normal(
                mean=np.zeros(latent_dimension), cov=np.identity(latent_dimension))
            self.intervention = False
        elif (latent_distribution == None) and (intervention):
            self.latent_distribution = multivariate_normal(mean=np.zeros(
                2*latent_dimension), cov=np.identity(2*latent_dimension))
            self.intervention = True
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

    def initialize_random_person_parameters(self, early_Q=np.empty(0), late_Q=np.empty(0), late_mean_value=1, q_share=0.0, constraint_type="none"):
        # TODO:  nochmal Q^tQ nehmen und dann den gewichteten Mittelwert bilden!
        # evtl. auch nochmal die Recovery mit unabhängigen Fähigkeiten testen
        # TODO: Um Werte kleiner zu machen Wurzel ziehen
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
            model = mirt_2pl_gain(
                latent_dimension=self.latent_dimension, item_dimension=late_Q.shape[0], Q=late_Q)
            model.set_parameters(
                {"item_parameters": {"q_matrix": late_Q}})
            q_corr_late = self.corr_from_Q(late_Q)
            if constraint_type == "esigma_spsi":
                sigma = make_sparse_spd_matrix(
                    dim=self.latent_dimension, alpha=0.1, norm_diag=True, smallest_coef=0.0, largest_coef=0.9)
                sigma = np.round(np.abs(sigma), 4)
                sigma = np.add(
                    (1-q_share)*sigma, q_share*q_corr_late)
                # TODO: How to know range for psi?
                c = 0.4+np.random.rand(1)*0.6
                rho = 0.4+0.5*np.random.rand(1)
                # create late sigma
                late_sigma_var = c*(1-rho)*np.diagonal(sigma)
                late_sigma_sd_vector = np.sqrt(late_sigma_var)
                sd_matrix = np.diag(late_sigma_sd_vector)
                inv_sd_matrix = np.linalg.inv(np.diag(late_sigma_sd_vector))
                late_sigma = np.dot(
                    np.dot(sd_matrix, sigma), sd_matrix)
                # Create random psi
                diag_psi = 0.5*c*rho*np.ones(D)
                psi = 2*np.random.rand(D, D)-1
                psi = np.tril(psi) + np.tril(psi, -1).transpose()
                psi[np.diag_indices_from(psi)] = diag_psi
                #fill in values
                correlation_matrix[0:D, 0:D] = sigma
                correlation_matrix[D:2*D, D:2*D] = late_sigma
                correlation_matrix[0:D, D:2*D] = psi
                correlation_matrix[D:2*D, 0:D] = psi.transpose()
                correlation_matrix = np.round(correlation_matrix, 4)
                try:
                    model.check_sigma(correlation_matrix, callback=False)
                except Exception:
                    return(self.initialize_random_person_parameters(
                        early_Q=early_Q, late_Q=late_Q, q_share=q_share, constraint_type=constraint_type))
            elif constraint_type == "diag_sum":
                correlation_matrix = model.fix_sigma(correlation_matrix)
                correlation_matrix[0:D, 0:D] = np.add(
                    (1-q_share)*correlation_matrix[0:D, 0:D], q_share*q_corr_early)
                correlation_matrix[D:2*D, D:2*D] = np.add(
                    (1-q_share)*correlation_matrix[D:2*D, D:2*D], q_share*q_corr_late)
            elif constraint_type == "unnormal_late":
                sigma = make_sparse_spd_matrix(
                    dim=self.latent_dimension, alpha=0.1, norm_diag=True, smallest_coef=0.0, largest_coef=0.9)
                sigma = np.abs(np.round(sigma, 5))
                psi = 2*np.random.rand(D, D) -1
                psi = np.round(psi, 5)
                late_sigma = 0.5*make_sparse_spd_matrix(
                    dim=self.latent_dimension, alpha=0.1, norm_diag=False, smallest_coef=0.0, largest_coef=0.9) 
                late_sigma = np.abs(np.round(late_sigma, 5))
                correlation_matrix[0:D, 0:D] = sigma
                correlation_matrix[D:2*D, D:2*D] = late_sigma
                correlation_matrix[0:D, D:2*D] = psi
                correlation_matrix[D:2*D, 0:D] = psi.transpose()
                try:
                    model.check_sigma(correlation_matrix, callback=False)
                except Exception:
                    return(self.initialize_random_person_parameters(
                        early_Q=early_Q, late_Q=late_Q, q_share=q_share, constraint_type=constraint_type))
            self.latent_distribution = multivariate_normal(
                mean=mean, cov=np.round(correlation_matrix, 4))
        return({"covariance": np.round(correlation_matrix, 4)})

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
