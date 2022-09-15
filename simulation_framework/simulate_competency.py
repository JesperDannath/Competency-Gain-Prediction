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

    def initialize_random_person_parameters(self):
        cov = make_sparse_spd_matrix(
            dim=self.latent_dimension, alpha=0.2, norm_diag=True, smallest_coef=-0.9, largest_coef=0.9)
        self.latent_distribution = multivariate_normal(mean=np.zeros(self.latent_dimension), cov=cov)
        return(cov)
    
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


        