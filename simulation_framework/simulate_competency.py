import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class respondent_population():

    def __init__(self, latent_dimension: int, latent_distribution=None, timesteps=1) -> None:
        if latent_distribution==None:
            self.latent_distribution = multivariate_normal(mean=np.ones(latent_dimension), cov=np.identity(latent_dimension))
        else:
            self.latent_distribution = latent_distribution
        self.latent_dimension = latent_dimension
        self.intervention=False
    
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
            return(self.latent_distribution.rvs(size=sample_size))

    def intervention(self, sample_size, conditional_distribution):
        """_summary_

        Args:
            sample_size (_type_): _description_
            conditional_distribution (_type_): _description_
        """
        self.latent_change = conditional_distribution(self.latent_distribution).rvs(size=sample_size)
        self.intervention = True


        