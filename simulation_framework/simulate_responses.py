import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class response_simulation():

    def __init__(self, population, item_dimension, early_item_params: dict = {}, late_item_params: dict = {}) -> None:
        self.population = population
        self.item_set = early_item_params
        self.item_dimension = item_dimension
        self.latent_dimension = self.population.latent_dimension
        self.early_model = mirt_2pl(self.item_dimension, self.latent_dimension)
        self.early_model.set_parameters({"item_parameters": early_item_params})

    def initialize_random_q_structured_matrix(self, structure="singular"):
        if structure == "singular":
            early_Q = np.ones((self.item_dimension, self.latent_dimension))
        elif structure == "seperated":
            item_frequencies = multinomial.rvs(self.item_dimension, p=[
                                               1/self.latent_dimension for i in range(0, self.latent_dimension)])
            early_Q = []
            for i, frequency in enumerate(item_frequencies):
                for repeat in range(0, frequency):
                    early_Q.append(
                        [0 if j != i else 1 for j in range(0, self.latent_dimension)])
            early_Q = np.array(early_Q)
        else:
            raise Exception("Q-Matrix structure not known")
        self.early_model.set_parameters(
            {"item_parameters": {"q_matrix": early_Q, "discrimination_matrix": early_Q}})

    def initialize_random_item_parameters(self, Q=np.empty(0)):
        if Q.size == 0:
            Q = self.early_model.item_parameters["q_matrix"]
        A = np.empty((self.item_dimension, self.latent_dimension))
        for i in range(0, self.item_dimension):
            A[i] = np.exp(multivariate_normal(mean=np.zeros(
                self.latent_dimension), cov=np.identity(self.latent_dimension)).rvs())
        A = np.multiply(A, Q)
        delta = multivariate_normal(mean=np.zeros(
            self.item_dimension), cov=np.identity(self.item_dimension)).rvs()
        item_parameters = {"discrimination_matrix": A,
                           "intercept_vetor": delta}
        self.early_model.item_parameters.update(item_parameters)
        return(self.early_model.item_parameters)

    def sample(self, sample_size) -> pd.DataFrame:
        sample = {}
        sample["latent_trait"] = self.population.sample(
            sample_size=sample_size)
        p_early = self.early_model.icc(sample["latent_trait"])
        #p_late = LFA_Curve(late_parameters, alpha, s)
        sample["early_responses"] = pd.DataFrame(bernoulli(p=p_early).rvs())
        #late_sample = bernoulli(p=p_late).rvs()
        return(sample)
