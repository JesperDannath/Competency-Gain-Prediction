import sys
import os
import pandas as pd
import numpy as np
import itertools
from scipy.stats import bernoulli
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl
    from mirt_2pl_gain import mirt_2pl_gain


class response_simulation():

    def __init__(self, population, item_dimension, early_item_params: dict = {}, late_item_params: dict = {}) -> None:
        self.population = population
        self.item_set = early_item_params
        self.item_dimension = item_dimension
        self.latent_dimension = self.population.latent_dimension
        self.early_model = mirt_2pl(self.item_dimension, self.latent_dimension)
        self.early_model.set_parameters({"item_parameters": early_item_params})
        self.late_model = mirt_2pl_gain(
            self.item_dimension, self.latent_dimension)
        self.late_model.set_parameters({"item_parameters": late_item_params})
        if self.item_dimension < self.latent_dimension:
            raise Exception("To few Items for given latent dimension")

    # TODO: Prevent All-zero columns
    def initialize_random_q_structured_matrix(self, structure="singular", ensure_id=False, early=True):
        if structure == "singular":
            Q = np.ones((self.item_dimension, self.latent_dimension))
        elif structure == "seperated":
            item_frequencies = multinomial.rvs(self.item_dimension, p=[
                                               1/self.latent_dimension for i in range(0, self.latent_dimension)])
            if 0 in item_frequencies:
                self.initialize_random_q_structured_matrix(
                    structure="seperated")
                return
            Q = []
            for i, frequency in enumerate(item_frequencies):
                for repeat in range(0, frequency):
                    Q.append(
                        [0 if j != i else 1 for j in range(0, self.latent_dimension)])
            Q = np.array(Q)
        elif structure == "pyramid":
            if (self.item_dimension < self.latent_dimension*2) & ensure_id:
                raise Exception(
                    "To few Items to ensure Identification for structrue: pyramid")
            # Items mit mehr skills sind ggf. unwahrscheinlicher (linearer abstieg)
            Q = []
            if ensure_id:
                Q = list(np.identity(self.latent_dimension))
                pyramid_items = self.item_dimension - self.latent_dimension
            else:
                pyramid_items = self. item_dimension
            stair_frequencies = multinomial.rvs(pyramid_items, p=[
                1/self.latent_dimension for i in range(0, self.latent_dimension)])
            if 0 in stair_frequencies:
                self.initialize_random_q_structured_matrix(structure="pyramid")
                return
            for i, frequency in enumerate(stair_frequencies):
                for repeat in range(0, frequency):
                    Q.append(
                        [1 if j <= i else 0 for j in range(0, self.latent_dimension)])
            Q = np.array(Q)
        # TODO: Add structure Chained
        # TODO: Add structure Full-Complexity
        elif (structure == "full"):
            n_pattern = 2**self.latent_dimension-1  # exclude all zero row
            if n_pattern > self.item_dimension:
                raise Exception("To few Items for structure full")
            pattern_frequencies = multinomial.rvs(self.item_dimension, p=[
                1/n_pattern for i in range(0, n_pattern)])
            if 0 in pattern_frequencies:
                self.initialize_random_q_structured_matrix(structure="full")
                return
            patterns = list(itertools.product(
                [0, 1], repeat=self.latent_dimension))
            patterns.remove(
                tuple([0 for i in range(0, self.latent_dimension)]))
            Q = []
            for i, pattern in enumerate(patterns):
                for j in range(0, pattern_frequencies[i]):
                    Q.append(pattern)
            Q = np.array(Q)
        else:
            raise Exception("Q-Matrix structure not known")
        if early:
            self.early_model.set_parameters(
                {"item_parameters": {"q_matrix": Q, "discrimination_matrix": Q}})
        else:
            self.late_model.set_parameters(
                {"item_parameters": {"q_matrix": Q, "discrimination_matrix": Q}})

    def get_Q(self, time="early"):
        if time not in ["early", "late"]:
            raise Exception("Invalid time selected")
        if time in ["early"]:
            return self.early_model.item_parameters["q_matrix"]
        if time in ["late"]:
            return self.late_model.item_parameters["q_matrix"]

    # def initialize_random_item_parameters(self, Q=np.empty(0)):
    #     if Q.size == 0:
    #         Q = self.early_model.item_parameters["q_matrix"]
    #     A = np.empty((self.item_dimension, self.latent_dimension))
    #     for i in range(0, self.item_dimension):
    #         A[i] = np.exp(multivariate_normal(mean=np.zeros(
    #             self.latent_dimension), cov=np.identity(self.latent_dimension)).rvs())
    #     A = np.multiply(A, Q)
    #     delta = multivariate_normal(mean=np.zeros(
    #         self.item_dimension), cov=np.identity(self.item_dimension)).rvs()
    #     item_parameters = {"discrimination_matrix": A,
    #                        "intercept_vector": delta}
    #     self.early_model.item_parameters.update(item_parameters)
    #     return(self.early_model.item_parameters)

    # # TODO: Prohibit all-zero rows
    # def initialize_random_item_parameters(self, Q=np.empty(0)):
    #     if Q.size == 0:
    #         Q = self.early_model.item_parameters["q_matrix"]
    #     # 1. Sample relative difficulties
    #     relative_difficulties_sample = np.sqrt(1)*self.population.sample(
    #         self.item_dimension)
    #     # t-Verteilung oder Gleichverteilung mal ausprobieren
    #     A = np.zeros((self.item_dimension, self.latent_dimension))
    #     delta = np.zeros(self.item_dimension)
    #     # 2. Get smallest relative difficulty per item
    #     for i in range(0, self.item_dimension):
    #         rel_item = relative_difficulties_sample[i]
    #         # np.where(rel_item >= np.median(rel_item))[0][0]
    #         rel_min = np.where(rel_item == np.min(rel_item))[0]
    #         # 3. Sample a_median
    #         a_min = np.exp(multivariate_normal(
    #             mean=0.5*np.ones(1), cov=np.identity(1)).rvs())
    #         # 4. Calculate delta
    #         delta[i] = -a_min*rel_item[rel_min]
    #         # 5. Calculate all other a_{i,j}'s
    #         A[i] = -delta[i]/rel_item
    #     item_parameters = {"discrimination_matrix": np.multiply(A, Q),
    #                        "intercept_vector": delta}
    #     self.early_model.item_parameters.update(item_parameters)
    #     if len(A[A <= 0]) > 0:
    #         raise Exception("Negative Discriminations sampled")
    #     return(self.early_model.item_parameters)

    # TODO: Prohibit all-zero rows
    def initialize_random_item_parameters(self, Q=np.empty(0), early=True):
        if Q.size == 0:
            if early:
                Q = self.early_model.item_parameters["q_matrix"]
            else:
                Q = self.late_model.item_parameters["q_matrix"]
        # 1. Sample relative difficulties
        delta = multivariate_normal(mean=np.zeros(
            1), cov=np.ones(1)).rvs(self.item_dimension)
        # t-Verteilung oder Gleichverteilung mal ausprobieren
        A = np.zeros((self.item_dimension, self.latent_dimension))
        #delta = np.zeros(self.item_dimension)
        # 2. Get smallest relative difficulty per item
        for i in range(0, self.item_dimension):
            A[i] = np.exp(multivariate_normal(
                mean=1*np.ones(self.latent_dimension), cov=0.5*np.identity(self.latent_dimension)).rvs())
        item_parameters = {"discrimination_matrix": np.multiply(A, Q),
                           "intercept_vector": delta}
        if len(A[A <= 0]) > 0:
            raise Exception("Negative Discriminations sampled")
        if early:
            self.early_model.item_parameters.update(item_parameters)
            return(self.early_model.item_parameters)
        else:
            self.late_model.item_parameters.update(item_parameters)
            return(self.late_model.item_parameters)

    def scale_discriminations(self, sigma_psi):
        self.late_model.set_parameters(
            {"person_parameters": {"covariance": sigma_psi}})
        # We have to change A, since it should be according to the
        # size or the diagonal values of COV(\theta + s)
        A = self.late_model.item_parameters["discrimination_matrix"]
        convolution_sigma_diag = np.diag(np.diag(
            self.late_model.get_cov("convolution")))
        scaled_A = np.dot(A, np.linalg.inv(convolution_sigma_diag))
        item_parameters = {"discrimination_matrix": scaled_A}
        self.late_model.item_parameters.update(item_parameters)

    def sample(self, sample_size) -> pd.DataFrame:
        sample = {}
        sample["latent_trait"], sample["latent_gain"] = self.population.sample(
            sample_size=sample_size)
        p_early = self.early_model.icc(theta=sample["latent_trait"])
        p_late = self.late_model.icc(
            theta=sample["latent_trait"], s=sample["latent_gain"], cross=False)
        sample["early_responses"] = pd.DataFrame(bernoulli(p=p_early).rvs())
        sample["late_responses"] = pd.DataFrame(bernoulli(p=p_late).rvs())
        sample["sample_size"] = sample_size
        sample["latent_dimension"] = self.latent_dimension
        sample["item_dimension"] = self.item_dimension
        if list(np.ones(sample_size)) in list(sample["early_responses"].transpose()):
            raise Exception("All-correct item identified")
        if list(np.zeros(sample_size)) in list(sample["early_responses"].transpose()):
            raise Exception("All-incorrect item identified")
        if list(np.ones(sample_size)) in list(sample["late_responses"].transpose()):
            raise Exception("All-correct item identified")
        if list(np.zeros(sample_size)) in list(sample["late_responses"].transpose()):
            raise Exception("All-incorrect item identified")
        return(sample)

    def get_item_parameters(self):
        return({"real"})
