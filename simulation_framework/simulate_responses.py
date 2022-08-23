
import sys
import os
from scipy.stats import bernoulli
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class response_simulation():

    def __init__(self, population, early_item_params, late_item_params=None) -> None:
        self.population = population
        self.item_set = early_item_params
        self.item_dimension = early_item_params["item_dimension"]
        self.latent_dimension = early_item_params["latent_dimension"]
        self.early_model = mirt_2pl(self.item_dimension, self.latent_dimension,
                                    A=early_item_params["A"], delta=early_item_params["delta"])

    def sample(self, sample_size):
        sample = {}
        sample["latent_trait"] = self.population.sample(
            sample_size=sample_size)
        p_early = self.early_model.icc(sample["latent_trait"])
        #p_late = LFA_Curve(late_parameters, alpha, s)
        sample["early_responses"] = bernoulli(p=p_early).rvs()
        #late_sample = bernoulli(p=p_late).rvs()
        return(sample)
