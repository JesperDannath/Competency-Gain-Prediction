import numpy as np
from simulate_competency import respondent_population
from simulate_responses import response_simulation


class item_response_simulation():

    def __init__(self, item_dimension, latent_dimension) -> None:
        self.respondend_population = respondent_population(
            latent_dimension=latent_dimension)
        self.response_simulation = response_simulation(
            population=self.respondend_population, item_dimension=item_dimension)
        self.latent_dimension = latent_dimension
        self.item_dimension = item_dimension

    def set_up(self, q_structure="custom", early_Q=np.empty(0), late_Q=np.empty(0), ensure_id=False, q_share=0):
        if (early_Q.size == 0) & (q_structure == "custom"):
            raise Exception("No Q-Matrix provided")
        if q_structure != "custom":
            self.response_simulation.initialize_random_q_structured_matrix(
                structure=q_structure, ensure_id=ensure_id)
            early_Q = self.response_simulation.get_Q("early")
        self.early_person_parameters = self.respondend_population.initialize_random_person_parameters(
            Q=early_Q, q_share=q_share)
        self.early_item_parameters = self.response_simulation.initialize_random_item_parameters(
            Q=early_Q)
        return(self.get_real_parameters())

    def sample(self, sample_size: int):
        sample_dict = self.response_simulation.sample(sample_size)
        return(sample_dict)

    def get_real_parameters(self):
        self.real_early_parameters = {
            "item_parameters": self.early_item_parameters, "person_parameters": self.early_person_parameters}
        parameter_dict = {"real_early_parameters": self.real_early_parameters}
        return(parameter_dict)