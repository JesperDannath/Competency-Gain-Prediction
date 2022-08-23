import numpy as np


def random_items_from_Q_matrix(self, Q_matrix):
    self.Q_matrix = Q_matrix


# TODO: Sample A in a way that corresponds to theta s.t. the response probabilites are uniform
def random_item_parameters(number_of_items, latent_dimension, A_range=[0, 3], delta_range=[-1, 1]):
    A = np.random.randint(low=A_range[0], high=A_range[1], size=(
        number_of_items, latent_dimension))
    delta = np.random.randint(low=delta_range[0], high=delta_range[1])
    return({"A": A, "delta": delta, "item_dimension": number_of_items, "latent_dimension": latent_dimension})
