import numpy as np

#TODO: Make all of this functions that return an item_parameters dictionary
class item_set():

    def __init__(self, number_of_items: int, latent_dimension: int, A=np.empty(0), delta=np.empty(0)) -> None:
        self.number_of_items = number_of_items
        self.latent_dimension = latent_dimension
        if np.size(A) == 0:
            self.A = np.ones((number_of_items, latent_dimension))
        else:
            self.A = A
        if np.size(delta) == 0: 
            self.delta = np.zeros(number_of_items)
        else:
            self.delta = delta

    def set_Q_matrix(self, Q_matrix):
        self.Q_matrix = Q_matrix

    def randomize_item_properties(self, A_range=[0,3], delta_range=[-1,1],constraints=None): #TODO: Sample A in a way that corresponds to theta s.t. the response probabilites are uniform
        self.A = np.random.randint(low=A_range[0], high=A_range[1], size=(self.number_of_items, self.latent_dimension))
        self.delta = np.random.randint(low=delta_range[0], high=delta_range[1])

    def get_A(self) -> np.array:
        return(self.A)
    
    def get_delta(self) -> np.array:
        return(self.delta)

    