import pandas as pd


class irt_model:
    """General IRT model
    """

    def __init__(self, item_dimension, latent_dimension) -> None:
        self.latent_dimension = latent_dimension
        self.item_dimension = item_dimension