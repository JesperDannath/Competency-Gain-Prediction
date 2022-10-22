import pandas as pd


class e_step():

    def __init__(self, incomplete_data: pd.DataFrame = pd.DataFrame()) -> None:
        self.incomplete_data = incomplete_data

    def set_incomplete_data(self, incomplete_data: list):
        self.incomplete_data = incomplete_data

    def step() -> dict:
        pass
