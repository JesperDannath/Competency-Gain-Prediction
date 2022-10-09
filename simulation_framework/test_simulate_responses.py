import unittest
import numpy as np
import pandas as pd
import os
import sys
from item_response_simulation import item_response_simulation
print(os.path.realpath("./models"))
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class test_mirt_2pl(unittest.TestCase):

    def setUp(self):
        self.item_response_simulation = item_response_simulation(
            latent_dimension=2, item_dimension=10)

    def test_item_response_simulation_singular(self):
        self.item_response_simulation.set_up(q_structure="singular")
        sample = self.item_response_simulation.sample(100)
        self.assertTrue(sample["early_responses"].shape == (100, 10))

    def test_item_response_simulation_pyramid(self):
        self.item_response_simulation.set_up(
            q_structure="pyramid", ensure_id=True, q_share=0.5)
        sample = self.item_response_simulation.sample(100)
        self.assertTrue(sample["early_responses"].shape == (100, 10))

    def test_item_response_simulation_full(self):
        self.item_response_simulation.set_up(
            q_structure="full", ensure_id=True, q_share=0.0)
        sample = self.item_response_simulation.sample(100)
        self.assertTrue(sample["early_responses"].shape == (100, 10))
        parameters = self.item_response_simulation.get_real_parameters()
        q_matrix = parameters["real_early_parameters"]["item_parameters"]["q_matrix"]
        self.assertTrue(q_matrix.shape == (10, 2))
        self.assertTrue([1, 1] in q_matrix.tolist())
        # Negative test for zero row
        self.assertFalse([0, 0] in q_matrix.tolist())


if __name__ == '__main__':
    unittest.main()
