import em_algorithm
import unittest
import sys
import os
sys.path.append(os.path.realpath("./models"))
if True:  # noqa: E402
    import simulation_experiment


class test_simula(unittest.TestCase):

    def setUp(self):
        pass

    def test_simulation_experiment(self):
        result_dict = simulation_experiment.mirt_simulation_experiment(
            sample_size=30, item_dimension=10, latent_dimension=2, q_type="full", stop_threshold=10, person_method="BFGS")


if __name__ == '__main__':
    unittest.main()
