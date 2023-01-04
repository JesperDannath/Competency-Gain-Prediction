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
            sample_size=30, item_dimension=10, latent_dimension=2, q_type="full", 
            stop_threshold=10, early_person_method="BFGS", late_person_method="BFGS",
            methods=["real_early", "pure_competency", "initial",
                 "late_em", "difference", "real_parameters"],
            gain_mean=1.5)


if __name__ == '__main__':
    unittest.main()


# mirt_simulation_experiment(
#         sample_size=sample_size, item_dimension=item_dimension, latent_dimension=latent_dimension,
#         q_type=q_type, stop_threshold=0.01,
#         early_person_method="BFGS",
#         late_person_method="BFGS",
#         sigma_constraint="early_constraint",
#         methods=["real_early", "pure_competency", "initial",
#                  "late_em", "difference", "real_parameters"],
#         gain_mean=1.5)