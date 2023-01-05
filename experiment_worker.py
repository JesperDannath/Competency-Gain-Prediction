import numpy as np
from simulation_framework.item_response_simulation import item_response_simulation
from simulation_framework.simulate_competency import respondent_population
from simulation_framework.simulate_responses import response_simulation
import pandas as pd
from simulation_experiment import mirt_simulation_experiment
from analyse_experiment import get_result_df
import datetime
from itertools import product


def repeat_mirt_experiment(experiment, repetitions: int, sample_sizes: list, item_dims: list, latent_dims: list, file: str=""):
    """Repeat the specified Competency Gain simulation experiment for a specified number of times with varying input parameters.
        Save the results in a .csv file. 

    Args:
        experiment (func): Simulation function, should return a result dictionary.
        repetitions (int): Number of reps
        sample_sizes (list, optional): Sample sizes to evaluate
        item_dims (list): Item dimensions to evaluate
        latent_dims (list): Latent dimensions to evaluate
        file (str, optional): File path for .csv storage. Can append to existing files. 
    """
    error_occured = False
    number_errors = 0
    errors = []
    multiple_result_dict = {}
    variant_list = list(product(sample_sizes, item_dims, latent_dims))
    if file != "":
        try:
            prior_results_df = pd.read_csv(file)
        except FileNotFoundError:
            prior_results_df = pd.DataFrame()
    else:
        prior_results_df = pd.DataFrame()
    experiment_df = prior_results_df.copy()
    for i in range(0, repetitions):
        for variant in variant_list:
            sample_size = variant[0]
            item_dimension = variant[1]
            latent_dimension = variant[2]
            if experiment_df.size > 0:
                prior_variant_results = experiment_df[(experiment_df["latent_dimension"] == latent_dimension) &
                                                      (experiment_df["item_dimension"] == item_dimension) &
                                                      (experiment_df["sample_size"] == sample_size)]
                if len(prior_variant_results["key"].unique()) < repetitions:
                    run_variant = True
                else:
                    run_variant = False
            else:
                run_variant = True
            if run_variant:
                key = datetime.datetime.now()
                try:
                    multiple_result_dict[key] = experiment(sample_size=sample_size,
                                                           item_dimension=item_dimension,
                                                           latent_dimension=latent_dimension)
                except Exception as e:
                    print("Exception occured")
                    print(e)
                    error_occured = True
                    number_errors += 1
                    errors.append(e)
                    continue
                variant_df = get_result_df({key: multiple_result_dict[key]})
                experiment_df = pd.concat((experiment_df, variant_df), axis=0)
                if file != "":
                    experiment_df.reset_index(drop=True)
                    experiment_df.to_csv(file, index=False)
    return(experiment_df, errors)


if __name__ == "__main__":
    q_type = username = input("Enter Q_type (full, pyramid, seperated):")

    def experiment(sample_size, item_dimension, latent_dimension): return mirt_simulation_experiment(
        sample_size=sample_size, item_dimension=item_dimension, latent_dimension=latent_dimension,
        q_type=q_type, stop_threshold=0.01,
        early_person_method="BFGS",
        late_person_method="BFGS",
        sigma_constraint="early_constraint",
        methods=["real_early", "pure_competency", "initial",
                 "late_em", "difference", "real_parameters"],
        gain_mean=1.5)
    #Settings that were used previously, may be changed for future simulation studies. 
    result_df, errors = repeat_mirt_experiment(experiment, repetitions=101,
                                               sample_sizes=[30, 100, 200], latent_dims=[2, 3],
                                               item_dims=[10, 20, 30], file="results/{0}_q.csv".format(q_type))
