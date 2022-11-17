import numpy as np
from simulation_framework.item_response_simulation import item_response_simulation
from simulation_framework.simulate_competency import respondent_population
from simulation_framework.simulate_responses import response_simulation
import models
import em_algorithm
import pandas as pd
import time
import scipy
from simulation_experiment import mirt_simulation_experiment
from analyse_experiment import get_result_df
from analyse_experiment import print_result_from_df
import datetime
from itertools import product


def repeat_mirt_experiment(experiment, repetitions, sample_sizes=[], item_dims=[], latent_dims=[], file=""):
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
    # if error_occured:
    #    pass #TODO Recursion for remaining variants
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
    result_df, errors = repeat_mirt_experiment(experiment, repetitions=30,
                                               sample_sizes=[30, 100, 200], latent_dims=[2, 3],
                                               item_dims=[10, 20, 30], file="results/{0}_q.csv".format(q_type))
