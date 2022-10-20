from unittest import result
from models.mirt_2pl import mirt_2pl
import numpy as np
from simulation_framework.item_response_simulation import item_response_simulation
from simulation_framework.simulate_competency import respondent_population
from simulation_framework.simulate_responses import response_simulation
from scipy.stats import multivariate_normal
import models
import em_algorithm
import pandas as pd
import time
from girth import multidimensional_twopl_mml
from girth import twopl_mml
import cma
import scipy


def params_to_vector(model):
    item_dimension = model.item_dimension
    latent_dimension = model.latent_dimension
    A_flat = model.item_parameters["discrimination_matrix"].flatten()
    A_flat = A_flat[A_flat != 0]
    A_cut = len(A_flat)
    delta_cut = A_cut+item_dimension
    delta = model.item_parameters["intercept_vector"]
    sigma_flat = model.person_parameters["covariance"][np.triu_indices(
        model.latent_dimension, k=1)]
    return(np.concatenate((A_flat, delta, sigma_flat), axis=0))


def vector_to_params(vector, model):
    item_dimension = model.item_dimension
    latent_dimension = model.latent_dimension
    q_matrix = model.item_parameters["q_matrix"]
    q_flat = q_matrix.flatten()
    A_cut = len(q_flat[q_flat != 0])
    delta_cut = A_cut+item_dimension
    A = vector[0:A_cut]
    A = model.fill_zero_discriminations(A).reshape(
        (item_dimension, latent_dimension))
    delta = vector[A_cut:delta_cut]
    item_parameters = {"discrimination_matrix": A, "intercept_vector": delta}
    corr = vector[delta_cut: len(vector)]
    sigma = model.corr_to_sigma(corr, False)
    return({"item_parameters": item_parameters, "person_parameters": {"covariance": sigma}})


def direct_marginal_optimization(model, response_data):
    # Get initial parameters
    x0 = params_to_vector(model)
    response_data = response_data.to_numpy()
    # Optimize with GA

    def marginal_l_func(input_vector):
        parameters = vector_to_params(input_vector, model)
        try:
            model.set_parameters(parameters)
        except Exception:
            return(np.inf)
        result = -1*model.marginal_response_loglikelihood(response_data)
        return(result)
    es = cma.CMAEvolutionStrategy(x0=x0, sigma0=0.5)
    es.optimize(marginal_l_func, maxfun=100000)
    # Get result
    result = es.result.xfavorite
    return(vector_to_params(result, model))


def rmse(y_pred: np.array, y_true: np.array) -> float:
    MSE = np.square(np.subtract(y_pred.flatten(), y_true.flatten())).mean()
    RMSE = np.sqrt(MSE)
    return(float(RMSE))


def experiment_performance(estimated_parameter_dict, real_parameter_dict):
    res_dict = {}
    if "item_parameters" in estimated_parameter_dict.keys():
        A_pred = estimated_parameter_dict["item_parameters"]["discrimination_matrix"]
        delta_pred = estimated_parameter_dict["item_parameters"]["intercept_vector"]

        A_true = real_parameter_dict["item_parameters"]["discrimination_matrix"]
        delta_true = real_parameter_dict["item_parameters"]["intercept_vector"]

        print("Absolute diff in A:")
        print(np.abs(A_true-A_pred))

        print("Absolute diff in delta:")
        print(np.abs(delta_true-delta_pred))

        res_dict["rmse_A"] = rmse(A_pred, A_true)
        res_dict["rmse_delta"] = rmse(delta_true, delta_pred)

    if "person_parameters" in estimated_parameter_dict.keys():
        sigma_pred = estimated_parameter_dict["person_parameters"]["covariance"]
        sigma_true = real_parameter_dict["person_parameters"]["covariance"]

        print("Absolute diff in sigma:")
        # TODO: Evtl. die Hauptdiagonale ausschließen!
        print(np.abs(sigma_true-sigma_pred))

        res_dict["rmse_sigma"] = rmse(sigma_true, sigma_pred)
    if len(res_dict.keys()) == 0:
        raise Exception("No performance to calculate")
    return(res_dict)


def standardize_parameters(parameter_dict):
    covariance = parameter_dict["person_parameters"]["covariance"]
    A = parameter_dict["item_parameters"]["discrimination_matrix"]
    if len(covariance.shape) <= 1:
        correlation_matrix = np.array([[1]])
        inv_sqrt_cov = np.array([[np.sqrt(1/covariance)]])
        A = np.multiply(A, inv_sqrt_cov)
    else:
        sd_vector = np.sqrt(covariance.diagonal())
        inv_sd_matrix = np.linalg.inv(np.diag(sd_vector))
        correlation_matrix = np.dot(
            np.dot(inv_sd_matrix, covariance), inv_sd_matrix)
        A = np.dot(A, np.linalg.inv(inv_sd_matrix))
    parameter_dict["item_parameters"]["discrimination_matrix"] = A
    parameter_dict["person_parameters"]["covariance"] = correlation_matrix
    return(parameter_dict)


def create_parameter_dict(estimated_early_parameters, real_early_parameters, estimated_late_parameters, real_late_parameters):
    parameter_dict = {"real_early_parameters": real_early_parameters,
                      "estimated_early_parameters": estimated_early_parameters}
    return(parameter_dict)


def create_performance_dict(parameter_dict, run_dict, sample=None, baselines=None, model=None):
    result_dict = parameter_dict
    result_dict["sample"] = sample
    # Model Performance
    result_dict["early_performance"] = {}
    result_dict["early_performance"]["rmse"] = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
                                                                      estimated_parameter_dict=result_dict["estimated_early_parameters"])
    # Baseline's Performance
    baselines["early_initial"]["performance"] = {"rmse": experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
                                                                                estimated_parameter_dict=baselines["early_initial"]["parameters"])}
    if "girth" in baselines.keys():
        girth_data = result_dict["sample"]["early_responses"].to_numpy(
        ).transpose()
        if model.latent_dimension == 1:
            girth_estimates = twopl_mml(girth_data)
        else:
            girth_estimates = multidimensional_twopl_mml(
                girth_data, n_factors=model.latent_dimension)
        girth_item_parameters = {
            "discrimination_matrix": girth_estimates["Discrimination"], "intercept_vector": girth_estimates["Difficulty"]}
        girth_parameters = {"item_parameters": girth_item_parameters}
        girth_estimated_covariance = np.cov(
            girth_estimates["Ability"], rowvar=True)
        girth_parameters["person_parameters"] = {
            "covariance": girth_estimated_covariance}
        girth_parameters = standardize_parameters(girth_parameters)
        girth_performance_rmse = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
                                                        estimated_parameter_dict=girth_parameters)
        baselines["girth"]["parameters"] = girth_parameters
        baselines["girth"]["performance"] = {"rmse": girth_performance_rmse}

    if "early_direct" in baselines.keys():
        dir_model = mirt_2pl(item_dimension=model.item_dimension,
                             latent_dimension=model.latent_dimension, Q=model.item_parameters["q_matrix"])
        direct_early_item_parameters = direct_marginal_optimization(
            dir_model, response_data=sample["early_responses"])
        direct_early_performance_rmse = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
                                                               estimated_parameter_dict=direct_early_item_parameters)
        baselines["early_direct"]["parameters"] = direct_early_item_parameters
        baselines["early_direct"]["performance"] = {
            "rmse": direct_early_performance_rmse}

    result_dict["baselines"] = baselines
    likelihood = calculate_marginal_likelihoods(model=model, response_data=sample["early_responses"], real_parameters=result_dict["real_early_parameters"],
                                                initial_parameters=baselines["early_initial"]["parameters"], estimated_parameters=result_dict["estimated_early_parameters"])
    result_dict["early_performance"]["marginal_likelihood"] = likelihood
    result_dict["early_performance"]["run"] = run_dict["early"]
    return(result_dict)


def calculate_marginal_likelihoods(model, response_data, real_parameters, initial_parameters, estimated_parameters):
    model.set_parameters(initial_parameters)
    initial_marginal_likelihood = model.marginal_response_loglikelihood(
        response_data=response_data.to_numpy())
    model.set_parameters(real_parameters)
    optimal_marginal_likelihood = model.marginal_response_loglikelihood(
        response_data=response_data.to_numpy())
    model.set_parameters(estimated_parameters)
    marginal_likelihood_estimated = model.marginal_response_loglikelihood(
        response_data=response_data.to_numpy())
    likelihood_dict = {"optimal": optimal_marginal_likelihood,
                       "estimated": marginal_likelihood_estimated, "initial": initial_marginal_likelihood}
    return(likelihood_dict)


def mirt_simulation_experiment(sample_size, item_dimension=20, latent_dimension=3,
                               q_type="seperated", girth=True, stop_threshold=0.2,
                               ensure_id=False, q_share=0.0) -> dict:

    # Simulate Responses
    simulation = item_response_simulation(
        item_dimension=item_dimension, latent_dimension=latent_dimension)
    parameter_dict = simulation.set_up(
        q_structure=q_type, q_share=q_share, ensure_id=ensure_id)
    sample = simulation.sample(sample_size=sample_size)

    # Define Population
    #population = respondent_population(latent_dimension=latent_dimension)
    real_latent_cov = parameter_dict["real_early_parameters"]["person_parameters"]["covariance"]
    print("Real latent covariance: {0}".format(real_latent_cov))

    # Sample responses
    #response_simulation_obj = response_simulation(population=population, item_dimension=item_dimension)
    #response_simulation_obj.initialize_random_q_structured_matrix(structure=q_type, ensure_id=ensure_id)
    #early_item_parameters = response_simulation_obj.initialize_random_item_parameters()
    parameter_dict["real_early_parameters"].update(
        {"item_dimension": item_dimension, "latent_dimension": latent_dimension})
    # {"item_parameters": early_item_parameters, "person_parameters": {"covariance": real_latent_cov}}
    real_early_parameters = parameter_dict["real_early_parameters"]

    # Fit Parameters
    # Initialize model
    model = models.mirt_2pl(latent_dimension=latent_dimension, item_dimension=item_dimension,
                            Q=real_early_parameters["item_parameters"]["q_matrix"])
    print("Covariance matrix is good: {0}".format(
        model.check_sigma(real_latent_cov)))
    model.initialize_from_responses(response_data=sample["early_responses"])
    initial_early_parameters = model.get_parameters()
    e_step = em_algorithm.e_step_ga_mml(model=model)
    m_step = em_algorithm.m_step_ga_mml(model)
    em = em_algorithm.em_algo(e_step=e_step, m_step=m_step, model=model)

    # Fit Model
    start_time = time.time()
    em.fit(sample["early_responses"], max_iter=100,
           stop_threshold=stop_threshold)
    run_time = (time.time() - start_time)

    # Measure Performance
    early_estimated_item_parameters = em.model.item_parameters
    early_estimated_person_parameters = em.model.person_parameters

    # Create Baselines
    baselines = {"early_initial": {"parameters": initial_early_parameters}}
    if girth == True:
        baselines["girth"] = {}

    # Create results
    early_estimated_parameters = em.model.get_parameters()

    parameter_dict = create_parameter_dict(estimated_early_parameters=early_estimated_parameters,
                                           real_early_parameters=real_early_parameters,
                                           estimated_late_parameters=None, real_late_parameters=None)
    run_dict = {"early": {"runtime": run_time,
                          "number_steps": em.n_steps}}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, baselines=baselines, model=model)

    return(performance_dict)
