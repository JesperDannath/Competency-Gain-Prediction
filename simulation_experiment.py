from unittest import result
import copy
import models
from models.mirt_2pl import mirt_2pl
import numpy as np
from models.mirt_2pl_gain import mirt_2pl_gain
from simulation_framework.item_response_simulation import item_response_simulation
from simulation_framework.simulate_competency import respondent_population
from simulation_framework.simulate_responses import response_simulation
from scipy.stats import multivariate_normal
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


def rmse(y_pred: np.array, y_true: np.array) -> float:
    MSE = np.square(np.subtract(y_pred.flatten(), y_true.flatten())).mean()
    RMSE = np.sqrt(MSE)
    return(float(RMSE))


def experiment_performance(estimated_parameter_dict={}, real_parameter_dict={}, early=True, latent_dimension=-1, empty=False):
    if empty:
        if early:
            return({"rmse_A": np.nan, "rmse_delta": np.nan, "rmse_sigma": np.nan})
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
        if early:
            sigma_pred = estimated_parameter_dict["person_parameters"]["covariance"]
            sigma_true = real_parameter_dict["person_parameters"]["covariance"]

            print("Absolute diff in sigma:")
            print(np.abs(sigma_true-sigma_pred))
            res_dict["rmse_sigma"] = rmse(sigma_true, sigma_pred)
        else:
            D = latent_dimension
            psi_true = real_parameter_dict["person_parameters"]["covariance"][0:D, D:2*D]
            psi_pred = estimated_parameter_dict["person_parameters"]["covariance"][0:D, D:2*D]
            res_dict["rmse_psi"] = rmse(psi_true, psi_pred)
            sigma_true = real_parameter_dict["person_parameters"]["covariance"][D:2*D, D:2*D]
            sigma_pred = estimated_parameter_dict["person_parameters"]["covariance"][D:2*D, D:2*D]
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
                      "estimated_early_parameters": estimated_early_parameters,
                      "estimated_late_parameters": estimated_late_parameters,
                      "real_late_parameters": real_late_parameters}
    return(parameter_dict)


def create_empty_model_parameter_dict():
    person_parameters = {"covariance": np.nan}
    item_parameters = {"discrimination_matrix": np.nan,
                       "intercept_vector": np.nan}
    parameters = {"item_parameters": item_parameters,
                  "person_parameters": person_parameters}
    return(parameters)


def create_performance_dict(parameter_dict, run_dict, sample=None, early_model=mirt_2pl, late_model=None):
    result_dict = copy.deepcopy(parameter_dict)
    #result_dict["sample"] = sample
    latent_dimension = parameter_dict["latent_dimension"]
    # Model Performance
    # Early
    result_dict["early_performance"] = {}
    if early_model.type != "empty":
        result_dict["early_performance"]["rmse"] = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
                                                                          estimated_parameter_dict=result_dict[
            "estimated_early_parameters"],
            latent_dimension=latent_dimension)
    else:
        result_dict["early_performance"]["rmse"] = experiment_performance(early=True,
                                                                          empty=True)
    # Late
    result_dict["late_performance"] = {}
    result_dict["late_performance"]["rmse"] = experiment_performance(real_parameter_dict=result_dict["real_late_parameters"],
                                                                     estimated_parameter_dict=result_dict[
                                                                         "estimated_late_parameters"],
                                                                     early=False, latent_dimension=latent_dimension)
    # Marginal Likelihood
    if early_model.type != "empty":
        early_likelihood = calculate_marginal_likelihoods(model=early_model, data=[sample["early_responses"]], real_parameters=result_dict["real_early_parameters"],
                                                          estimated_parameters=result_dict["estimated_early_parameters"])
        result_dict["early_performance"]["early_marginal_likelihood"] = early_likelihood
    else:
        result_dict["early_performance"]["early_marginal_likelihood"] = {
            "optimal": np.nan, "estimated": np.nan}
    if late_model.type != "empty":
        late_data = [sample["late_responses"],
                     parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"]]
        late_likelihood = calculate_marginal_likelihoods(model=late_model, data=late_data, real_parameters=result_dict["real_late_parameters"],
                                                         estimated_parameters=result_dict["estimated_late_parameters"])
        result_dict["late_performance"]["late_marginal_likelihood"] = late_likelihood
    else:
        result_dict["late_performance"]["late_marginal_likelihood"] = {
            "optimal": np.nan, "estimated": np.nan}
    # Individual Level
    theta_pred = parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"].to_numpy(
    )
    theta_real = sample["latent_trait"]
    rmse_theta = rmse(theta_pred, theta_real)
    s_hat = parameter_dict["estimated_late_parameters"]["person_parameters"]["s_estimated"].to_numpy(
    )
    s_pred_train = parameter_dict["estimated_late_parameters"]["person_parameters"]["s_pred_train"].to_numpy(
    )
    s_real = sample["latent_gain"]
    rmse_s_hat = rmse(s_hat, s_real)
    rmse_s_pred_train = rmse(s_pred_train, s_real)
    result_dict["early_performance"]["individual"] = {}
    result_dict["late_performance"]["individual"] = {}
    result_dict["early_performance"]["individual"]["rmse"] = rmse_theta
    result_dict["late_performance"]["individual"]["rmse_estimated"] = rmse_s_hat
    result_dict["late_performance"]["individual"]["rmse_pred_train"] = rmse_s_pred_train
    # Runtime
    result_dict["early_performance"]["run"] = run_dict["early"]
    result_dict["late_performance"]["run"] = run_dict["late"]
    return(result_dict)


def calculate_marginal_likelihoods(model, data: list, real_parameters, estimated_parameters):
    # model.set_parameters(initial_parameters)
    # initial_marginal_likelihood = model.marginal_response_loglikelihood(
    #     *data)
    model.set_parameters(real_parameters)
    optimal_marginal_likelihood = model.marginal_response_loglikelihood(
        *data)
    model.set_parameters(estimated_parameters)
    marginal_likelihood_estimated = model.marginal_response_loglikelihood(
        *data)
    likelihood_dict = {"optimal": optimal_marginal_likelihood,
                       "estimated": marginal_likelihood_estimated}
    return(likelihood_dict)


def fit_early_model(sample, parameter_dict, stop_threshold, person_method, sigma_constraint="unconstrained"):
    # Fit Parameters from Early_Model
    # Initialize model
    latent_dimension = parameter_dict["latent_dimension"]
    item_dimension = parameter_dict["item_dimension"]
    if sigma_constraint == "identity":
        init_sigma = False
    else:
        init_sigma = True
    early_model = models.mirt_2pl(latent_dimension=latent_dimension, item_dimension=item_dimension,
                                  Q=parameter_dict["real_early_parameters"]["item_parameters"]["q_matrix"],
                                  sigma=np.identity(latent_dimension))
    print("Covariance matrix is good: {0}".format(
        early_model.check_sigma(parameter_dict["real_early_parameters"]["person_parameters"]["covariance"])))
    early_model.initialize_from_responses(
        response_data=sample["early_responses"], sigma=init_sigma)
    early_initial_parameters = early_model.get_parameters()
    e_step = em_algorithm.e_step_ga_mml(model=early_model)
    m_step = em_algorithm.m_step_ga_mml(
        early_model, sigma_constraint=sigma_constraint)
    em = em_algorithm.em_algo(e_step=e_step, m_step=m_step, model=early_model)

    # Fit early Model
    start_time = time.time()
    em.fit([sample["early_responses"]], max_iter=30,
           stop_threshold=stop_threshold, person_method=person_method)
    run_time = (time.time() - start_time)
    theta_hat = early_model.predict_competency(sample["early_responses"])

    # Measure Performance
    early_estimated_parameters = em.model.get_parameters()
    early_estimated_parameters["person_parameters"].update(
        {"theta": pd.DataFrame(theta_hat)})

    run_dict = {"runtime": run_time,
                "number_steps": em.n_steps}
    return(early_estimated_parameters, early_initial_parameters, run_dict, early_model)


def fit_late_model(sample, parameter_dict, stop_threshold, person_method,
                   sigma_constraint, real_theta=False, gain_mean=1):
    # Fit late Model
    estimated_early_sigma = parameter_dict["estimated_early_parameters"]["person_parameters"]["covariance"]
    item_dimension = parameter_dict["item_dimension"]
    latent_dimension = parameter_dict["latent_dimension"]

    # Initialize Model
    late_model = mirt_2pl_gain(item_dimension=item_dimension, latent_dimension=latent_dimension, mu=gain_mean,
                               early_sigma=estimated_early_sigma,
                               Q=parameter_dict["real_late_parameters"]["item_parameters"]["q_matrix"])
    late_model.initialize_from_responses(
        late_response_data=sample["late_responses"], early_response_data=sample["early_responses"],
        convolution_variance=sample["convolution_variance"], sigma=False)
    late_initial_parameters = late_model.get_parameters()
    e_step = em_algorithm.e_step_ga_mml_gain(
        model=late_model)
    m_step = em_algorithm.m_step_ga_mml_gain(
        late_model, sigma_constraint=sigma_constraint)
    em = em_algorithm.em_algo(e_step=e_step, m_step=m_step, model=late_model)

    # Decide on Competency
    if real_theta:
        theta = pd.DataFrame(sample["latent_trait"])
    else:
        theta = parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"]

    # Fit late model
    start_time = time.time()
    em.fit([sample["late_responses"], theta], max_iter=30,
           stop_threshold=stop_threshold, person_method=person_method)
    run_time = (time.time() - start_time)
    s_hat = late_model.predict_gain(
        sample["late_responses"], theta)

    s_pred_train = late_model.predict_gain(
        theta=theta)
    # Measure Performance
    late_estimated_parameters = em.model.get_parameters()
    late_estimated_parameters["person_parameters"].update(
        {"s_estimated": pd.DataFrame(s_hat), "s_pred_train": pd.DataFrame(s_pred_train)})

    # Baselines
    run_dict = {"runtime": run_time,
                "number_steps": em.n_steps}

    return(late_estimated_parameters, late_initial_parameters, run_dict, late_model)


def mirt_simulation_experiment(sample_size, item_dimension=20, latent_dimension=3,
                               q_type="seperated",
                               methods=["late_em", "initial", "difference",
                                        "real_early", "real_parameters"],
                               stop_threshold=2,
                               ensure_id=False, q_share=0.0, early_person_method="newton_raphson", late_person_method="ga",
                               sigma_constraint="early_constraint",
                               gain_mean=1) -> dict:
    """Simulation Experiment that uses a number of specified baselines and has various options for differentiating. 

    Args:
        sample_size (int): Sample Size
        q_type (str, optional): Type of Q-matrix that is used Defaults to "seperated".
        methods (list, optional): List of parameter estimation methods. Defaults to ["late_em", "initial", "difference", "real_early", "real_parameters"].
        stop_threshold (int, optional): Threshold for EM-stopping criterion. Defaults to 2.
        ensure_id (bool, optional): Ensure that Q-matrix is identifiable (especially pyramid Q-matrix). Defaults to False.
        q_share (float, optional): How strong to orinet the real discriminations on the Q-matrix. Defaults to 0.0.
        early_person_method (str, optional): Optimization method in M-Step for early EM person parameters. Defaults to "newton_raphson".
        late_person_method (str, optional):  Optimization method in M-Step for late EM person parameters. Defaults to "ga".
        sigma_constraint (str, optional): Type of covariance matrix used. Defaults to "early_constraint".
        gain_mean (int, optional): Mean competency gain of the simulation. Defaults to 1.

    Returns:
        dict: _description_
    """
    # Simulate Responses
    if sigma_constraint == "esigma_spsi":
        sigma_type = sigma_constraint
    elif sigma_constraint == "early_constraint":
        sigma_type = "unnormal_late"
    else:
        sigma_type = "none"
    simulation = item_response_simulation(
        item_dimension=item_dimension, latent_dimension=latent_dimension)
    parameter_dict = simulation.set_up(
        q_structure=q_type, q_share=q_share, ensure_id=ensure_id, sigma_constraint=sigma_type, gain_mean=gain_mean)
    sample = simulation.sample(sample_size=sample_size)

    # Define Population
    real_latent_cov = parameter_dict["real_late_parameters"]["person_parameters"]["covariance"]
    print("Real latent covariance: {0}".format(real_latent_cov))

    result_dict = {"sample": sample}
    if "late_em" in methods:
        print("Start late EM")
        performance_dict_le = late_em_optimization(sample=copy.deepcopy(sample), parameter_dict=copy.deepcopy(parameter_dict), stop_threshold=stop_threshold,
                                                   early_person_method=early_person_method, late_person_method=late_person_method,
                                                   sigma_constraint=sigma_constraint, real_theta=False, gain_mean=np.ones(latent_dimension)*gain_mean)
        result_dict["late_em"] = performance_dict_le
    if "initial" in methods:
        print("Start initial baseline")
        performance_dict_init = initial_params_baseline(
            sample=copy.deepcopy(sample), parameter_dict=copy.deepcopy(parameter_dict),
            sigma_constraint=sigma_constraint, gain_mean=np.ones(latent_dimension)*gain_mean)
        result_dict["initial"] = performance_dict_init
    if "difference" in methods:
        print("Start difference baseline")
        performance_dict_diff = two_mirt_2pl_baseline(
            sample=copy.deepcopy(sample), parameter_dict=copy.deepcopy(parameter_dict), early_person_method=early_person_method,
            sigma_constraint=sigma_constraint, stop_threshold=stop_threshold, gain_mean=np.ones(latent_dimension)*gain_mean)
        result_dict["difference"] = performance_dict_diff
    if "real_early" in methods:
        print("Start real early baseline")
        performance_dict_re = real_early_params_baseline(
            sample=copy.deepcopy(sample), parameter_dict=copy.deepcopy(parameter_dict), late_person_method=late_person_method,
            sigma_constraint=sigma_constraint, stop_threshold=stop_threshold, gain_mean=np.ones(latent_dimension)*gain_mean)
        result_dict["real_early"] = performance_dict_re
    if "pure_competency" in methods:
        print("Start pure competency baseline")
        performance_dict_pc = pure_competency_baseline(
            sample=copy.deepcopy(sample), parameter_dict=copy.deepcopy(parameter_dict), late_person_method=late_person_method,
            early_person_method=early_person_method,
            sigma_constraint=sigma_constraint, stop_threshold=stop_threshold, gain_mean=np.ones(latent_dimension)*gain_mean)
        result_dict["pure_competency"] = performance_dict_pc
    if "real_parameters" in methods:
        print("Start real Parameters baseline")
        performance_dict_rp = real_parameters_baseline(
            sample=copy.deepcopy(sample), parameter_dict=copy.deepcopy(parameter_dict), late_person_method=late_person_method,
            early_person_method=early_person_method,
            sigma_constraint=sigma_constraint, stop_threshold=stop_threshold, gain_mean=np.ones(latent_dimension)*gain_mean)
        result_dict["real_parameters"] = performance_dict_rp
    return(result_dict)


def pure_competency_baseline(sample, parameter_dict, early_person_method, late_person_method,
                             sigma_constraint, stop_threshold, gain_mean):
    """
    Pure Competency Baseline that restrics the covariance to not allow inter-competency correlations.
    """
    # Fit Parameters from Early_Model
    early_estimated_parameters, early_initial_parameters, early_run, early_model = fit_early_model(
        parameter_dict=copy.deepcopy(parameter_dict), sample=sample, stop_threshold=stop_threshold, person_method=early_person_method, sigma_constraint="identity")
    parameter_dict.update(
        {"estimated_early_parameters": early_estimated_parameters})
    # Fit late Model
    late_estimated_parameters, late_initial_parameters, late_run, late_model = fit_late_model(
        parameter_dict=copy.deepcopy(parameter_dict), sample=sample, stop_threshold=stop_threshold,
        person_method="ga", sigma_constraint="diagonal", real_theta=False, gain_mean=gain_mean)
    parameter_dict.update(
        {"estimated_late_parameters": late_estimated_parameters})
    run_dict = {"early": early_run, "late": late_run}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def initial_params_baseline(sample, parameter_dict, sigma_constraint, gain_mean):
    """Initial Parameters as baseline.
    """
    # Initialize model
    latent_dimension = parameter_dict["latent_dimension"]
    item_dimension = parameter_dict["item_dimension"]
    early_model = models.mirt_2pl(latent_dimension=latent_dimension, item_dimension=item_dimension,
                                  Q=parameter_dict["real_early_parameters"]["item_parameters"]["q_matrix"])
    early_model.initialize_from_responses(
        response_data=sample["early_responses"])
    early_initial_parameters = early_model.get_parameters()
    theta_hat = early_model.predict_competency(sample["early_responses"])
    early_initial_parameters["person_parameters"].update(
        {"theta": pd.DataFrame(theta_hat)})
    # Late Model
    # parameter_dict["estimated_early_parameters"]["person_parameters"]["covariance"]
    estimated_early_sigma = early_model.person_parameters["covariance"]
    item_dimension = parameter_dict["item_dimension"]
    latent_dimension = parameter_dict["latent_dimension"]

    # Initialize Model
    late_model = mirt_2pl_gain(item_dimension=item_dimension, latent_dimension=latent_dimension, mu=gain_mean,
                               early_sigma=estimated_early_sigma)
    late_model.initialize_from_responses(
        late_response_data=sample["late_responses"], early_response_data=sample["early_responses"],
        convolution_variance=sample["convolution_variance"],
        sigma=False)
    late_initial_parameters = late_model.get_parameters()
    s_hat = late_model.predict_gain(
        sample["late_responses"], pd.DataFrame(theta_hat))
    s_pred_train = late_model.predict_gain(theta=pd.DataFrame(theta_hat))
    late_initial_parameters["person_parameters"].update(
        {"s_estimated": pd.DataFrame(s_hat), "s_pred_train": pd.DataFrame(s_pred_train)})

    parameter_dict.update(
        {"estimated_early_parameters": early_initial_parameters})
    parameter_dict.update(
        {"estimated_late_parameters": late_initial_parameters})
    run_dict = {"early": {"runtime": 0, "number_steps": 0},
                "late": {"runtime": 0, "number_steps": 0}}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def late_em_optimization(sample, parameter_dict, stop_threshold, early_person_method,
                         late_person_method, sigma_constraint, real_theta=False, gain_mean=1):
    """Late EM Method (with unrestricted Covariance.)
    """
    # Fit Parameters from Early_Model
    early_estimated_parameters, early_initial_parameters, early_run, early_model = fit_early_model(
        parameter_dict=copy.deepcopy(parameter_dict), sample=sample, stop_threshold=stop_threshold, person_method=early_person_method, sigma_constraint=sigma_constraint)
    parameter_dict.update(
        {"estimated_early_parameters": copy.deepcopy(early_estimated_parameters)})
    # Fit late Model
    late_estimated_parameters, late_initial_parameters, late_run, late_model = fit_late_model(
        parameter_dict=copy.deepcopy(parameter_dict), sample=sample, stop_threshold=stop_threshold,
        person_method=late_person_method, sigma_constraint=sigma_constraint, real_theta=real_theta, gain_mean=gain_mean)
    parameter_dict.update(
        {"estimated_late_parameters": late_estimated_parameters})
    run_dict = {"early": early_run, "late": late_run}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def two_mirt_2pl_baseline(sample, parameter_dict, stop_threshold, early_person_method, sigma_constraint, gain_mean=np.empty(0)):
    """Difference Baseline. Uses two normal MIRT-2PL models to estiate gain parameters.
    """
    # Estimate early parameters with standard-procedure
    early_estimated_parameters, early_initial_parameters, early_run, early_model = fit_early_model(sample=sample,
                                                                                                   parameter_dict=copy.deepcopy(
                                                                                                       parameter_dict),
                                                                                                   stop_threshold=stop_threshold,
                                                                                                   person_method=early_person_method, sigma_constraint=sigma_constraint)
    early_theta_hat = early_model.predict_competency(
        sample["early_responses"], strict_variance=False)
    parameter_dict.update(
        {"estimated_early_parameters": early_estimated_parameters})
    early_estimated_parameters["person_parameters"].update(
        {"theta": pd.DataFrame(early_theta_hat)})
    changed_parameter_dict = copy.deepcopy(parameter_dict)
    changed_parameter_dict["real_early_parameters"]["item_parameters"][
        "q_matrix"] = parameter_dict["real_late_parameters"]["item_parameters"]["q_matrix"]
    changed_parameter_dict["real_early_parameters"]["item_parameters"][
        "discrimination_matrix"] = parameter_dict["real_late_parameters"]["item_parameters"]["discrimination_matrix"]
    changed_sample = copy.deepcopy(sample)
    changed_sample["early_responses"] = sample["late_responses"]
    late_estimated_parameters, late_initial_parameters, late_run, late_model = fit_early_model(sample=changed_sample,
                                                                                               parameter_dict=changed_parameter_dict, stop_threshold=stop_threshold,
                                                                                               person_method=early_person_method, sigma_constraint=sigma_constraint)
    D = parameter_dict["latent_dimension"]

    # Shift Discriminations and Covariance accoring to real variance
    conv_sigma = late_model.person_parameters["covariance"]
    conv_A = late_model.item_parameters["discrimination_matrix"]
    real_conv_sigma_var = sample["convolution_variance"]
    conv_sd_vector = np.sqrt(real_conv_sigma_var)
    conv_sd_matrix = np.diag(conv_sd_vector)
    conv_sigma_scaled = np.round(np.dot(
        np.dot(conv_sd_matrix, conv_sigma), conv_sd_matrix), 5)
    conv_A_scaled = np.dot(conv_A, np.linalg.inv(conv_sd_matrix).transpose())
    scaled_parameters = {"person_parameters": {"covariance": conv_sigma_scaled},
                         "item_parameters": {"discrimination_matrix": conv_A_scaled}}
    late_model.set_parameters(scaled_parameters)

    late_theta_hat = late_model.predict_competency(
        sample["late_responses"], strict_variance=False) + gain_mean
    # Double late theta because var(late_theta) = var(early_theta + gain) = var(early_theta) + var(gain) - 2*cov(early_theta, gain)
    s_hat = late_theta_hat - early_theta_hat
    late_estimated_parameters["item_parameters"]["discrimination_matrix"] = conv_A_scaled
    late_delta = late_estimated_parameters["item_parameters"]["intercept_vector"]
    translated_delta = late_delta - \
        np.dot(late_estimated_parameters["item_parameters"]
               ["discrimination_matrix"], np.ones(D))
    late_estimated_parameters["item_parameters"]["intercept_vector"] = translated_delta
    sigma_psi = pd.DataFrame(np.concatenate(
        (early_theta_hat, s_hat), axis=1)).cov().to_numpy()
    late_estimated_parameters["person_parameters"]["covariance"] = sigma_psi

    ######################
    item_dimension = parameter_dict["item_dimension"]
    actual_late_model = mirt_2pl_gain(item_dimension=item_dimension, latent_dimension=D,
                                      mu=gain_mean,
                                      early_sigma=early_estimated_parameters["person_parameters"]["covariance"],
                                      Q=parameter_dict["real_late_parameters"]["item_parameters"]["q_matrix"])
    # try:
    actual_late_model.set_parameters(late_estimated_parameters)
    s_pred_train = actual_late_model.predict_gain(
        theta=pd.DataFrame(early_theta_hat))
    late_estimated_parameters["person_parameters"].update(
        {"s_estimated": pd.DataFrame(s_hat), "s_pred_train": pd.DataFrame(s_pred_train)})

    parameter_dict.update(
        {"estimated_late_parameters": late_estimated_parameters})
    run_dict = {"early": early_run, "late": late_run}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=actual_late_model)
    return(performance_dict)


def real_early_params_baseline(sample, parameter_dict, stop_threshold, late_person_method, sigma_constraint, gain_mean):
    """Real Early Parameters Baseline. Uses real early covariance and real individual initial competencies to estimate competency gain.
    """
    # theta = sample["latent_trait"]
    estimated_early_parameters = copy.deepcopy(
        parameter_dict["real_early_parameters"])
    #
    parameter_dict.update(
        {"estimated_early_parameters": estimated_early_parameters})
    late_estimated_parameters, late_initial_parameters, late_run, late_model = fit_late_model(sample=sample,
                                                                                              parameter_dict=copy.deepcopy(
                                                                                                  parameter_dict),
                                                                                              stop_threshold=stop_threshold,
                                                                                              person_method=late_person_method,
                                                                                              sigma_constraint=sigma_constraint,
                                                                                              real_theta=True, gain_mean=gain_mean)
    parameter_dict.update(
        {"estimated_late_parameters": late_estimated_parameters})
    # Create empty early parameters
    estimated_early_parameters = create_empty_model_parameter_dict()
    estimated_early_parameters["person_parameters"]["theta"] = pd.DataFrame(
        sample["latent_trait"])
    parameter_dict.update(
        {"estimated_early_parameters": estimated_early_parameters})
    early_run = {"runtime": np.nan, "number_steps": np.nan}
    run_dict = {"early": early_run, "late": late_run}
    early_model = mirt_2pl(
        item_dimension=parameter_dict["item_dimension"], latent_dimension=parameter_dict["item_dimension"], empty=True)
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def real_parameters_baseline(sample, parameter_dict, late_person_method, early_person_method,
                             sigma_constraint, stop_threshold, gain_mean):
    """Real Parameters Baseline. Uses the real model parameters to estimate the individual latent traits. 
    """
    estimated_early_parameters = copy.deepcopy(
        parameter_dict["real_early_parameters"])
    estimated_late_parameters = copy.deepcopy(
        parameter_dict["real_late_parameters"])
    early_model = mirt_2pl(
        item_dimension=parameter_dict["item_dimension"], latent_dimension=parameter_dict["latent_dimension"])
    early_model.set_parameters(estimated_early_parameters)
    late_model = mirt_2pl_gain(item_dimension=parameter_dict["item_dimension"],
                               latent_dimension=parameter_dict["latent_dimension"], mu=gain_mean)
    late_model.set_parameters(estimated_late_parameters)
    # Fit Competency and Gain
    theta_hat = early_model.predict_competency(sample["early_responses"])
    s_estimated = late_model.predict_gain(
        response_data=sample["late_responses"], theta=pd.DataFrame(sample["latent_trait"]))
    s_pred_train = late_model.predict_gain(
        theta=pd.DataFrame(sample["latent_trait"]))
    estimated_early_parameters["person_parameters"].update(
        {"theta": pd.DataFrame(theta_hat)})
    estimated_late_parameters["person_parameters"].update(
        {"s_estimated": pd.DataFrame(s_estimated), "s_pred_train": pd.DataFrame(s_pred_train)})
    parameter_dict.update(
        {"estimated_early_parameters": estimated_early_parameters})
    parameter_dict.update(
        {"estimated_late_parameters": estimated_late_parameters})
    early_run = {"runtime": np.nan, "number_steps": np.nan}
    late_run = {"runtime": np.nan, "number_steps": np.nan}
    run_dict = {"early": early_run, "late": late_run}
    early_model = mirt_2pl(
        item_dimension=parameter_dict["item_dimension"], latent_dimension=parameter_dict["latent_dimension"])
    early_model.set_parameters({"item_parameters": estimated_early_parameters["item_parameters"],
                                "person_parameters": estimated_early_parameters["person_parameters"]})
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def direct_marginal_optimization(model, response_data):
    """Baseline  that performs direct optimzation of the Marginal Log-Likelihood without EM-Algorithm."""
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
