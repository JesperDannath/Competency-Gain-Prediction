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


def experiment_performance(estimated_parameter_dict, real_parameter_dict, early=True, latent_dimension=-1):
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
            # TODO: Evtl. die Hauptdiagonale ausschließen!
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
                      "estimated_early_parameters": estimated_early_parameters}
    return(parameter_dict)


def create_performance_dict(parameter_dict, run_dict, sample=None, baselines=None, early_model=None, late_model=None):
    result_dict = copy.deepcopy(parameter_dict)
    #result_dict["sample"] = sample
    latent_dimension = parameter_dict["latent_dimension"]
    # Model Performance
    # Early
    result_dict["early_performance"] = {}
    result_dict["early_performance"]["rmse"] = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
                                                                      estimated_parameter_dict=result_dict[
                                                                          "estimated_early_parameters"],
                                                                      latent_dimension=latent_dimension)
    # Late
    result_dict["late_performance"] = {}
    result_dict["late_performance"]["rmse"] = experiment_performance(real_parameter_dict=result_dict["real_late_parameters"],
                                                                     estimated_parameter_dict=result_dict[
                                                                         "estimated_late_parameters"],
                                                                     early=False, latent_dimension=latent_dimension)

    # # Baseline's Performance
    # baselines["early"]["initial"]["performance"] = {"rmse": experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
    #                                                                                estimated_parameter_dict=baselines["early"][
    #     "initial"]["parameters"],
    #     latent_dimension=latent_dimension)}
    # baselines["late"]["initial"]["performance"] = {"rmse": experiment_performance(real_parameter_dict=result_dict["real_late_parameters"],
    #                                                                               estimated_parameter_dict=baselines["late"][
    #     "initial"]["parameters"],
    #     latent_dimension=latent_dimension, early=False)}
    # if "girth" in baselines["early"].keys():
    #     girth_data = result_dict["sample"]["early_responses"].to_numpy(
    #     ).transpose()
    #     if early_model.latent_dimension == 1:
    #         girth_estimates = twopl_mml(girth_data)
    #     else:
    #         girth_estimates = multidimensional_twopl_mml(
    #             girth_data, n_factors=early_model.latent_dimension)
    #     girth_item_parameters = {
    #         "discrimination_matrix": girth_estimates["Discrimination"], "intercept_vector": girth_estimates["Difficulty"]}
    #     girth_parameters = {"item_parameters": girth_item_parameters}
    #     girth_estimated_covariance = np.cov(
    #         girth_estimates["Ability"], rowvar=True)
    #     girth_parameters["person_parameters"] = {
    #         "covariance": girth_estimated_covariance}
    #     girth_parameters = standardize_parameters(girth_parameters)
    #     girth_performance_rmse = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
    #                                                     estimated_parameter_dict=girth_parameters, latent_dimension=latent_dimension)
    #     baselines["early"]["girth"]["parameters"] = girth_parameters
    #     baselines["early"]["girth"]["performance"] = {
    #         "rmse": girth_performance_rmse}

    # if "early_direct" in baselines["early"].keys():
    #     dir_model = mirt_2pl(item_dimension=early_model.item_dimension,
    #                          latent_dimension=early_model.latent_dimension, Q=early_model.item_parameters["q_matrix"])
    #     direct_early_item_parameters = direct_marginal_optimization(
    #         dir_model, response_data=sample["early_responses"])
    #     direct_early_performance_rmse = experiment_performance(real_parameter_dict=result_dict["real_early_parameters"],
    #                                                            estimated_parameter_dict=direct_early_item_parameters, latent_dimension=latent_dimension)
    #     baselines["early"]["early_direct"]["parameters"] = direct_early_item_parameters
    #     baselines["early"]["early_direct"]["performance"] = {
    #         "rmse": direct_early_performance_rmse}

    # result_dict["baselines"] = baselines
    # Marginal Likelihood
    early_likelihood = calculate_marginal_likelihoods(model=early_model, data=[sample["early_responses"]], real_parameters=result_dict["real_early_parameters"],
                                                      estimated_parameters=result_dict["estimated_early_parameters"])
    result_dict["early_performance"]["early_marginal_likelihood"] = early_likelihood
    late_data = [sample["late_responses"],
                 parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"]]
    late_likelihood = calculate_marginal_likelihoods(model=late_model, data=late_data, real_parameters=result_dict["real_late_parameters"],
                                                     estimated_parameters=result_dict["estimated_late_parameters"])
    result_dict["late_performance"]["late_marginal_likelihood"] = late_likelihood
    # Individual Level
    theta_pred = parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"].to_numpy(
    )
    theta_real = sample["latent_trait"]
    rmse_theta = rmse(theta_pred, theta_real)
    s_pred = parameter_dict["estimated_late_parameters"]["person_parameters"]["s"].to_numpy(
    )
    s_real = sample["latent_gain"]
    rmse_s = rmse(s_pred, s_real)
    result_dict["early_performance"]["individual"] = {}
    result_dict["late_performance"]["individual"] = {}
    result_dict["early_performance"]["individual"]["rmse"] = rmse_theta
    result_dict["late_performance"]["individual"]["rmse"] = rmse_s
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


def fit_early_model(sample, parameter_dict, stop_threshold, person_method, sigma_constraint):
    # Fit Parameters from Early_Model
    # Initialize model
    latent_dimension = parameter_dict["latent_dimension"]
    item_dimension = parameter_dict["item_dimension"]
    early_model = models.mirt_2pl(latent_dimension=latent_dimension, item_dimension=item_dimension,
                                  Q=parameter_dict["real_early_parameters"]["item_parameters"]["q_matrix"])
    print("Covariance matrix is good: {0}".format(
        early_model.check_sigma(parameter_dict["real_early_parameters"]["person_parameters"]["covariance"])))
    early_model.initialize_from_responses(
        response_data=sample["early_responses"])
    early_initial_parameters = early_model.get_parameters()
    e_step = em_algorithm.e_step_ga_mml(model=early_model)
    m_step = em_algorithm.m_step_ga_mml(early_model)
    em = em_algorithm.em_algo(e_step=e_step, m_step=m_step, model=early_model)

    # Fit early Model
    start_time = time.time()
    em.fit([sample["early_responses"]], max_iter=100,
           stop_threshold=stop_threshold, person_method=person_method)
    run_time = (time.time() - start_time)
    theta_hat = early_model.predict_competency(sample["early_responses"])

    # Measure Performance
    early_estimated_item_parameters = em.model.item_parameters
    early_estimated_person_parameters = em.model.person_parameters
    early_estimated_person_parameters.update(
        {"theta": pd.DataFrame(theta_hat)})

    early_estimated_parameters = em.model.get_parameters()

    run_dict = {"runtime": run_time,
                "number_steps": em.n_steps}

    return(early_estimated_parameters, early_initial_parameters, run_dict, early_model)


def fit_late_model(sample, parameter_dict, stop_threshold, person_method, sigma_constraint, real_theta=False):
    # Fit late Model
    estimated_early_sigma = parameter_dict["estimated_early_parameters"]["person_parameters"]["covariance"]
    item_dimension = parameter_dict["item_dimension"]
    latent_dimension = parameter_dict["latent_dimension"]

    # Initialize Model
    late_model = mirt_2pl_gain(item_dimension=item_dimension, latent_dimension=latent_dimension, mu=1,
                               early_sigma=estimated_early_sigma, Q=parameter_dict["real_late_parameters"]["item_parameters"]["q_matrix"])
    # TODO: Check if theta_hat can be used
    late_model.initialize_from_responses(
        response_data=sample["late_responses"], sigma=False)
    late_initial_parameters = late_model.get_parameters()
    e_step = em_algorithm.e_step_ga_mml_gain(
        model=late_model)
    m_step = em_algorithm.m_step_ga_mml_gain(
        late_model, sigma_constraint=sigma_constraint)
    em = em_algorithm.em_algo(e_step=e_step, m_step=m_step, model=late_model)

    # Decide on Competency
    if real_theta:
        theta = parameter_dict["sample"]["theta"]
    else:
        theta = parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"]

    # Fit late model
    start_time = time.time()
    em.fit([sample["late_responses"], theta], max_iter=100,
           stop_threshold=stop_threshold, person_method=person_method)
    run_time = (time.time() - start_time)
    s_hat = late_model.predict_gain(
        sample["late_responses"], parameter_dict["estimated_early_parameters"]["person_parameters"]["theta"])  # TODO: implement this

    # Measure Performance
    late_estimated_item_parameters = em.model.item_parameters
    late_estimated_person_parameters = em.model.person_parameters
    late_estimated_person_parameters.update({"s": pd.DataFrame(s_hat)})

    # Baselines
    run_dict = {"runtime": run_time,
                "number_steps": em.n_steps}

    late_estimated_parameters = em.model.get_parameters()
    return(late_estimated_parameters, late_initial_parameters, run_dict, late_model)


def mirt_simulation_experiment(sample_size, item_dimension=20, latent_dimension=3,
                               q_type="seperated", methods=["late_em", "initial", "difference"], stop_threshold=0.2,
                               ensure_id=False, q_share=0.0, person_method="newton_raphson",
                               sigma_constraint="early_constraint", real_theta=False) -> dict:
    # Simulate Responses
    simulation = item_response_simulation(
        item_dimension=item_dimension, latent_dimension=latent_dimension)
    parameter_dict = simulation.set_up(
        q_structure=q_type, q_share=q_share, ensure_id=ensure_id)
    sample = simulation.sample(sample_size=sample_size)

    # Define Population
    real_latent_cov = parameter_dict["real_late_parameters"]["person_parameters"]["covariance"]
    print("Real latent covariance: {0}".format(real_latent_cov))

    result_dict = {"sample": sample}
    if "late_em" in methods:
        performance_dict_le = late_em_optimization(sample=sample, parameter_dict=parameter_dict, stop_threshold=stop_threshold,
                                                   person_method=person_method, sigma_constraint=sigma_constraint, real_theta=False)
        result_dict["late_em"] = performance_dict_le
    if "initial" in methods:
        performance_dict_init = initial_params_baseline(
            sample=sample, parameter_dict=parameter_dict, sigma_constraint=sigma_constraint)
        result_dict["initial"] = performance_dict_init
    if "difference":
        performance_dict_diff = two_mirt_2pl_baseline(
            sample=sample, parameter_dict=parameter_dict, person_method=person_method, sigma_constraint=sigma_constraint, stop_threshold=stop_threshold)
        result_dict["difference"] = performance_dict_diff
    return(result_dict)


def initial_params_baseline(sample, parameter_dict, sigma_constraint):
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
    estimated_early_sigma = parameter_dict["estimated_early_parameters"]["person_parameters"]["covariance"]
    item_dimension = parameter_dict["item_dimension"]
    latent_dimension = parameter_dict["latent_dimension"]

    # Initialize Model
    late_model = mirt_2pl_gain(item_dimension=item_dimension, latent_dimension=latent_dimension, mu=1,
                               early_sigma=estimated_early_sigma)
    # TODO: Check if theta_hat can be used
    late_model.initialize_from_responses(
        response_data=sample["late_responses"], sigma=False)
    late_initial_parameters = late_model.get_parameters()
    s_hat = late_model.predict_gain(
        sample["late_responses"], pd.DataFrame(theta_hat))
    late_initial_parameters["person_parameters"].update(
        {"s": pd.DataFrame(s_hat)})

    parameter_dict.update(
        {"estimated_early_parameters": early_initial_parameters})
    parameter_dict.update(
        {"estimated_late_parameters": late_initial_parameters})
    run_dict = {"early": {"runtime": 0, "number_steps": 0},
                "late": {"runtime": 0, "number_steps": 0}}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def late_em_optimization(sample, parameter_dict, stop_threshold, person_method, sigma_constraint, real_theta=False):
    # Fit Parameters from Early_Model
    early_estimated_parameters, early_initial_parameters, early_run, early_model = fit_early_model(
        parameter_dict=parameter_dict, sample=sample, stop_threshold=stop_threshold, person_method=person_method, sigma_constraint=sigma_constraint)
    parameter_dict.update(
        {"estimated_early_parameters": early_estimated_parameters})
    # Fit late Model
    late_estimated_parameters, late_initial_parameters, late_run, late_model = fit_late_model(
        parameter_dict=parameter_dict, sample=sample, stop_threshold=stop_threshold,
        person_method=person_method, sigma_constraint=sigma_constraint, real_theta=real_theta)
    parameter_dict.update(
        {"estimated_late_parameters": late_estimated_parameters})
    run_dict = {"early": early_run, "late": late_run}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=late_model)
    return(performance_dict)


def two_mirt_2pl_baseline(sample, parameter_dict, stop_threshold, person_method, sigma_constraint):
    # Estimate early parameters with standard-procedure
    early_estimated_parameters, early_initial_parameters, early_run, early_model = fit_early_model(sample=sample,
                                                                                                   parameter_dict=parameter_dict, stop_threshold=stop_threshold,
                                                                                                   person_method=person_method, sigma_constraint=sigma_constraint)
    early_theta_hat = early_model.predict_competency(sample["early_responses"])
    parameter_dict.update(
        {"estimated_early_parameters": early_estimated_parameters})
    early_estimated_parameters["person_parameters"].update(
        {"theta": pd.DataFrame(early_theta_hat)})
    changed_parameter_dict = copy.deepcopy(parameter_dict)
    changed_parameter_dict["real_early_parameters"]["item_parameters"][
        "q_matrix"] = parameter_dict["real_late_parameters"]["item_parameters"]["q_matrix"]
    changed_sample = copy.deepcopy(sample)
    changed_sample["early_responses"] = sample["late_responses"]
    late_estimated_parameters, late_initial_parameters, late_run, late_model = fit_early_model(sample=changed_sample,
                                                                                               parameter_dict=changed_parameter_dict, stop_threshold=stop_threshold,
                                                                                               person_method=person_method, sigma_constraint=sigma_constraint)
    late_theta_hat = late_model.predict_competency(sample["late_responses"])
    s_hat = late_theta_hat - early_theta_hat
    s_hat = s_hat/np.std(s_hat, axis=0)
    D = parameter_dict["latent_dimension"]
    psi = pd.DataFrame(np.concatenate(
        (early_theta_hat, s_hat), axis=1)).corr().to_numpy()[0:D, D:2*D]
    late_sigma = pd.DataFrame(s_hat).corr().to_numpy()
    sigma_psi = np.identity(2*D)
    sigma_psi[0:D, 0:D] = early_estimated_parameters["person_parameters"]["covariance"]
    sigma_psi[0:D, D:2*D] = psi
    sigma_psi[D:2*D, 0:D] = np.transpose(psi)
    sigma_psi[D:2*D, D:2*D] = late_sigma
    late_estimated_parameters["person_parameters"]["covariance"] = sigma_psi
    late_estimated_parameters["person_parameters"].update(
        {"s": pd.DataFrame(s_hat)})
    ######################
    item_dimension = parameter_dict["item_dimension"]
    actual_late_model = mirt_2pl_gain(item_dimension=item_dimension, latent_dimension=D, mu=1,
                                      early_sigma=early_estimated_parameters["person_parameters"]["covariance"],
                                      Q=parameter_dict["real_late_parameters"]["item_parameters"]["q_matrix"])
    try:
        actual_late_model.set_parameters(late_estimated_parameters)
    except Exception:
        late_estimated_parameters["person_parameters"]["covariance"] = pd.DataFrame(np.concatenate(
            (early_theta_hat, s_hat), axis=1)).corr().to_numpy()
    parameter_dict.update(
        {"estimated_late_parameters": late_estimated_parameters})
    run_dict = {"early": early_run, "late": late_run}
    performance_dict = create_performance_dict(
        parameter_dict=parameter_dict, run_dict=run_dict, sample=sample, early_model=early_model, late_model=actual_late_model)
    return(performance_dict)


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
