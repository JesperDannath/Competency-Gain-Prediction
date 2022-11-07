import pandas as pd
import numpy as np


def print_result_from_dict(result_dict, description=""):
    # Description
    print("------------------------------------")
    print("##### Results for {0}".format(description))
    # Experiment Properties:
    latent_dimension = result_dict["sample"]["latent_dimension"]
    item_dimension = result_dict["sample"]["item_dimension"]
    sample_size = result_dict["sample"]["sample_size"]
    print("Latent dimension: {0},  Item dimension: {1}, sample size {2} \\".format(
        latent_dimension, item_dimension, sample_size))
    # Performance/time
    ep_dict = result_dict["early_performance"]
    lp_dict = result_dict["late_performance"]
    early_runtime = np.round(ep_dict["run"]["runtime"], 2)
    early_steps = ep_dict["run"]["number_steps"]
    print("Early runtime: {0} seconds, {1} steps, {2} seconds per step \\".format(
        early_runtime, early_steps, np.round(early_runtime/early_steps, 2)))
    late_runtime = np.round(lp_dict["run"]["runtime"], 2)
    late_steps = lp_dict["run"]["number_steps"]
    print("Late runtime: {0} seconds, {1} steps, {2} seconds per step \\".format(
        late_runtime, late_steps, np.round(late_runtime/late_steps, 2)))
    # Performance/results
    # Early Likelihood
    early_l_optimal = np.round(
        ep_dict["early_marginal_likelihood"]["optimal"], 2)
    early_l_estimated = np.round(
        ep_dict["early_marginal_likelihood"]["estimated"], 2)
    early_l_real = np.round(ep_dict["early_marginal_likelihood"]["initial"], 2)
    print("Early Marginal Likelihood, Optimal: {0}, Estimated: {1}, Initial {2} \\".format(
        early_l_optimal, early_l_estimated, early_l_real))
    # Late Likelihood
    late_l_optimal = np.round(
        lp_dict["late_marginal_likelihood"]["optimal"], 2)
    late_l_estimated = np.round(
        lp_dict["late_marginal_likelihood"]["estimated"], 2)
    late_l_real = np.round(lp_dict["late_marginal_likelihood"]["initial"], 2)
    print("Late Marginal Likelihood, Optimal: {0}, Estimated: {1}, Initial {2}".format(
        late_l_optimal, late_l_estimated, late_l_real))
    #print("Performance: rmse-mean = {0} \\".format(np.round(np.mean(np.array(list(result_dict["early_performance"].values()))), 4)))
    rmse_early_model = ep_dict["rmse"]

    early_rmse_frame = pd.DataFrame(
        columns=["rmse_A", "rmse_delta", "rmse_sigma"])
    early_rmse_frame = early_rmse_frame.append(
        rmse_early_model, ignore_index=True)
    index = ["estimated"]
    for baseline in list(result_dict["baselines"]["early"].keys()):
        rmse_baseline = result_dict["baselines"]["early"][baseline]["performance"]["rmse"]
        early_rmse_frame = early_rmse_frame.append(
            rmse_baseline, ignore_index=True)
        index.append(baseline)
    early_rmse_frame.index = index
    early_rmse_frame.columns = ["early_A", "early_delta", "early_sigma"]

    rmse_late_model = lp_dict["rmse"]

    late_rmse_frame = pd.DataFrame(
        columns=["rmse_A", "rmse_delta", "rmse_sigma"])
    late_rmse_frame = late_rmse_frame.append(
        rmse_late_model, ignore_index=True)
    index = ["estimated"]
    for baseline in list(result_dict["baselines"]["late"].keys()):
        rmse_baseline = result_dict["baselines"]["late"][baseline]["performance"]["rmse"]
        late_rmse_frame = late_rmse_frame.append(
            rmse_baseline, ignore_index=True)
        index.append(baseline)
    late_rmse_frame.index = index
    late_rmse_frame.columns = ["late_A", "late_delta", "late_sigma", "psi"]
    rmse_frame = pd.concat((early_rmse_frame, late_rmse_frame), axis=1)
    print(rmse_frame.to_markdown())
    print("####")
    # Individual
    rmse_theta = ep_dict["individual"]["rmse"]
    rmse_gain = lp_dict["individual"]["rmse"]
    print("Performance on Individual Level \\")
    print("rmse theta: {0} \\".format(rmse_theta))
    print("rmse gain: {0}".format(rmse_gain))


def parse_method_dict(performance_dict):
    ep_dict = performance_dict["early_performance"]
    lp_dict = performance_dict["late_performance"]
    early_runtime = np.round(ep_dict["run"]["runtime"], 2)
    early_steps = ep_dict["run"]["number_steps"]
    late_runtime = np.round(lp_dict["run"]["runtime"], 2)
    late_steps = lp_dict["run"]["number_steps"]
    early_l_optimal = np.round(
        ep_dict["early_marginal_likelihood"]["optimal"], 2)
    early_l_estimated = np.round(
        ep_dict["early_marginal_likelihood"]["estimated"], 2)
    late_l_optimal = np.round(
        lp_dict["late_marginal_likelihood"]["optimal"], 2)
    late_l_estimated = np.round(
        lp_dict["late_marginal_likelihood"]["estimated"], 2)
    rmse_early_model = ep_dict["rmse"]
    rmse_late_model = lp_dict["rmse"]
    rmse_early_A = rmse_early_model["rmse_A"]
    rmse_early_delta = rmse_early_model["rmse_delta"]
    rmse_early_sigma = rmse_early_model["rmse_sigma"]
    rmse_late_A = rmse_late_model["rmse_A"]
    rmse_late_delta = rmse_late_model["rmse_delta"]
    rmse_late_psi = rmse_late_model["rmse_psi"]
    rmse_late_sigma = rmse_late_model["rmse_sigma"]
    rmse_theta = ep_dict["individual"]["rmse"]
    rmse_gain_estimated = lp_dict["individual"]["rmse_estimated"]
    rmse_gain_pred_train = lp_dict["individual"]["rmse_pred_train"]
    performance_vector = np.array([early_runtime, late_runtime, early_steps, late_steps, early_l_optimal, early_l_estimated,
                                   late_l_optimal, late_l_estimated,
                                   rmse_early_A, rmse_early_delta, rmse_early_sigma,
                                   rmse_late_A, rmse_late_delta, rmse_late_psi, rmse_late_sigma, rmse_theta, rmse_gain_estimated, rmse_gain_pred_train])
    performance_df = pd.DataFrame(
        data=pd.DataFrame(performance_vector).transpose())
    performance_df.columns = ["early_runtime", "late_runtime", "early_steps", "late_steps", "early_l_optimal", "early_l_estimated",
                              "late_l_optimal", "late_l_estimated",
                              "rmse_early_A", "rmse_early_delta", "rmse_early_sigma",
                              "rmse_late_A", "rmse_late_delta", "rmse_psi", "rmse_late_sigma", "rmse_theta", "rmse_gain_estimated", "rmse_gain_pred_train"]
    return(performance_df)


def get_result_df(_result_dict):
    keylist = list(_result_dict.keys())
    latent_dimension = _result_dict[keylist[0]]["sample"]["latent_dimension"]
    item_dimension = _result_dict[keylist[0]]["sample"]["item_dimension"]
    sample_size = _result_dict[keylist[0]]["sample"]["sample_size"]
    meta = {"latent_dimension": latent_dimension,
            "item_dimension": item_dimension, "sample_size": sample_size}
    result_df = pd.DataFrame()
    for key in keylist:
        result_dict = _result_dict[key]
        method_list = list(result_dict.keys())
        method_list.remove("sample")
        for method in method_list:
            method_info_vector = np.array(
                [latent_dimension, item_dimension, sample_size, key, method])
            method_info_df = pd.DataFrame(
                data=pd.DataFrame(method_info_vector).transpose())
            method_info_df.columns = [
                "latent_dimension", "item_dimension", "sample_size", "key", "method"]
            performance_dict = result_dict[method]
            method_row = parse_method_dict(performance_dict)
            method_row = pd.concat((method_info_df, method_row), axis=1)
            result_df = pd.concat((result_df, method_row), axis=0)
    return(result_df)


def print_result_from_df(result_df, groupby=["method"], reference_method="late_em", description=""):
    # Description
    print("------------------------------------")
    print("##### Results for {0}".format(description))
    reference_method = result_df[result_df["method"]
                                 == reference_method].mean()
    initial = result_df[result_df["method"] == "initial"].mean()
    # Experiment Properties:
    latent_dimension = reference_method["latent_dimension"]
    item_dimension = reference_method["item_dimension"]
    sample_size = reference_method["sample_size"]
    print("Latent dimension: {0},  Item dimension: {1}, sample size {2} \\".format(
        latent_dimension, item_dimension, sample_size))
    # Performance/time
    early_runtime = np.round(reference_method["early_runtime"], 2)
    early_steps = reference_method["early_steps"]
    print("Early runtime: {0} seconds, {1} steps, {2} seconds per step \\".format(
        early_runtime, early_steps, np.round(early_runtime/early_steps, 2)))
    late_runtime = np.round(reference_method["late_runtime"], 2)
    late_steps = reference_method["late_steps"]
    print("Late runtime: {0} seconds, {1} steps, {2} seconds per step \\".format(
        late_runtime, late_steps, np.round(late_runtime/late_steps, 2)))
    # Performance/results
    # Early Likelihood
    early_l_optimal = np.round(reference_method["early_l_optimal"], 2)
    early_l_estimated = np.round(reference_method["early_l_estimated"], 2)
    early_l_initial = np.round(initial["early_l_estimated"], 2)
    print("Early Marginal Likelihood, Optimal: {0}, Estimated: {1}, Initial {2} \\".format(
        early_l_optimal, early_l_estimated, early_l_initial))
    # Late Likelihood
    late_l_optimal = np.round(reference_method["late_l_optimal"], 2)
    late_l_estimated = np.round(reference_method["late_l_estimated"], 2)
    late_l_initial = np.round(initial["late_l_estimated"], 2)
    print("Late Marginal Likelihood, Optimal: {0}, Estimated: {1}, Initial {2}".format(
        late_l_optimal, late_l_estimated, late_l_initial))
    #print("Performance: rmse-mean = {0} \\".format(np.round(np.mean(np.array(list(result_dict["early_performance"].values()))), 4)))

    rmse_columns = ["rmse_early_A", "rmse_early_delta", "rmse_early_sigma",
                    "rmse_late_A", "rmse_late_delta", "rmse_late_sigma", "rmse_psi"]
    rmse_df = result_df.groupby(groupby)[rmse_columns].mean()
    rmse_df.columns = ["early_A", "early_delta", "early_sigma",
                       "late_A", "late_delta", "late_sigma", "psi"]
    print(rmse_df.to_markdown())
    print("####")
    # Individual
    rmse_theta = reference_method["rmse_theta"]
    rmse_gain_estimated = reference_method["rmse_gain_estimated"]
    rmse_gain_pred_train = reference_method["rmse_gain_pred_train"]
    print("Performance on Individual Level \\")
    print("rmse theta: {0} \\".format(rmse_theta))
    print("rmse gain estimated: {0} \\".format(rmse_gain_estimated))
    print("rmse gain predicted: {0}".format(rmse_gain_pred_train))
