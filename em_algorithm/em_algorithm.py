import pandas as pd
import numpy as np
import copy
from sklearn.covariance import log_likelihood


class em_algorithm():

    def __init__(self, e_step: object, m_step: object, model: object) -> None:
        """Initializes em_algorithm object
        Args:
            e_step (e_step): Object that calculates an e-step. Offers a function step
            m_step (m_step): Object that calculates an m-step. Offers a function step
        """
        self.e_step = e_step
        self.m_step = m_step
        self.model = model

    def fit(self, data: pd.DataFrame, hyper_params: dict = None, person_method="newton_raphson", max_iter=100, stop_threshold=0.1) -> dict:
        """Fit the EM-Algorithm for some incomplete data for the specified incomplete_data_model.

        Args:
            data (pd.DataFrame): DataFrame of the incomplete data
            hyper_params (dict): Dictionary with Hyperparameters for the e_step and m_step. 
            max_iter (int, optional): Maximum Iterations. Defaults to 1000.

        Returns:
            dict: _description_
        """
        converged = False
        i = 1
        candidate_count = 0
        self.e_step.set_incomplete_data(data)
        current_parameters = {"item_parameters": self.model.item_parameters,
                              "person_parameters": self.model.person_parameters}
        marginal_loglikelihood = self.model.marginal_response_loglikelihood(
            *data)
        while (not converged) and i <= max_iter:
            print("EM Iteration {0}".format(i+1))
            last_step_parameters = copy.deepcopy(current_parameters)
            last_step_marginal_loglikelihood = marginal_loglikelihood.copy()
            # print("E-step")
            posterior_expectation = self.e_step.step(
                *data, iter=i)
            # print("M-step")
            current_parameters, log_likelihood = self.m_step.step(
                pe_functions=posterior_expectation, person_method=person_method)
            self.model.set_parameters(current_parameters)
            marginal_loglikelihood = self.model.marginal_response_loglikelihood(
                *data)
            marginal_loglikelihood_diff = abs(
                marginal_loglikelihood - last_step_marginal_loglikelihood)
            abs_parameter_diff, parameter_diff = self.give_parameter_diff(
                current_parameters=current_parameters, last_step_parameters=last_step_parameters)
            # if (parameter_diff <= 0.2) or (i >= max_iter-1):
            # Enforce that the Likelihood increases
            # if marginal_loglikelihood < last_step_marginal_loglikelihood:
            #    print("Degenerated step!")
            #    self.model.set_parameters(last_step_parameters)
            #    current_parameters = copy.deepcopy(last_step_parameters)
            #    marginal_loglikelihood = last_step_marginal_loglikelihood.copy()
            # TODO: make stopping criterion in percent-change
            if (marginal_loglikelihood_diff <= stop_threshold) or (i >= max_iter):
                candidate_count += 1
                if candidate_count >= 3:
                    converged = True
            else:
                candidate_count = 0
            # if (np.sum(np.array(parameter_diff) >= np.array(stop_criterion)) == 0) and i >= 10:
            #    converged = True
            i = i+1
            print("Step: {0}: current parameter_diff: {1}, current marginal loglikelihood: {2}".format(
                i, abs_parameter_diff,  marginal_loglikelihood))
            self.n_steps = i+1

    def give_parameter_diff(self, current_parameters, last_step_parameters):
        current_A = current_parameters["item_parameters"]["discrimination_matrix"]
        current_delta = current_parameters["item_parameters"]["intercept_vector"]
        current_sigma = current_parameters["person_parameters"]["covariance"]
        current_parameters_flat = np.concatenate(
            (current_A.flatten(), current_delta.flatten(), current_sigma.flatten()), axis=0)

        last_A = last_step_parameters["item_parameters"]["discrimination_matrix"]
        last_delta = last_step_parameters["item_parameters"]["intercept_vector"]
        last_sigma = last_step_parameters["person_parameters"]["covariance"]
        last_parameters_flat = np.concatenate(
            (last_A.flatten(), last_delta.flatten(), last_sigma.flatten()), axis=0)
        abs_parameter_diff = np.sum(
            np.abs(current_parameters_flat-last_parameters_flat))
        parameter_diff = np.sum(
            current_parameters_flat-last_parameters_flat)
        return(abs_parameter_diff, parameter_diff)
