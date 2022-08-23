import pandas as pd
import numpy as np
from sklearn.covariance import log_likelihood


class em_algorithm():

    def __init__(self, e_step: object, m_step: object, incomplete_data_model: object) -> None:
        """Initializes em_algorithm object
        Args:
            e_step (e_step): Object that calculates an e-step. Offers a function step
            m_step (m_step): Object that calculates an m-step. Offers a function step
        """
        self.e_step = e_step
        self.m_step = m_step

    def fit(self, data: pd.DataFrame, hyper_params: dict, max_iter=1000) -> dict:
        """Fit the EM-Algorithm for some incomplete data for the specified incomplete_data_model.

        Args:
            data (pd.DataFrame): DataFrame of the incomplete data
            hyper_params (dict): Dictionary with Hyperparameters for the e_step and m_step. 
            max_iter (int, optional): Maximum Iterations. Defaults to 1000.

        Returns:
            dict: _description_
        """
        converged = False
        i = 0
        while (not converged) and i <= max_iter:
            last_step_parameters = current_parameters
            posterior_expectation = self.e_step.step(data=data)
            current_parameters, log_likelihood = self.m_step.step(
                data=data, posterior_expectation=posterior_expectation)
            parameter_diff = [np.sum(np.abs(current_parameters[i]-last_step_parameters[i]))
                              for i in range(0, len(current_parameters))]
            if parameter_diff <= 0.1:
                converged = True
            # if (np.sum(np.array(parameter_diff) >= np.array(stop_criterion)) == 0) and i >= 10:
            #    converged = True
            i = i+1
            print("Step: {0}: current parameter_diff: {1}, current data likelihood: {2}".format(
                i, parameter_diff, log_likelihood))
