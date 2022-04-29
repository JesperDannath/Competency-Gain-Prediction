import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn import metrics

"""Algortihm for Knowledge Growth prediction.
"""

###############################################################################
# ICC'S:
# Define Item Characteristic Curve Function based on vectors a, b, theta.


def ICC(parameters, theta):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
        theta (_type_): _description_
    """
    a, b = parameters
    try:
        item_dim = len(b)
    except Exception:
        item_dim = 1
    theta_expand = np.transpose(np.tile(theta, (item_dim, 1)))
    ability_item_product = np.subtract(theta_expand, b)
    sigma = 1/(1+np.exp(-np.multiply(a, (ability_item_product))))
    return(sigma)


def LFA_Curve(parameters, alpha, s):
    """ICC that corresponds to the Concept of LFA (Learning Factor Analysis) and Features prior ability as well as item and person paramers.

    Args:
        parameters (list): list of the format [beta, gamma]. beta is the vector of Item-Easiness. gamma is the vector of item-training effectiveness. 
        alpha (np.ndarray): prior ability values. 
        s (np.ndarray): Person training-extend. 
    """
    beta, gamma = parameters
    try:
        item_dim = len(beta)
    except Exception:
        item_dim = 1
    s_expand = np.transpose(np.tile(s, (item_dim, 1)))
    sigmoid_arg = np.add(np.expand_dims(alpha, axis=1),
                         np.add(beta, np.multiply(gamma, s_expand)))
    sigma = 1/(1+np.exp(-1*sigmoid_arg))
    return(sigma)


########################################################################
# Prior Distributions
def generate_prior_norm(loc, scale, K, dist=norm):
    prior_dist = dist(loc, scale)
    start_point = loc-2.5*scale
    end_point = loc+2.5*scale
    side_points = np.linspace(start_point, end_point, num=K+1)
    quadratures = [(side_points[i]+side_points[i-1]) /
                   2 for i in range(1, len(side_points))]
    quadrature_dist = [prior_dist.cdf(
        side_points[i])-prior_dist.cdf(side_points[i-1]) for i in range(1, len(side_points))]
    quadrature_dist[0] = prior_dist.cdf(side_points[1])
    quadrature_dist[K-1] = 1-prior_dist.cdf(side_points[K-1])
    return(quadratures, quadrature_dist)


######################################################################
# EM-Algorithm

# Likelihood for the M-Step's
def complete_data_log_likelihood(ability_weight, correct_response_weight, current_parameters, quadratures, quadrature_probs, ICC):
    # Number of Quadratures or factors of the latent prior
    K = len(ability_weight)
    J = len(current_parameters[0])  # Number of Items
    ICC_values = ICC(current_parameters, quadratures)
    log_likelihood = 0
    for j in range(0, J):
        # vector: should be equal in length to # of Items
        r_j = correct_response_weight[j, :]
        log_likelihood += np.dot(r_j, np.log(ICC_values[:, j])) + np.dot(np.subtract(ability_weight, r_j), np.log(
            (1-ICC_values[:, j]))) + np.dot(ability_weight*np.ones((1, K)), np.log(quadrature_probs))
    return(log_likelihood)


def m_step_mmle(expectation, current_parameters, prior_dist, ICC):
    """M-Step for a classic UIRT-Model with a latent prior distribution like in Hanson 2000

    Args:
        expectation (list): Expectation for key model constants from the E-Step
        current_parameters (list): Current Parameters from last M-Sterp or initiation
        prior_dist (list): List of the format [quadratures, quadrature_weights]
        ICC (function): ICC for the respective UIRT Model. 
    """
    ability_weight, correct_response_weight = expectation
    # Define negative log-likelihood in relation to x = (a, b)

    def nll(x): return -1*complete_data_log_likelihood(ability_weight, correct_response_weight,
                                                       current_parameters=[x[0: len(current_parameters[0])], x[len(current_parameters[0]): len(
                                                           current_parameters[0])+len(current_parameters[1])]],  # a=x[0:len(a)], b=x[len(a):len(x)],
                                                       quadratures=prior_dist[0], quadrature_probs=prior_dist[1],
                                                       ICC=ICC)
    #a_t, b_t = current_parameters
    nll(np.concatenate(current_parameters, axis=0))
    x0 = np.concatenate(current_parameters, axis=0)
    res = minimize(nll, x0=x0, method='Nelder-Mead')  # BFGS
    new_parameters = [res.x[0: len(current_parameters[0])], res.x[len(
        current_parameters[0]): len(current_parameters[0])+len(current_parameters[1])]]
    #a_hat = res.x[0:len(a_t)]
    #b_hat = res.x[len(a_t): len(res.x)]
    likelihood = res.fun
    return(new_parameters, likelihood)


def m_step_late_mmle(expectation, current_parameters, prior_dist, ICC):
    """M-Step for a LFA-Model with a latent prior distribution like in Hanson 2000

    Args:
        expectation (list): Expectation for key model constants from the E-Step
        current_parameters (list): Current Parameters from last M-Step or initiation
        prior_dist (list): List of the format [quadratures, quadrature_weights]
        ICC (function): ICC for the respective LFA Model. 
    """
    ability_weight, correct_response_weight, expected_alpha = expectation
    #N, J = U.shape
    # Define negative log-likelihood in relation to x = (a, b)

    def nll(x): return -1*complete_data_log_likelihood(ability_weight, correct_response_weight,
                                                       current_parameters=[x[0: len(current_parameters[0])], x[len(current_parameters[0]): len(
                                                           current_parameters[0])+len(current_parameters[1])]],  # a=x[0:len(a)], b=x[len(a):len(x)],
                                                       quadratures=prior_dist[0], quadrature_probs=prior_dist[1],
                                                       ICC=lambda parameters, s: ICC(parameters, expected_alpha, s))
    nll(np.concatenate(current_parameters, axis=0))
    x0 = np.concatenate(current_parameters, axis=0)
    res = minimize(nll, x0=x0, method='Nelder-Mead')  # BFGS
    new_parameters = [res.x[0: len(current_parameters[0])], res.x[len(
        current_parameters[0]): len(current_parameters[0])+len(current_parameters[1])]]
    likelihood = res.fun
    return(new_parameters, likelihood)


# E-Step's
def conditional_ability_probability(response_pattern, current_parameters, prior_dist, ICC):
    quadratures, quadrature_dist = prior_dist
    K = len(quadratures)
    conditional_answer_probability = np.empty(shape=K)
    for k in range(0, K):
        conditional_answer_probability[k] = np.prod(np.power(ICC(current_parameters, quadratures[k]), response_pattern))*np.prod(
            np.power(1-ICC(current_parameters, quadratures[k]), 1-response_pattern))
    p_sum = np.sum(conditional_answer_probability)
    conditional_ability_p = [
        conditional_answer_probability[k]/p_sum for k in range(0, K)]
    return(conditional_ability_p)


def e_step_mmle(response_data, current_parameters, prior_dist, ICC):
    """E-Step for a classic UIRT-Model with a latent prior distribution like in Hanson 2000

    Args:
        response_data (pd.DataFrame): Dichotomous Data for Item Responses, rows=respondents, cols=Items
        current_parameters (list): Current Parameters from last M-Sterp or initiation
        prior_dist (list): List of the format [quadratures, quadrature_weights]
        ICC (function): ICC for the respective UIRT Model. 
    """
    quadratures = prior_dist[0]
    #ICC_values = ICC(current_parameters, quadratures)
    # Calculate Expected Values
    N = response_data.shape[0]
    K = len(quadratures)
    c_a_p = np.empty(shape=(N, K))
    for i, response_pattern in enumerate(response_data.to_numpy()):
        c_a_p[i] = conditional_ability_probability(
            response_pattern, current_parameters, prior_dist, ICC)
    # There should be K ability weights
    ability_weights = np.sum(np.array(c_a_p), axis=0)
    correct_response_weights = np.dot(np.transpose(
        response_data), np.array(c_a_p))  # should be a K * J Matrix

    return(ability_weights, correct_response_weights)


def e_step_late_mmle(data, current_parameters, prior_dist, ICC):
    """E-Step for a LFA-Model with a latent prior distribution

    Args:
        response_data (pd.DataFrame): Dichotomous Data for Item Responses, rows=respondents, cols=Items
        current_parameters (list): Current Parameters from last M-Sterp or initiation
        prior_dist (list): List of the format [quadratures, quadrature_weights]
        ICC (function): ICC for the respective LFA Model. 
    """
    response_data, alpha_data = data
    quadratures, quadrature_probs = prior_dist
    # Calculate Expected Values
    N = response_data.shape[0]
    K = len(quadratures)
    c_a_p = np.empty(shape=(N, K))
    for i, response_pattern in enumerate(response_data.to_numpy()):
        c_a_p[i] = conditional_ability_probability(
            response_pattern, current_parameters, prior_dist, lambda parameters, s: ICC(parameters, [alpha_data[i]], s))
    # There should be K ability weights
    ability_weights = np.sum(np.array(c_a_p), axis=0)
    expected_alpha = [(1/N*quadrature_probs[k]) *
                      np.sum(alpha_data*c_a_p[:, k]) for k in range(0, K)]
    correct_response_weights = np.dot(np.transpose(
        response_data), np.array(c_a_p))  # should be a K * J Matrix

    return(ability_weights, correct_response_weights, expected_alpha)


# Main EM-Algorithm
def em_algorithm(response_data, starting_parameters, prior_dist,
                 e_step,
                 m_step=m_step_mmle, stop_criterion=[0.01, 0.01], ICC=ICC, max_iter=90):
    """Basic formulation of the EM-Algorithm for estimimating Parameters for IRT

    Args:
        response_data (pd.DataFrame or np.array): Response data from Item's 
        starting_parameters (np.array): Initial Parameters to be handed over to the EM-steps. End-result might depend on these.
        e_step (function): Algorithm for calculating the current expected 
        m_step (function): Algorithm for finding the best paramaters given the current expectation. 
    """
    converged = False
    current_parameters = starting_parameters
    i = 0
    while (not converged) and i <= max_iter:
        last_step_parameters = current_parameters
        expectation = e_step(current_parameters, prior_dist, ICC)
        current_parameters, log_likelihood = m_step(
            expectation, current_parameters, prior_dist, ICC)
        parameter_diff = [np.sum(np.abs(current_parameters[i]-last_step_parameters[i]))
                          for i in range(0, len(current_parameters))]
        if (np.sum(np.array(parameter_diff) >= np.array(stop_criterion)) == 0) and i >= 10:
            converged = True
        i = i+1
        print("Step: {0}: current parameter_diff: {1}, current data likelihood: {2}".format(
            i, parameter_diff, log_likelihood))
    return(current_parameters)

###################################################################################
# Fitting Algorithms for Person Parameters

# Fit ability
# Log-Liklihood for ability


def answer_log_likelihood(theta, answer_vector, item_parameters, prior_dist):
    ICC_values = ICC(item_parameters, theta)
    log_likelihood = np.dot(answer_vector, np.log(ICC_values)[0]) + np.dot(
        (1-answer_vector), np.log(1-ICC_values)[0]) + np.log(prior_dist.pdf(theta))
    return(log_likelihood)


def fit_ability(response_data, item_parameters, prior_dist=norm(0, 1)):
    """Given the estimated item-parameters for a UIRT-Model and some response_data, this function will estimate the latent ability for every respondent.

    Args:
        response_data (pd.DataFrame or np.array): Response data from Item's 
        item_parameters (list): list of estimated item_parameters
        prior_dist (scipy.stats.distribution, optional): Prior Distribution for latent trait in given sample. If nothing else is apparent this should be equal to the prior
                                                         dist that was used in the estimation process . Defaults to norm(0,1).
    """
    ability_vector = np.empty(response_data.shape[0])
    for i, response_pattern in enumerate(response_data.to_numpy()):
        def nll(x): return -1*answer_log_likelihood(x,
                                                    response_pattern, item_parameters, prior_dist)
        res = minimize(nll, x0=np.random.normal(0, 1), method='BFGS')
        ability_vector[i] = res.x
    ability = pd.DataFrame(ability_vector, columns=["ability"])
    ability = ability.set_index(response_data.index)
    return(ability)


# Fit Practise
def LFA_answer_log_likelihood(s, alpha, answer_vector, item_parameters, prior_dist):
    LFA_values = LFA_Curve(item_parameters, alpha, s)
    log_likelihood = np.dot(answer_vector, np.log(LFA_values)[
                            0]) + np.dot((1-answer_vector), np.log(1-LFA_values)[0]) + np.log(prior_dist.pdf(s))
    return(log_likelihood)


def fit_practise_ammount(response_data, alpha, item_parameters, prior_dist):
    """Given the estimated item-parameters for a UIRT-Model and some response_data, this function will estimate the latent ability for every respondent.

    Args:
        response_data (pd.DataFrame or np.array): Response data from Item's 
        item_parameters (list): list of estimated item_parameters
        prior_dist (scipy.stats.distribution, optional): Prior Distribution for latent trait in given sample. If nothing else is apparent this should be equal to the prior
                                                         dist that was used in the estimation process . Defaults to norm(0,1).
    """
    ability_vector = np.empty(response_data.shape[0])
    for i, response_pattern in enumerate(response_data.to_numpy()):
        def nll(x): return -1*LFA_answer_log_likelihood(x, alpha,
                                                        response_pattern, item_parameters, prior_dist)
        res = minimize(nll, x0=prior_dist.rvs(), method='Nelder-Mead')
        ability_vector[i] = res.x
    return(pd.Series(ability_vector))


####################################################################################
# All together Model Class
class knowledge_growth_model():

    def __init__(self) -> None:
        pass

    def fit(self, early_response_data, late_response_data,
            alpha_loc=0, alpha_scale=1, alpha_k=10, s_loc=2, s_scale=1, s_k=15):
        """Fit the UIRT Model on Data for late and early item responses. Different kinds of
           normal priors can be used by specifying their mean and scale.

        Args:
            early_response_data (pd.DataFrame): _description_
            late_response_data (pd.DataFrame): _description_
            alpha_loc (int, optional): _description_. Defaults to 0.
            alpha_scale (int, optional): _description_. Defaults to 1.
            alpha_k (int, optional): _description_. Defaults to 10.
            s_loc (int, optional): _description_. Defaults to 2.
            s_scale (int, optional): _description_. Defaults to 1.
            s_k (int, optional): _description_. Defaults to 15.
        """
        # Set model-wide Parameters for prior's
        self.alpha_loc = alpha_loc
        self.alpha_scale = alpha_scale
        self.s_loc = s_loc
        self.s_scale = s_scale
        prior_b = np.random.normal(
            loc=0, scale=1, size=early_response_data.shape[1])
        prior_a = np.random.uniform(
            low=0, high=2, size=early_response_data.shape[1])
        print("Fit early Item Parameters:")
        self.a_hat, self.b_hat = em_algorithm(response_data=early_response_data,
                                              starting_parameters=[
                                                  prior_a, prior_b],
                                              e_step=lambda current_parameters, prior_dist, ICC: e_step_mmle(
                                                  early_response_data, current_parameters, prior_dist, ICC),
                                              prior_dist=generate_prior_norm(alpha_loc, alpha_scale, alpha_k))
        print("Fit early Person Parameters")
        self.alpha = fit_ability(early_response_data, [
                                 self.a_hat, self.b_hat], prior_dist=norm(alpha_loc, alpha_scale))
        prior_beta = -1 * \
            np.random.exponential(scale=1, size=late_response_data.shape[1])
        prior_gamma = np.random.exponential(
            scale=1, size=late_response_data.shape[1])
        print("Fit late Item Parameters")
        alpha_for_late = pd.DataFrame(late_response_data.index).set_index(
            late_response_data.index).join(self.alpha)["ability"].to_numpy()
        alpha_for_late = np.nan_to_num(
            alpha_for_late, nan=np.nanmean(alpha_for_late))
        self.beta_hat, self.gamma_hat = em_algorithm(response_data=late_response_data,
                                                     starting_parameters=[
                                                         prior_beta, prior_gamma],
                                                     prior_dist=generate_prior_norm(
                                                         s_loc, s_scale, s_k),
                                                     ICC=LFA_Curve,
                                                     e_step=lambda current_parameters, prior_dist, ICC: e_step_late_mmle(
                                                         [late_response_data, alpha_for_late], current_parameters, prior_dist, ICC),
                                                     m_step=m_step_late_mmle)
        print("Fit late Person Parameters")
        self.s_hat = fit_practise_ammount(late_response_data, alpha_for_late, [
                                          self.beta_hat, self.gamma_hat], prior_dist=norm(s_loc, s_scale))

    def predict_answers(self, early_response_data, s_estimate=[], late_icc=LFA_Curve, cutoff=0.5, return_p=False):
        alpha = fit_ability(early_response_data, [
            self.a_hat, self.b_hat], prior_dist=norm(self.alpha_loc, self.alpha_scale))

        if len(s_estimate) == 0:
            p = np.squeeze(late_icc([self.beta_hat, self.gamma_hat],
                                    alpha, np.mean(self.s_hat)))
        else:
            p = np.squeeze(late_icc([self.beta_hat, self.gamma_hat],
                                    alpha.to_numpy().transpose()[0], s_estimate))  # Alpha hat evtl. falsche Dimension
        prediction = (p >= cutoff)
        if return_p:
            return(pd.DataFrame(p, index=early_response_data.index))
        else:
            return(pd.DataFrame(prediction.astype("int64"), index=early_response_data.index))

    def get_training_performance(self, early_response_data, late_response_data, s_estimate=[], return_performance=False):
        """Predict late responses and calculate performance measures to compare models. 

        Args:
            early_response_data (pd.DataFrame): Early Response Data
            late_response_data (pd.DataFrame): Late Response Data
            s_estimate (list, optional): If a list of s-estimates is given, the ammount of practise per person is considered for prediction. Defaults to [].
        """
        prediction = self.predict_answers(
            early_response_data, s_estimate=s_estimate)
        prediction = prediction.drop(
            prediction.index.difference(late_response_data.index))
        prediction_shift_index = prediction.reindex(
            late_response_data.index).dropna(how="all")
        p = self.predict_answers(
            early_response_data, s_estimate=s_estimate, return_p=True)
        p = p.drop(
            p.index.difference(late_response_data.index))
        p_shift_index = p.reindex(
            late_response_data.index).dropna(how="all")
        late_response_data_drop = late_response_data.drop(
            late_response_data.index.difference(prediction_shift_index.index)).to_numpy()
        prediction_shift_index = prediction_shift_index.to_numpy()
        auc_score = metrics.roc_auc_score(late_response_data_drop.flatten().astype(
            np.int), p_shift_index.to_numpy().flatten())
        print("Accuracy per Question: \n{0} \nOverall acuracy: {1}".format(
            1-np.mean(np.abs(prediction_shift_index-late_response_data_drop), axis=0), 1-np.mean(np.abs(prediction_shift_index-late_response_data_drop).flatten())))
        print("AUC-Score: {0}".format(auc_score))
        if return_performance:
            return({"Accuracy": 1-np.mean(np.abs(prediction_shift_index-late_response_data_drop).flatten()), "AUC": auc_score})

    def predict_early_ability(self, early_response_data):
        alpha = fit_ability(early_response_data, [
            self.a_hat, self.b_hat], prior_dist=norm(self.alpha_loc, self.alpha_scale))
        return(alpha)
