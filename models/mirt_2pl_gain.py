from tkinter import Variable
from mirt_2pl import mirt_2pl
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats.qmc import MultivariateNormalQMC
from scipy.optimize import minimize


class mirt_2pl_gain(mirt_2pl):

    def __init__(self, item_dimension: int, latent_dimension: int,  mu: np.array = np.empty(0),  A=np.empty(0), Q=np.empty(0),
                 delta=np.empty(0), early_sigma=np.empty(0), late_sigma=np.empty(0), latent_corr=np.empty(0)):
        super().__init__(item_dimension, latent_dimension=latent_dimension, A=A, Q=Q)
        self.person_parameters["early_sigma"] = early_sigma
        self.type = "gain"
        self.person_parameters["convolution_variance"] = np.empty(0)
        self.initialize_gain_parameters(early_sigma, late_sigma, latent_corr)
        self.update_conditional_covariance()

    def initialize_from_responses(self, late_response_data: pd.DataFrame, early_response_data: pd.DataFrame,
                                  convolution_variance=np.empty(0), sigma=True, logit=True):
        super().initialize_from_responses(late_response_data, sigma)
        D = self.latent_dimension
        if convolution_variance.size == 0:
            p_early = np.mean(early_response_data, axis=1)
            p_late = np.mean(late_response_data, axis=1)
            p_diff = np.subtract(p_late, p_early)
            if not logit:
                # Var(p_late) = Var(p_diff) + Var(p_early) + 2*Cov(p_diff, p_early)
                var_p_early = np.var(p_early)
                var_p_late = np.var(p_late)
                var_p_diff = np.var(p_diff)
                cov_early_diff = pd.DataFrame(
                    np.stack((p_diff, p_early), axis=1)).cov().to_numpy()[0, 1]
                early_share = var_p_early/var_p_late
                late_share = (var_p_diff)/var_p_late
                cov_share = cov_early_diff/var_p_late
            else:
                p_early[p_early == 0] = np.min(p_early[p_early != 0])
                p_early[p_early == 1] = np.max(p_early[p_early != 1])
                p_late[p_late == 0] = np.min(p_late[p_late != 0])
                p_late[p_late == 1] = np.max(p_late[p_late != 1])
                logit_early = np.log(np.divide(p_early, 1 -
                                               p_early))
                logit_late = np.log(np.divide(p_late, 1 -
                                              p_late))
                logit_diff = logit_late - logit_early
                var_logit_early = np.var(logit_early)
                var_logit_late = np.var(logit_late)
                var_logit_diff = np.var(logit_diff)
                cov_early_diff = pd.DataFrame(
                    np.stack((logit_diff, logit_early), axis=1)).cov().to_numpy()[0, 1]
                early_share = var_logit_early/var_logit_late
                late_share = (var_logit_diff)/var_logit_late
                cov_share = cov_early_diff/var_logit_late
            # rho und p sollten vectoren sein!
            late_sigma_var = np.ones(D)*late_share/early_share
            late_sigma_sd_vector = np.sqrt(late_sigma_var)
            sd_matrix = np.diag(late_sigma_sd_vector)
            late_sigma = np.dot(
                np.dot(sd_matrix, self.person_parameters["early_sigma"]), sd_matrix)
            self.person_parameters["late_sigma"] = late_sigma
            self.person_parameters["covariance"][D:2*D, D:2*D] = late_sigma
            psi_diag = np.ones(D)*cov_share/early_share
            psi = np.zeros((D, D))
            psi[np.diag_indices_from(psi)] = psi_diag
            self.person_parameters["covariance"][0:D, D:2*D] = psi
            self.person_parameters["covariance"][D:2*D, 0:D] = psi.transpose()
        else:
            self.person_parameters["convolution_variance"] = convolution_variance
            late_variance = convolution_variance - 1
            sigma_psi = np.identity(2*D)
            if (late_variance > 0).all():
                late_sigma = np.diag(late_variance)
            else:
                late_sigma = np.identity(D)
                psi_diag = (late_variance-1)/2
                psi = np.diag(psi_diag)
                sigma_psi[0:D, D:2*D] = psi
                sigma_psi[D:2*D, 0:D] = psi.transpose()
            sigma_psi[0:D, 0:D] = self.person_parameters["early_sigma"]
            sigma_psi[D:2*D, D:2*D] = late_sigma
            self.person_parameters["covariance"] = sigma_psi
            try:
                self.check_sigma()
            except Exception:
                # Sigma_psi might violate the convolution variance constraint in this case
                sigma_psi = 0.5 * \
                    np.diag(
                        np.concatenate((convolution_variance, convolution_variance), axis=0))
                self.person_parameters["covariance"] = sigma_psi

    def initialize_gain_parameters(self, early_sigma=np.empty(0), late_sigma=np.empty(0), latent_corr=np.empty(0)):
        D = self.latent_dimension
        mu = np.ones(2*D)
        mu[0:D] = 0
        sigma_psi = np.identity(self.latent_dimension*2)
        sigma_psi[sigma_psi != 1] = 0.5
        if early_sigma.size > 0:
            sigma_psi[0:D, 0:D] = early_sigma
        if late_sigma.size > 0:
            sigma_psi[D:2*D, D:2*D] = late_sigma
        if latent_corr.size > 0:
            sigma_psi[0:D, D:2*D] = latent_corr
            sigma_psi[D:2*D, 0:D] = latent_corr.transpose()
        person_parameter_update = {"covariance": sigma_psi, "mean": mu}
        self.set_parameters({"person_parameters": person_parameter_update})

    # TODO: Maybe reduce number of times the matrix calculations are done here
    def update_conditional_covariance(self):
        D = self.latent_dimension
        psi = self.get_cov("corr")
        try:
            inv_early_cov = np.linalg.inv(self.get_cov("early"))
            inv_late_cov = np.linalg.inv(self.get_cov("late"))
        except Exception:
            print(self.get_cov("late"))
            raise Exception("Conditional cov could not be calculated")
        mu = self.person_parameters["mean"][D:2*D]
        early_mean_predictor = np.dot(psi, inv_late_cov)
        early_conditional_cov = self.get_cov(
            "early") - np.dot(np.dot(psi, inv_late_cov), psi.transpose())
        # TODO: Check where Psi needs to be transposed!
        late_conditional_cov = self.get_cov(
            "late") - np.dot(np.dot(psi.transpose(), inv_early_cov), psi)
        person_parameters = {"early_conditional_covariance": early_conditional_cov,
                             "early_conditional_mean_predictor": early_mean_predictor,
                             "late_conditional_covariance": late_conditional_cov}
        self.set_parameters({"person_parameters": person_parameters})

    def check_sigma(self, sigma: np.array = np.empty(0), enforce_early_cov=False, callback=True):
        D = self.latent_dimension
        if sigma.size == 0:
            sigma = self.person_parameters["covariance"]
        else:
            if (sigma[0:D, 0:D] != self.person_parameters["covariance"][0:D, 0:D]).any() and enforce_early_cov:
                raise Exception("Early Covariance not correct")
        if not sigma.shape == (2*D, 2*D):
            raise Exception("Covariance is of wrong shape")
        if not np.array_equal(sigma.transpose(), sigma):
            if callback:
                print(sigma)
            raise Exception("Covariance is not symmetric")
        if not np.all(np.linalg.eigvals(sigma) >= 0):
            if callback:
                print(sigma)
            raise Exception("New Covariance not positive semidefinite")
        if self.person_parameters["convolution_variance"].size != 0:
            convolution_matrix = np.concatenate(
                (np.identity(D), np.identity(D)), axis=1)
            convolution_covariance = np.dot(convolution_matrix, sigma)
            convolution_covariance = np.dot(
                convolution_covariance, convolution_matrix.transpose())
            variance_diff = np.abs(
                np.diag(convolution_covariance)-self.person_parameters["convolution_variance"])
            if (variance_diff > 0.01).any():
                raise Exception("Convolution variance has wrong values")
        return(True)

    def corr_to_sigma(self, corr_vector, check=True, type="only_late"):
        D = self.latent_dimension
        if type == "only_late":
            sigma_psi_triu = np.triu(self.person_parameters["covariance"], k=1)
            psi = corr_vector[0:D**2].reshape((D, D))
            sigma_psi_triu[0:D, D:2*D] = psi
            late_sigma_corr_u = corr_vector[D**2:]
            late_sigma_triu = np.identity(D)
            np.place(late_sigma_triu,
                     mask=np.triu(np.ones(D), k=1).astype(np.bool), vals=late_sigma_corr_u)
            sigma_psi_triu[D:2*D, D:2*D] = late_sigma_triu
            sigma_psi = super().corr_to_sigma(
                sigma_psi_triu[np.triu_indices_from(sigma_psi_triu, k=1)], check=check)
        elif type == "only_psi_off_diag":
            sigma_psi = self.person_parameters["covariance"]
            psi = sigma_psi[0:D, D:2*D]
            psi[np.tril_indices_from(psi, k=1)] = corr_vector
            psi[np.triu_indices_from(
                psi, k=-1)] = psi.transpose()[np.triu_indices_from(psi, k=-1)]
            sigma_psi[0:D, D:2*D] = psi
            sigma_psi[D:2*D, 0:D] = psi.transpose()
        elif type == "fixed_convolution_variance":
            psi = corr_vector[0:D**2]
            psi = psi.reshape((D, D))
            late_sigma_triu = corr_vector[D**2: len(corr_vector)]
            late_sigma_diag = self.person_parameters["convolution_variance"] - \
                1 - 2*np.diag(psi)
            late_sigma = np.diag(late_sigma_diag)
            late_sigma[np.triu_indices_from(late_sigma, k=1)] = late_sigma_triu
            late_sigma[np.tril_indices_from(late_sigma, k=-1)] = late_sigma.transpose()[
                np.tril_indices_from(late_sigma, k=-1)]
            sigma_psi = self.person_parameters["covariance"]
            sigma_psi[D:2*D, D:2*D] = late_sigma
            sigma_psi[0:D, D:2*D] = psi
            sigma_psi[D:2*D, 0:D] = psi.transpose()
        elif type == "diagonal":
            late_sigma_diag = corr_vector
            late_sigma = np.diag(corr_vector)
            psi_diag = (self.person_parameters["convolution_variance"] -
                        1 - late_sigma_diag)/2
            psi = np.diag(psi_diag)
            sigma_psi = np.identity(2*D)
            sigma_psi[D:2*D, D:2*D] = late_sigma
            sigma_psi[0:D, D:2*D] = psi
            sigma_psi[D:2*D, 0:D] = psi.transpose()
        elif type == "only_psi":
            pass
        elif type == "full":
            sigma_psi = super().corr_to_sigma(corr_vector)
        else:
            raise Exception("Corr type for sigma not known")
        return(sigma_psi)

    def fix_sigma(self, sigma_psi, sigma_constraint="early_constraint", convolution_constraint=True):
        D = self.latent_dimension
        J = self.item_dimension
        # if sigma_constraint != "esigma_spsi":
        #     sigma_psi = super().fix_sigma(sigma_psi)
        if (sigma_constraint == "early_constraint") & (not convolution_constraint):
            sigma_psi[0:D, 0:D] = self.person_parameters["covariance"][0:D, 0:D]
        if convolution_constraint:
            convolution_variance = self.person_parameters["convolution_variance"]
            early_sigma = self.person_parameters["early_sigma"]
            A = self.item_parameters["discrimination_matrix"]
            AA = np.concatenate((A, A), axis=1)

            def func(Lambda):
                Lambda = Lambda.reshape((2*D, 2*D))
                new_sigma = np.dot(
                    np.dot(Lambda, sigma_psi), Lambda.transpose())
                early_diff = np.sum(
                    np.square(new_sigma[0:D, 0:D].flatten() - early_sigma.flatten()))
                # late_diag_diff = np.sum(
                #     np.square(np.diagonal(new_sigma[D:2*D, D:2*D]) - late_diag))
                convolution_matrix = np.concatenate(
                    (np.identity(D), np.identity(D)), axis=1)
                convolution_sigma = np.dot(convolution_matrix, new_sigma)
                convolution_sigma = np.dot(
                    convolution_sigma, convolution_matrix.transpose())
                convolution_variance_diff = np.sum(
                    np.square(convolution_variance - np.diag(convolution_sigma)))
                # Regularise by Q-Matrix
                QQ = np.concatenate(
                    (self.item_parameters["q_matrix"], self.item_parameters["q_matrix"]), axis=1)
                inv_Lambda = np.linalg.inv(Lambda)
                rot_QQ = np.dot(QQ, inv_Lambda)
                q_violation_sum = np.sum(np.square(rot_QQ[np.where(QQ == 0)]))
                # Regularise by A Matrix
                rot_AA = np.dot(AA, inv_Lambda)
                rot_A1 = rot_AA[:, 0:D]
                rot_A2 = rot_AA[:, D:2*D]
                rot_A_diff = np.sum(
                    np.square(rot_A1.flatten() - rot_A2.flatten()))
                return(early_diff + convolution_variance_diff + q_violation_sum + rot_A_diff)
            x0 = np.identity(2*D).flatten()
            Lambda = minimize(
                fun=func, x0=x0, method="BFGS").x.reshape((2*D, 2*D))
            sigma_psi = np.dot(np.dot(Lambda, sigma_psi), Lambda.transpose())
            inv_Lambda = np.linalg.inv(Lambda)
            rot_AA = np.dot(AA, inv_Lambda)
            rot_A1 = rot_AA[:, 0:D]
            rot_A2 = rot_AA[:, D:2*D]
            rot_A = np.mean(np.stack((rot_A1, rot_A2), axis=2), axis=2)
            self.item_parameters["discrimination_matrix"] = rot_A
        return(sigma_psi)

    # def icc(self, theta, s, A=np.empty(0), delta=np.empty(0), cross=True):
    #     if not cross:
    #         icc_values = super().icc(theta=theta+s, A=A, delta=delta)
    #     else:
    #         competency = np.tile(s, (theta.shape[0], 1))
    #         competency = competency.reshape(
    #             (s.shape[0], theta.shape[0], self.latent_dimension))  # TODO: Müsste nicht theta.shape[0] vorne stehen?
    #         #competency = np.expand_dims(theta, axis=1)
    #         competency = np.add(competency, theta)
    #         competency = competency.reshape(
    #             s.shape[0]*theta.shape[0], self.latent_dimension)
    #         icc_values = super().icc(theta=competency, A=A, delta=delta)
    #         icc_values = icc_values.reshape(
    #             (s.shape[0], theta.shape[0], self.item_dimension))
    #     return(icc_values)

    def icc(self, theta, s, A=np.empty(0), delta=np.empty(0), cross=True, save=False):
        if not cross:
            icc_values = super().icc(theta=theta+s, A=A, delta=delta, save=save)
        else:
            s_tile, theta_tile = self.tile_competency_gain(s=s, theta=theta)
            competency = np.add(s_tile, theta_tile)
            competency = competency.reshape(
                s.shape[0]*theta.shape[0], self.latent_dimension)
            icc_values = super().icc(theta=competency, A=A, delta=delta, save=save)
            icc_values = icc_values.reshape(
                (theta.shape[0], s.shape[0], A.shape[0]))
        return(icc_values)

    def sample_gain(self, sample_size=1, qmc=False):
        D = self.latent_dimension
        cov = self.person_parameters["covariance"][D:2*D, D:2*D]
        mu = self.person_parameters["mean"][D:2*D]
        if not qmc:
            sample = multivariate_normal.rvs(
                size=sample_size, mean=mu, cov=cov)
            # if self.latent_dimension == 1:
            #    sample = np.expand_dims(sample, axis=1)
        else:
            sample = MultivariateNormalQMC(
                mean=mu, cov=cov, engine=None).random(sample_size)
        # Ensure correct dimensionality if sample_siz==1 or latent_dim==1
        sample = sample.reshape((sample_size, self.latent_dimension))
        return(sample)

    def get_cov(self, type="full"):
        D = self.latent_dimension
        sigma_psi = self.person_parameters["covariance"]
        if type == "full":
            return(sigma_psi)
        elif type == "early":
            return(sigma_psi[0:D, 0:D])
        elif type == "late":
            return(sigma_psi[D:2*D, D:2*D])
        elif type == "corr":
            return(sigma_psi[0:D, D:2*D])
        elif type == "early_conditional":
            return(self.person_parameters["early_conditional_covariance"])
        elif type == "late_conditional":
            return(self.person_parameters["late_conditional_covariance"])
        elif type == "full_cross":
            return(self.person_parameters["covariance"])
        elif type == "convolution":
            convolution_matrix = np.concatenate(
                (np.identity(D), np.identity(D)), axis=1)
            convolution_covariance = np.dot(convolution_matrix, sigma_psi)
            convolution_covariance = np.dot(
                convolution_covariance, convolution_matrix.transpose())
            return(convolution_covariance)
        else:
            raise Exception("Covariance type does not exist")

    def get_mean(self, type="full"):
        mu = self.person_parameters["mean"]
        D = self.latent_dimension
        if type == "full":
            return(mu)
        elif type == "early":
            return(mu[0:D])
        elif type == "late":
            return(mu[D:2*D])

    def latent_density(self, type="full", theta: np.array = np.empty(0), s: np.array = np.empty(0),
                       sigma: np.array = np.empty(0), mu: np.array = np.empty(0), save=False):
        D = self.latent_dimension
        if sigma.size == 0:
            sigma = self.get_cov(type)
        if type in ["full", "early", "late"]:
            mu = self.get_mean(type)
            if type == "full":
                x = np.concatenate((theta, s), axis=0)
            elif type == "early":
                x = theta
            elif type == "late":
                x = s
        elif type == "early_conditional":
            if s.size == 0:
                raise Exception("Gain not specified")
            mu = self.get_mean("early") + np.dot(
                self.person_parameters["early_conditional_mean_predictor"], (s-self.get_mean("late")).transpose()).transpose()
            #mu = mu.transpose()
            x = theta
        elif type == "late_conditional":
            raise Exception("Not implemented yet")
        elif type == "full_cross":
            mu = self.get_mean("full")
            s_tile, theta_tile = self.tile_competency_gain(s=s, theta=theta)
            x = np.concatenate((s_tile, theta_tile), axis=2).reshape(
                (s.shape[0]*theta.shape[0], 2*D))
            density = super().latent_density(x, sigma, mu, save=save).reshape(
                (s.shape[0], theta.shape[0]))
            return(density)
        else:
            raise Exception("density type not known")
        return super().latent_density(x, sigma, mu, save=save)

    def response_matrix_probability(self, s, theta, response_matrix: np.array, A=np.empty(0), delta=np.empty(0)) -> np.array:
        """Calculate a matrix of response-vector probabilities. One entry in the resulting matrix
        reflects the response vector probability for response-vector i for a person with competency j

        Args:
            theta (np.array): competency matrix of shape (number_response_shapes, latent_dimension)
            s (np.array): competency-gain matrix of shape (competency_gain_sample_size, latent_dimension)
            response_matrix (np.array): response-matrix of shape (number_response_shapes, item_dimension)
            A (_type_, optional): _description_. Defaults to np.empty(0).
            delta (_type_, optional): _description_. Defaults to np.empty(0).

        Returns:
            np.array: _description_
        """
        if A.size == 0:
            A = self.item_parameters["discrimination_matrix"]
        if delta.size == 0:
            delta = self.item_parameters["intercept_vector"]
        if theta.shape[0] != response_matrix.shape[0]:
            raise Exception(
                "Prior ability and Response matrix don't have same length")
        correct_response_probabilities = self.icc(
            theta=theta, s=s, A=A, delta=delta, cross=True)  # TODO: Hier anpassen an tiling!
        response_matrix = np.expand_dims(response_matrix, axis=1)
        probability_vector = np.add(np.multiply(correct_response_probabilities, response_matrix),
                                    np.multiply(np.subtract(1, correct_response_probabilities), np.subtract(1, response_matrix)))
        probability = np.prod(probability_vector, axis=2)
        # TODO: Log-option
        return(probability.transpose())

    # TODO: Immer ein tiling verwenden und Methoden mit tiling anstelle von cross definieren, sicherer!
    def tile_competency_gain(self, s, theta):
        s_tile = np.tile(s, reps=(theta.shape[0], 1))
        s_tile = s_tile.reshape(
            (theta.shape[0], s.shape[0], self.latent_dimension))
        theta_tile = np.tile(theta, (1, 1, s.shape[0])).reshape(
            (theta.shape[0], s.shape[0], theta.shape[1]))
        return(s_tile, theta_tile)

    def realize_interpretation(interpretation="lower gain bound"):
        pass

    def marginal_response_loglikelihood(self, response_data, theta, N=1000):
        s_sample = self.sample_gain(N, qmc=False)
        response_matrix_prob = self.response_matrix_probability(
            theta=theta, s=s_sample, response_matrix=response_data)
        conditional_competency_density = self.latent_density(
            type="early_conditional", s=s_sample, theta=theta)
        # Integration for each data-element
        response_matrix_prob = np.mean(np.multiply(
            response_matrix_prob, conditional_competency_density), axis=0)
        log_marginal_vector_probabilities = np.log(response_matrix_prob)
        # sum over data-elements
        response_loglikelihood = np.sum(log_marginal_vector_probabilities)
        return(response_loglikelihood)

    def answer_log_likelihood(self, s, theta, answer_vector):
        # TODO: check why so many values are returned!
        ICC_values = self.icc(theta=np.expand_dims(
            theta, axis=0), s=np.expand_dims(s, axis=0), cross=False)[0]
        latent_density = self.latent_density(
            theta=theta, s=s, type="full", save=True)
        # log_likelihood = np.dot(answer_vector, np.log(ICC_values + np.float64(1.7976931348623157e-309))[0]) + np.dot(
        #     (1-answer_vector), np.log(1-ICC_values + np.float64(1.7976931348623157e-309))[0]) + np.log(latent_density)
        log_likelihood = np.dot(answer_vector, np.log(ICC_values + np.float64(1.7976931348623157e-309))) + np.dot(
            (1-answer_vector), np.log(1-ICC_values + np.float64(1.7976931348623157e-309))) + np.log(latent_density)
        return(log_likelihood)

    def predict_gain(self, response_data: pd.DataFrame = pd.DataFrame(), theta: pd.DataFrame = pd.DataFrame()) -> np.array:
        """Given the estimated item-parameters for a MIRT-Model and some response_data, this function will estimate the latent ability for every respondent.

        Args:
            response_data (pd.DataFrame or np.array): Response data from Item's 
        """
        if response_data.size > 0:
            gain_matrix = np.zeros(
                shape=(response_data.shape[0], self.latent_dimension))
            for i, response_pattern in enumerate(response_data.to_numpy()):
                def nll(x): return -1*self.answer_log_likelihood(s=x,
                                                                 theta=theta.to_numpy()[
                                                                     i],
                                                                 answer_vector=response_pattern)
                x0 = self.sample_gain()
                res = minimize(nll, x0=x0, method='BFGS')
                gain_matrix[i] = res.x
        else:
            D = self.latent_dimension
            late_conditional_covariance = self.person_parameters["late_conditional_covariance"]
            early_sigma = self.person_parameters["covariance"][0:D, 0:D]
            inv_early_sigma = np.linalg.inv(early_sigma)
            psi = self.person_parameters["covariance"][0:D, D: 2*D]
            mu = self.person_parameters["mean"]
            gain_matrix = np.expand_dims(mu[D:2*D], axis=1) + np.dot(np.dot(psi, inv_early_sigma),
                                                                     theta.to_numpy().transpose())
            gain_matrix = gain_matrix.transpose()
        return(gain_matrix)
