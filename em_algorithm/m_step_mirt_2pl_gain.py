from hashlib import new
from m_step import m_step
from m_step_mirt_2pl import m_step_ga_mml
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
import random
import sys
import os
import cma
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl_gain import mirt_2pl_gain


class m_step_ga_mml_gain(m_step_ga_mml):

    def __init__(self, model: mirt_2pl_gain, sigma_constraint) -> None:
        super().__init__(model)
        self.sigma_constraint = sigma_constraint

    def q_0(self, pe_functions):
        D = self.model.latent_dimension
        if self.sigma_constraint == "early_constraint":
            type = "only_late"
        elif self.sigma_constraint == "unconstrained":
            type = "full"
        elif self.sigma_constraint == "esigma_spsi":
            type = "only_psi_off_diag"
        else:
            raise Exception("Sigma constraint not known")

        def func(corr_vector):
            sigma_psi = self.model.corr_to_sigma(corr_vector, type=type)
            sigma_psi[0:D, 0:D] = self.model.person_parameters["covariance"][0:D, 0:D]
            return pe_functions["q_0"](np.reshape(
                sigma_psi, newshape=(
                    2*self.model.latent_dimension, 2*self.model.latent_dimension)))
        return(func)

    def q_0_cholesky(self, pe_functions):
        def func(cholesky_sigma_psi_vector):
            D = self.model.latent_dimension
            cholesky_sigma = np.identity(2*D)
            cholesky_sigma[np.tril_indices_from(
                cholesky_sigma)] = cholesky_sigma_psi_vector
            sigma = np.dot(cholesky_sigma, cholesky_sigma.transpose())
            return pe_functions["q_0"](np.reshape(
                sigma, newshape=(
                    2*D, 2*D)))
        return(func)

    def q_0_gradient_cholesky(self, pe_functions):
        def func(cholesky_sigma_psi_vector):
            D = self.model.latent_dimension
            cholesky_sigma_psi = np.identity(2*D)
            cholesky_sigma_psi[np.tril_indices_from(
                cholesky_sigma_psi)] = cholesky_sigma_psi_vector
            sigma_psi = np.dot(cholesky_sigma_psi,
                               cholesky_sigma_psi.transpose())
            if self.sigma_constraint == "early_constraint":
                sigma_psi[0:D, 0:D] = self.model.person_parameters["covariance"][0:D, 0:D]
            # Apply chain rule

            def outer_func(C_vector):
                C = C_vector.reshape((2*D, 2*D))
                sigma_psi = np.dot(C, C.transpose())
                # TODO: Is this correct?
                #sigma_psi[0:D, 0:D] = self.model.person_parameters["covariance"][0:D, 0:D]
                return(sigma_psi.flatten())
            chain2 = approx_fprime(f=outer_func, xk=cholesky_sigma_psi.flatten(),
                                   epsilon=1.4901161193847656e-20).reshape(((2*D)**2, 2*D, 2*D))
            q_0_grad = pe_functions["q_0_grad"](sigma_psi)
            #q_0_grad[0:D, 0:D] = 0
            gradient = np.sum(np.sum(np.multiply(
                q_0_grad, chain2), axis=1), axis=1).reshape((2*D, 2*D))
            gradient = gradient[np.tril_indices_from(gradient)]
            return(gradient.flatten())
        return(func)
