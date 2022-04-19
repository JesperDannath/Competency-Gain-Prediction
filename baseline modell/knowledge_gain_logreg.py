""" This is an implementation for a logistic regression model which
predicts the probability of a correct response in a post-test based
on the knowledge in the pre-test and a predicted knowledge gain, where
the gain is a linear function of pre-test knowledge and auxiliary
features. """

# Knowledge Gain Logistic Model
# Copyright (C) 2021 Benjamin Paaßen
# German Research Center for Artificial Intelligence (DFKI)
#
# This implementation is currently for internal use only.
# This is a special copy for Jesper Dannath's Master's thesis.

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from scipy.optimize import minimize
from qirt import QIRT
from closed_form_vae_irt import VAE_IRT

class RBF(BaseEstimator, TransformerMixin):
    """ Sets up a set of radial basis functions over a given
    range. The equation of the kth radial basis function is:

    f_k(x) = exp(-psi * (x - c[k]) ** 2)

    where c[k] is the 'center' of the kth basis function, meaning
    the value where f_k(x) = 1.

    Parameters
    ----------
    lo: float
        The minimum value that is expected as input.
    hi: float
        The maximum value that is expected as input.
    num_basis_functions: int
        The number of radial basis functions. These will be
        linearly spaced over the given range.
    sigma: float (default = 1.)
        The number of inter-center gaps before the basis function value
        shrinks to 1/e.

    Attributes
    ----------
    c_: ndarray
        The locations of the RBF centers.
    psi_: float
        The constant in the exponential. This is derived from lo, hi, and sigma.

    """
    def __init__(self, lo, hi, num_basis_functions, sigma = 1.):
        self.lo = lo
        self.hi = hi
        self.num_basis_functions = num_basis_functions
        self.sigma = sigma

    def fit(self, X = None, Y = None):
        """ Fits this set of RBF functions. No data necessary. """
        if self.num_basis_functions <= 1:
            self.c_ = np.array([0.5 * (self.lo + self.hi)])
            delta   = 0.5 * (self.hi - self.lo)
        else:
            self.c_   = np.linspace(self.lo, self.hi, self.num_basis_functions)
            delta     = (self.hi - self.lo) / (self.num_basis_functions - 1)
        self.psi_ = 1. / (delta * self.sigma) ** 2
        return self

    def transform(self, X):
        """ Applies the radial basis functions to all dimensions of the given input matrix.

        Parameters
        ----------
        X: ndarray
            An m x n matrix of input values.

        Returns
        -------
        Phi: ndarray
            An m x n x self.num_basis_functions tensor of RBF values, representing the given input.

        """
        cexpand = self.c_
        for dim in range(len(X.shape)):
            cexpand = np.expand_dims(cexpand, 0)
        Phi = np.expand_dims(X, -1) - cexpand
        Phi = np.exp(-self.psi_ * np.square(Phi))
        return Phi


class Sigmoids(BaseEstimator, TransformerMixin):
    """ Sets up a set of sigmoid basis functions over a given
    range. The equation of the kth basis function is:

    f_k(x) = 1 / (1 + exp(-beta * (x - c[k])))

    where c[k] is the 'center' of the kth basis function, meaning
    the value where f_k(x) = .5. A special case is the first
    basis function, which is constant 1 over the entire range.

    Parameters
    ----------
    lo: float
        The minimum value that is expected as input.
    hi: float
        The maximum value that is expected as input.
    num_basis_functions: int
        The number of radial basis functions. These will be
        linearly spaced over the given range.
    sigma: float (default = 1.)
        The number of inter-center gaps before the basis function value
        shrinks to 1/(e + 1).

    Attributes
    ----------
    c_: ndarray
        The locations of the RBF centers. Note that these are num_basis_functions -1
        because the first basis function is constant 1.
    beta_: float
        The constant in the exponential. This is derived from lo, hi, and sigma.

    """
    def __init__(self, lo, hi, num_basis_functions, sigma = 1.):
        self.lo = lo
        self.hi = hi
        self.num_basis_functions = num_basis_functions
        self.sigma = sigma

    def fit(self, X = None, Y = None):
        """ Fits this set of sigmoid functions. No data necessary. """
        if self.num_basis_functions <= 1:
            self.c_ = np.array([])
            delta   = self.hi - self.lo
        if self.num_basis_functions == 2:
            self.c_ = np.array([0.5 * (self.lo + self.hi)])
            delta   = 0.5 * (self.hi - self.lo)
        else:
            self.c_   = np.linspace(self.lo, self.hi, self.num_basis_functions - 1)
            delta     = (self.hi - self.lo) / (self.num_basis_functions - 2)
        self.beta_ = 1. / (delta * self.sigma)
        return self

    def transform(self, X):
        """ Applies the radial basis functions to all dimensions of the given input matrix.

        Parameters
        ----------
        X: ndarray
            An m x n matrix of input values.

        Returns
        -------
        Phi: ndarray
            An m x n x self.num_basis_functions tensor of RBF values, representing the given input.

        """
        if self.num_basis_functions <= 1:
            ones = np.ones_like(X)
            ones = np.expand_dims(ones, -1)
            return ones

        cexpand = self.c_
        for dim in range(len(X.shape)):
            cexpand = np.expand_dims(cexpand, 0)
        Phi = np.expand_dims(X, -1) - cexpand
        Phi = 1. / (1. + np.exp(-self.beta_ * Phi))
        # attach a constant 1 dimension
        ones = np.ones_like(X)
        Phi = np.concatenate((np.expand_dims(ones, -1), Phi), -1)
        return Phi


class QGainIRT(BaseEstimator):
    """ A gain item response theory model where we model
    pre-test knowledge and knowledge gain during a course.
    The pre-test knowledge is modelled by multiplying the
    Q matrix with the response matrix. The knowledge gain
    is modelled via a generalized, non-negative linear regression
    over an RBF model of the pre-test knowledge and of
    auxiliary features. Difficulties of pre-test and post-test
    knowledge are adjusted to have equal means.


    Parameters
    ----------
    Qpre: ndarray
        A matrix with one row per pre-test question and one
        column per knowledge component. Q[j, k] = 1 means that
        question j requires knowledge component k. Otherwise,
        Q[j, k] should be zero.
    Qpost: ndarray
        A matrix with one row per post-test question and one
        column per knowledge component. Q[j, k] = 1 means that
        question j requires knowledge component k. Otherwise,
        Q[j, k] should be zero.
    L: int (default = 16)
        The number of RBF kernels to describe the non-linear
        relationship between pre-test knowlegde and knowledge
        gain. The kernel centers are linearly spaced across
        the range of pre-test knowledge.
    sigma: float (default = 1.)
        The width of each RBF kernel, expressed in units of
        inter-center ranges.
    C: float (default = 1.)
        The inverse L2 regularization applied during fitting
        the QIRT models for pre- and post-test.
    regul: float (default = 1E-3)
        The L2 regularization applied for fitting the gain
        prediction model.
    match_mean_difficulties: bool (default = False)
        If set to True, the average post-test difficulty is matched
        to the average pre-test difficulty before training the
        knowledge gain model. Note that this may make the gain
        estimation more accurate but also makes the q_post_
        potentially disfunctional on its own.
    aux_bounds: list (default = None)
        A list of bounds to be used for the auxiliary features (if given).
        Each bound is a pair of a lower and an upper bound (inf values are permitted).
        If not provided, -inf to +inf is used.

    Attributes
    ----------
    qirt_pre_: class QIRT
        A QIRT model for the pre-test.
    qirt_post_: class QIRT
        A QIRT model for the post-test.
    rbf_: class RBF
        The RBF functions to represent the pre-test knowledge.
    W_: ndarray
        The RBF weights for knowledge gain prediction.
    U_: ndarray
        The auxiliary feature weights for knowlegde gain
        prediction.

    """
    def __init__(self, Qpre, Qpost, L = 16, sigma = 1., C = 1., regul = 1E-3, match_mean_difficulties = False, aux_bounds = None):
        self.Qpre  = Qpre
        self.Qpost = Qpost
        self.L     = L
        self.sigma = sigma
        self.C     = C
        self.regul = regul
        self.match_mean_difficulties = match_mean_difficulties
        self.aux_bounds = aux_bounds

    def fit(self, Xpre, Xpost, Xaux = None):
        """ Fits this model to the given data.

        Parameters
        ----------
        Xpre: ndarray
            A pre-test response matrix with Xpre[i, j] = 1 if
            student i answered question j correctly and Xpre[i, j] = 0,
            otherwise.
        Xpost: ndarray
            A post-test response matrix with Xpost[i, j] = 1 if
            student i answered question j correctly and Xpost[i, j] = 0,
            otherwise.
        Xaux: ndarray (default = None)
            An auxiliary feature tensor where Xaux[i, k, l] is the lth
            feature value for the kth skill of student i. If all knowledge
            components have the same feature, just repeat the features K times.

        Returns
        -------
        self

        """
        # check input
        m, npre  = Xpre.shape
        _, npost = Xpost.shape
        if Xpost.shape[0] != m:
            raise ValueError('There were %d post-test students but %d pre-test students.' % (Xpost.shape[0], m))
        if npre != self.Qpre.shape[0]:
            raise ValueError('There were %d items in the pre-test but %d rows in the Qpre matrix.' % (npre, self.Qpre.shape[0]))
        if npost != self.Qpost.shape[0]:
            raise ValueError('There were %d items in the post-test but %d rows in the Qpost matrix.' % (npost, self.Qpost.shape[0]))

        K = self.Qpre.shape[1]
        if K != self.Qpost.shape[1]:
            raise ValueError('The Qpre matrix had %d columns but the Qpost matrix had %d columns.' % (K, self.Qpost.shape[1]))

        if Xaux is not None:
            if Xaux.shape[0] != m:
                raise ValueError('Auxiliary feature matrix had %d rows but the response matrices had %d rows.' % (Xaux.shape[0], m))
            if Xaux.shape[1] != K:
                raise ValueError('Auxiliary feature matrix Xaux needs as many columns as there are skills.')
            naux = Xaux.shape[2]
        else:
            naux = 0

        if self.aux_bounds is not None:
            if len(self.aux_bounds) != K * naux:
                raise ValueError('expected %d aux bounds but got %d' % (K * naux, len(self.aux_bounds)))

        # train QIRT models for both pre- and post-test.
        self.qirt_pre_  = QIRT(self.Qpre, self.C)
        self.qirt_pre_.fit(Xpre)
        self.qirt_post_ = QIRT(self.Qpost, self.C)
        self.qirt_post_.fit(Xpost)

        # if so desired, adjust the difficulties in the post model
        if self.match_mean_difficulties:
            nonzerospre  = self.qirt_pre_.a_ > 1E-3
            nonzerospost = self.qirt_post_.a_ > 1E-3
            bpre  = self.qirt_pre_.difficulties()[nonzerospre]
            bpost = self.qirt_post_.difficulties()[nonzerospost]
            self.qirt_post_.b_[nonzerospost] = (bpost - np.mean(bpost) + np.mean(bpre)) * self.qirt_post_.a_[nonzerospost]

        # compute pre-test knowledge
        Theta     = self.qirt_pre_.encode(Xpre)

        # set up RBF model
        self.rbf_ = RBF(0., 1., self.L, self.sigma)
        self.rbf_.fit()
        Phi       = self.rbf_.transform(Theta)

        # set up objective function for optimization
        pos  = Xpost > 0.5
        neg  = Xpost < 0.5
        nans = np.isnan(Xpost)
        Qnorm = self.Qpost * np.expand_dims(1. / np.sum(self.Qpost, 1) * self.qirt_post_.a_, 1)
        if Xaux is None:
            def objective(w):
                # re-shape parameters to a matrix
                W = np.reshape(w[:K*self.L], (K, self.L))
                # compute current estimate for knowledge gain
                Gain = np.einsum('ikl,kl -> ik', Phi, W)
                # compute current estimate of post-test knowledge
                Theta_post = Theta + Gain
                # compute current estimate for probability of being correct
                P = self.qirt_post_.decode_proba(Theta_post)
                # compute objective function
                f  = -np.sum(np.log(P[pos]))-np.sum(np.log(1. - P[neg])) + 0.5 * self.regul * np.sum(np.square(W))
                # compute gradient
                Delta = P - Xpost
                Delta[nans] = 0.
                gradW = np.einsum('ij,jk,ikl->kl', Delta, Qnorm, Phi) + self.regul * W
                # ravel and return
                grad = np.ravel(gradW)
                return f, grad
        else:
            def objective(wu):
                # re-shape parameters to a matrix
                W = np.reshape(wu[:K*self.L], (K, self.L))
                U = np.reshape(wu[K*self.L:K*(self.L+naux)], (K, naux))
                # compute current estimate for knowledge gain
                Gain = np.einsum('ikl,kl -> ik', Phi, W) + np.einsum('ikl,kl -> ik', Xaux, U)
                # compute current estimate of post-test knowledge
                Theta_post = Theta + Gain
                # compute current estimate for probability of being correct
                P = self.qirt_post_.decode_proba(Theta_post)
                # compute objective function
                f  = -np.sum(np.log(P[pos])) - np.sum(np.log(1. - P[neg])) + 0.5 * self.regul * (np.sum(np.square(W)) + np.sum(np.square(U)))
                # compute gradient
                Delta = P - Xpost
                Delta[nans] = 0.
                gradW = np.einsum('ij,jk,ikl->kl', Delta, Qnorm, Phi)  + self.regul * W
                gradU = np.einsum('ij,jk,ikl->kl', Delta, Qnorm, Xaux) + self.regul * U
                # ravel and return
                grad = np.concatenate((np.ravel(gradW), np.ravel(gradU)))
                return f, grad

        # set up bounds for optimization
        bounds = [(0., np.inf)] * (K*self.L)
        if Xaux is not None:
            if self.aux_bounds is None:
                bounds = bounds + [(-np.inf, np.inf)] * (K*naux)
            else:
                bounds = bounds + self.aux_bounds
        init_params = np.zeros(K*self.L+K*naux)

        # optimize
        res = minimize(objective, init_params , jac = True, bounds = bounds)

        if res.success is False:
            raise ValueError('optimization was unsuccessful! ' + str(res))

        # extract results
        self.W_ = np.reshape(res.x[:K*self.L], (K, self.L))
        if Xaux is not None:
            self.U_ = np.reshape(res.x[K*self.L:K*(self.L+naux)], (K, naux))

        return self


    def encode(self, Xpre):
        """ Encodes pre-test knowledge.

        Parameters
        ----------
        Xpre: ndarray
            A pre-test response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
        Theta: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        """
        return self.qirt_pre_.encode(Xpre)

    def predict_gain(self, Theta, Xaux = None):
        """ Predicts the knowledge gain for the given pre-test knowledge
        and auxiliary features.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.
        Xaux: ndarray (default = None)
            An auxiliary feature tensor where Xaux[i, k, l] is the lth
            feature value for the kth skill of student i. If all knowledge
            components have the same feature, just repeat the features K times.

        Returns
        -------
        Gains: ndarray
            A matrix with one row per student and one column per knowlegde
            component, where Gains[i, k] is the predicted knowlegde gain
            for student i in knowlegde component k.

        """
        # compute RBF kernels
        Phi = self.rbf_.transform(Theta)
        # compute predicted learning gain
        Gain = np.einsum('ikl,kl -> ik', Phi, self.W_)
        if Xaux is not None:
            Gain += np.einsum('ikl,kl -> ik', Xaux, self.U_)
        return Gain


    def decode(self, Thetapost):
        """ Decodes the given post-test abilities to predictions.

        Parameters
        ----------
        Thetapost: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        Returns
        -------
        Y: ndarray
            An predicted response matrix where Y[i, j] = 1 if student i
            is predicted to answer correctly to item j and Y[i, j] = 0,
            otherwise.

        """
        return self.qirt_post_.decode(Thetapost)

    def decode_proba(self, Thetapost):
        """ Decodes the given post-test abilities to probabilities.

        Parameters
        ----------
        Thetapost: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        Returns
        -------
        P: ndarray
            A probability matrix where P[i, j] is the predicted probability
            of student i getting item j right.

        """
        return self.qirt_post_.decode_proba(Thetapost)


    def predict(self, Xpre, Xaux = None):
        """ Predicts post-test results for the given pre-test knowledge.

        Parameters
        ----------
        Xpre: ndarray
            A pre-test response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.
        Xaux: ndarray (default = None)
            An auxiliary feature tensor where Xaux[i, k, l] is the lth
            feature value for the kth skill of student i. If all knowledge
            components have the same feature, just repeat the features K times.

        Returns
        -------
        Ypost: ndarray
            An predicted response matrix where Y[i, j] = 1 if student i
            is predicted to answer correctly to item j and Y[i, j] = 0,
            otherwise.

        """
        Thetapre  = self.encode(Xpre)
        Gain      = self.predict_gain(Thetapre, Xaux)
        Thetapost = Thetapre + Gain
        return self.decode(Thetapost)

    def predict_proba(self, Xpre, Xaux = None):
        """ Predicts post-test probabilities for the given pre-test knowledge.

        Parameters
        ----------
        Xpre: ndarray
            A pre-test response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.
        Xaux: ndarray (default = None)
            An auxiliary feature tensor where Xaux[i, k, l] is the lth
            feature value for the kth skill of student i. If all knowledge
            components have the same feature, just repeat the features K times.

        Returns
        -------
        Ppost: ndarray
            A probability matrix where P[i, j] is the predicted probability
            of student i getting item j of the post-test right.

        """
        Thetapre  = self.encode(Xpre)
        Gain      = self.predict_gain(Thetapre, Xaux)
        Thetapost = Thetapre + Gain
        return self.decode_proba(Thetapost)
