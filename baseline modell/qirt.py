""" This is an implementation of a simple Q-matrix item response theory
scheme where knowledge is represented by the rate of correct answers for
a knowledge component. """

# Copyright (C) 2021
# Benjamin Paaßen
# DFKI
# Only intended for internal use in the KIPerWeb project
# This is a special copy for Jesper Dannath's Master's thesis.

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2021, Benjamin Paaßen'
__license__ = 'Internal Use Only'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@dfki.de'

class QIRT(BaseEstimator):
    """ A simple item response theory model, where we assume
    that student knowlegde is connected to responses via the
    following two equations. First:

    Theta = np.dot(X, Q)

    where Theta is student knowledge, X are the student responses,
    and Q is a matrix coupling items to skills. Second:

    P(X[i, j] = 1) = sigma(a[j] * np.dot(Theta[i, :], Q[j, :]) - b[j])

    where sigma is the logistic function, b[j] is the difficulty
    of item j, and a[j] is the discrimination of item j.

    Because Q is assumed to be given in advance, such a model
    can be fitted to data by pre-computing

    np.dot(Theta[i, :], Q[j, :])

    for all i and j and then fitting a[j] and b[j] via a simple
    logistic regression.

    Parameters
    ----------
    Q: ndarray
        A skill matrix where Q[j, k] = 1 if task j requires skill
        k and Q[j, k] = 0, otherwise.
    C: float (default = 1.)
        The inverse regularization strength for the logistic
        regression.

    Attributes
    ----------
    p_: ndarray
        The average success rates in each column, ignoring nans.
    b_: ndarray
        A vector where b[j] is the difficulty of task j.
        Note that b must be divided by a to get the 'classic'
        difficulty value of IRT. We don't do this here for numerical
        stability in edge cases of very small a.
    a_: ndarray
        A vector where a[j] is the discrimination parameter of task j.

    """
    def __init__(self, Q, C = 1.):
        self.Q = Q
        self.C = C

    def fit(self, X):
        """ Fits difficulties and discrimination parameters
        to the given data.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
            self.

        """
        m, n = X.shape
        if self.Q.shape[0] != n:
            raise ValueError('The given response matrix had %d columns but the given Q matrix has %d rows' % (n, self.Q.shape[0]))
        K = self.Q.shape[1]

        # get item-wise success rates for pre-test
        self.p_ = np.nanmean(X, 0)

        # fill nans with average item success rate
        nans = np.isnan(X)
        nonnans = np.logical_not(nans)
        if np.any(nans):
            X = np.copy(X)
            for j in range(n):
                X[nans[:, j], j] = self.p_[j]

        # train difficulties (and discriminations) for pre-test first.
        # This is possible by computing the task-relevant knowledge and
        # training a separate logistic regression for each item.
        self.qnorm_cols = np.sum(self.Q, 0)
        self.qnorm_cols[self.qnorm_cols < 1E-3] = 1.
        self.qnorm_cols = np.expand_dims(self.qnorm_cols, 0)

        self.qnorm_rows = np.sum(self.Q, 1)
        self.qnorm_rows[self.qnorm_rows < 1E-3] = 1.
        self.qnorm_rows = np.expand_dims(self.qnorm_rows, 0)

        Theta = np.dot(X, self.Q) / self.qnorm_cols
        Z     = np.dot(Theta, self.Q.T) / self.qnorm_rows
        self.b_ = np.zeros(n)
        self.a_ = np.zeros(n)
        for j in range(n):
            # check if there are only correct or only wrong responses for one item
            Xj = np.expand_dims(Z[nonnans[:, j], j], 1)
            Yj = X[nonnans[:, j], j]
            if np.all(Yj < 0.5):
                # if all answers are wrong, set discrimination to zero and difficulty high
                self.a_[j] = 0.
                self.b_[j] = 3.
            elif np.all(Yj > 0.5):
                # if all answers are wrong, set discrimination to zero and difficulty low
                self.a_[j] = 0.
                self.b_[j] = -3.
            else:
                # otherwise, train a logistic regression
                modelj = LogisticRegression(C = self.C)
                modelj.fit(Xj, Yj)
                self.a_[j] = max(.1, modelj.coef_[0, 0])
                self.b_[j] = -modelj.intercept_[0]
        return self

    def Q(self):
        """ Returns the Q matrix stored in this model.

        Returns
        ----------
        Q: ndarray
            A skill matrix where Q[j, k] = 1 if task j requires skill
            k and Q[j, k] = 0, otherwise.

        """
        return self.Q_

    def difficulties(self):
        """ Returns the difficulties stored in this model.
        Note: These are the 'classic' IRT difficulties, meaning
        self.b_ / self.a_.

        Returns
        ----------
        b: ndarray
            A difficulty vector where b[j] is the difficulty of
            item j.

        """
        b = np.copy(self.b_)
        nonzeros = self.a_ > 1E-3
        b[nonzeros] = b[nonzeros] / self.a_[nonzeros]
        return b

    def encode(self, X):
        """ Encodes the given responses to abilities.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
        Theta: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        """
        nans = np.isnan(X)
        if np.any(nans):
            X = np.copy(X)
            for j in range(X.shape[1]):
                X[nans[:, j], j] = self.p_[j]
        Theta = np.dot(X, self.Q) / self.qnorm_cols
        return Theta

    def decode_logits(self, Theta):
        """ Decodes the given abilities to logit-probabilities of
        success on every item.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        Returns
        -------
        Z: ndarray
            A logit response matrix where Z[i, j] is the predicted
            logit probability of student i getting item j right.

        """
        Z = np.dot(Theta, self.Q.T) / self.qnorm_rows
        Z = Z * np.expand_dims(self.a_, 0) - np.expand_dims(self.b_, 0)
        return Z

    def decode(self, Theta):
        """ Decodes the given abilities to predictions.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        Returns
        -------
        Y: ndarray
            An predicted response matrix where Y[i, j] = 1 if student i
            is predicted to answer correctly to item j and Y[i, j] = 0,
            otherwise.

        """
        Y = self.decode_logits(Theta)
        Y[Y >  0.] = 1.
        Y[Y <= 0.] = 0.
        return Y

    def decode_proba(self, Theta):
        """ Decodes the given abilities to probabilities.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] corresponds to the
            ability of student i at skill k.

        Returns
        -------
        P: ndarray
            A probability matrix where P[i, j] is the predicted probability
            of student i getting item j right.

        """
        Z = self.decode_logits(Theta)
        return 1. / (1. + np.exp(-Z))

    def predict(self, X):
        """ Predicts responses from actual responses.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
        Y: ndarray
            An predicted response matrix where Y[i, j] = 1 if student i
            is predicted to answer correctly to item j and Y[i, j] = 0,
            otherwise.

        """
        Theta = self.encode(X)
        return self.decode(Theta)

    def predict_proba(self, X):
        """ Predicts probabilities from actual responses.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i responded
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
        P: ndarray
            A probability matrix where P[i, j] is the predicted probability
            of student i getting item j right.

        """
        Theta = self.encode(X)
        return self.decode_proba(Theta)
