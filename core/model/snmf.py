#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class NMF(NormalModel):
    """
    Using `Sparse Non-negative Matrix Factorization <http://membres-timc.imag.fr/Olivier.Francois/snmf/
    index.htm>`_ model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :parameter alpha: constant that multiplies the regularization terms
    :type alpha: float

    :parameter beta: constant that multiplies the regularization terms
    :type beta: float

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self,
                 train,
                 test):
        self.train = train
        self.test = test

        self.alpha = 0.9
        self.beta = 0.8

    def run(self):
        from numpy import array, reshape
        from sklearn.decomposition import non_negative_factorization
        from sklearn.ensemble import GradientBoostingRegressor

        ##########################Learning#################################
        W, H, _ = non_negative_factorization(X=self.train.data,
                                             n_components=self.train.data.shape[1],
                                             regularization='transformation',
                                             alpha=2 * self.alpha + self.beta,
                                             l1_ratio=self.beta / (2 * self.alpha + self.beta))

        Y = reshape(self.train.occupancy, (-1,))
        gblsr = GradientBoostingRegressor(loss='ls', n_estimators=500).fit(W, Y)
        ####################################################################

        #############################Prediction#############################

        W, _, _ = non_negative_factorization(X=self.test.data,
                                             H=H,
                                             n_components=self.test.data.shape[1],
                                             regularization='transformation',
                                             alpha=2 * self.alpha + self.beta,
                                             l1_ratio=self.beta / (2 * self.alpha + self.beta))
        Y = gblsr.predict(W)
        Y[Y < 0] = 0
        predict_occupancy = array(Y)
        ####################################################################

        return reshape(predict_occupancy, (-1, 1))
