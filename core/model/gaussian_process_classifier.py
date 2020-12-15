#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class GPC(NormalModel):
    """
    Using `Gaussian Process Classifier <https://scikit-learn.org/stable/modules/generated/
    sklearn.gaussian_process.GaussianProcessClassifier.html>`_ model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self,
                 train,
                 test):

        self.train = train
        self.test = test

    def run(self):
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels \
            import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

        X = np.array(self.train.data)
        Y = np.array(self.train.occupancy).flatten()
        kernel = RBF() + RBF() * ExpSineSquared() + RationalQuadratic() + WhiteKernel()
        gp = GaussianProcessClassifier(kernel=kernel,
                                       optimizer='fmin_l_bfgs_b').fit(X, Y)

        predict_occupancy = gp.predict(np.array(self.test.data))
        return np.reshape(predict_occupancy, (-1, 1))
