#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class SVM(NormalModel):
    """
    Using `Support Vector Machine <https://scikit-learn.org/stable/modules/generated/
    sklearn.svm.LinearSVC.html>`_ model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :parameter gamma: kernel coefficient for ``'rbf'``, ``'poly'``, or ``'sigmoid'``
    :type gamma: float or ``'auto'``

    :parameter kernel: specifies the kernel type to be used in the algorithm.
                       It must be one of ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``, ``'precomputed'``
    :type kernel: str

    :parameter penalty_error: penalty parameter C of the error term.
    :type penalty_error: float

    :parameter n_estimators: estimators used for predictions.
    :type n_estimators: int

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self, train, test):
        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.gamma = 'auto'
        self.kernel = 'linear'
        self.penalty_error = 1
        self.n_estimators = 10

    # the model must have a method called run, and return the predicted result
    def run(self):
        from sklearn.svm import SVC
        from sklearn.ensemble import BaggingClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from numpy import reshape

        if len(self.train.occupancy.shape) == 2 and self.train.occupancy.shape[1] == 1:
            self.train.change_occupancy(reshape(self.train.occupancy, (self.train.occupancy.shape[0],)))

        classifier = OneVsRestClassifier(BaggingClassifier(
            SVC(kernel=self.kernel, C=self.penalty_error, gamma=self.gamma, verbose=True),
            max_samples=1.0 / self.n_estimators, n_estimators=self.n_estimators, bootstrap=False))

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        predict_occupancy.shape += (1,)

        return predict_occupancy


class SVR(NormalModel):
    """
    Using `Support Vector Regression <https://scikit-learn.org/stable/modules/generated/
    sklearn.svm.LinearSVR.html>`_ model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :parameter gamma: kernel coefficient for ``'rbf'``, ``'poly'``, or ``'sigmoid'``
    :type gamma: float or ``'auto'``

    :parameter kernel: specifies the kernel type to be used in the algorithm.
                       It must be one of ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``, ``'precomputed'``
    :type kernel: str

    :parameter penalty_error: penalty parameter C of the error term.
    :type penalty_error: float

    :parameter n_estimators: estimators used for predictions.
    :type n_estimators: int

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self, train, test):
        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.gamma = 'auto'
        self.kernel = 'linear'
        self.penalty_error = 1
        self.n_estimators = 10

    # the model must have a method called run, and return the predicted result
    def run(self):
        from sklearn.svm import SVR
        from sklearn.ensemble import BaggingRegressor
        from sklearn.multiclass import OneVsRestClassifier
        from numpy import reshape

        if len(self.train.occupancy.shape) == 2 and self.train.occupancy.shape[1] == 1:
            self.train.change_occupancy(reshape(self.train.occupancy, (self.train.occupancy.shape[0],)))

        classifier = OneVsRestClassifier(BaggingRegressor(
            SVR(kernel=self.kernel, C=self.penalty_error, gamma=self.gamma, verbose=True),
            max_samples=1.0 / self.n_estimators, n_estimators=self.n_estimators, bootstrap=False))

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        predict_occupancy.shape += (1,)

        return predict_occupancy
