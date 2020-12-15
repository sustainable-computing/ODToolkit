#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class RandomForest(NormalModel):
    """
    Using `Random Forest <https://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.RandomForestClassifier.html>`_ model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :parameter estimator: number of estimators in Random Forest
    :type estimator: int

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self, train, test):
        from numpy import reshape

        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.estimator = 200
        if len(self.train.occupancy.shape) == 2 and self.train.occupancy.shape[1] == 1:
            self.train.change_occupancy(reshape(self.train.occupancy, (self.train.occupancy.shape[0],)))

    # the model must have a method called run, and return the predicted result
    def run(self):
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators=self.estimator)

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        if len(predict_occupancy.shape) == 1:
            predict_occupancy.shape += (1,)

        return predict_occupancy
