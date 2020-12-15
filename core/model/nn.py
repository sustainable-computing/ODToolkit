#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class NN(NormalModel):
    """
    Using `Multi-layer Perception Classifier <https://scikit-learn.org/stable/modules/generated/
    sklearn.neural_network.MLPClassifier.html>`_ model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :parameter solver: the solver for weight optimization. Choice of ``'lbfgs'``, ``'sgd'``, or ``'adam'``
    :type solver: str

    :parameter alpha: l2 penalty (regularization term) parameter
    :type alpha: float

    :parameter batch_size: size of minibatches for stochastic optimizers
    :type batch_size: int or ``'auto'``

    :parameter activation: activation function for the hidden layer.
                           Choice of ``'identity'``, ``'logistic'``, ``'tanh'``, or ``'relu'``
    :type activation: str

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self, train, test):
        from numpy import reshape
        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.solver = 'adam'
        self.alpha = 0.0001
        self.batch_size = 'auto'
        self.activation = 'logistic'
        if len(self.train.occupancy.shape) == 2 and self.train.occupancy.shape[1] == 1:
            self.train.change_occupancy(reshape(self.train.occupancy, (self.train.occupancy.shape[0],)))

    # the model must have a method called run, and return the predicted result
    def run(self):
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(solver=self.solver,
                                   alpha=self.alpha,
                                   hidden_layer_sizes=(75,),
                                   batch_size=self.batch_size,
                                   activation=self.activation)

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        if len(predict_occupancy.shape) == 1:
            predict_occupancy.shape += (1,)

        return predict_occupancy
