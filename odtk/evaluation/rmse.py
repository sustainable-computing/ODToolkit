#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class RMSE(OccupancyEvaluation):
    """
    Calculate the `Root Mean Square Error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_
    between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: RMSE score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        from numpy import sqrt

        return sqrt(((self.truth - self.predict) ** 2).mean())
