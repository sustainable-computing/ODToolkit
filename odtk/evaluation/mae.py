#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class MAE(OccupancyEvaluation):
    """
    Calculate the `Mean Absolute Error <https://en.wikipedia.org/wiki/Mean_absolute_error>`_
    between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: MAE score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        return abs(self.truth - self.predict).mean()
