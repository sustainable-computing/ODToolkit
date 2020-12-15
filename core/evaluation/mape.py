#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class MAPE(OccupancyEvaluation):
    """
    Calculate the `Mean Absolute Percentage Error <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_
    between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: MAPE score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        return abs(1 - self.predict[occupied_index] / self.truth[occupied_index]).mean()
