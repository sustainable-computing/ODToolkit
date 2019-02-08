#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class MASE(OccupancyEvaluation):
    """
    Calculate the `Mean Absolute Scaled Error <https://en.wikipedia.org/wiki/Mean_absolute_scaled_error>`_
    between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: MASE score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        mae = abs(self.truth - self.predict).mean()
        denominator = self.truth.shape[0] / (self.truth.shape[0] - 1)
        denominator *= abs(self.truth[1:] - self.truth[:-1]).mean()
        return mae / denominator
