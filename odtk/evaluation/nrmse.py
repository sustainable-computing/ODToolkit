#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class NRMSE(OccupancyEvaluation):
    """
    Calculate the `Normalized Root Mean Square Error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_
    between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :parameter mode: the mode of nRMSE. Can select ``'minmax'`` or ``'mean'``
    :type mode: str

    :rtype: float
    :return: nRMSE score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth
        self.mode = "minmax"

    def run(self):
        from numpy import sqrt

        if self.mode == "minmax":
            return sqrt(((self.truth - self.predict) ** 2).mean()) / (self.truth.max() - self.truth.min())
        elif self.mode == "mean":
            return sqrt(((self.truth - self.predict) ** 2).mean()) / self.truth.mean()
