#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from core.evaluation.superclass import *


class TruePositive(BinaryEvaluation):
    """
    Calculate the True-Positive between prediction and ground truth

    The True-Positive indicate the proportion of the actual occupied states that are correctly identified

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: int
    :return: number of entries that is TP
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        return (self.predict[occupied_index] > 0).sum()


class TrueNegative(BinaryEvaluation):
    """
    Calculate the True-Negative between prediction and ground truth

    The True-Negative indicate the proportion of the actual unoccupied states that are correctly identified

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: int
    :return: number of entries that is TN
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        return (self.predict[unoccupied_index] == 0).sum()


class FalsePositive(BinaryEvaluation):
    """
    Calculate the False-Positive between prediction and ground truth

    The False-Positive indicate the proportion of the actual unoccupied states that are identified as occupied

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: int
    :return: number of entries that is FP
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        return (self.predict[unoccupied_index] > 0).sum()


class FalseNegative(BinaryEvaluation):
    """
    Calculate the False-Negative between prediction and ground truth

    The False-Negative indicate the proportion of the actual occupied states that are identified as unoccupied

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: int
    :return: number of entries that is FN
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        return (self.predict[occupied_index] == 0).sum()


class Recall(BinaryEvaluation):
    """
    Calculate the Recall between prediction and ground truth

    Recall is the percentage of the true occupied states which are identified by TP/(TP+FN)

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: Recall score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        tp = (self.predict[occupied_index] > 0).sum()
        fn = (self.predict[occupied_index] == 0).sum()
        return tp / (tp + fn)


class Fallout(BinaryEvaluation):
    """
    Calculate the Fallout between prediction and ground truth

    The Fallout are identified by FP/(FP+TN)

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: Fallout score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        tn = (self.predict[unoccupied_index] == 0).sum()
        fp = (self.predict[unoccupied_index] > 0).sum()
        return fp / (fp + tn)


class Missrate(BinaryEvaluation):
    """
    Calculate the Missrate between prediction and ground truth

    The Missrate are identified by 1 - Recall

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: Missrate score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        tp = (self.predict[occupied_index] > 0).sum()
        fn = (self.predict[occupied_index] == 0).sum()
        return fn / (tp + fn)


class Selectivity(BinaryEvaluation):
    """
    Calculate the Selectivity between prediction and ground truth

    The Selectivity are identified by 1 - Fallout

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: Selectivity score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        tn = (self.predict[unoccupied_index] == 0).sum()
        fp = (self.predict[unoccupied_index] > 0).sum()
        return tn / (fp + tn)


class Precision(BinaryEvaluation):
    """
    Calculate the Precision between prediction and ground truth

    The Precision indicates the percentage of occupancy predictions which are correct by TP/(TP+FP)

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: Precision score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        unoccupied_index = ~occupied_index
        tp = (self.predict[occupied_index] > 0).sum()
        fp = (self.predict[unoccupied_index] > 0).sum()
        return tp / (tp + fp)


class F1Score(BinaryEvaluation):
    """
    Calculate the F1 Score between prediction and ground truth

    The F1 Score are identified by 2 * TP / (2 * TP + FP + FN)

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: F1 Score score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        unoccupied_index = ~occupied_index
        tp = (self.predict[occupied_index] > 0).sum()
        fn = (self.predict[occupied_index] == 0).sum()
        fp = (self.predict[unoccupied_index] > 0).sum()
        return 2 * tp / (2 * tp + fn + fp)


class Accuracy(BinaryEvaluation):
    """
    Calculate the Accuracy between prediction and ground truth

    The Accuracy are identified by percentage of correct prediction

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: float
    :return: Accuracy score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        tn = (self.predict[~occupied_index] == 0).sum()
        tp = (abs(self.predict[occupied_index] - self.truth[occupied_index]) <= 0).sum()
        return (tn + tp) / self.truth.shape[0]


class AccuracyTolerance(OccupancyEvaluation):
    """
    Calculate the AccuracyTolerance between prediction and ground truth

    The AccuracyTolerance are identified same as Accuracy, but with differences smaller than the given tolerance
    will be considered as a correct prediction

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :parameter tolerance: the maximum differences between prediction and truth to mark as correct
    :type tolerance: int

    :rtype: float
    :return: AccuracyTolerance score
    """
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth
        self.tolerance = 0

    def run(self):
        occupied_index = self.truth > 0
        tn = (self.predict[~occupied_index] == 0).sum()
        tp = (abs(self.predict[occupied_index] - self.truth[occupied_index]) <= self.tolerance).sum()
        return (tn + tp) / self.truth.shape[0]
