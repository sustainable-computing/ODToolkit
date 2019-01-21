from odtk.evaluation.superclass import *


class TruePositive(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        return (self.predict[occupied_index] > 0).sum()


class TrueNegative(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        return (self.predict[unoccupied_index] == 0).sum()


class FalsePositive(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        return (self.predict[unoccupied_index] > 0).sum()


class FalseNegative(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        return (self.predict[occupied_index] == 0).sum()


class Recall(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        tp = (self.predict[occupied_index] > 0).sum()
        fn = (self.predict[occupied_index] == 0).sum()
        return tp / (tp + fn)


class Fallout(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        tn = (self.predict[unoccupied_index] == 0).sum()
        fp = (self.predict[unoccupied_index] > 0).sum()
        return fp / (fp + tn)


class Missrate(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        tp = (self.predict[occupied_index] > 0).sum()
        fn = (self.predict[occupied_index] == 0).sum()
        return fn / (tp + fn)


class Selectivity(BinaryEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        unoccupied_index = self.truth < 0.5
        tn = (self.predict[unoccupied_index] == 0).sum()
        fp = (self.predict[unoccupied_index] > 0).sum()
        return tn / (fp + tn)


class Precision(BinaryEvaluation):
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
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        tn = (self.predict[~occupied_index] == 0).sum()
        tp = (abs(self.predict[occupied_index] - self.truth[occupied_index]) <= 0).sum()
        return (tn + tp) / self.truth.shape[0]


class AccuracyTolerance(OccupancyEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth
        self.tolerance = 0

    def run(self):
        occupied_index = self.truth > 0
        tn = (self.predict[~occupied_index] == 0).sum()
        tp = (abs(self.predict[occupied_index] - self.truth[occupied_index]) <= self.tolerance).sum()
        return (tn + tp) / self.truth.shape[0]
