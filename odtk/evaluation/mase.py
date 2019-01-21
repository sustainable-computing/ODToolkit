from odtk.evaluation.superclass import *
from numpy import sqrt


class MASE(OccupancyEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        mae = abs(self.truth - self.predict).mean()
        denominator = self.truth.shape[0] / (self.truth.shape[0] - 1)
        denominator *= abs(self.truth[1:] - self.truth[:-1]).mean()
        return mae / denominator
