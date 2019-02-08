from odtk.evaluation.superclass import *


class NRMSE(OccupancyEvaluation):
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
