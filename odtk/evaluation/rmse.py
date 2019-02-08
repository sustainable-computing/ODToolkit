from odtk.evaluation.superclass import *


class RMSE(OccupancyEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        from numpy import sqrt

        return sqrt(((self.truth - self.predict) ** 2).mean())
