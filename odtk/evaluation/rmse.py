from odtk.evaluation.superclass import *
from numpy import sqrt


class RMSE(OccupancyEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        return sqrt(((self.truth - self.predict) ** 2).mean())
