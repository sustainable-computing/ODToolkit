from odtk.evaluation.superclass import *
from numpy import sqrt


class MAE(OccupancyEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        return abs(self.truth - self.predict).mean()
