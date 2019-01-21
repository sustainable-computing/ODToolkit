from odtk.evaluation.superclass import *
from numpy import sqrt


class MAPE(OccupancyEvaluation):
    def __init__(self, predict, truth):
        self.predict = predict
        self.truth = truth

    def run(self):
        occupied_index = self.truth > 0
        return abs(1 - self.predict[occupied_index] / self.truth[occupied_index]).mean()
