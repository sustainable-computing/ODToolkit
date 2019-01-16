from odtk.model.superclass import *
from seasonal import fit_seasons, adjust_seasons

class SDHOC(NormalModel):
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.tl = 1

    def run(self):
        pass
