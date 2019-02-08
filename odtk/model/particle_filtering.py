from odtk.model.superclass import *


class PF(NormalModel):

    def __init__(self,
                 train,
                 test):
        from numpy import amax

        self.train = train
        self.test = test
        self.number_of_hidden_states = int(amax(train.occupancy)) + 1

    def run(self):
        from . import hmm_core
        from numpy import reshape

        hmm = hmm_core.HMM_Core(number_of_hidden_states=self.number_of_hidden_states)
        hmm.learn(hidden_seq=np.array(self.train.occupancy, int).flatten(),
                  emission_seq=self.train.data)

        predict_occupancy = hmm.pf_predict(emission_seq=np.array(self.test.data))

        return np.reshape(predict_occupancy, (-1, 1))


class PF_DA(DomainAdaptiveModel):

    def __init__(self,
                 source,
                 target_retrain,
                 target_test):
        from numpy import amax

        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.number_of_hidden_states = int(amax(source.occupancy)) + 1

    def run(self):
        from . import hmm_core
        from numpy import reshape

        hmm_source = hmm_core.HMM_Core(number_of_hidden_states=self.number_of_hidden_states)

        hmm_source.learn(hidden_seq=np.array(self.source.occupancy, int).flatten(),
                         emission_seq=self.source.data)

        hmm_target = hmm_core.HMM_Core(number_of_hidden_states=self.number_of_hidden_states)

        hmm_target.prior(hmm_source)

        hmm_target.learn(hidden_seq=np.array(self.target_retrain.occupancy, int).flatten(),
                         emission_seq=self.target_retrain.data)

        predict_occupancy = hmm_target.pf_predict(emission_seq=np.array(self.target_test.data))

        return reshape(predict_occupancy, (-1, 1))
