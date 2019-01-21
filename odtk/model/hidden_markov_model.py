from odtk.model import hmm_core
from odtk.model.superclass import *
from odtk.modifier import change


class HMM(NormalModel):

    def __init__(self,
                 train,
                 test,

                 header=0,
                 emission_model='gaussian',
                 emission_params={}):

        self.train = train
        self.test = test
        
        self.__emission_params = {}
        if emission_model == 'gaussian':
            self.__emission_model = hmm_core.Gaussian
        if emission_model == 'gamma':
            self.__emission_model = hmm_core.Gamma
        if emission_model == 'categorical':
            self.__emission_model = hmm_core.Categorical
            self.__emission_params = {'K':2}
            
        if isinstance(header, int):
            self.feature_col = header
        elif isinstance(header, str):
            self.feature_col = train.header[header]
        else:
            raise ValueError("The type of header is not int or str")
            
    def run(self):
    
        hmm = hmm_core.HMM_Core(number_of_hidden_states=2, emission_model=self.__emission_model, emission_params=self.__emission_params)

        change.change_to_binary(self.train)

        hmm.learn(hidden_seq=np.array(self.train.occupancy[:, 0], int), 
                  emission_seq=self.train.data[:, self.feature_col])

        change.change_to_binary(self.test)

        predict_occupancy = hmm.viterbi_predict(emission_seq=self.test.data[:, self.feature_col])

        return predict_occupancy
        
class HMM_DA(DomainAdaptiveModel):

    def __init__(self,
                 source,
                 target_retrain,
                 target_test,

                 header=0,
                 emission_model='gaussian',
                 emission_params={}):

        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        
        self.__emission_params = {}
        if emission_model == 'gaussian':
            self.__emission_model = hmm_core.Gaussian
        if emission_model == 'gamma':
            self.__emission_model = hmm_core.Gamma
        if emission_model == 'categorical':
            self.__emission_model = hmm_core.Categorical
            self.__emission_params = {'K':2}
            
        if isinstance(header, int):
            self.feature_col = header
        elif isinstance(header, str):
            self.feature_col = train.header[header]
        else:
            raise ValueError("The type of header is not int or str")
            
    def run(self):

        hmm_source = hmm_core.HMM_Core(number_of_hidden_states=2, emission_model=self.__emission_model, emission_params=self.__emission_params)

        change.change_to_binary(self.source)

        hmm_source.learn(hidden_seq=np.array(self.train.occupancy[:, 0], int), 
                  emission_seq=self.train.data[:, self.feature_col])

        hmm_target = hmm_core.HMM_Core(number_of_hidden_states=2, emission_model=self.__emission_model, emission_params=self.__emission_params)

        hmm_target.prior(hmm_source)

        change.change_to_binary(self.target_retrain)
        
        hmm_target.learn(hidden_seq=np.array(self.target_retrain.occupancy[:, 0], int), 
                  emission_seq=self.target_retrain.data[:, self.feature_col])

        change.change_to_binary(self.target_test)

        predict_occupancy = hmm_target.viterbi_predict(emission_seq=self.target_test.data[:, self.feature_col])

        return predict_occupancy
