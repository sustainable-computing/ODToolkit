import hmm_core

from odtk.data.dataset import Dataset


def hidden_markov_model(train,
                        test,
                        
                        emission='gaussian',
                        emission_params = {},
                        
                        train_start=0,
                        train_end=-1,
                        
                        test_start=0,
                        test_end=-1,
                        
                        feature_col=0):

    if not isinstance(train, Dataset) or not isinstance(test, Dataset):
        raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")
        
        
    emission_type = ()
    if emission == 'gaussian':
        emission_type = (hmm_core.Gaussian, {})
    if emission == 'gamma':
        emission_type = (hmm_core.Gamma, {})
    if emission == 'categorical':
        emission_type = (hmm_core.Categorical, emission_params) 

    hmm = hmm_core.HMM(number_of_hidd_states=2, emission_type=emission_type)
    
    hmm.learn(emi_seqs=[train.data[train_start:train_end, feature_col]], hidd_seqs=[train.occupancy[train_start:train_end]])

    predict_occupancy = hmm.viterbi_predict(emi_seqs=[test.data[test_start:test_end, feature_col]])[0]

    return predict_occupancy

