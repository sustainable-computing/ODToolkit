from odtk.model import hmm_core
from odtk.data.dataset import Dataset


def da_hidden_markov_model(src_train,
                           trgt_train,
                           trgt_test,

                           emission='gaussian',
                           emission_params={},

                           src_train_start=0,
                           src_train_end=-1,

                           trgt_train_start=0,
                           trgt_train_end=-1,

                           trgt_test_start=0,
                           trgt_test_end=-1,

                           feature_col=0):
    if not isinstance(src_train, Dataset) or not isinstance(trgt_train, Dataset) or not isinstance(trgt_test, Dataset):
        raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")

    emission_type = ()
    if emission == 'gaussian':
        emission_type = (hmm_core.Gaussian, {})
    if emission == 'gamma':
        emission_type = (hmm_core.Gamma, {})
    if emission == 'categorical':
        emission_type = (hmm_core.Categorical, emission_params)

    src_hmm = hmm_core.HMM(number_of_hidd_states=2, emission_type=emission_type)

    src_hmm.learn(emi_seqs=[src_train.data[src_train_start:src_train_end, feature_col]],
                  hidd_seqs=[src_train.occupancy[src_train_start:src_train_end]])

    trgt_hmm = hmm_core.HMM(number_of_hidd_states=2, emission_type=emission_type)
    trgt_hmm.prior(prior=(src_hmm.A, src_hmm.B, src_hmm.PI))

    trgt_hmm.learn(emi_seqs=[trgt_train.data[trgt_train_start:trgt_train_end, feature_col]],
                   hidd_seqs=[trgt_train.occupancy[trgt_train_start:trgt_train_end]])

    predict_occupancy = trgt_hmm.viterbi_predict(
        emi_seqs=[trgt_test.data[trgt_test_start:trgt_test_end, feature_col]])[0]

    return predict_occupancy
