from sklearn import svm
from odtk.data.dataset import Dataset


def support_vector_machine(train,
                           test,
                           retrain=None,
                           gamma='auto',
                           kernel='linear',
                           penalty_error=1):

    if not isinstance(train, Dataset) or not isinstance(test, Dataset) or \
            retrain is not None and not isinstance(retrain, Dataset):
        raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")

    classifier = svm.SVC(kernel=kernel, C=penalty_error, gamma=gamma)

    classifier.fit(train.data, train.occupancy)

    if retrain is not None:
        classifier.fit(retrain.data, retrain.occupancy)

    predict_occupancy = classifier.predict(test.data)

    return predict_occupancy
