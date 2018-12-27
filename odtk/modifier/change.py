from odtk.data.dataset import Dataset
from numpy import zeros, isnan, arange


def change_to_one_hot(dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    if not dataset.labelled:
        return
    occupancy = dataset.occupancy
    if occupancy.shape[1] != 1:
        return

    new_occupancy = zeros((occupancy.shape[0], int(occupancy[~isnan(occupancy)].max()) + 1))
    new_occupancy[arange(occupancy.shape[0]), occupancy.T.astype(int)] = 1

    dataset.change_occupancy(new_occupancy)


def change_to_label(dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    if not dataset.labelled:
        return
    occupancy = dataset.occupancy
    if occupancy.shape[1] == 1:
        return

    new_occupancy = occupancy.argmax(axis=1)
    new_occupancy.shape += (1,)

    dataset.change_occupancy(new_occupancy)


def change_to_binary(dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    if not dataset.labelled:
        return
    occupancy = dataset.occupancy
    if occupancy.shape[1] != 1:
        raise ValueError("Dataset occupancy must be label rather than one hot")

    occupancy[occupancy > 0] = 1

    dataset.change_occupancy(occupancy)
