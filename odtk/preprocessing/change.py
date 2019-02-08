# Change the occupancy data in odtk.data.dataset.Dataset to one hot encoding
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
# Return:
#     No return


def change_to_one_hot(dataset):
    from ..data import Dataset
    from numpy import zeros, isnan, arange

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


# Change the occupancy data in odtk.data.dataset.Dataset to label encoding
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
# Return:
#     No return


def change_to_label(dataset):
    from ..data import Dataset

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


# Change the occupancy data in odtk.data.dataset.Dataset to binary
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
# Return:
#     No return


def change_to_binary(dataset):
    from ..data import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    if not dataset.labelled:
        return
    occupancy = dataset.occupancy
    if occupancy.shape[1] != 1:
        raise ValueError("Dataset occupancy must be label rather than one hot")

    occupancy[occupancy > 0] = 1

    dataset.change_occupancy(occupancy)
