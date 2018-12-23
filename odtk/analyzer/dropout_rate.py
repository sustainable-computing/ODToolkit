from odtk.data.dataset import Dataset
from numpy import isnan


def dropout_rate(dataset, total=False):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")
    if total:
        data = dataset.data
        drop_row = ((~isnan(data)).sum(axis=1) != data.shape[1]).sum()
        return drop_row / data.shape[0]
    else:
        result = {}
        rooms = dataset.room

        for room in rooms:
            data = dataset[room][0]
            drop_row = ((~isnan(data)).sum(axis=1) != data.shape[1]).sum()
            result[room] = drop_row / data.shape[0]

        return result
