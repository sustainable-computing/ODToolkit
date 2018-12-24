from odtk.data.dataset import Dataset
from numpy import isnan, where, arange, maximum, nonzero


def fill(dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    data = dataset.data
    mask = isnan(data.T)
    idx = where(~mask, arange(mask.shape[1]), 0)
    maximum.accumulate(idx, axis=1, out=idx)
    data.T[mask] = data.T[nonzero(mask)[0], idx[mask]]
    dataset.change_values(data)
