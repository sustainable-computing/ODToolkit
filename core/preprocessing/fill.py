#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def fill(dataset):
    """
    Fill all nan value in the sensor data for given core.data.dataset.Dataset

    :parameter dataset: Dataset object that wants to fill in missing values
    :type dataset: core.data.dataset.Dataset

    :return: None
    """
    from ..data import Dataset
    from numpy import isnan, where, arange, maximum, nonzero

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    data = dataset.data

    for _ in range(2):
        mask = isnan(data.T)
        idx = where(~mask, arange(mask.shape[1]), 0)
        maximum.accumulate(idx, axis=1, out=idx)
        data.T[mask] = data.T[nonzero(mask)[0], idx[mask]]
        data = data[::-1]

    dataset.change_values(data)
