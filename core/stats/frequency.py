#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def frequency(dataset, dataset_level=True):
    """
    Compute the average sample frequency base on the given dataset

    :parameter dataset: Dataset object that want to compute the average frequency.
                        The average frequency is the average second of all consecutive timestamp
    :type dataset: core.data.dataset.Dataset

    :parameter dataset_level: decide the result is separate for each room in room_list or
                              combine for the whole dataset together
    :type dataset_level: bool

    :rtype: str or dict(str, str)
    :return: the room name with its corresponding average sampling frequency
    """
    from core.data.dataset import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    result = {}
    rooms = dataset.room_list
    time_col = dataset.time_column_index

    for room in rooms:
        data = dataset[room][0][:, time_col]
        result[room] = (data[1:] - data[:-1]).mean()

    if dataset_level:
        return sum(result.values()) / len(result.values())
    else:
        return result
