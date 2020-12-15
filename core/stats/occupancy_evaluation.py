#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def occupancy_distribution_evaluation(dataset, dataset_level=True):
    """
    Compute the distribution of the occupancy level on given Dataset

    :parameter dataset: Dataset object that want to compute the occupancy distribution
    :type dataset: core.data.dataset.Dataset

    :parameter dataset_level: decide the result is separate for each room in room_list or
                              combine for the whole dataset together
    :type dataset_level: bool

    :rtype: dict(int, str) or dict(str, dict(int, str))
    :return: the room name with its each possible occupancy level corresponding to distribution
    """
    from core.data.dataset import Dataset
    from numpy import unique

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    result = {}
    rooms = dataset.room_list
    all_entries = 0

    if not dataset.labelled:
        return None

    for room in rooms:
        occupancy = dataset[room][1]
        value, count = unique(occupancy, return_counts=True)
        result[room] = {}
        non_zero = 0

        all_entries += occupancy.shape[0]

        for i in range(value.shape[0]):
            result[room][value[i]] = (count[i], count[i] / occupancy.shape[0])
            if value[i]:
                non_zero += count[i]

        result[room]["occupied"] = (non_zero, non_zero / occupancy.shape[0])

    if dataset_level:
        summarize = {}
        for values in result.values():
            for possible_occupancy in values.keys():
                summarize[possible_occupancy] = summarize.get(possible_occupancy, 0) + values[possible_occupancy][0]

        for possible_occupancy in summarize.keys():
            summarize[possible_occupancy] = (summarize[possible_occupancy], summarize[possible_occupancy] / all_entries)
        return summarize
    else:
        return result
