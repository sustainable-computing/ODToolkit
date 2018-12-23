from odtk.data.dataset import Dataset
from numpy import unique

def occupancy_evaluation(dataset, total=True):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    result = {}
    rooms = dataset.room
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

    if total:
        summarize = {}
        for values in result.values():
            for possible_occupancy in values.keys():
                if possible_occupancy not in summarize.keys():
                    summarize[possible_occupancy] = values[possible_occupancy][0]
                else:
                    summarize[possible_occupancy] += values[possible_occupancy][0]

        for possible_occupancy in summarize.keys():
            summarize[possible_occupancy] = (summarize[possible_occupancy], summarize[possible_occupancy] / all_entries)
        return summarize
    else:
        return result
