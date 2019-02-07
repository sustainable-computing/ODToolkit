# Compute the average sample frequency base on the given dataset
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
#     dataset_level: decide the result is separate for each room or combine the whole dataset together
# Return:
#     a float number indicates the sample frequency in seconds


def frequency(dataset, dataset_level=True):
    from odtk.data.dataset import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    result = {}
    rooms = dataset.room
    time_col = dataset.time_column

    for room in rooms:
        data = dataset[room][0][:, time_col]
        result[room] = (data[1:] - data[:-1]).mean()

    if dataset_level:
        return sum(result.values()) / len(result.values())
    else:
        return result
