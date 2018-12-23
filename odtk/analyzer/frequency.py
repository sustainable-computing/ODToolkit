from odtk.data.dataset import Dataset


def frequency(dataset, total=True):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    result = {}
    rooms = dataset.room
    time_col = dataset.time_column

    for room in rooms:
        data = dataset[room][0][:, time_col]
        result[room] = (data[1:] - data[:-1]).mean()

    if total:
        return sum(result.values())
    else:
        return result
