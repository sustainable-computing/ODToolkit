from odtk.data.dataset import Dataset
from numpy import isnan, where
from datetime import datetime


# threshold in seconds
def gap_detect(dataset, threshold, detail=False):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    result = {}
    rooms = dataset.room
    time_col = dataset.time_column

    for room in rooms:
        data = dataset[room][0]

        if detail:
            result[room] = {}
            sensors = dataset.header
            sensor_dict = dataset.header_info
            for sensor in sensors:
                if sensor == sensors[time_col]:
                    continue
                result[room][sensor] = []

                valid_data = data[(~isnan(data[:, [time_col, sensor_dict[sensor]]])).sum(axis=1) == 2, time_col]
                indices = where(valid_data[1:] - valid_data[:-1] >= threshold)[0]
                for period in indices:
                    result[room][sensor].append((str(datetime.fromtimestamp(valid_data[period])),
                                                 str(datetime.fromtimestamp(valid_data[period + 1]))))

        else:
            result[room] = []
            valid_data = data[(~isnan(data)).sum(axis=1) == data.shape[1], time_col]
            indices = where(valid_data[1:] - valid_data[:-1] >= threshold)[0]
            for period in indices:
                result[room].append((str(datetime.fromtimestamp(valid_data[period])),
                                     str(datetime.fromtimestamp(valid_data[period + 1]))))

    return result
