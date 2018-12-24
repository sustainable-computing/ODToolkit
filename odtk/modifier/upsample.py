from odtk.data.dataset import Dataset
from numpy import array, concatenate, full, nan, isnan, interp


def upsample(dataset, target_frequency, algorithm="linear"):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    new_data = array([], dtype=float)
    new_data.shape = (0, len(dataset.header))
    new_occupancy = array([], dtype=float)
    new_occupancy.shape = (0, 1)
    rooms = dataset.room
    detail_room = dataset.room_info
    time_col = dataset.time_column

    for room in rooms:
        data, occupancy = dataset[room]
        data = concatenate((data, occupancy), axis=1)
        start_t = data[0, time_col]
        end_t = data[-1, time_col]

        edited_data = full([int((end_t - start_t) // target_frequency) + 1, new_data.shape[1] + 1], nan)
        data[:, time_col] = (data[:, time_col] - start_t) // target_frequency
        edited_data[data[:, time_col].astype(int)] = data

        if algorithm == "linear":
            edited_data = edited_data.T

            mask = ~isnan(edited_data)
            xp = mask.ravel().nonzero()[0]
            fp = edited_data[~isnan(edited_data)]
            x = isnan(edited_data).ravel().nonzero()[0]

            edited_data[isnan(edited_data)] = interp(x, xp, fp)

            edited_data = edited_data.T

        edited_data[:, time_col] = edited_data[:, time_col] * target_frequency + start_t

        detail_room[room] = (new_data.shape[0], new_data.shape[0] + edited_data.shape[0])
        new_data = concatenate((new_data, edited_data[:, :-1]), axis=0)
        occupancy = edited_data[:, -1].round()
        occupancy.shape += (1,)
        new_occupancy = concatenate((new_occupancy, occupancy), axis=0)

    dataset.change_values(new_data)
    dataset.change_occupancy(new_occupancy)
    dataset.change_room_info(detail_room)
