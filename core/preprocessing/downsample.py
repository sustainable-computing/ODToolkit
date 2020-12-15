#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def downsample(dataset, target_frequency, algorithm="mean"):
    """
    Downsampling the sampling frequency (decrease the number of rows) of given core.data.dataset.Dataset

    :parameter dataset: Dataset object that wants to downsample
    :type dataset: core.data.dataset.Dataset

    :parameter target_frequency: sampling frequency in second that the dataset wants to become
    :type target_frequency: int

    :parameter algorithm: downsampling algorithm. Only ``'mean'`` is available for now
    :type algorithm: str

    :return: None
    """
    from ..data import Dataset
    from numpy import array, concatenate, full, nan, isnan, interp
    from pandas import DataFrame

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    new_data = array([], dtype=float)
    new_data.shape = (0, len(dataset.feature_list))
    new_occupancy = array([], dtype=float)
    new_occupancy.shape = (0, 1)
    rooms = dataset.room_list
    detail_room = dataset.room_mapping
    time_col = dataset.time_column_index

    for room in rooms:
        data, occupancy = dataset[room]
        data = concatenate((data, occupancy), axis=1)
        start_t = data[0, time_col]
        end_t = data[-1, time_col]

        edited_data = full([int((end_t - start_t) // target_frequency) + 1, new_data.shape[1] + 1], nan)
        data[:, time_col] = ((data[:, time_col] - start_t) // target_frequency).astype(int)

        if algorithm == "mean":
            df = DataFrame(data)
            df = df.groupby(time_col).mean()
            time = array(df.index, dtype=int)
            data = concatenate((array(df, dtype=float), full((time.shape[0], 1), 0)), axis=1)
            data[:, time_col + 1:] = data[:, time_col:-1]

            data[:, time_col] = time.astype(float)
            edited_data[time, :] = data

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
    dataset.change_room_mapping(detail_room)
