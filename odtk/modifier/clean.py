from odtk.data.dataset import Dataset
from odtk.modifier.fill import fill
from numpy import percentile, isnan, nonzero, nan


def clean(dataset, auto_fill=True):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    data = dataset.data
    time_col = dataset.time_column

    for i in range(data.shape[1]):
        if i == time_col:
            continue
        column = nonzero(~isnan(data[:, i]))[0]
        q25, q75 = percentile(data[column, i], (25, 75))
        iqr = q75 - q25
        data[column[nonzero(data[column, i] < (q25 - 1.5 * iqr))[0]], i] = nan
        data[column[nonzero(data[column, i] > (q75 + 1.5 * iqr))[0]], i] = nan
        dataset.change_values(data)

    if auto_fill:
        fill(dataset)
