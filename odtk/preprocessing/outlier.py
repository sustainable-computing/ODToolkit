# Remove potential outliers using IQR
#
# Parameters:
#   dataset: odtk.data.dataset.Dataset() or feature_list list
#     auto_fill: Whether automatically fill the outliers or leave it nan
#     ratio: IQR ratio in order to mark value as an outlier
# Return:
#     No return


def remove_outlier(dataset, auto_fill=True, ratio=1.5):
    from ..data import Dataset
    from .fill import fill
    from numpy import percentile, isnan, nonzero, nan

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    data = dataset.data
    time_col = dataset.time_column_index

    for i in range(data.shape[1]):
        if i == time_col:
            continue
        column = nonzero(~isnan(data[:, i]))[0]
        q25, q75 = percentile(data[column, i], (25, 75))
        iqr = q75 - q25
        data[column[nonzero(data[column, i] < (q25 - ratio * iqr))[0]], i] = nan
        data[column[nonzero(data[column, i] > (q75 + ratio * iqr))[0]], i] = nan
        dataset.change_values(data)

    if auto_fill:
        fill(dataset)
