# Load raw data from disk
#
# Parameters:
#     file_name: The name of the raw data file
#     time_column_index: The column index for the timestamp in given raw data file
#     mode: The raw data format, support 'csv'
#     feature_list: Indicate whether the raw data include feature_list at the first line
#     room_name: The name of room_list for the raw data file
#     tz: The time zone offset that need to fix for the raw data file
# Return:
#     odtk.data.dataset.Dataset object contains one room_list data


def import_data(file_name, time_column_index=None, mode='csv', header=True, room_name=None, tz=0):
    from csv import reader
    from dateutil.parser import parse
    from numpy import nan, asarray
    from .dataset import Dataset

    if mode == 'csv':
        with open(file_name, 'r') as input_file:
            csv_reader = reader(input_file, delimiter=',')
            feature_name = []
            data = []
            if header:
                feature_name = next(csv_reader)[:-1]

            for line in csv_reader:
                if not len(line):
                    continue

                for i in range(len(line)):
                    if i == time_column_index:
                        line[i] = parse(line[i]).timestamp() + tz * 60 * 60
                    elif not len(line[i]):
                        line[i] = nan
                    else:
                        try:
                            line[i] = float(line[i])
                        except ValueError:
                            line[i] = nan

                data.append(line)
            data = asarray(data, dtype=float)

            if not len(feature_name):
                feature_name = list(range(data.shape[1]))

        dataset = Dataset()
        dataset.add_room(data[:, :-1], occupancy=data[:, -1], header=False, room_name=room_name)
        dataset.set_feature_name(feature_name)
        dataset.time_column_index = time_column_index
        return dataset
