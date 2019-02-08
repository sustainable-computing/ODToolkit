# Parameters:
#     dataset: odtk.data.dataset.Dataset()
#     threshold: maximum different second between two consecutive value
#     gaps: All gaps detected by odtk.stats.gap_detect
# Return:
#     {room_name: sensors} where sensors = {sensor_name: (uptime in string, uptime in seconds, uptime ratio)}


def uptime(dataset, threshold, gaps=None):
    from .gap_detect import gap_detect
    from datetime import timedelta

    if gaps is None:
        gaps = gap_detect(dataset, threshold, sensor_level=True)

    result = {}

    time_col = dataset.time_column_index
    for room in gaps.keys():
        data = dataset[room][0][:, time_col]
        total_uptime = data[-1] - data[0]
        result[room] = {}

        for sensor in gaps[room].keys():
            for gap in gaps[room][sensor]:
                result[room][sensor] = result[room].get(sensor, 0) + gap[2]
            sensor_uptime = total_uptime - result[room].get(sensor, 0)
            result[room][sensor] = (str(timedelta(0, sensor_uptime)), sensor_uptime, sensor_uptime / total_uptime)

    return result
