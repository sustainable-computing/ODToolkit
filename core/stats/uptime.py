# Parameters:
#     dataset: core.data.dataset.Dataset()
#     threshold: maximum different second between two consecutive value
#     gaps: All gaps detected by core.stats.gap_detect
# Return:
#     {room_name: sensors} where sensors = {sensor_name: (uptime in string, uptime in seconds, uptime ratio)}


def uptime(dataset, threshold, gaps=None):
    """
    Compute the uptime in the given dataset.
    Uptime is the length of time a sensor reported value

    :parameter dataset: Dataset object that want to compute the uptime
    :type dataset: core.data.dataset.Dataset

    :parameter threshold: the maximum time differences in seconds between two consecutive timestamp
                          to not mark them as a gap
    :type threshold: int

    :parameter gaps: a dictionary result from the core.stats.gap_detect
    :type gaps: dict(str, list(str)) or dict(str, dict(str, list(str)))

    :rtype: dict(str, tuple(str)) or dict(str, dict(str, tuple(str)))
    :return: the room name corresponds to the name of sensor with its corresponding uptime
    """
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
