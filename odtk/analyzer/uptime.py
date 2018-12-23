from odtk.analyzer.gap_detect import gap_detect
from datetime import timedelta


def uptime(dataset, frequency, gaps=None):
    if gaps is None:
        gaps = gap_detect(dataset, frequency, detail=True)

    result = {}

    time_col = dataset.time_column
    for room in gaps.keys():
        data = dataset[room][0][:, time_col]
        total_uptime = data[-1] - data[0]
        result[room] = {}

        for sensor in gaps[room].keys():
            for gap in gaps[room][sensor]:
                result[room][sensor] = result[room].get(sensor, 0) + gap[2]
            sensor_uptime = total_uptime - result[room][sensor]
            result[room][sensor] = (str(timedelta(0, sensor_uptime)), sensor_uptime, sensor_uptime / total_uptime)

    return result
