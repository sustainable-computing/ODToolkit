from odtk.analyzer.gap_detect import gap_detect
from datetime import timedelta


# Parameters:
#   dataset: odtk.data.dataset.Dataset()
#   frequency: maximum different second between two consecutive value
#   gap: All gaps detected by odtk.analyzer.gap_detect
# Return:
#   {room_name: sensors} where sensors = {sensor_name: (uptime in string, uptime in seconds, uptime ratio)}
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
