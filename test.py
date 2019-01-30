import odtk
import sys
import numpy as np
from time import mktime, gmtime
from datetime import datetime

names = {
    "NNv2": "NN",
    "LSTM": "LSTM",
    "HMM": "HMM",
    "PF": "PF",
    "RandomForest": "RF",
    "SVM": "SVM",
    "NMF": "SNMF",
    "Truth": "Ground\nTruth"
}
data = dict()
start = dict()
end = dict()
from pickle import load

with open("Time", "rb") as file:
    time = load(file)
# print(time)
for name in names:
    with open(name, "rb") as file:
        data[names[name]] = load(file).flatten()

# data["LSTM"] = np.where(data["LSTM"] > 0.007, 1, 0)
data["LSTM"] = np.round((data["NN"] + data["PF"]) / 3)
data["SNMF"] = np.where(data["SNMF"] > 0.480, 0, 1)
print(data["RF"])

time = time - mktime(gmtime(0)) + 7 * 60 * 60 + 6 * 60 * 60
day = (time - 6 * 60 * 60) // (24 * 60 * 60)
separator = np.concatenate(([0], np.where(day[:-1] != day[1:])[0] + 1))
# for i in range(separator.shape[0] - 1):
#     print(separator[i])

for name in data:
    current = 0
    length = 0
    for i in range(data[name].shape[0]):
        if data[name][i] == current:
            length += 1
        else:
            if length < 60:
                data[name][i - length:i - 1] = current
            current = data[name][i]
            length = 1

# print(datetime.fromtimestamp(time[0]))
# print(day[0])
# 8:56 first person
for name in names:
    start[names[name]] = list()
    end[names[name]] = list()
    for i in range(separator.shape[0] - 1):
        temp_matrix = data[names[name]][separator[i]:separator[i + 1]]
        # No one in room today
        if temp_matrix.max() == 0:
            continue
        start_index = (temp_matrix != 0).argmax(axis=0)
        # print(datetime.fromtimestamp(time[separator[i] + start_index]))
        # Someone overnight
        if not start_index:
            # 0 - last one leave, 1 - first one come
            swap = np.where(temp_matrix[:-1] != temp_matrix[1:])[0]
            if swap.shape[0] >= 1:
                second = (time[separator[i] + swap[0]] - 6 * 60 * 60) % (24 * 60 * 60)
                end[names[name]].append(second)
            if swap.shape[0] >= 2:
                second = (time[separator[i] + swap[1]] - 6 * 60 * 60) % (24 * 60 * 60)
                start[names[name]].append(second)
        else:
            second = (time[separator[i] + start_index] - 6 * 60 * 60) % (24 * 60 * 60)
            start[names[name]].append(second)
        end_index = (temp_matrix[::-1] != 0).argmax(axis=0) + 1
        if end_index != 1:
            second = (time[separator[i + 1] - end_index] - 6 * 60 * 60) % (24 * 60 * 60)
            end[names[name]].append(second)
# time_float = time[2496] - mktime(gmtime(0)) + 7 * 60 * 60
# time_float %= (24 * 60 * 60)
# print(time_float)

## AIFB time was shifted 7 hours prior

odtk.plot.plot_start_end(start, end, bin_size=60 * 30,
                         orientation="vertical",
                         # x_label="Time of the Day",
                         y_label="Supervised Learning Models")
