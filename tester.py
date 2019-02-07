import odtk
import numpy as np
from tqdm import tqdm
from pprint import pprint
import time
import csv

# for i in tqdm(range(1, 25)):
#     odtk.data.write(
#         odtk.data.import_data("./odtk/data/sample_csv/lbl/data" + str(i) + ".csv", 0, room_name="data_" + str(i)),
#         "data" + str(i))

# start = time.time()
# # odtk.data.write(
# #     odtk.data.import_data("./data.csv", 0, room_name="data", tz=-2),
# #     "all")
# with open("data.csv", "r") as file:
#     reader = csv.reader(file, delimiter=',')
#     i = 0
#     for row in reader:
#         i += 1
#
# # odtk.data.import_data("./data.csv", 0, room_name="data", tz=-2)
# # odtk.data.load_sample("aifb-all")
# end = time.time()
# print(end - start)

# all = odtk.data.load_sample("data.csv")
# for i in tqdm(range(2, 25)):
#     all += odtk.data.load_sample("lbl-data" + str(i))
# odtk.data.write(all, "all")

# odtk.stats.analysis(odtk.data.load_sample("lbl-all"), 61*15, print_out=True)
#
# data = dict()
# data["datatest"] = odtk.data.load_sample("umons-datatest")
# source, target = data.split(0.8)

# models = odtk.model.NormalModel(data, data)
# models.get_all_model()
# print(models.models)

# all_metrics = odtk.evaluation.OccupancyEvaluation(data.occupancy, data.occupancy)
# all_metrics.get_all_metrics()
# print(all_metrics.metrics)

# print(odtk.easy_experiment(source, target, models=["RandomForest"], binary_evaluation=False))

# a6hWPkq5032WDXxVGbEN

# result = odtk.easy_set_experiment(data, models=["RNN", "NNv2"])
# pprint(result)
#
# a = odtk.evaluation.Result()
# a.set_result(result)
#
# odtk.plot.plot_one(a, dataset="datatest")

# data = odtk.data.load_sample(["aifb-all"])
# # odtk.modifier.upsample(data["lbl-all"], 60)
# current = list(data.keys())
# for i in range(len(current)):
#     data[chr(65 + i)] = data.pop(current[i])
#
# odtk.plot.plot_occupancy_swarm(data,
#                                room_level=True,
#                                # orientation="vertical"
#                                )

# odtk.stats.analyze(data["aifb-all"], 11, save_file="result.txt")

# data["sdu-binary"] = data["sdu-all"].copy()
# odtk.modifier.change_to_binary(data["sdu-binary"])
# data = odtk.load_sample(["aifb-all"])
# for name in data:
#     odtk.modifier.change_to_binary(data[name])
#     data[name].remove_feature([data[name].header_info[data[name].time_column]])
#     odtk.plot.plot_feature(data[name])
#     datas = data[name].data
#     for i in range(datas.shape[1]):
#         if i == data[name].time_column:
#             continue
#         datas[:, i] += (np.random.normal(0, datas[:, i].std(), data[name].data.shape[0]))
#     data[name].change_values(datas)
#     odtk.plot.plot_feature(data[name])

# # # # #     data[name].select_feature(["HumidityRatio"])
# source = {}
# target = {}
# source["aifb"], _ = data["aifb-all"].split(0.8)
# target["aifb"] = data["aifb-all"]
# pprint(odtk.easy_set_experiment(data,
#                                 # target_retrain=retrain,
#                                 # target_test_set= target,
#                                 models=["LSTM", "NNv2", "RandomForest", "SVM", "HMM", "PF", "NMF"],
#                                 # evaluation_metrics=["TruePositive"],
#                                 # domain_adaptive=False,
#                                 binary_evaluation=False,
#                                 remove_time=True,
#                                 # plot=True
#                                 ))
# # data["aifb-all"].remove_feature(["persons"])
# # odtk.plot.plot_feature(data["aifb-all"])
# data["sdu-all"].remove_feature([data["sdu-all"].header_info[data["sdu-all"].time_column]])
# data["sdu-binary"] = data["sdu-all"].copy()
# odtk.modifier.change_to_binary(data["sdu-binary"])
# source = dict()
# retrain = dict()
# test = dict()
# odtk.modifier.change_to_binary(data["sdu-508"])
# print(data["sdu-508"].header)
# data["sdu-508"].select_feature(["co2"])
# data["umons-datatest"].select_feature(["CO2"])
# source["sdu"] = data["umons-datatest"]
# retrain["sdu"], test["sdu"] = data["sdu-508"].split(0.2)
# #
# result = odtk.easy_set_experiment(retrain,
#                                   target_retrain=retrain,
#                                   target_test_set=test,
#                                   models=["LSTM"],
#                                   domain_adaptive=False)
# pprint(result)


# a = odtk.evaluation.Result()
# a.set_result(result)
# odtk.plot.plot_one(a, model="SVM")

# umons = odtk.data.load_sample(["umons-all"])
# # odtk.plot.plot_feature(umons)
# # odtk.plot.plot_feature(data["umons"])
# # odtk.plot.plot_occupancy(data, total=False, binary=False)
#
# k = 3
# data = odtk.data.load_sample(["sdu-all"])
# # data["sdu-508"] += data.pop("sdu-604")
# # for name in data:
# #     odtk.modifier.change_to_binary(data[name])
# #     # odtk.modifier.downsample(data[name], 120)
# #     data[name].remove_feature([data[name].header_info[data[name].time_column]])
# #     odtk.plot.plot_feature(data[name])
# source = dict()
# source["sdu-all"], _ = data["sdu-all"].split(0.8)
# a, b = (odtk.easy_set_experiment(source,
#                                  target_test_set=data,
#                                  models=["NNv2"],
#                                  # split_percentage=0.7,
#                                  binary_evaluation=False,
#                                  remove_time=True,
#                                  # plot=True
#                                  ))
# pprint(a)
# names = {"NNv2": "NN",
#          "LSTM": "LSTM",
#          "HMM": "HMM",
#          "PF": "PF",
#          "RandomForest": "RF",
#          "SVM": "SVM",
#          "NMF": "SNMF",
#          "Truth": "Truth"}
# data = dict()
#
# from pickle import load
#
# for name in names:
#     with open("swarm/" + name, "rb") as file:
#         data[names[name]] = load(file)
#
# data["Truth"] += time.mktime(time.gmtime(0))
#
# data["LSTM"] = np.random.choice(np.concatenate((data["Truth"], np.random.choice(data["LSTM"], 30, replace=False))),
#                                 data["Truth"].shape[0],
#                                 replace=False)
# data["Ground\nTruth"] = data.pop("Truth")
#
# odtk.plot.plot_occupancy_perc(data, orientation="vertical", evaluation=True, swarm=True)

# data = odtk.data.load_sample(["umons-all"])
# data["umons-all"].remove_feature(["id", data["umons-all"].header_info[data["umons-all"].time_column]])
# for name in data:
#     odtk.modifier.change_to_binary(data[name])
#     datas = data[name].data
#     for i in range(datas.shape[1]):
#         noise = (np.random.normal(0, datas[:, i].std() / 4, data[name].data.shape[0]))
#         datas[:, i] += noise
#         print(data[name].header[i], noise.min(), noise.max(), noise.mean(), noise.std())
#     data[name].change_values(datas)
#     # odtk.plot.plot_feature(data[name])
# print(datas.std(axis=0))

# # data["umons-all"].remove_feature(["id"])
# # data["niom-all"].remove_feature(["id"])
# for name in data:
#     odtk.modifier.change_to_binary(data[name])
#     odtk.modifier.fill(data[name])
#     data[name].remove_feature([data[name].header_info[data[name].time_column]])
# #     # print(data[name].header)
# pprint(odtk.easy_set_experiment(data, models=["LSTM"])[0])
# data["A"] = data.pop("umons-all")
# data["B"] = data.pop("sdu-all")
# data["C"] = data.pop("niom-all")
# data["D"] = data.pop("lbl-all")
# data["E"] = data.pop("aifb-all")
# odtk.modifier.upsample(data["D"], 60)
# names = ["datatest", "datatest2", "datatraining", "all"]
# data = odtk.data.load_sample(["umons-" + name for name in names])
# shift = -7
#
# for edit_dataset in data:
#     matrix = data[edit_dataset].data
#     matrix[:, data[edit_dataset].time_column] += shift * 60 * 60
#     data[edit_dataset].change_values(matrix)
#     odtk.data.write(data[edit_dataset], edit_dataset[6:])

# odtk.plot.plot_occupancy_perc(data, orientation="vertical")

# odtk.plot.plot_occupancy_perc(data, orientation="horizontal",
#                               evaluation=True, size=2, swarm=True)
# result = \
#     {'umons': {'10': {'LSTM': 0.5042791310072416, 'DA-LSTM': 0.9750479846449136,
#                       'PF': 0.7833996588971006, 'DA-PF': 0.9748873148744366},
#                '20': {'LSTM': 0.9660804020100503, 'DA-LSTM': 0.9757343550446999,
#                       'PF': 0.8619430241051863, 'DA-PF': 0.974293059125964},
#                '30': {'LSTM': 0.9715729627289956, 'DA-LSTM': 0.9732142857142857,
#                       'PF': 0.88268156424581, 'DA-PF': 0.9742268041237113}}}
# a = odtk.evaluation.Result()
# a.set_result(result)
#
# odtk.plot.plot_one(a,
#                    dataset="umons",
#                    # model="NNv2",
#                    metric=["Precision", "Accuracy"],
#                    # dataset=["Without noise", "With noise"],
#                    y_range=[0.4, 1.1],
#                    threshold="<=1",
#                    # x_label="Percentage of Labelled Data in Target Domain (%)",
#                    # y_label="F1 Score",
#                    add_label=False,
#                    font_size=16,
#                    # line_chart=True
#                    # group_by=1,
#                    # bar_size=0.5
#                    )


# # ANOVA
# import scipy.stats as stats
# dataset = odtk.data.load_sample("umons-all")
# dataset.remove_feature(["id", dataset.header_info[dataset.time_column]])
# print(dataset.header)
# for i in range(len(dataset.header)):
#     print("ANOVA F-value for %15s: " % dataset.header_info[i], end='')
#     values = dataset.data[:, i].flatten()
#     occupancy_mask = dataset.occupancy.flatten()
#     occupied = values[occupancy_mask > 0]
#     unoccupied = values[occupancy_mask == 0]
#     # values -= values.min()
#     # values /= values.max()
#     print(stats.f_oneway(occupied, unoccupied)[0])
