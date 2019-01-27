import odtk
import numpy as np
from tqdm import tqdm
from pprint import pprint

# for i in tqdm(range(1, 25)):
#     odtk.data.write(
#         odtk.data.import_data("./odtk/data/sample_csv/lbl/data" + str(i) + ".csv", 0, room_name="data_" + str(i)),
#         "data" + str(i))


# odtk.data.write(
#     odtk.data.import_data("./data.csv", 0, room_name="data", tz=-2),
#     "all")

# all = odtk.data.load_sample("data.csv")
# for i in tqdm(range(2, 25)):
#     all += odtk.data.load_sample("lbl-data" + str(i))
# odtk.data.write(all, "all")

# odtk.analyzer.analyze(odtk.data.load_sample("lbl-all"), 61*15, save_file="result.txt")
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

data = odtk.data.load_sample(["umons-all", "sdu-all", "niom-all", "aifb-all", "lbl-all"])
odtk.modifier.upsample(data["lbl-all"], 60)
current = list(data.keys())
for i in range(len(current)):
    data["Data Set " + chr(65 + i)] = data.pop(current[i])

odtk.plot.plot_occupancy_perc(data,
                              # room_level=True,
                              orientation="vertical"
                              )
# odtk.analyzer.analyze(data["aifb-all"], 11, save_file="result.txt")

# # data["sdu-binary"] = data["sdu-all"].copy()
# # odtk.modifier.change_to_binary(data["sdu-binary"])
# for name in data:
#     # odtk.modifier.change_to_binary(data["sdu-all"])
#     data[name].remove_feature([data[name].header_info[data[name].time_column], "id"])
#     # odtk.plot.plot_feature(data[name])
#     data[name].change_values(data[name].data *
#                              np.abs(np.random.normal(1, 0.5, data[name].data.shape)))
#     odtk.plot.plot_feature(data[name])
# # #     data[name].select_feature(["HumidityRatio"])
# pprint(odtk.easy_set_experiment(data, models=["RandomForest", "HMM", "LSTM", "NNv2", "NMF", "SVM", "PF"]))
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

# umons = odtk.data.load_sample("umons-all")
# odtk.plot.plot_feature(umons)
# odtk.plot.plot_feature(data["umons"])
# odtk.plot.plot_occupancy(data, total=False, binary=False)

# data = odtk.data.load_sample(["sdu-all"])
# for name in data:
#     odtk.modifier.change_to_binary(data[name])
#     odtk.modifier.downsample(data[name], 120)
#     data[name].remove_feature([data[name].header_info[data[name].time_column]])
# pprint(odtk.easy_set_experiment(data, models=["LSTM"]))

# a = odtk.evaluation.Result()
# a.set_result(result)
#
# odtk.plot.plot_one(a,
#                    # metric="Accuracy",
#                    # model="NNv2",
#                    metric="F1Score",
#                    dataset=["Without noise", "With noise"],
#                    y_range=[0.2, 1.2],
#                    threshold="<=1",
#                    # x_label="Supervised Learning NN\nPercentage of Labelled Data in Target Domain (%)",
#                    y_label="F1 Score",
#                    # add_label=False,
#                    font_size=16,
#                    group_by=1
#                    )
