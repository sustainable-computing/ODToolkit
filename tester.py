import odtk
import numpy as np
from tqdm import tqdm
from pprint import pprint

# for i in tqdm(range(1, 25)):
#     odtk.data.write(
#         odtk.data.import_data("./odtk/data/sample_csv/lbl/data" + str(i) + ".csv", 0, room_name="data_" + str(i)),
#         "data" + str(i))

# all = odtk.data.load_sample("lbl-data1")
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

# data = odtk.data.load_sample(["umons-datatraining", "umons-datatest"])
# for name in data:
#     data[name].remove_feature([data[name].header_info[data[name].time_column], "id"])
# # data["aifb-all"].remove_feature(["persons"])
# # odtk.plot.plot_feature(data["aifb-all"])
# # data["sdu-all"].remove_feature([data["sdu-all"].header_info[data["sdu-all"].time_column]])
# # data["sdu-binary"] = data["sdu-all"].copy()
# # odtk.modifier.change_to_binary(data["sdu-binary"])
# source = dict()
# retrain = dict()
# test = dict()
# source["umons"] = data["umons-datatraining"]
# retrain["umons"], test["umons"] = data["umons-datatest"].split(0.1)
#
# result = odtk.easy_set_experiment(retrain,
#                                   target_retrain=retrain,
#                                   target_test_set=test,
#                                   models=["PF", "LSTM"],
#                                   domain_adaptive=False)
# pprint(result)


# a = odtk.evaluation.Result()
# a.set_result(result)
# odtk.plot.plot_one(a, model="SVM")

# umons = odtk.data.load_sample("umons-all")
# odtk.plot.plot_feature(umons)
# odtk.plot.plot_feature(data["umons"])
# odtk.plot.plot_occupancy(data, total=False, binary=False)

result = \
    {'15 sec': {'NN': {'Accuracy': 0.9981758482305728,
                         'F1Score': 0.9936305732484076,
                         'Fallout': 0.0016313213703099511,
                         'FalseNegative': 7,
                         'FalsePositive': 23,
                         'Missrate': 0.002982530890498509,
                         'Precision': 0.9902666102412188,
                         'Recall': 0.9970174691095015,
                         'Selectivity': 0.99836867862969,
                         'TrueNegative': 14076,
                         'TruePositive': 2340}},
     '30 sec': {'NN': {'Accuracy': 0.9970817120622568,
                         'F1Score': 0.9897959183673469,
                         'Fallout': 0.0022688598979013048,
                         'FalseNegative': 2,
                         'FalsePositive': 4,
                         'Missrate': 0.006825938566552901,
                         'Precision': 0.9864406779661017,
                         'Recall': 0.9931740614334471,
                         'Selectivity': 0.9977311401020987,
                         'TrueNegative': 1759,
                         'TruePositive': 291}},
     '60 sec': {'NN': {'Accuracy': 0.9980544747081712,
                         'F1Score': 0.9932088285229203,
                         'Fallout': 0.0014188422247446084,
                         'FalseNegative': 3,
                         'FalsePositive': 5,
                         'Missrate': 0.00510204081632653,
                         'Precision': 0.9915254237288136,
                         'Recall': 0.9948979591836735,
                         'Selectivity': 0.9985811577752554,
                         'TrueNegative': 3519,
                         'TruePositive': 585}},
     '90 sec': {'NN': {'Accuracy': 0.9974461875228019,
                         'F1Score': 0.9910828025477707,
                         'Fallout': 0.0017028522775649213,
                         'FalseNegative': 3,
                         'FalsePositive': 4,
                         'Missrate': 0.007653061224489796,
                         'Precision': 0.989821882951654,
                         'Recall': 0.9923469387755102,
                         'Selectivity': 0.998297147722435,
                         'TrueNegative': 2345,
                         'TruePositive': 389}},
     '120 sec': {'NN': {'Accuracy': 0.9970817120622568,
                          'F1Score': 0.9897959183673469,
                          'Fallout': 0.0022688598979013048,
                          'FalseNegative': 2,
                          'FalsePositive': 4,
                          'Missrate': 0.006825938566552901,
                          'Precision': 0.9864406779661017,
                          'Recall': 0.9931740614334471,
                          'Selectivity': 0.9977311401020987,
                          'TrueNegative': 1759,
                          'TruePositive': 291}}}
a = odtk.evaluation.Result()
a.set_result(result)

odtk.plot.plot_one(a, model="NN",
                   metric=["F1Score", "Accuracy"],
                   # legend=["Missrate", "Fallout"],
                   threshold="<=1",
                   x_label="Data set frequency",
                   y_range=[0.98, 1],
                   line=["F1Score", "Accuracy"]
                   )
