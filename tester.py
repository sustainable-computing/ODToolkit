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

# data = odtk.data.load_sample(["umons-all"])
# for name in data:
#     # data[name].remove_feature([data[name].header_info[data[name].time_column]])
#     data[name].select_feature(["HumidityRatio"])
# pprint(odtk.easy_set_experiment(data, models=["RandomForest"]))
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

data = odtk.data.load_sample(["sdu-all"])
for name in data:
    data[name].remove_feature([data[name].header_info[data[name].time_column]])
    odtk.modifier.change_to_binary(data[name])
pprint(odtk.easy_set_experiment(data, models=["SVM", "LSTM"]))

# result = \
#     {'umons-all': {'RandomForest': {'Accuracy': 0.9961089494163424,
#                                     'F1Score': 0.9863013698630136,
#                                     'Fallout': 0.0011350737797956867,
#                                     'FalseNegative': 12,
#                                     'FalsePositive': 4,
#                                     'Missrate': 0.02040816326530612,
#                                     'Precision': 0.993103448275862,
#                                     'Recall': 0.9795918367346939,
#                                     'Selectivity': 0.9988649262202043,
#                                     'TrueNegative': 3520,
#                                     'TruePositive': 576},
#                    'HMM': {'Accuracy': 0.9982976653696498,
#                            'F1Score': 0.9940627650551315,
#                            'Fallout': 0.0014188422247446084,
#                            'FalseNegative': 2,
#                            'FalsePositive': 5,
#                            'Missrate': 0.003401360544217687,
#                            'Precision': 0.9915397631133672,
#                            'Recall': 0.9965986394557823,
#                            'Selectivity': 0.9985811577752554,
#                            'TrueNegative': 3519,
#                            'TruePositive': 586},
#                    'PF': {'Accuracy': 0.9973249027237354,
#                           'F1Score': 0.9906700593723494,
#                           'Fallout': 0.0019863791146424517,
#                           'FalseNegative': 4,
#                           'FalsePositive': 7,
#                           'Missrate': 0.006802721088435374,
#                           'Precision': 0.988155668358714,
#                           'Recall': 0.9931972789115646,
#                           'Selectivity': 0.9980136208853575,
#                           'TrueNegative': 3517,
#                           'TruePositive': 584},
#                    'NNv2': {'Accuracy': 0.9980544747081712,
#                             'F1Score': 0.9932088285229203,
#                             'Fallout': 0.0014188422247446084,
#                             'FalseNegative': 3,
#                             'FalsePositive': 5,
#                             'Missrate': 0.00510204081632653,
#                             'Precision': 0.9915254237288136,
#                             'Recall': 0.9948979591836735,
#                             'Selectivity': 0.9985811577752554,
#                             'TrueNegative': 3519,
#                             'TruePositive': 585},
#                    'RNN': {'Accuracy': 0.9980544747081712,
#                            'F1Score': 0.9932088285229203,
#                            'Fallout': 0.0014188422247446084,
#                            'FalseNegative': 3,
#                            'FalsePositive': 5,
#                            'Missrate': 0.00510204081632653,
#                            'Precision': 0.9915254237288136,
#                            'Recall': 0.9948979591836735,
#                            'Selectivity': 0.9985811577752554,
#                            'TrueNegative': 3519,
#                            'TruePositive': 585},
#                    'SVM': {'Accuracy': 0.9970817120622568,
#                            'F1Score': 0.9898819561551433,
#                            'Fallout': 0.0031214528944381384,
#                            'FalseNegative': 1,
#                            'FalsePositive': 11,
#                            'Missrate': 0.0017006802721088435,
#                            'Precision': 0.9816053511705686,
#                            'Recall': 0.9982993197278912,
#                            'Selectivity': 0.9968785471055619,
#                            'TrueNegative': 3513,
#                            'TruePositive': 587},
#                    'NMF': {'Accuracy': 0.9229085603112841,
#                            'F1Score': 0.6864490603363007,
#                            'Fallout': 0.021566401816118047,
#                            'FalseNegative': 241,
#                            'FalsePositive': 76,
#                            'Missrate': 0.4098639455782313,
#                            'Precision': 0.8203309692671394,
#                            'Recall': 0.5901360544217688,
#                            'Selectivity': 0.978433598183882,
#                            'TrueNegative': 3448,
#                            'TruePositive': 347}}}
#
# a = odtk.evaluation.Result()
# a.set_result(result)
#
# odtk.plot.plot_one(a,
#                    dataset="umons-all",
#                    metric=["Precision", "Recall", "F1Score"],
#                    # legend=["Missrate", "Fallout"],
#                    y_range=[0.95, 1.05],
#                    threshold="<=1",
#                    x_label="Data set feature",
#                    y_label="F1 Score",
#                    add_label=False
#                    )
