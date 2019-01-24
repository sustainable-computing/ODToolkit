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


result = \
{ 'Humidity': {'Without': {'Accuracy': 0.995136186770428,
                          'F1Score': 0.9828178694158075,
                          'Fallout': 0.0011350737797956867,
                          'FalseNegative': 16,
                          'FalsePositive': 4,
                          'Missrate': 0.027210884353741496,
                          'Precision': 0.9930555555555556,
                          'Recall': 0.9727891156462585,
                          'Selectivity': 0.9988649262202043,
                          'TrueNegative': 3520,
                          'TruePositive': 572},
              'With only': {'Accuracy': 0.8056906614785992,
                            'F1Score': 0.23540669856459331,
                            'Fallout': 0.09477866061293984,
                            'FalseNegative': 465,
                            'FalsePositive': 334,
                            'Missrate': 0.7908163265306123,
                            'Precision': 0.26914660831509846,
                            'Recall': 0.20918367346938777,
                            'Selectivity': 0.9052213393870602,
                            'TrueNegative': 3190,
                            'TruePositive': 123}},
 'HumidityRatio': {'Without': {'Accuracy': 0.9953793774319066,
                               'F1Score': 0.9836909871244636,
                               'Fallout': 0.0011350737797956867,
                               'FalseNegative': 15,
                               'FalsePositive': 4,
                               'Missrate': 0.025510204081632654,
                               'Precision': 0.9930675909878682,
                               'Recall': 0.9744897959183674,
                               'Selectivity': 0.9988649262202043,
                               'TrueNegative': 3520,
                               'TruePositive': 573},
                   'With only': {'Accuracy': 0.6670719844357976,
                                 'F1Score': 0.33186920448999513,
                                 'Fallout': 0.3181044267877412,
                                 'FalseNegative': 248,
                                 'FalsePositive': 1121,
                                 'Missrate': 0.4217687074829932,
                                 'Precision': 0.2327173169062286,
                                 'Recall': 0.5782312925170068,
                                 'Selectivity': 0.6818955732122588,
                                 'TrueNegative': 2403,
                                 'TruePositive': 340}},
 'Light': {'Without': {'Accuracy': 0.9270428015564203,
                       'F1Score': 0.7098646034816247,
                       'Fallout': 0.022417707150964812,
                       'FalseNegative': 221,
                       'FalsePositive': 79,
                       'Missrate': 0.3758503401360544,
                       'Precision': 0.8228699551569507,
                       'Recall': 0.6241496598639455,
                       'Selectivity': 0.9775822928490352,
                       'TrueNegative': 3445,
                       'TruePositive': 367},
           'With only': {'Accuracy': 0.9965953307392996,
                         'F1Score': 0.9880341880341881,
                         'Fallout': 0.0011350737797956867,
                         'FalseNegative': 10,
                         'FalsePositive': 4,
                         'Missrate': 0.017006802721088437,
                         'Precision': 0.993127147766323,
                         'Recall': 0.9829931972789115,
                         'Selectivity': 0.9988649262202043,
                         'TrueNegative': 3520,
                         'TruePositive': 578}},
 'CO2': {'Without': {'Accuracy': 0.9910019455252919,
                     'F1Score': 0.9676855895196507,
                     'Fallout': 0.000851305334846765,
                     'FalseNegative': 34,
                     'FalsePositive': 3,
                     'Missrate': 0.05782312925170068,
                     'Precision': 0.9946140035906643,
                     'Recall': 0.9421768707482994,
                     'Selectivity': 0.9991486946651532,
                     'TrueNegative': 3521,
                     'TruePositive': 554},
         'With only': {'Accuracy': 0.8818093385214008,
                       'F1Score': 0.49269311064718163,
                       'Fallout': 0.038024971623155504,
                       'FalseNegative': 352,
                       'FalsePositive': 134,
                       'Missrate': 0.5986394557823129,
                       'Precision': 0.6378378378378379,
                       'Recall': 0.4013605442176871,
                       'Selectivity': 0.9619750283768445,
                       'TrueNegative': 3390,
                       'TruePositive': 236}},
'Temperature': {'Without': {'Accuracy': 0.9968385214007782,
                            'F1Score': 0.9888983774551665,
                            'Fallout': 0.0011350737797956867,
                            'FalseNegative': 9,
                            'FalsePositive': 4,
                            'Missrate': 0.015306122448979591,
                            'Precision': 0.9931389365351629,
                            'Recall': 0.9846938775510204,
                            'Selectivity': 0.9988649262202043,
                            'TrueNegative': 3520,
                            'TruePositive': 579},
                'With only': {'Accuracy': 0.8745136186770428,
                              'F1Score': 0.4901185770750988,
                              'Fallout': 0.049943246311010214,
                              'FalseNegative': 340,
                              'FalsePositive': 176,
                              'Missrate': 0.5782312925170068,
                              'Precision': 0.5849056603773585,
                              'Recall': 0.4217687074829932,
                              'Selectivity': 0.9500567536889898,
                              'TrueNegative': 3348,
                              'TruePositive': 248}}}

a = odtk.evaluation.Result()
a.set_result(result)

odtk.plot.plot_one(a,
                   metric="F1Score",
                   # legend=["Missrate", "Fallout"],
                   # threshold="<=1",
                   x_label="Data set feature",
                   y_range=[0.2, 1.2],
                   y_label="F1 Score"
                   )
