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

# data = odtk.data.load_sample(["umons-all","sdu-all", "niom-all", "aifb-all", "lbl-all"])
# # odtk.modifier.upsample(data["lbl-all"], 60)
# current = list(data.keys())
# for i in range(len(current)):
#     data[chr(65 + i)] = data.pop(current[i])
#
# odtk.plot.plot_occupancy_swarm(data,
#                                room_level=True,
#                                # orientation="vertical"
#                                )

# odtk.analyzer.analyze(data["aifb-all"], 11, save_file="result.txt")

# data["sdu-binary"] = data["sdu-all"].copy()
# odtk.modifier.change_to_binary(data["sdu-binary"])
# data = odtk.load_sample(["umons-all"])
# for name in data:
#     # odtk.modifier.change_to_binary(data["sdu-all"])
#     data[name].remove_feature([data[name].header_info[data[name].time_column], "id"])
#     # odtk.plot.plot_feature(data[name])
#     datas = data[name].data
#     for i in range(datas.shape[1]):
#         if i == data[name].time_column:
#             continue
#         datas[:, i] += (np.random.normal(0, datas[:, i].std()/4, data[name].data.shape[0]))
#     data[name].change_values(datas)
# odtk.plot.plot_feature(data[name])
# # # #     data[name].select_feature(["HumidityRatio"])
# pprint(odtk.easy_set_experiment(data, models=["RandomForest", "HMM", "LSTM", "NNv2", "SVM", "PF"]))
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

# data = odtk.data.load_sample(["umons-all"])
# for name in data:
#     odtk.modifier.change_to_binary(data[name])
#     odtk.modifier.downsample(data[name], 120)
#     data[name].remove_feature([data[name].header_info[data[name].time_column]])
# a, b = (odtk.easy_set_experiment(data,
#                                 models=["NMF"],
#                                 plot=True,
#                                 ))

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
#     with open(name, "rb") as file:
#         data[names[name]] = load(file)
#
# data["LSTM"] = np.random.choice(np.concatenate((data["Truth"], np.random.choice(data["LSTM"], 30, replace=False))),
#                                 data["Truth"].shape[0],
#                                 replace=False)
#
# odtk.plot.plot_occupancy_perc(data, orientation="horizontal",
#                               evaluation=True, size=2, swarm=True)
result = \
    {'Without noise': {'RF': {'Accuracy': 0.9961089494163424,
                                        'F1Score': 0.9863013698630136,
                                        'Fallout': 0.0011350737797956867,
                                        'FalseNegative': 12,
                                        'FalsePositive': 4,
                                        'Missrate': 0.02040816326530612,
                                        'Precision': 0.993103448275862,
                                        'Recall': 0.9795918367346939,
                                        'Selectivity': 0.9988649262202043,
                                        'TrueNegative': 3520,
                                        'TruePositive': 576},
                       'HMM': {'Accuracy': 0.9982976653696498,
                               'F1Score': 0.9940627650551315,
                               'Fallout': 0.0014188422247446084,
                               'FalseNegative': 2,
                               'FalsePositive': 5,
                               'Missrate': 0.003401360544217687,
                               'Precision': 0.9915397631133672,
                               'Recall': 0.9965986394557823,
                               'Selectivity': 0.9985811577752554,
                               'TrueNegative': 3519,
                               'TruePositive': 586},
                       'PF': {'Accuracy': 0.9973249027237354,
                              'F1Score': 0.9906700593723494,
                              'Fallout': 0.0019863791146424517,
                              'FalseNegative': 4,
                              'FalsePositive': 7,
                              'Missrate': 0.006802721088435374,
                              'Precision': 0.988155668358714,
                              'Recall': 0.9931972789115646,
                              'Selectivity': 0.9980136208853575,
                              'TrueNegative': 3517,
                              'TruePositive': 584},
                       'NN': {'Accuracy': 0.9980544747081712,
                                'F1Score': 0.9932088285229203,
                                'Fallout': 0.0014188422247446084,
                                'FalseNegative': 3,
                                'FalsePositive': 5,
                                'Missrate': 0.00510204081632653,
                                'Precision': 0.9915254237288136,
                                'Recall': 0.9948979591836735,
                                'Selectivity': 0.9985811577752554,
                                'TrueNegative': 3519,
                                'TruePositive': 585},
                       'LSTM': {'Accuracy': 0.9980544747081712,
                                'F1Score': 0.9932088285229203,
                                'Fallout': 0.0014188422247446084,
                                'FalseNegative': 3,
                                'FalsePositive': 5,
                                'Missrate': 0.00510204081632653,
                                'Precision': 0.9915254237288136,
                                'Recall': 0.9948979591836735,
                                'Selectivity': 0.9985811577752554,
                                'TrueNegative': 3519,
                                'TruePositive': 585},
                       'SVM': {'Accuracy': 0.9970817120622568,
                               'F1Score': 0.9898819561551433,
                               'Fallout': 0.0031214528944381384,
                               'FalseNegative': 1,
                               'FalsePositive': 11,
                               'Missrate': 0.0017006802721088435,
                               'Precision': 0.9816053511705686,
                               'Recall': 0.9982993197278912,
                               'Selectivity': 0.9968785471055619,
                               'TrueNegative': 3513,
                               'TruePositive': 587},
                       'SNMF': {'Accuracy': 0.9229085603112841,
                               'F1Score': 0.6864490603363007,
                               'Fallout': 0.021566401816118047,
                               'FalseNegative': 241,
                               'FalsePositive': 76,
                               'Missrate': 0.4098639455782313,
                               'Precision': 0.8203309692671394,
                               'Recall': 0.5901360544217688,
                               'Selectivity': 0.978433598183882,
                               'TrueNegative': 3448,
                               'TruePositive': 347}},
     'sdu-all': {'SNMF': {'Accuracy': 0.4038895730706076,
                         'F1Score': 0.6402240896358543,
                         'Fallout': 0.09200938619241694,
                         'FalseNegative': 5677,
                         'FalsePositive': 745,
                         'Missrate': 0.4983759108067773,
                         'Precision': 0.8846570676575322,
                         'Recall': 0.5016240891932228,
                         'Selectivity': 0.9079906138075831,
                         'TrueNegative': 7352,
                         'TruePositive': 5714},
                 'NN': {'Accuracy': 0.6139675697865353,
                          'F1Score': 0.7146595865731082,
                          'Fallout': 0.6858095590959614,
                          'FalseNegative': 1970,
                          'FalsePositive': 5553,
                          'Missrate': 0.17294355192695987,
                          'Precision': 0.6291572058234273,
                          'Recall': 0.8270564480730401,
                          'Selectivity': 0.3141904409040385,
                          'TrueNegative': 2544,
                          'TruePositive': 9421},
                 'PF': {'Accuracy': 0.6672311165845649,
                        'F1Score': 0.6366336078892811,
                        'Fallout': 0.095714462146474,
                        'FalseNegative': 5710,
                        'FalsePositive': 775,
                        'Missrate': 0.5012729347730664,
                        'Precision': 0.8799566294919455,
                        'Recall': 0.49872706522693355,
                        'Selectivity': 0.904285537853526,
                        'TrueNegative': 7322,
                        'TruePositive': 5681},
                 'RF': {'Accuracy': 0.6451149425287356,
                                  'F1Score': 0.6613787700744223,
                                  'Fallout': 0.2814622699765345,
                                  'FalseNegative': 4637,
                                  'FalsePositive': 2279,
                                  'Missrate': 0.4070757615661487,
                                  'Precision': 0.7477028672644747,
                                  'Recall': 0.5929242384338513,
                                  'Selectivity': 0.7185377300234654,
                                  'TrueNegative': 5818,
                                  'TruePositive': 6754},
                 'LSTM': {'Accuracy': 0.6740045155993432,
                          'F1Score': 0.642788867022772,
                          'Fallout': 0.08373471656168952,
                          'FalseNegative': 5675,
                          'FalsePositive': 678,
                          'Missrate': 0.49820033359669913,
                          'Precision': 0.8939630903972474,
                          'Recall': 0.5017996664033009,
                          'Selectivity': 0.9162652834383105,
                          'TrueNegative': 7419,
                          'TruePositive': 5716},
                 'SVM': {'Accuracy': 0.6729782430213465,
                         'F1Score': 0.6488980566713712,
                         'Fallout': 0.08558725453871804,
                         'FalseNegative': 5680,
                         'FalsePositive': 693,
                         'Missrate': 0.4986392766218945,
                         'Precision': 0.891786383510306,
                         'Recall': 0.5013607233781056,
                         'Selectivity': 0.9144127454612819,
                         'TrueNegative': 7404,
                         'TruePositive': 5711},
                 'HMM': {'Accuracy': 0.6729782430213465,
                         'F1Score': 0.6419864052581316,
                         'Fallout': 0.08595776213412375,
                         'FalseNegative': 5677,
                         'FalsePositive': 696,
                         'Missrate': 0.4983759108067773,
                         'Precision': 0.8914196567862714,
                         'Recall': 0.5016240891932228,
                         'Selectivity': 0.9140422378658762,
                         'TrueNegative': 7401,
                         'TruePositive': 5714}},
     'sdu-noise': {'NN': {'Accuracy': 0.6705665024630542,
                            'F1Score': 0.6381467703753805,
                            'Fallout': 0.08521674694331234,
                            'FalseNegative': 5730,
                            'FalsePositive': 690,
                            'Missrate': 0.5030287068738478,
                            'Precision': 0.8913556920170052,
                            'Recall': 0.49697129312615224,
                            'Selectivity': 0.9147832530566876,
                            'TrueNegative': 7407,
                            'TruePositive': 5661},
                   'HMM': {'Accuracy': 0.6716954022988506,
                           'F1Score': 0.6377123442808607,
                           'Fallout': 0.07879461528961344,
                           'FalseNegative': 5760,
                           'FalsePositive': 638,
                           'Missrate': 0.5056623650250197,
                           'Precision': 0.898229382676663,
                           'Recall': 0.49433763497498023,
                           'Selectivity': 0.9212053847103866,
                           'TrueNegative': 7459,
                           'TruePositive': 5631},
                   'LSTM': {'Accuracy': 0.6538382594417077,
                            'F1Score': 0.6209260507979322,
                            'Fallout': 0.1086822279856737,
                            'FalseNegative': 5866,
                            'FalsePositive': 880,
                            'Missrate': 0.5149679571591608,
                            'Precision': 0.862607338017174,
                            'Recall': 0.4850320428408393,
                            'Selectivity': 0.8913177720143263,
                            'TrueNegative': 7217,
                            'TruePositive': 5525},
                   'PF': {'Accuracy': 0.6571736453201971,
                          'F1Score': 0.6143722943722943,
                          'Fallout': 0.07558354946276398,
                          'FalseNegative': 6069,
                          'FalsePositive': 612,
                          'Missrate': 0.5327890439820911,
                          'Precision': 0.8968655207280081,
                          'Recall': 0.46721095601790885,
                          'Selectivity': 0.924416450537236,
                          'TrueNegative': 7485,
                          'TruePositive': 5322},
                   'RF': {'Accuracy': 0.646243842364532,
                                    'F1Score': 0.6690985888451569,
                                    'Fallout': 0.3054217611461035,
                                    'FalseNegative': 4421,
                                    'FalsePositive': 2473,
                                    'Missrate': 0.38811342287771045,
                                    'Precision': 0.7381128878534364,
                                    'Recall': 0.6118865771222896,
                                    'Selectivity': 0.6945782388538965,
                                    'TrueNegative': 5624,
                                    'TruePositive': 6970},
                   'SVM': {'Accuracy': 0.573121921182266,
                           'F1Score': 0.4560256326423854,
                           'Fallout': 0.05125355069778931,
                           'FalseNegative': 7904,
                           'FalsePositive': 415,
                           'Missrate': 0.6938811342287771,
                           'Precision': 0.8936442849820605,
                           'Recall': 0.3061188657712229,
                           'Selectivity': 0.9487464493022107,
                           'TrueNegative': 7682,
                           'TruePositive': 3487},
                   'SNMF': {'Accuracy': 0.6729269293924466,
                           'F1Score': 0.6418296246347494,
                           'Fallout': 0.08571075707051995,
                           'FalseNegative': 5680,
                           'FalsePositive': 694,
                           'Missrate': 0.4986392766218945,
                           'Precision': 0.891647150663544,
                           'Recall': 0.5013607233781056,
                           'Selectivity': 0.9142892429294801,
                           'TrueNegative': 7403,
                           'TruePositive': 5711}},
     'With mul noise': {'HMM': {'Accuracy': 0.9785992217898832,
                                'F1Score': 0.93026941362916,
                                'Fallout': 0.024687854710556185,
                                'FalseNegative': 1,
                                'FalsePositive': 87,
                                'Missrate': 0.0017006802721088435,
                                'Precision': 0.870919881305638,
                                'Recall': 0.9982993197278912,
                                'Selectivity': 0.9753121452894438,
                                'TrueNegative': 3437,
                                'TruePositive': 587},
                        'LSTM': {'Accuracy': 0.8431420233463035,
                                 'F1Score': 0.43782837127845886,
                                 'Fallout': 0.08633910820789549,
                                 'FalseNegative': 338,
                                 'FalsePositive': 304,
                                 'Missrate': 0.5748299319727891,
                                 'Precision': 0.45126353790613716,
                                 'Recall': 0.42517006802721086,
                                 'Selectivity': 0.9136608917921045,
                                 'TrueNegative': 3217,
                                 'TruePositive': 250},
                        'SNMF': {'Accuracy': 0.6675583657587548,
                                'F1Score': 0.4436304436304436,
                                'Fallout': 0.3757094211123723,
                                'FalseNegative': 43,
                                'FalsePositive': 1324,
                                'Missrate': 0.07312925170068027,
                                'Precision': 0.29159978598180847,
                                'Recall': 0.9268707482993197,
                                'Selectivity': 0.6242905788876277,
                                'TrueNegative': 2200,
                                'TruePositive': 545},
                        'NN': {'Accuracy': 0.9477140077821011,
                                 'F1Score': 0.8367501898253606,
                                 'Fallout': 0.05051078320090806,
                                 'FalseNegative': 37,
                                 'FalsePositive': 178,
                                 'Missrate': 0.06292517006802721,
                                 'Precision': 0.7558299039780522,
                                 'Recall': 0.9370748299319728,
                                 'Selectivity': 0.9494892167990919,
                                 'TrueNegative': 3346,
                                 'TruePositive': 551},
                        'PF': {'Accuracy': 0.9764105058365758,
                               'F1Score': 0.9233201581027668,
                               'Fallout': 0.026390465380249715,
                               'FalseNegative': 4,
                               'FalsePositive': 93,
                               'Missrate': 0.006802721088435374,
                               'Precision': 0.8626292466765141,
                               'Recall': 0.9931972789115646,
                               'Selectivity': 0.9736095346197503,
                               'TrueNegative': 3431,
                               'TruePositive': 584},
                        'RF': {'Accuracy': 0.9686284046692607,
                                         'F1Score': 0.8945216680294358,
                                         'Fallout': 0.024971623155505107,
                                         'FalseNegative': 41,
                                         'FalsePositive': 88,
                                         'Missrate': 0.06972789115646258,
                                         'Precision': 0.8614173228346457,
                                         'Recall': 0.9302721088435374,
                                         'Selectivity': 0.9750283768444948,
                                         'TrueNegative': 3436,
                                         'TruePositive': 547},
                        'SVM': {'Accuracy': 0.9654669260700389,
                                'F1Score': 0.8780068728522337,
                                'Fallout': 0.01844494892167991,
                                'FalseNegative': 77,
                                'FalsePositive': 65,
                                'Missrate': 0.13095238095238096,
                                'Precision': 0.8871527777777778,
                                'Recall': 0.8690476190476191,
                                'Selectivity': 0.9815550510783201,
                                'TrueNegative': 3459,
                                'TruePositive': 511}},
     'With noise': {'HMM': {'Accuracy': 0.9985408560311284,
                            'F1Score': 0.9949152542372881,
                            'Fallout': 0.0014188422247446084,
                            'FalseNegative': 1,
                            'FalsePositive': 5,
                            'Missrate': 0.0017006802721088435,
                            'Precision': 0.9915540540540541,
                            'Recall': 0.9982993197278912,
                            'Selectivity': 0.9985811577752554,
                            'TrueNegative': 3519,
                            'TruePositive': 587},
                    'SNMF': {'Accuracy': 0.6675583657587548,
                            'F1Score': 0.4436304436304436,
                            'Fallout': 0.3757094211123723,
                            'FalseNegative': 43,
                            'FalsePositive': 1324,
                            'Missrate': 0.07312925170068027,
                            'Precision': 0.29159978598180847,
                            'Recall': 0.9268707482993197,
                            'Selectivity': 0.6242905788876277,
                            'TrueNegative': 2200,
                            'TruePositive': 545},
                    'LSTM': {'Accuracy': 0.9365272373540856,
                             'F1Score': 0.7987663839629915,
                             'Fallout': 0.05419977298524404,
                             'FalseNegative': 70,
                             'FalsePositive': 191,
                             'Missrate': 0.11904761904761904,
                             'Precision': 0.7306064880112835,
                             'Recall': 0.8809523809523809,
                             'Selectivity': 0.945800227014756,
                             'TrueNegative': 3333,
                             'TruePositive': 518},
                    'NN': {'Accuracy': 0.9973249027237354,
                             'F1Score': 0.9907016060862215,
                             'Fallout': 0.002553916004540295,
                             'FalseNegative': 2,
                             'FalsePositive': 9,
                             'Missrate': 0.003401360544217687,
                             'Precision': 0.984873949579832,
                             'Recall': 0.9965986394557823,
                             'Selectivity': 0.9974460839954598,
                             'TrueNegative': 3515,
                             'TruePositive': 586},
                    'PF': {'Accuracy': 0.9958657587548638,
                           'F1Score': 0.9853321829163072,
                           'Fallout': 0.0,
                           'FalseNegative': 17,
                           'FalsePositive': 0,
                           'Missrate': 0.02891156462585034,
                           'Precision': 1.0,
                           'Recall': 0.9710884353741497,
                           'Selectivity': 1.0,
                           'TrueNegative': 3524,
                           'TruePositive': 571},
                    'RF': {'Accuracy': 0.9965953307392996,
                                     'F1Score': 0.9881756756756757,
                                     'Fallout': 0.0031214528944381384,
                                     'FalseNegative': 3,
                                     'FalsePositive': 11,
                                     'Missrate': 0.00510204081632653,
                                     'Precision': 0.9815436241610739,
                                     'Recall': 0.9948979591836735,
                                     'Selectivity': 0.9968785471055619,
                                     'TrueNegative': 3513,
                                     'TruePositive': 585},
                    'SVM': {'Accuracy': 0.9946498054474708,
                            'F1Score': 0.9816053511705686,
                            'Fallout': 0.005959137343927355,
                            'FalseNegative': 1,
                            'FalsePositive': 21,
                            'Missrate': 0.0017006802721088435,
                            'Precision': 0.9654605263157895,
                            'Recall': 0.9982993197278912,
                            'Selectivity': 0.9940408626560726,
                            'TrueNegative': 3503,
                            'TruePositive': 587}}}
a = odtk.evaluation.Result()
a.set_result(result)

odtk.plot.plot_one(a,
                   # metric="Accuracy",
                   # model="NNv2",
                   metric="F1Score",
                   dataset=["Without noise", "With noise"],
                   y_range=[0.2, 1.2],
                   threshold="<=1",
                   # x_label="Supervised Learning NN\nPercentage of Labelled Data in Target Domain (%)",
                   y_label="F1 Score",
                   # add_label=False,
                   font_size=16,
                   group_by=1
                   )
