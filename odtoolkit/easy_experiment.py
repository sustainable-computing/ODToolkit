#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .model.superclass import *
from .evaluation.superclass import *
from .data.load_sample import load_sample


def easy_experiment(source,
                    target_test,
                    target_retrain=None,
                    domain_adaptive=False,
                    models="all",
                    binary_evaluation=True,
                    evaluation_metrics="all",
                    thread_num=4,
                    remove_time=True,
                    plot=False
                    ):
    """
    A function for researcher to fast test all models on one dataset and evaluate by all metrics

    :parameter source: the source domain with full knowledge for training the model
    :type source: odtoolkit.data.dataset.Dataset

    :parameter target_retrain: the labelled ground truth Dataset in the target domain for re-training the model
    :type target_retrain: ``None`` or odtoolkit.data.dataset.Dataset

    :parameter target_test: the Dataset in the rest of the target domain for testing by using sensor data only
    :type target_test: odtoolkit.data.dataset.Dataset

    :parameter domain_adaptive: indicate whether use normal supervised learning model
                                or domain-adaptive semi-supervised learning model
    :type domain_adaptive: bool

    :parameter binary_evaluation: indicate whether use binary evaluation metrics
                                  or occupancy count metrics
    :type binary_evaluation: bool

    :parameter models: choose the models want to use in this experiment. If ``'all'`` then all model with
                       selected superclass will add to the experiment.
    :type models: str, list(str)

    :parameter evaluation_metrics: choose the evaluation metrics want to use in this experiment. If ``'all'`` then
                                   all metrics with selected superclass will add to the experiment.
    :type evaluation_metrics: str, list(str)

    :parameter thread_num: the maximum number of threads can use to speed up
    :type thread_num: int

    :parameter remove_time: decide whether remove the time column when predicting occupancy level
    :type remove_time: bool

    :rtype: list(dict(str, dict(str, score)), dict(str, numpy.ndarray))
    :return: first is the final score of the metrics by all models, and
             second is the prediction result
    """
    if domain_adaptive and target_retrain is None:
        raise ValueError("Domain Adaptive model must have target_retrain dataset")

    # test_time = target_test.data[:, target_test.time_column].flatten()
    if remove_time:
        if source.time_column_index is not None:
            source.remove_feature([source.feature_mapping[source.time_column_index]])
        if target_test.time_column_index is not None:
            target_test.remove_feature([target_test.feature_mapping[target_test.time_column_index]])
        if target_retrain is not None and target_retrain.time_column_index is not None:
            target_retrain.remove_feature([target_retrain.feature_mapping[target_retrain.time_column_index]])

    if domain_adaptive:
        model = DomainAdaptiveModel(source, target_retrain, target_test, thread_num=thread_num)
    else:
        model = NormalModel(source, target_test, thread_num=thread_num)

    if models == "all":
        model.get_all_model()
    else:
        model.add_model(models)

    results = model.run_all_model()

    # if plot:
    #     plot_dict = dict()
    #     from pickle import dump
    #     for model in results:
    #         print(results[model].shape, test_time.shape)
    #         # plot_dict[model] = test_time[results[model].flatten() > 0]
    #         # with open(model, 'wb') as file:
    #             # dump(plot_dict[model], file)
    #             # dump(results[model], file)
    #     plot_dict["Truth"] = test_time[target_test.occupancy.flatten() > 0]
    #     with open("Truth", 'wb') as file:
    #         dump(plot_dict["Truth"], file)
    #
    #     # plot_occupancy_distribution(plot_dict, orientation="horizontal",
    #     #                     evaluation=True, size=2, swarm=True)
    total_result = dict()

    for model_result in results.keys():
        total_result[model_result] = dict()

        if binary_evaluation:
            metrics = BinaryEvaluation(results[model_result], target_test.occupancy)
        else:
            metrics = OccupancyEvaluation(results[model_result], target_test.occupancy)

        if evaluation_metrics == "all":
            metrics.get_all_metrics()
        else:
            metrics.add_metrics(evaluation_metrics)

        total_result[model_result] = metrics.run_all_metrics()

    return total_result, results


def easy_set_experiment(source_set,
                        target_test_set=None,
                        split_percentage=0.8,
                        target_retrain=None,
                        domain_adaptive=False,
                        models="all",
                        binary_evaluation=True,
                        evaluation_metrics="all",
                        thread_num=4,
                        remove_time=True,
                        plot=False):
    """
    A function for researcher to fast test all models on all dataset and evaluate by all metrics.
    Please make sure all keys in *source_set*, *target_test_set*, and *target_retrain* are the same

    :parameter source_set: the set of source domain with full knowledge for training the model
    :type source_set: dict(str, odtoolkit.data.dataset.Dataset)

    :parameter target_retrain: the labelled ground truth Dataset in the target domain for re-training the model
    :type target_retrain: ``None`` or dict(str, odtoolkit.data.dataset.Dataset)

    :parameter target_test_set: the set of Datasets in the rest of the target domain for
                                testing by using sensor data only. If ``None`` then split source domain to get
                                new source domain and target domain
    :type target_test_set: dict(str, odtoolkit.data.dataset.Dataset)

    :parameter split_percentage: percentage of the row in the first part
    :type split_percentage: float

    :parameter domain_adaptive: indicate whether use normal supervised learning model
                                or domain-adaptive semi-supervised learning model
    :type domain_adaptive: bool

    :parameter binary_evaluation: indicate whether use binary evaluation metrics
                                  or occupancy count metrics
    :type binary_evaluation: bool

    :parameter models: choose the models want to use in this experiment. If ``'all'`` then all model with
                       selected superclass will add to the experiment.
    :type models: str, list(str)

    :parameter evaluation_metrics: choose the evaluation metrics want to use in this experiment. If ``'all'`` then
                                   all metrics with selected superclass will add to the experiment.
    :type evaluation_metrics: str, list(str)

    :parameter thread_num: the maximum number of threads can use to speed up
    :type thread_num: int

    :parameter remove_time: decide whether remove the time column when predicting occupancy level
    :type remove_time: bool

    :parameter plot: unused
    :type plot: bool

    :rtype: list(dict(str, dict(str, dict(str, score))), dict(str, dict(str, numpy.ndarray)))
    :return: first is the final score of the metrics by all Datasets, all models, and
             second is the prediction result
    """
    if source_set == "all":
        source_set = load_sample(source_set)
    elif not isinstance(source_set, dict):
        raise TypeError("Source must be 'all' or a dictionary")

    if target_retrain is None:
        target_retrain = dict()
    elif not isinstance(target_retrain, dict):
        raise TypeError("Target_retrain must be None or a dictionary")

    if target_test_set is None:
        target_test_set = dict()
    elif not isinstance(target_test_set, dict):
        raise TypeError("Target_test_set must be None or a dictionary")

    results = dict()
    pred = dict()
    for dataset in source_set.keys():
        if target_test_set.get(dataset, None) is None:
            source, target = source_set[dataset].split(split_percentage)
            results[dataset], pred[dataset] = easy_experiment(source,
                                                              target,
                                                              target_retrain=target_retrain.get(dataset, None),
                                                              domain_adaptive=domain_adaptive,
                                                              models=models,
                                                              binary_evaluation=binary_evaluation,
                                                              evaluation_metrics=evaluation_metrics,
                                                              thread_num=thread_num,
                                                              remove_time=remove_time,
                                                              plot=plot)
        else:
            results[dataset], pred[dataset] = easy_experiment(source_set[dataset],
                                                              target_test_set[dataset],
                                                              target_retrain=target_retrain.get(dataset, None),
                                                              domain_adaptive=domain_adaptive,
                                                              models=models,
                                                              binary_evaluation=binary_evaluation,
                                                              evaluation_metrics=evaluation_metrics,
                                                              thread_num=thread_num,
                                                              remove_time=remove_time,
                                                              plot=plot)
    return results, pred
