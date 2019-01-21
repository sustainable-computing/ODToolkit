from odtk.model.superclass import *
from odtk.evaluation.superclass import *
from odtk.data.load_sample import load_sample


def easy_experiment(source,
                    target_test,
                    target_retrain=None,
                    domain_adaptive=False,
                    models="all",
                    binary_evaluation=True,
                    evaluation_metrics="all",
                    thread_num=4):
    if domain_adaptive and target_retrain is None:
        raise ValueError("Domain Adaptive model must have target_retrain dataset")

    if domain_adaptive:
        model = DomainAdaptiveModel(source, target_retrain, target_test, thread_num=thread_num)
    else:
        model = NormalModel(source, target_test, thread_num=thread_num)

    if models == "all":
        model.get_all_model()
    else:
        model.add_model(models)

    results = model.run_all_model()

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

    return total_result


def easy_set_experiment(source_set,
                        target_test_set=None,
                        split_percentage=0.8,
                        target_retrain=None,
                        domain_adaptive=False,
                        models="all",
                        binary_evaluation=True,
                        evaluation_metrics="all",
                        thread_num=4):
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
    for dataset in source_set.keys():
        if target_test_set.get(dataset, None) is None:
            source, target = source_set[dataset].split(split_percentage)
            results[dataset] = easy_experiment(source,
                                               target,
                                               target_retrain=target_retrain.get(dataset, None),
                                               domain_adaptive=domain_adaptive,
                                               models=models,
                                               binary_evaluation=binary_evaluation,
                                               evaluation_metrics=evaluation_metrics,
                                               thread_num=thread_num)
        else:
            results[dataset] = easy_experiment(source_set[dataset],
                                               target_test_set[dataset],
                                               target_retrain=target_retrain.get(dataset, None),
                                               domain_adaptive=domain_adaptive,
                                               models=models,
                                               binary_evaluation=binary_evaluation,
                                               evaluation_metrics=evaluation_metrics,
                                               thread_num=thread_num)
    return results
