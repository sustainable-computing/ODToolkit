#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def auto_clean(dataset, target_frequency):
    """
    The full preprocessing for the given core.data.dataset.Dataset

    :parameter dataset: Dataset object that wants to perform preprocessing
    :type dataset: core.data.dataset.Dataset

    :parameter target_frequency: sampling frequency in second that the dataset wants to become
    :type target_frequency: int

    :return: None
    """
    from .downsample import downsample
    from .fill import fill
    from .ontology import ontology
    from .upsample import upsample
    from .outlier import remove_outlier
    from ..stats import frequency

    remove_outlier(dataset)
    overall_frequency = frequency(dataset, dataset_level=True)

    if overall_frequency > target_frequency:
        upsample(dataset, target_frequency)
    else:
        downsample(dataset, target_frequency)

    fill(dataset)

    ontology(dataset)
