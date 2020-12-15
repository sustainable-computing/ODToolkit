#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def save_dataset(dataset, file_name):
    """
    Save a core.data.dataset.Dataset object to local disk as a binary file

    :type dataset: core.data.dataset.Dataset
    :param dataset: Dataset object that want to save to local disk as a binary file

    :type file_name: str
    :param file_name: name of the binary file

    :return: None
    """
    from pickle import dump

    with open(file_name, 'wb') as file:
        dump(dataset, file)


def read_dataset(file_name):
    """
    Load a core.data.dataset.Dataset object from local disk binary file

    :type file_name: str
    :param file_name: name of the binary file

    :rtype: core.data.dataset.Dataset
    :return: Dataset object that load from a binary file
    """
    from pickle import load

    with open(file_name, 'rb') as file:
        dataset = load(file)
    return dataset
