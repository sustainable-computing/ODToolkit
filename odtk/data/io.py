#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def save_dataset(dataset, file_name):
    """
    Save a odtk.data.dataset.Dataset object to local disk as a binary file

    :type dataset: odtk.data.dataset.Dataset
    :param dataset: Dataset object that want to save to local disk as a binary file

    :type file_name: str
    :param file_name: name of the binary file

    :return: None
    """
    from pickle import dump

    with open(file_name, 'wb') as file:
        dump(dataset, file)


# Load a odtk.data.dataset.Dataset object from local disk
#
# Parameters:
#     file_name: The name of target file
# Return:
#     odtk.data.dataset.Dataset


def read_dataset(file_name):
    """
    Load a odtk.data.dataset.Dataset object from local disk binary file

    :type file_name: str
    :param file_name: name of the binary file

    :rtype: odtk.data.dataset.Dataset
    :return: Dataset object that load from a binary file
    """
    from pickle import load

    with open(file_name, 'rb') as file:
        dataset = load(file)
    return dataset
