#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def load_sample(sample_name):
    """
    Load one or more core.data.dataset.Dataset object from the sample folder

    :type sample_name: str or list(str)
    :type sample_name: str or list(str)
    :param sample_name: name(s) of the binary file in binary_dataset

    :rtype: core.data.dataset.Dataset or dict(str, core.data.dataset.Dataset)
    :return: Dataset object(s) that load from a binary file. If a list of name is provided, then a dictionary
             with their name as key and corresponding Dataset is returned
    """
    from .io import read_dataset
    from os.path import abspath, join, basename, isfile
    from os import listdir

    if isinstance(sample_name, str):
        sample_dir = join(abspath(__file__).rstrip(basename(__file__)), "binary_dataset")
        if sample_name == "all":
            all_data = dict()
            for name in listdir(sample_dir):
                if isfile(join(sample_dir, name)):
                    all_data[name] = read_dataset(join(sample_dir, name))
                else:
                    all_data[name] = read_dataset(join(sample_dir, name, "all"))
            return all_data
        sample_name = sample_name.split('-')
        directory = join(sample_dir, *sample_name)
        if isfile(directory):
            return read_dataset(directory)
        else:
            raise FileNotFoundError("Dataset not found in built-in library")

    result = {}
    for required_name in sample_name:
        result[required_name] = load_sample(required_name)
    return result
