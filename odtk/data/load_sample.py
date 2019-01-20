from odtk.data.read import read
from os.path import abspath, join, basename, isfile
import os


def load_sample(sample_name):
    sample_dir = join(abspath(__file__).rstrip(basename(__file__)), "sample_csv")
    sample_name = sample_name.split('-')
    directory = join(sample_dir, *sample_name)
    if isfile(directory):
        return read(directory)
    print("Data not defined")
