import odtk
import sys
import numpy as np


data = odtk.data.load_sample("umons-all")

train, test = data.split(0.8)

train.remove_feature(['id', 'date', 'Temperature', 'Humidity', 'Light', 'HumidityRatio'])
test.remove_feature(['id', 'date', 'Temperature', 'Humidity', 'Light', 'HumidityRatio'])

print(odtk.easy_experiment(train, test, models=["nmf"]))
