from odtk.modifier.downsample import *
from odtk.modifier.regulate import *
from odtk.modifier.upsample import *
from odtk.modifier.clean import *
from odtk.analyzer.frequency import *


def auto_clean(dataset, target_frequency):
    clean(dataset)
    overall_frequency = frequency(dataset, total=True)

    if overall_frequency > target_frequency:
        upsample(dataset, target_frequency)
    else:
        downsample(dataset, target_frequency)

    # regulate(dataset)
