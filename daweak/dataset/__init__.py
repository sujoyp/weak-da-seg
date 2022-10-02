###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

from .gta5 import GTA5Segmentation
from .cityscapes import CityscapesSegmentation
from .synthia import SYNTHIADataSet

datasets = {
    'gta5': GTA5Segmentation,
    'cityscapes': CityscapesSegmentation,
    'synthia': SYNTHIADataSet,
}


def get_dataset(name, **kwargs):
    return datasets[name.lower()](name, **kwargs)
