###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image


class GTA5Segmentation(data.Dataset):
    def __init__(
            self,
            dataset=None,
            path=None,
            split=None,
            mode=None,
            data_root=None,
            max_iters=None,
            size=(256, 256),
            use_pixeladapt=False
    ):
        self.dataset = dataset
        self.path = path
        self.split = split
        self.mode = mode
        self.data_root = data_root
        self.size = size
        self.ignore_label = 255
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.use_pixeladapt = use_pixeladapt

        # load image list
        list_path = osp.join(self.data_root, '%s_list/%s.txt' % (self.dataset, self.split))
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        # map label IDs to the format of Cityscapes
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # load dataset
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.path, "images/%s" % name)
            label_file = osp.join(self.path, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            if use_pixeladapt:
                p_img_file = osp.join(
                    'data/cycleGAN_output/gta5_deeplab',
                    "images/%s" % name
                )
                self.files.append({
                    "img": p_img_file,
                    "label": label_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.size, Image.BICUBIC)
        label = label.resize(self.size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name
