import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image


class SYNTHIADataSet(data.Dataset):
    def __init__(
            self,
            root,
            path,
            max_iters=None,
            size=(321, 321),
            mean=(128, 128, 128),
            scale=True,
            mirror=True,
            ignore_label=255,
            mode=None,
            split=None,
            data_root=None,
            use_pixeladapt=False
    ):
        self.root = root
        self.path = path
        self.size = size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.use_pixeladapt = use_pixeladapt

        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(osp.join(path, 'train.txt'), 'r')]
        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.path, "RGB/%s" % name)
            label_file = osp.join(self.path, "LABELS_cityscapes/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            if self.use_pixeladapt:
                p_img_file = osp.join(
                    'data/cycleGAN_output/syn_deeplab', "RGB/%s" % name
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

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name
