import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
import cv2

class SynthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        # todo:check mean
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def convert(self, label):
        labels = {(0, 0, 0): 0, (70, 130, 180): 1, (70, 70, 70): 2, (128, 64, 128): 3, (244, 35, 232): 4,
                  (64, 64, 128): 5, (107, 142, 35): 6, (153, 153, 153): 7, (0, 0, 142): 8, (220, 220, 0): 9,
                  (220, 20, 60): 10, (119, 11, 32): 11, (0, 0, 230): 12, (250, 170, 160): 13, (128, 64, 64): 14,
                  (250, 170, 30): 15, (152, 251, 152): 16, (255, 0, 0): 17, (0, 0, 70): 18, (0, 60, 100): 19,
                  (0, 80, 100): 20, (102, 102, 156): 21, (102, 102, 156): 22}

        h, w, _ = label.shape
        new_label = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                try:
                    a = labels[(label[i, j][0], label[i, j][1], label[i, j][2])]
                except KeyError:
                    a = -100
                new_label[i, j] = a
        return new_label
    def color_code(self, image):
        labels = {(0, 0, 0): 0, (70, 130, 180): 1, (70, 70, 70): 2, (128, 64, 128): 3, (244, 35, 232): 4,
                  (64, 64, 128): 5, (107, 142, 35): 6, (153, 153, 153): 7, (0, 0, 142): 8, (220, 220, 0): 9,
                  (220, 20, 60): 10, (119, 11, 32): 11, (0, 0, 230): 12, (250, 170, 160): 13, (128, 64, 64): 14,
                  (250, 170, 30): 15, (152, 251, 152): 16, (255, 0, 0): 17, (0, 0, 70): 18, (0, 60, 100): 19,
                  (0, 80, 100): 20, (102, 102, 156): 21, (102, 102, 156): 22}
        labels = [list(k) for k, v in labels.items()]

        colour_codes = np.array(labels)
        x = colour_codes[image.astype(int)]
        print("color_code x =", x)
        return x

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = cv2.imread(datafiles["label"], cv2.IMREAD_UNCHANGED)[:, :, 2]
        gt_lab = np.asarray(imageio.imread(gt_lab, format='PNG-FI'))
        name = datafiles["name"]


        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = cv2.resize(label, self.crop_size, cv2.INTER_NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.int32)
       
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
       
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name
