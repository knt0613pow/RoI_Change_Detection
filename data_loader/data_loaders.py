from torchvision import datasets, transforms
from base import BaseDataLoader
import json
import torch
import os
from torchvision.io import read_image
from PIL import Image
import numpy as np
import utils
import glob
from utils.util import read_json

class ImageLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = ImageDataset(self.data_dir, trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.root = data_dir
        self.image_list =  glob.glob(f'{self.root}/*/*.jpg')

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):

        image_path = self.image_list[idx]
        print(image_path)
        i1 = self.transform(Image.open(image_path).convert("RGB"))

        label_path = self.image_list[idx][:-3] + 'json'
        json_1 = read_json(label_path)
        I1_label ={}
        boxes1 = []
        labels1 = []
        for obj in json_1["shapes"]:
            labels1.append(0)
            xyxy = obj['points']
            xyxy = np.array(xyxy)
            xmin = np.min(xyxy[:,0])
            xmax = np.max(xyxy[:,0])
            ymin = np.min(xyxy[:,1])
            ymax = np.max(xyxy[:,1])
            boxes1.append([xmin, ymin, xmax, ymax])
        I1_label["boxes"] = torch.as_tensor(boxes1, dtype = torch.float32)
        I1_label["labels"] = torch.as_tensor(labels1, dtype = torch.int64)

        return i1,I1_label, image_path
    