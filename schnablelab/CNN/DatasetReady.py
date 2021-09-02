# -*- coding: UTF-8 -*-

"""
Defining the Dataset for training pytorch based models
"""
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class LeafcountingDataset(Dataset):
    """leaf counting dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the comma separated csv file without header. The 1st column is image file name and the 2nd column is the annotation/label. 
            root_dir (string): Directory with all the images.
        """
        self.csv = pd.read_csv(csv_file)#, header=None)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv.iloc[idx, 0]
        image = Image.open(self.root_dir/img_name)
        if len(image.getbands()) == 4:
            image = image.convert('RGB')
        label = self.csv.iloc[idx, 1].astype('float32').reshape(-1,)

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

class ObjectDetectionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        csv_file:
            csv solumns follow https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        at least contains: fn, boxes, labels, area, iscrowd, masks(required for mask-RNN)
        root_dir:
            where the images in csv_file are located
        '''
        self.csv = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.csv.loc[idx, 'fn']
        image = Image.open(self.root_dir/img_name)
        if len(image.getbands()) == 4:
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        tips = json.loads(self.csv.loc[idx, 'targets'])
        labels_series = pd.Series(np.unique(tips['label']))
        labels_series.index += 1
        labels_dict = {y:x for x,y in labels_series.items()}

        boxes, labels = [], []
        for x, y, label in zip(tips['x'], tips['y'], tips['label']):
            boxes.append([x-15, y-15, x+15, y+15])
            labels.append(labels_dict[label])
        boxes = torch.tensor(boxes, dtype=torch.float)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.tensor(labels)
        image_id = torch.tensor([idx])
        # zeros: False, ones: True
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["area"] = area
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        return image, target

class InstanceSegmentationDataset(Dataset):
    '''
    add mask, keypoints
    '''
    pass