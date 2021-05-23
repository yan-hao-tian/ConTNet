import os.path as osp

import torch
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torchvision

import os
import json
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class MyDataSet(Dataset):

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def build_loader(data_path, autoaug, batch_size, workers):

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert batch_size % world_size == 0, f'The batch size is indivisible by world size {batch_size} // {world_size}'
    
    train_transform = create_transform(input_size=224,
                                       is_training=True,
                                       auto_augment=autoaug)
    # train_dataset = MyDataSet(root_dir='./mini-imagenet', csv_name='new_train.csv', json_path='./mini-imagenet/classes_name.json', transform=train_transform)
    train_dataset = datasets.ImageFolder(osp.join(data_path, 'train'), transform=train_transform)
    
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size // world_size, 
                              shuffle=False,
                              num_workers=workers, 
                              pin_memory=True, 
                              sampler=train_sampler)

    val_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                        ])
    val_dataset = datasets.ImageFolder(osp.join(data_path, 'val'), transform=val_transform)
    # val_dataset = MyDataSet(root_dir='./mini-imagenet', csv_name='new_val.csv', json_path='./mini-imagenet/classes_name.json', transform=val_transform)
    val_sampler = distributed.DistributedSampler(val_dataset, world_size, rank)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size // world_size, 
                            shuffle=False,
                            num_workers=workers, 
                            pin_memory=True,
                            sampler=val_sampler)

    return train_loader, val_loader
