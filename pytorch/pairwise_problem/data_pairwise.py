import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.train_dir = osp.join(self.root_dir, 'compatibility_train.txt')
        self.valid_dir = osp.join(self.root_dir, 'compatibility_valid.txt')
        self.train_json = osp.join(self.root_dir, 'train.json')
        self.valid_json = osp.join(self.root_dir, 'valid.json')
        self.transforms = self.get_data_transforms()
        # self.labels = LabelEncoder()
        # self.y = self.labels.fit_transform(self.y)
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms



    def create_dataset(self):
        # train_json
        meta_file = open(osp.join(self.root_dir, self.train_json), 'r')
        train_json = json.load(meta_file)
        set_id_to_item_id_train = {}
        for set in train_json:
            set_id_to_item_id_train[set['set_id']] = []
            for item in set['items']:
                set_id_to_item_id_train[set['set_id']].append(item['item_id'])
        # valid_json
        meta_file = open(osp.join(self.root_dir, self.valid_json), 'r')
        valid_json = json.load(meta_file)
        set_id_to_item_id_valid = {}
        for set in valid_json:
            set_id_to_item_id_valid[set['set_id']] = []
            for item in set['items']:
                set_id_to_item_id_valid[set['set_id']].append(item['item_id'])

        X_train = []
        y_train = []
        X_val = []
        y_val = []
        for line in open(self.train_dir, "r"):  # train data
            line = line.rstrip("\n")
            label, items = line.split(' ', 1)
            items = items.lstrip()
            if len(items.split(' ')) < 2:
                print(items)
                continue
            else:
                this_comp_items = []
                for item in items.split(' '):
                    this_comp_items.append(item.split('_'))
                i = 0
                while i < len(this_comp_items):
                    j = i
                    while j < len(this_comp_items):
                        if i != j:
                            set_id, index = this_comp_items[i][0], this_comp_items[i][1]
                            x1 = set_id_to_item_id_train[set_id][int(index) - 1]
                            set_id, index = this_comp_items[j][0], this_comp_items[j][1]
                            x2 = set_id_to_item_id_train[set_id][int(index) - 1]
                            X_train.append([x1, x2])
                            y_train.append(int(label))
                        j += 1
                    i += 1

        for line in open(self.valid_dir, "r"):  # validation data
            line = line.rstrip("\n")
            label, items = line.split(' ', 1)
            items = items.lstrip()
            if len(items.split(' ')) < 2:
                print(items)
                continue
            else:
                this_comp_items = []
                for item in items.split(' '):
                    this_comp_items.append(item.split('_'))
                i = 0
                while i < len(this_comp_items):
                    j = i
                    while j < len(this_comp_items):
                        if i != j:
                            set_id, index = this_comp_items[i][0], this_comp_items[i][1]
                            x1 = set_id_to_item_id_valid[set_id][int(index)-1]
                            set_id, index = this_comp_items[j][0], this_comp_items[j][1]
                            x2 = set_id_to_item_id_valid[set_id][int(index)-1]
                            X_val.append([x1, x2])
                            y_val.append(int(label))
                        j += 1
                    i += 1


        print('len of train: {}, # len of val: {}'.format(len(y_train), len(y_val)))

        return X_train, X_val, torch.tensor(y_train), torch.tensor(y_val)


########################################################################
# For Pairwise Compatibility Classification

class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_train[item][0])
        X1 = self.transform(Image.open(file_path+'.jpg'))
        file_path = osp.join(self.image_dir, self.X_train[item][1])
        X2 = self.transform(Image.open(file_path+'.jpg'))
        X = torch.cat((X1, X2), 0)
        return X, self.y_train[item]

class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')
    def __len__(self):
        return len(self.X_test)
    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item][0])
        X1 = self.transform(Image.open(file_path+'.jpg'))
        file_path = osp.join(self.image_dir, self.X_test[item][1])
        X2 = self.transform(Image.open(file_path+'.jpg'))
        X = torch.cat((X1, X2), 0)
        return X, self.y_test[item]




def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size)
                                 for x in ['train', 'test']}
    return dataloaders, dataset_size, dataset

