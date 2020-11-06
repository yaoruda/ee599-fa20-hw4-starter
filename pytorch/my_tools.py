import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
from utils import Config
from model import torch_model, Ruda_Model
from data import get_dataloader

# model = MyNetwork().to(device)
# model = torch.load('/home/ubuntu/ee599-fa20-hw4-starter/models/best_model.pth')
# print('success!')

from torchvision.models import resnet50, resnet34

model = resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 153)
model.load_state_dict(torch.load('../log_history/ResNet34-freeze-1/best_model.pth', map_location=torch.device('cpu')))


dataloaders, classes, dataset_size, dataset = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'],
                                                        num_workers=Config['num_workers'])
torch.save({
            'model': model,
            'label': dataset.labels.classes_
        }, '../log_history/ResNet34-freeze-1/ResNet34-freeze-1.pth')

# model = torch.load('/Users/yaoruda/Documents/Labs/data/models/best_model.pth')
# model = torch.load('/Users/yaoruda/OneDrive - University of Southern California/EE599DLS/HW/ee599-fa20-hw4-starter/log_history/MyModel-2/test_model.pth', map_location=torch.device('cpu'))
# model.eval()
print(model)
print('Success!')