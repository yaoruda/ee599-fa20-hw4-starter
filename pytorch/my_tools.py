import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp

# model = MyNetwork().to(device)
model = torch.load(torch.load('/home/ubuntu/ee599-fa20-hw4-starter/models/best_model.pth'))
torch.save(model, '/home/ubuntu/ee599-fa20-hw4-starter/models/test_model.pth', _use_new_zipfile_serialization=False)
print('success!')


# model = torch.load('/Users/yaoruda/Documents/Labs/data/models/best_model.pth')
# model = torch.load('/Users/yaoruda/OneDrive - University of Southern California/EE599DLS/HW/ee599-fa20-hw4-starter/log_history/MyModel-2/best_model.pth')
# model.eval()
# print(model.fc1.weight)
