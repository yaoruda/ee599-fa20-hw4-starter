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
from pairwise_problem.model import Ruda_Model as pair_model
from data import get_dataloader



def gen_autolab_model():
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

    print(model)
    print('Success!')

def plot_model():
    from torchviz import make_dot, make_dot_from_trace

    model = pair_model()
    x = torch.randn(1, 6, 224, 224)
    y = model(x)
    plot = make_dot(y.mean(), params=dict(model.named_parameters()))
    plot.view()

#

# gen_autolab_model()
plot_model()
