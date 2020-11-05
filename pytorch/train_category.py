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

from logger import logger, logger_acc


def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    info_dict = {
        'phase': [],
        'epoch': [],
        'loss': [],
        'acc': [],
    }

    for epoch in range(num_epochs):

        if Config['half_finetune']:
            if epoch == 15:
                # unfreeze the ResNet34 layers
                for name, value in model.named_parameters():
                    if (name != 'fc.weight') and (name != 'fc.bias'):
                        value.requires_grad = True
                for name, param in model.named_parameters():
                    print(name, param.requires_grad)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            info = 'epoch:{} {} - Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc)
            logger.info(info)
            info_dict['phase'].append(phase)
            info_dict['epoch'].append(epoch)
            info_dict['loss'].append(round(epoch_loss, 4))
            info_dict['acc'].append(round(epoch_acc.item(), 4))
            logger_acc.info(info_dict)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # same model for each epoch
        model_name = 'model_epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), osp.join(Config['checkpoint_path'], model_name))
        print('Model saved at: {}'.format(osp.join(Config['checkpoint_path'], model_name)))
        # save best model
        torch.save(best_model_wts, osp.join(Config['checkpoint_path'], 'best_model.pth'))
        print('Best Model saved at: {}'.format(osp.join(Config['checkpoint_path'], 'best_model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))


if __name__ == '__main__':

    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'],
                                                        num_workers=Config['num_workers'])

    if Config['ruda_model']:
        model = Ruda_Model()
    else:
        model = torch_model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)  # repleace the fc layer to fit this problem

    # print(model)
    summary(model, input_size=(3, 224, 224))


    if Config['half_finetune']:
        # freeze the ResNet34 layers
        for name, value in model.named_parameters():
            if (name != 'fc.weight') and (name != 'fc.bias'):
                value.requires_grad = False

        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    if torch.cuda.is_available():
        print("USE GPU")
    else:
        print("USE CPU")

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'],
                dataset_size=dataset_size)
