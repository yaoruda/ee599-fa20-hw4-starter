import numpy as np
import os
import os.path as osp
import argparse


Config ={}

local = False
TX = True
Config['debug'] = False
Config['finetune'] = True
Config['half_finetune'] = True
Config['ruda_model'] = False

Config['num_epochs'] = 30
Config['batch_size'] = 8000
Config['learning_rate'] = 0.01
Config['num_workers'] = 8

Config['use_cuda'] = True

if local:
    Config['root_path'] = '/Users/yaoruda/Documents/Labs/data/polyvore_outfits'
    Config['checkpoint_path'] = '/Users/yaoruda/Documents/Labs/data/models'
else:
    if TX:
        Config['root_path'] = '/home/ubuntu/dataset/polyvore_outfits'
        Config['checkpoint_path'] = '/home/ubuntu/ee599-fa20-hw4-starter/models'
    else:
        Config['root_path'] = '/home/ec2-user/data/polyvore_outfits'
        Config['checkpoint_path'] = '/home/ec2-user/ee599-fa20-hw4-starter/models'


Config['meta_file'] = 'polyvore_item_metadata.json'



