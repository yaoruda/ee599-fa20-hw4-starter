import numpy as np
import os
import os.path as osp
import argparse

Config ={}

local = False

if local:
    Config['root_path'] = '/Users/yaoruda/Documents/Labs/data/polyvore_outfits'
    Config['checkpoint_path'] = '/Users/yaoruda/Documents/Labs/data/models'
else:
    Config['root_path'] = '/root/data/polyvore_outfits'
    Config['checkpoint_path'] = '/home/ec2-user/ee599-fa20-hw4-starter/models'

Config['meta_file'] = 'polyvore_item_metadata.json'


Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 2
Config['batch_size'] = 32

Config['learning_rate'] = 0.001
Config['num_workers'] = 5

