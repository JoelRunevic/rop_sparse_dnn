import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

import os
import argparse

from models import *
from sparse_functions import *


path = './sparse_matrices'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=False)

net = ResNet18_sparse()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True
# Load model
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# Print weights as Matrices
# useful: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
for i, layer in enumerate(get_sparse_conv2d_layers(net)):
    weight = layer.weight
    weight_mm = weight.cpu().detach().view(weight.size(0), -1).t()
    print(i)
    #print(weight_mm)
    sio.mmwrite(path+"/layer_{}.mtx".format(i), weight_mm)

