import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

from models import *
from sparse_functions import *

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
    weight_mm = weight.view(weight.size(0), -1).t()
    print(weight.view(weight.size(0), -1).t().shape)
    print(weight_mm)
