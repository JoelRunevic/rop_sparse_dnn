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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--path', default = './checkpoint/ckpt.pth', help='path to checkpoint')
parser.add_argument('--net', default='resnet18',  help='network')
parser.add_argument('--cifar100', action ='store_true', help='use cifar10')

args = parser.parse_args()

if args.cifar100:
    num_classes = 100
else:
    num_classes = 10
    
if args.net in "resnet18":
    net = ResNet18_sparse(num_classes = num_classes)
elif args.net == "resnet50":
    net = ResNet50_sparse(num_classes = num_classes)

#net = ResNet18_sparse()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

net = net.to(device)
#if device == 'cuda':
net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True
# Load model
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.path, map_location='cpu')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

input_shape = 32
# Print weights as Matrices
# useful: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
for i, layer in enumerate(get_sparse_conv2d_layers(net)):
    M = input_shape**2
    input_shape = input_shape / layer.stride[0]
    weight = layer.weight
    weight_mm = weight.cpu().detach().view(weight.size(0), -1).t()
    print(i)
    #print(weight_mm)
    sio.mmwrite(path+"/layer_{}.mtx".format(i), weight_mm)
    f= open(path+"/layer_{}.mtx".format(i), 'r+')
    content = f.read()
    f.seek(0)
    #x = "M: {} K: {} N {}".format(weight_mm.shape[1], weight_mm.shape[0], int(M))
    x = "M: {} K: {} N {}".format(int(M), weight_mm.shape[1], weight_mm.shape[0])
    f.write(x + '\n' + content)
    f.close()

