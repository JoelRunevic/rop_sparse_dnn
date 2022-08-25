import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
#from main import test

from sparse_functions import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', default='resnet18',  help='network')
parser.add_argument('--cifar100', action ='store_true', help='use cifar10')
parser.add_argument('--ckpt', default='checkpoint/ckpt.', help='path to training checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if not args.cifar100:
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
else:
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 100
# Model
print('==> Building model..')
print(args.net, num_classes)
if args.net in "resnet18":
    net = ResNet18_sparse(num_classes = num_classes)
elif args.net == "resnet50":
    net = ResNet50_sparse(num_classes = num_classes)
elif args.net == "vgg19":
    net = VGG_sparse('VGG19', num_classes = num_classes)
elif args.net == "mobilenet":
    net = MobileNetV2_sparse(num_classes = num_classes)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.ckpt)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            

def remove_small_values(net, thresh=1e-3, verbose=False):
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        weight_npy = np.abs(layer._weight.data.cpu().numpy()) 
        #thresh = np.percentile(weight_npy.flatten(), prune_percent)
        # Get indexs of all weights to be zeroed and then zero them
        idxs = weight_npy < thresh
        layer._mask.data[idxs] = 0

        if verbose:
            num_nonzero = layer._mask.sum().item()
            print(num_nonzero, num_total)
            sparsity = 100.0 * (1 - (num_nonzero / num_total))
            print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))

def print_layer_sparsity(net):
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_total = len(layer._mask)
        num_nonzero = layer._mask.sum().item()
        print(num_nonzero, num_total)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))

def get_number_zero(weights):
    uniques, counts = np.unique(weights, return_counts=True)
    return counts[np.where(uniques == 0)]

def get_sparsity(weights):
    return get_number_zero(weights) / weights.size

def get_sparsity(net):
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        print(get_sparsity(layer.weight.cpu().detach().numpy()))

epoch = 0
print_layer_sparsity(net)
test(epoch)

remove_small_values(net, verbose=True)
test(epoch)
