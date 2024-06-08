''' Trying to find early bird tickets with Rosko pruning '''

import torch 
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import copy 

from collections import deque

from custom_utils import *
from sparse_functions import *

from datetime import datetime

# General network variables.
device = get_device()
num_classes = 10

# Rosko variables.
group_size = 16
pruning_percent = 87.5

# Early bird tickets variables.
early_bird_epochs = 150
queue_length = 5

# Hamming distance function.
def hamming_distance(mask1, mask2):
    return (mask1 != mask2).sum().item()


# Preparing the dataset.
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Initializing the network.
net = creating_network(device, num_classes)

# Initializing training dynamics.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Queue for early bird tickets.
mask_queue = deque(maxlen=queue_length)
distances = []

# Training the network.
for epoch in range(early_bird_epochs):
    # Computing the training and test accuracy.
    train_acc, train_loss, _ = train_iter(epoch, net, optimizer, criterion, trainloader, device)
    test_acc, test_loss, _ = test_iter(epoch, net, criterion, testloader, device)
    scheduler.step()

    # Pruning the network with Rosko. 
    rop_pruning(net, pruning_percent, group_size, True, False)

    # Adding the masks to the queue
    conv_layers = get_sparse_conv2d_layers(net)
    masks = [copy.deepcopy(layer._mask) for layer in conv_layers]
    mask_queue.append(masks)

    # Resetting the network mask to torch.ones 
    for layer in conv_layers:
        layer._mask = nn.Parameter(torch.ones_like(layer._mask))

    # Computing the average hamming distance between the masks in the queue.
    if len(mask_queue) == queue_length:
        ref_mask_lst = mask_queue[-1]
        total_dist = 0
        total_elems = 0 

        for i in range(queue_length - 1):
            curr_mask_lst = mask_queue[i]

            for curr_mask_lst_mask, ref_mask_lst_mask in zip(curr_mask_lst, ref_mask_lst):
                total_dist += hamming_distance(curr_mask_lst_mask, ref_mask_lst_mask)
                total_elems += curr_mask_lst_mask.size(0)

        average_dist = total_dist / total_elems

        distances.append(average_dist)


print(distances)