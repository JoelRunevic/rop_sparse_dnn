''' Finding Rosko Tickets '''
# Example Usage: python finding_tickets.py --lr 0.1 --p 88 --g 16 --d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import detectors # this library is necessary for timm to load up the correct pretrained model.
import timm 

import torchvision
import torchvision.transforms as transforms

from datetime import datetime

import os
import argparse

import copy 

from models import *
from sparse_functions import *

from utils import progress_bar

from sparse_functions import *
from pytorchtools import EarlyStopping

import pickle

####################################              TRAINING FUNCTIONS             ####################################

def creating_network(device, num_classes):
    net = ResNet18_sparse(num_classes = num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True 
    return net 

def replace_conv_with_sparse(model):
    def _replace_recursively(model):
        for name, layer in list(model.named_children()):  # Iterate over named modules
            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels
                out_channels = layer.out_channels
                kernel_size = (layer.kernel_size)[0]
                stride = (layer.stride)[0] 
                padding = (layer.padding)[0]

                # Defining a sparse layer.
                new_sparse_layer = SparseConv2d(in_channels, out_channels, kernel_size,
                                                stride, padding, None, False)

                # Copy weights and bias (if exists).
                layer_weight_data = layer.weight.data.reshape(-1)
                new_sparse_layer._weight.data.copy_(layer_weight_data)
                if layer.bias is not None:
                    new_sparse_layer.bias.data.copy_(layer.bias.data)

                # Replace the original Conv2d layer
                setattr(model, name, new_sparse_layer)
            else:
                _replace_recursively(layer)
    
    _replace_recursively(model)

def create_pretrained_network(device):
    # Getting the pre-trained weights.
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    
    # Replacing the conv layers with sparse conv layers.
    replace_conv_with_sparse(model)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model


def get_network_sparsity(network):
    for i, layer in enumerate(get_sparse_conv2d_layers(network)):
        num_total = len(layer._mask)
        num_nonzero = layer._mask.sum().item()
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                    sparsity))
        print(num_nonzero, num_total, num_nonzero/num_total)

def get_layer_sparsity(num_layer, layer):
    total = len(layer._mask)
    num_non_zero = layer._mask.sum().item()
    print(f"Layer Number: {num_layer+1} has sparsity: {1 - (num_non_zero / total)}")


def reinitialize_model_with_mask(net, initial_net_state):
    copied_original_layers = copy.deepcopy(get_sparse_conv2d_layers(net))
    net.load_state_dict(initial_net_state)
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        layer._mask = copied_original_layers[i]._mask

        # total_elems = layer.out_channels * layer.in_channels * layer.kernel_size * layer.kernel_size

        # w_reinitialized = layer._weight.view(layer.out_channels, layer.in_channels, layer.kernel_size, layer.kernel_size)
        
        # copied_layer = copied_original_layers[i]
        # w_original = copied_layer._weight.view(copied_layer.out_channels, copied_layer.in_channels,
        #                                        copied_layer.kernel_size, copied_layer.kernel_size)
        
        # total_equal = (w_reinitialized == w_original).sum().item()

        # print(total_equal, total_elems, total_equal / total_elems, "\n")



def training_loop(network):
    EPOCHS = 200
    patience = 15
    net_stats = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    

    # Training the model per epoch.
    for epoch in range(EPOCHS):
        train_acc, train_loss, _ = train_iter(epoch, network, optimizer, criterion)
        test_acc, test_loss, _ = test_iter(epoch, network, criterion)
        scheduler.step()

        early_stopping(test_loss, network)

        net_stats.append({
            "Train Accuracy": train_acc, 
            "Test Accuracy": test_acc,
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Epoch": epoch
        })

        if early_stopping.early_stop:
            break
    
    return net_stats

def train_iter(epoch, network, optimizer, criterion):
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    print('\nEpoch: %d' % epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    normalized_loss = train_loss / len(trainloader)

    return acc, normalized_loss, epoch

def test_iter(epoch, network, criterion):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    acc = 100.*correct/total
    normalized_loss = test_loss / len(testloader)
    
    return acc, normalized_loss, epoch

########################################################################################################################

####################################              SETUP + ARGUMENT HANDLING               ####################################

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--p', default=88, type=int, help='Prune percentage')
parser.add_argument('--g', default=16, type=int, help='Group size')
parser.add_argument('--d', action='store_true', help='If True, then CIFAR 10, else CIFAR 100')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 10 if args.d else 100

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y:%H:%M:%S")

print(args)


####################################              PREPARING THE DATA               ####################################
# Data
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


if args.d:
    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    print("Prepared CIFAR 10 data.\n")
    
else:
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    print("Prepared CIFAR 100 data.\n")


# ##################################################################################################################

# ####################################              PRUNING LOOP                ####################################



# CIFAR 10, so we randomly initialize the network as per usual.
if args.d:
    net_to_prune = creating_network(device, num_classes)

# If CIFAR 100, then load the CIFAR 100 pretrained model.
else:
    net_to_prune = create_pretrained_network(device)

torch.save(net_to_prune.state_dict(), f'experiment_states/initial_network_state_{dt_string}.pth')


prune_percentages = np.arange(10, rounddown(args.p)+10, 10)
prune_percentages = np.append(prune_percentages, args.p)

pruning_stats = []
for prune_stage, prune in enumerate(prune_percentages):
    print(f"Pruning Stage: {prune_stage + 1} / {len(prune_percentages)}\n\n")

    # Training.
    stat = training_loop(net_to_prune)
    pruning_stats.append(stat)

    # Pruning; last two arguments are (verbose, mobilenet).         
    rop_pruning(net_to_prune, prune_percentages[prune_stage], args.g, True, False)

    # Reinitialization.
    initial_network_state = torch.load(f'experiment_states/initial_network_state_{dt_string}.pth')
    reinitialize_model_with_mask(net_to_prune, initial_network_state)

# Model has been pruned.
print(f"\n\n\nModel has been pruned.\n\n\n")
torch.save(net_to_prune.state_dict(), f'experiment_models/pruned_model_{dt_string}.pth')

# Saving pruning stats.
with open(f'experiment_lists/pruning_stats_{dt_string}.pkl', 'wb') as f:
    pickle.dump(pruning_stats, f)


##################################################################################################################


# ####################################           TRAINING INITIAL MODEL        ###################################

# For CIFAR 100 (i.e. args.d == False), we require to use a pretrained model.
if args.d:
    initial_net_to_train = creating_network(device, num_classes)
else:
    initial_net_to_train = create_pretrained_network(device)

initial_network_state = torch.load(f'experiment_states/initial_network_state_{dt_string}.pth')
initial_net_to_train.load_state_dict(initial_network_state)

initial_stats = training_loop(initial_net_to_train)
print("\n\n\n Initial model has been retrained without pruning.\n\n\n")

# Saving trained model.
torch.save(initial_net_to_train.state_dict(), f'experiment_models/trained_initial_model_{dt_string}.pth')

# Saving training stats.
with open(f'experiment_lists/training_initial_model_{dt_string}.pkl', 'wb') as f:
    pickle.dump(initial_stats, f)


##################################################################################################################


# ####################################           TRAINING PRUNED MODEL        ###################################
    
# Loading the pruned model with mask; pretrained network for CIFAR 100.
if args.d:
    pruned_net_to_train = creating_network(device, num_classes)
else:
    pruned_net_to_train = create_pretrained_network(device)

pruned_network_state = torch.load(f'experiment_models/pruned_model_{dt_string}.pth')
pruned_net_to_train.load_state_dict(pruned_network_state)

# Reinitializing weights to what they were before.
initial_network_state = torch.load(f'experiment_states/initial_network_state_{dt_string}.pth')
reinitialize_model_with_mask(pruned_net_to_train, initial_network_state)

# Training the pruned model.
pruned_stats = training_loop(pruned_net_to_train)
print("\n\n\n Pruned model has been retrained without further pruning.\n\n\n")

# Saving trained model.
torch.save(pruned_net_to_train.state_dict(), f'experiment_models/trained_pruned_model_{dt_string}.pth')

# Saving training stats.
with open(f'experiment_lists/training_pruned_model_{dt_string}.pkl', 'wb') as f:
    pickle.dump(pruned_stats, f)


# Saving the config.
config = {
    "lr": args.lr,
    "p": args.p, 
    "g": args.g,
    "CIFAR-10": args.d
}

with open(f'experiment_config/config_{dt_string}.pkl', 'wb') as f:
    pickle.dump(config, f)












