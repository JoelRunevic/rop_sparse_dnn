'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from datetime import datetime

import os
import argparse

import copy 

from models import *
from utils import progress_bar

from sparse_functions import *

# import EarlyStopping
from pytorchtools import EarlyStopping

import pickle

# Example Usage: python main.py --rop --p 75

# TODO: change the group size from 18 to powers of 2, do hyperparam search, ask Vikas about commented out part.

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--p', default=0, type=int, help='Prune percentage')
parser.add_argument('--rop', action='store_true', help='use rop pruning')
parser.add_argument('--layer' , action='store_true', help='use layer pruning')
parser.add_argument('--filter', action ='store_true', help='prune entire filters')
parser.add_argument('--cifar100', action ='store_true', help='use cifar10')
parser.add_argument('--net', default='resnet18',  help='network')

parser.add_argument('--es', action='store_true', help='use early stopping')
parser.add_argument('--t', default = 0, type=int, help='percent of sigma values to prune')
parser.add_argument('--svd', default = -1, type=int, help='epoch to apply svd')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

patience = 15
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True)

has_stopped = False

def get_pruning_config(epoch=10, iters = 10):
    global args
    #Pruning
    if args.p:
            print("Prune percent: {}".format(args.p))

    prune_percentages = np.arange(10, rounddown(args.p)+10, 10)
    prune_epochs = np.arange(epoch, epoch + round(args.p/ 10)*iters, iters)
    #handle last pruning stage
    if args.p%10 != 0:
      prune_percentages = np.append(prune_percentages, args.p)#[10, 20, 30, 40, 50, 60, 70]                          
      prune_epochs = np.append(prune_epochs, roundup(args.p))	# TODO fix to handle svd stuff
    #prune_percentages = np.arange(10, args.p+10, 10) #[10, 20, 30, 40, 50, 60, 70]
    #prune_epochs = np.arange(10, args.p+10, 10)
    #prune_epochs[0] = 0 # = [10, 20, 30, 40, 50, 60, 70]
    return prune_epochs, prune_percentages


prune_epochs, prune_percentages = get_pruning_config(10, 10)
curr_prune_stage = 0

# print(f"Prune epochs: {prune_epochs}", f"Prune percentages: {prune_percentages}")

# Function for resetting network to initialization, but copying over mask.

def reinitialize_model_with_mask(net):
    copied_original_layers = copy.deepcopy(get_sparse_conv2d_layers(net))
    net.load_state_dict(initial_network_state)
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        layer._mask = copied_original_layers[i]._mask


# Training
def train(epoch, net):
    global curr_prune_stage
    global has_stopped
    global prune_epochs
    global prune_percentages
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if args.svd == epoch:
        print("applying svd")
        svd_pruning(net, 50)
    if (epoch in prune_epochs) and (has_stopped is True):
        print('Pruning Stage {}'.format(curr_prune_stage))

        if args.layer:
            smallest_magnitude_pruning(net, prune_percentages[curr_prune_stage])
        if args.rop:
            rop_pruning(net, prune_percentages[curr_prune_stage],  multi_prune= args.net == "mobilenet")
        if args.filter:
            struct_pruning(net, prune_percentages[curr_prune_stage])
        curr_prune_stage += 1

        # Resetting network to initialization, but copying over mask.
        # TODO: Is this really resetting the weights?
        reinitialize_model_with_mask(net)
        print("Reinitialized unpruned weights to initial values.")



def no_prune_train(epoch, network):
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



def no_prune_test(epoch, network):
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



def test(epoch, net):
    global best_acc
    global has_stopped
    global prune_epochs
    global prune_percentages
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

    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    if has_stopped ==  False and args.svd == -1:
        early_stopping(test_loss, net)

    
    if early_stopping.early_stop and has_stopped == False:
        print("Early stopping")
        
        ###  I COMMENTED THIS OUT; I AM NOT SURE WHAT IT DOES. ####

        # svd_pruning(net, 90, True)
        prune_epochs, prune_percentages = get_pruning_config(epoch+2, 5)
        print(epoch, prune_epochs, prune_percentages)

        has_stopped = True

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')
        if args.cifar100:
            cifar="CIFAR100"
        else:
            cifar="CIFAR10"
        if args.rop:
             torch.save(state, './checkpoint/ckpt_{}_{}_rop_{}.pth'.format(args.net, cifar, args.p))
        if args.layer:
             torch.save(state, './checkpoint/ckpt_{}_{}_layer_{}.pth'.format(args.net, cifar, args.p))
        if args.filter:
             torch.save(state, './checkpoint/ckpt_{}_{}_filter_{}.pth'.format(args.net, cifar, args.p))
        else:
            torch.save(state, './checkpoint/ckpt_{}_{}_no_pruning.pth'.format(args.net, cifar))
        best_acc = acc

# Datetime string for saving models.
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y:%H:%M:%S")

# Saving model weights:
initial_network_state = copy.deepcopy(net.state_dict())
torch.save(initial_network_state, f'experiment_models/initial_model_{dt_string}.pth')

# Pruning step.
for epoch in range(start_epoch, start_epoch+200):
    train(epoch, net)
    test(epoch, net)
    scheduler.step()

# Pruning is done.
verbose = True
for i, layer in enumerate(get_sparse_conv2d_layers(net)):
    if verbose:
            num_total = len(layer._mask)
            num_nonzero = layer._mask.sum().item()
            print(num_nonzero, num_total)
            sparsity = 100.0 * (1 - (num_nonzero / num_total))
            print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))

print("Finished pruning the model.")

def no_prune_training_loop(network):
    EPOCHS = 200
    patience = 15
    net_stats = []

    optimizer = optim.SGD(network.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    

    # Training the model without doing any pruning.
    for epoch in range(EPOCHS):
        train_acc, train_loss, _ = no_prune_train(epoch, network)
        test_acc, test_loss, _ = no_prune_test(epoch, network)
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

# Saving the pruned model.
torch.save(net.state_dict(), f'experiment_models/pruned_model_{dt_string}.pth')

# Reinitializing the pruned network to initial weights, keeping the learned mask.
reinitialize_model_with_mask(net)

# Training pruned model, saving the trained model & statistics.
pruned_net_stats = no_prune_training_loop(net)
torch.save(net.state_dict(), f'experiment_models/retrained_pruned_model_{dt_string}.pth')
with open(f'experiment_lists/retrained_pruned_model_{dt_string}.pkl', 'wb') as f:
    pickle.dump(pruned_net_stats, f)
print("\nFinished retraining the pruned model and saving all the statistics.\n")


# Training the initialized model without pruning, saving the trained model & statistics.
initial_net = ResNet18_sparse(num_classes = num_classes)
initial_net = initial_net.to(device)
if device == 'cuda':
    initial_net = torch.nn.DataParallel(initial_net)
    cudnn.benchmark = True

initial_net.load_state_dict(initial_network_state)
initial_net_stats = no_prune_training_loop(initial_net)
torch.save(initial_net.state_dict(), f'experiment_models/retrained_initial_model_{dt_string}.pth')
with open(f'experiment_lists/retrained_initial_model_{dt_string}.pkl', 'wb') as f:
    pickle.dump(initial_net_stats, f)
print("\nFinished retraining the initial model and saving all the statistics.\n")


if args.cifar100:
    cifar="CIFAR100"
else:
    cifar="CIFAR10"
if args.rop:
     print("{} {}% ROP pruning on {}  Acc: {}%".format(args.net, args.p, cifar, best_acc))
if args.layer:
     print("{} {}% layer pruning on {} Acc: {}%".format(args.net,args.p, cifar, best_acc))
if args.filter:
    print("{} {}% filter pruning on {} Acc: {}%".format(args.net, args.p, cifar, best_acc))
else:
     print("{} on {} Acc: {}%".format(args.net, cifar, best_acc))

### EVERYTHING HERE IS TEMP CODE. ### 
# initial_net_state = torch.load('experiment_models/initial_model_24_03_2024:01:12:45.pth')
# net.load_state_dict(initial_net_state)

# initial_net_stats = no_prune_training_loop(net)
# tmp_dt_string = "24_03_2024:01:12:45"

# torch.save(net.state_dict(), f'experiment_models/retrained_initial_model_{tmp_dt_string}.pth')
# with open('experiment_lists/retrained_initial_model_{tmp_dt_string}.pkl', 'wb') as f:
#     pickle.dump(initial_net_stats, f)
# print("\nFinished retraining the initial model and saving all the statistics.\n")


#####################################