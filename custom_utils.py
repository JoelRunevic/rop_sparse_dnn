from models import *

from parralel_axis import *
from utils import *

import torch.backends.cudnn as cudnn
import torch.optim as optim

from pytorchtools import EarlyStopping

import copy 

# Just a file of util functions that I often use.

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device 

def creating_network(device, num_classes):
    net = ResNet18_sparse(num_classes = num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True 
    return net    

def get_sparse_conv2d_layers(net):
    '''
    Helper function which returns all SparseConv2d layers in the net.
    Use this below to implement layerwise pruning.
    '''
    sparse_conv_layers = []
    for layer in net.children():
        if isinstance(layer, SparseConv2d):
            sparse_conv_layers.append(layer)
        else:
            child_layers = get_sparse_conv2d_layers(layer)
            sparse_conv_layers.extend(child_layers)
    
    return sparse_conv_layers

def get_layers(net, layer_type):
    '''
    Helper function which returns all layers of type "layer_type" in the net.
    '''
    layers = []
    for layer in net.children():
        if isinstance(layer, layer_type):
            layers.append(layer)
        else:
            child_layers = get_layers(layer, layer_type)
            layers.extend(child_layers)
    return layers



def reinitialize_model_with_mask(net, initial_net_state):
    copied_original_layers = copy.deepcopy(get_sparse_conv2d_layers(net))
    net.load_state_dict(initial_net_state)
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        layer._mask = copied_original_layers[i]._mask

def get_network_sparsity(network):
    for i, layer in enumerate(get_sparse_conv2d_layers(network)):
        num_total = len(layer._mask)
        num_nonzero = layer._mask.sum().item()
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                    sparsity))
        print(num_nonzero, num_total, num_nonzero/num_total)

def training_loop(network, optimizer, trainloader, testloader, device):
    EPOCHS = 200
    patience = 15
    net_stats = []

    criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    

    # Training the model per epoch.
    for epoch in range(EPOCHS):
        train_acc, train_loss, _ = train_iter(epoch, network, optimizer, criterion, trainloader, device)
        test_acc, test_loss, _ = test_iter(epoch, network, criterion, testloader, device)
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

def train_iter(epoch, network, optimizer, criterion, trainloader, device):
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

def test_iter(epoch, network, criterion, testloader, device):
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