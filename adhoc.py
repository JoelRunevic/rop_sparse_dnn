import pickle 
import torch 
import copy 
import torch.backends.cudnn as cudnn

from models import *

from parralel_axis import *
from utils import *

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

dt_string = "14_04_2024:16:38:07"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10

with open(f'experiment_config/config_{dt_string}.pkl', 'rb') as f:
    config = pickle.load(f)

initial_network_state = torch.load(f'experiment_states/initial_network_state_{dt_string}.pth')
# pruned_model_state = torch.load(f'experiment_models/pruned_model_{dt_string}.pth')
pruned_model_state = torch.load(f'experiment_models/trained_pruned_model_{dt_string}.pth')

network = creating_network(device, num_classes)
network.load_state_dict(pruned_model_state)

get_network_sparsity(network)



