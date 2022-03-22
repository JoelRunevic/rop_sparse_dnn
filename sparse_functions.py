import numpy as np
from models import *

from parralel_axis import *
from utils import *

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


def smallest_magnitude_pruning(net, prune_percent, verbose=True):
    
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent)
        
        # PART 3.3: Implement pruning by settings elements in layer._mask
        #           to zero corresponding to the smallest magnitude elements
        #           in layer._weight. This must be implement in a way that 
        #           allows the function to be called multiple times during the
        #           training phase in order to make the weights progressively
        #           sparser over time. Note: to update variable such as
        #           layer._mask and layer._weight, do the following:
        #           layer._mask.data[idx] = 0
        
        # Find upperbound of weights to be pruned
        weight_npy = np.abs(layer._weight.data.cpu().numpy()) 
        thresh = np.percentile(weight_npy.flatten(), prune_percent)
        # Get indexs of all weights to be zeroed and then zero them
        idxs = weight_npy < thresh
        layer._mask.data[idxs] = 0

        if verbose:
            num_nonzero = layer._mask.sum().item()
            print(num_nonzero, num_total)
            sparsity = 100.0 * (1 - (num_nonzero / num_total))
            print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))

def get_prune_group(row, prune_percent):
    thresh = np.percentile(row.flatten(), prune_percent)
    # Get indexs of all weights to be zeroed and then zero them
    idxs = row < thresh
    return idxs


def rop_pruning(net, prune_percent, verbose=True, multi_prune = False):
    
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent)
                
        # Find upperbound of weights to be pruned
        #weight_npy = np.abs(layer._weight.data.cpu().numpy())
        #thresh = np.percentile(weight_npy.flatten(), prune_percent)
        # Get indexs of all weights to be zeroed and then zero them
        #idxs = weight_npy < thresh
        weight_npy = np.abs(layer.weight.cpu().detach().numpy())
        weight_npy = weight_npy.reshape(weight_npy.shape[0], weight_npy.shape[1], -1)
        #idxs = np.empty_like(weight_npy)
        #print(weight_npy.shape)
        #print(layer.weight.shape)
        #for j in range(0, weight_npy.shape[0]):
            #for k in range(0, weight_npy.shape[1]):
                #idxs[j,k] = get_prune_group(weight_npy[j,k,:], prune_percent)
        #idxs = np.apply_along_axis(get_prune_group, 2, weight_npy, prune_percent)
        within_filter = False
        if True:
            group_size = get_closest_split(weight_npy.shape[0], 18)
            if group_size is None:
                # Handle very small filters
                group_size = weight_npy.shape[0]
            weights = weight_npy.reshape(weight_npy.shape[0], -1).T
            #print(weights[0])
            weights = weights.flatten().reshape([-1, group_size])
            #print(weights[0])
            #idxs = np.array([get_prune_group(row, prune_percent) for row in weights])
            if multi_prune:
                idxs = np.array([get_prune_group(row, prune_percent) for row in weights]) 
            else:
                idxs = parallel_apply_along_axis(get_prune_group, 1, weights, prune_percent)
            idxs = idxs.flatten().reshape(-1, weight_npy.shape[0]).T
            idxs = idxs.reshape(weight_npy.shape)
        elif weight_npy.shape[-1] == 1:
            group_size = get_closest_split(weight_npy.shape[0] * weight_npy.shape[1], 9)
            #weights = weight_npy.reshape(weight_npy.shape[0], weight_npy.shape[1]) #np.concatenate(np.split(weight_npy, group_size, 0))
            weights = weight_npy.flatten().reshape([-1, group_size])
            idxs = np.array([get_prune_group(row, prune_percent) for row in weights]) #parallel_apply_along_axis(get_prune_group, 0, weights, prune_percent)
            idxs = idxs.reshape(weight_npy.shape)
        elif within_filter:
            group_size = get_closest_split(weight_npy.shape[2] * weight_npy.shape[1], 18)
            weights = weight_npy.flatten().reshape([-1, group_size])
            idxs = np.array([get_prune_group(row, prune_percent) for row in weights]) 
            idxs = idxs.reshape(weight_npy.shape)
        else:
            group_size = get_closest_split(weight_npy.shape[0], 18)
            weights = weight_npy.reshape(weight_npy.shape[0], -1).T
            #print(weights[0])
            weights = weights.flatten().reshape([-1, group_size])
            #print(weights[0])
            #idxs = np.array([get_prune_group(row, prune_percent) for row in weights])
            idxs = parallel_apply_along_axis(get_prune_group, 1, weights, prune_percent)
            idxs = idxs.flatten().reshape(-1, weight_npy.shape[0]).T
            idxs = idxs.reshape(weight_npy.shape)
            #with np.printoptions(threshold=np.inf):
                #print(idxs.reshape(weight_npy.shape[0], -1).T)
        #print(idxs)
        #idxs = idxs > 0.5 
        layer._mask.data[idxs.flatten()] = 0
        #print(idxs[0,0])
        #print(idxs.flatten()[0:18])
        #print(idxs[0][0])
        #print(layer.mask[0][0])
        #print(layer.weight.cpu().detach().numpy()[0][0])


        if verbose:
            num_nonzero = layer._mask.sum().item()
            print(num_nonzero, num_total)
            sparsity = 100.0 * (1 - (num_nonzero / num_total))
            print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))


def struct_group_pruning(filter, prune_percent):
  #filters = filter.sum(axis=1)
  filters = np.amax(filter, axis=1)
  thresh = np.percentile(filters, prune_percent)
  #indx = filter.sum(axis=1) < thresh
  indx = np.amax(filter, axis=1) < thresh
  return indx


def struct_pruning(net, prune_percent, verbose=True):
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent)

        weight_npy = np.abs(layer.weight.cpu().detach().numpy())
        weight_npy = weight_npy.reshape(weight_npy.shape[0], weight_npy.shape[1], -1)
        idxs = np.zeros_like(weight_npy)
        for j in range (0, weight_npy.shape[0]):
            #print(idxs.shape)
            #print(struct_group_pruning(weight_npy[i], prune_percent))
            idxs[j][struct_group_pruning(weight_npy[i], prune_percent)] = True
        #print(idxs)
        idxs = idxs > 0.5
        #print(idxs)
        #print(idxs.reshape(weight_npy.shape[0],weight_npy.shape[1], weight_npy.shape[2], weight_npy.shape[3]))
        layer._mask.data[idxs.flatten()] = 0
        
        if verbose:
            num_nonzero = layer._mask.sum().item()
            #print(num_nonzero, num_total)
            sparsity = 100.0 * (1 - (num_nonzero / num_total))
            print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))
