import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import math
import numpy as np
from scipy.io import savemat
import time

from utils_bfp import *

def bfp_quantize_weights(w, exponent_bit, mantissa_bit, group_size):
    # im2col weights
    weight_matrix = w.reshape(w.shape[0], -1)


    pad_amount  = 0
    # Pad weights to match group size
    if weight_matrix.shape[1] % group_size != 0:
        pad_amount = group_size - weight_matrix.shape[1] % group_size
        pad = np.zeros((weight_matrix.shape[0], pad_amount))
        weight_matrix = np.concatenate((weight_matrix, pad), axis = 1)


    # reshape for groups
    weight_matrix_shape = weight_matrix.shape
    print(weight_matrix_shape)
    weight_matrix = weight_matrix.reshape(-1, group_size)
        
    # Apply bfp quantization to groups
    weight_matrix = bfp_quantize(torch.tensor(weight_matrix).cuda(), EXPONENT_WIDTH=exponent_bit, MANTISSA_WIDTH=mantissa_bit, quant_dim=1)
    #unsahpe groups
    weight_matrix = weight_matrix.reshape(weight_matrix_shape)

    # remove_padding
    if pad_amount > 0:
        weight_matrix  = weight_matrix[:,0:weight_matrix.shape[1]-pad_amount]
    # reshape weights back to normal
    return weight_matrix.reshape(w.shape)


if __name__ == "__main__":
    input = torch.tensor(np.random.randn(4,3,1,1))

    print(input.reshape(input.shape[0], -1))
    print(bfp_quantize_weights(input, 4,4, 2).reshape(input.shape[0], -1))
