import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.autograd import Variable

import sys
sys.path.append("/home/kunglab/training/rop_sparse_dnn")
from BFP_quantized import *

def _make_pair(x):
    if hasattr(x, '__len__'):
        return x
    else:
        return (x, x)

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=1, groups = None, bias=False):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        self.groups = groups
        n = out_channels*in_channels*kernel_size*kernel_size

        # Note: making _weight and _mask one-dimensional makes implementation
        # of pruning significantly easier. Internally, we performing 'view'
        # operations to convert them to 4D tensors before convolution.
        self._weight = nn.Parameter(torch.Tensor(n))
        stdv = 1. / math.sqrt(in_channels)
        self._weight.data.uniform_(-stdv, stdv)
        self.register_buffer('_mask', torch.ones(n))


    def forward(self, x):
        if self.groups:
            return F.conv2d(x, self.weight, stride=self.stride,
                            padding=self.padding, groups=self.groups)
        else:
            #qweight = BFP_quantize(self._weight, exponent_bit=4, mantissa_bit=8, group_size=4, category='WGT',  partition='channel', max_exp=0, min_exp=-15)
            #qinput = BFP_quantize(x.cuda(), exponent_bit=4, mantissa_bit=8, group_size=1, category='ACT',  partition='channel', max_exp=0, min_exp=-15).to(torch.device("cpu"))
            #qinput = qinput.cuda()
            #return F.conv2d(qinput, self.weight, stride=self.stride,
            #                padding=self.padding)
            return F.conv2d(x, self.weight, stride=self.stride,
                            padding=self.padding)
                    
    @property
    def weight(self):
        w = self._weight.view(self.out_channels, self.in_channels,
                              self.kernel_size, self.kernel_size)
        return self.mask * w
    
    @property
    def mask(self):
        m = self._mask.view(self.out_channels, self.in_channels,
                            self.kernel_size, self.kernel_size)
        return Variable(m, requires_grad=False)
