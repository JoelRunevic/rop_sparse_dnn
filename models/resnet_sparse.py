'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.autograd import Variable
## Code Cell 3.1

def _make_pair(x):
    if hasattr(x, '__len__'):
        return x
    else:
        return (x, x)

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=1, bias=False):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        n = out_channels*in_channels*kernel_size*kernel_size

        # Note: making _weight and _mask one-dimensional makes implementation
        # of pruning significantly easier. Internally, we performing 'view'
        # operations to convert them to 4D tensors before convolution.
        self._weight = nn.Parameter(torch.Tensor(n))
        stdv = 1. / math.sqrt(in_channels)
        self._weight.data.uniform_(-stdv, stdv)
        self.register_buffer('_mask', torch.ones(n))


    def forward(self, x):
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

## Code Cell 3.2

def sparse_conv_block(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1):
    '''
    Replaces 3x3 nn.Conv2d with 3x3 SparseConv2d
    '''
    return nn.Sequential(
        SparseConv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class SparseConvNet(nn.Module):
    '''
    A 9 layer CNN using the sparase_conv_block function above.
    PART 3.1: Implement!
    '''
    def __init__(self):
        super(SparseConvNet, self).__init__()
        # PART 3.1: Implement!
        self.model = nn.Sequential(
            sparse_conv_block(3,32),
            sparse_conv_block(32,32),
            sparse_conv_block(32,64, stride=2),
            sparse_conv_block(64,64),
            sparse_conv_block(64,64),
            sparse_conv_block(64,128, stride=2),
            sparse_conv_block(128,128),
            sparse_conv_block(128,256),
            sparse_conv_block(256,256),
            # Added this layer in to fix dimension issues
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        '''
        PART 3.1: Implement!
        '''
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SparseConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SparseConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.sparse_conv_block(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = SparseConv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_sparse():
    return ResNet(BasicBlock, [2, 2, 2, 2])
    #return ResNet(sparse_conv_block, [2,2,2,2])


#def ResNet34():
#    return ResNet(BasicBlock, [3, 4, 6, 3])


#def ResNet50():
#    return ResNet(Bottleneck, [3, 4, 6, 3])


#def ResNet101():
#    return ResNet(Bottleneck, [3, 4, 23, 3])


#def ResNet152():
#    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18_sparse()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

test()
