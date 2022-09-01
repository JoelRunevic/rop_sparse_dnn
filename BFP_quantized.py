# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import math
import numpy as np
from scipy.io import savemat
import time

################
kExponentWidth = 4
kMantissaWidth = 5 - 1  # total 5 bits, plus the left most 1 
ExpOffset = -18
SAMPLE_RESULTS = True
SAVED_PATH = "/home/jovyan/user/pytorch-cifar/buffer_results/"
################

def min_value(ExpOffset, exp_bias):
    min_exp = -1* ( 2**(kExponentWidth-1)) + 2 + exp_bias + ExpOffset;
    min_value = 2**(min_exp);
    return torch.tensor(min_value);

def max_value(ExpOffset, exp_bias):
    max_exp = ( 2**(kExponentWidth-1) -1 ) + exp_bias + ExpOffset
    max_value = 2**(max_exp) * (2 - 1/(2**kMantissaWidth))
    return torch.tensor(max_value);


class GROUP_FPQuantize(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, exponent_bit, mantissa_bit, group_size, category=None, layer_id=None, epoch=None, partition=None, max_exp=None, min_exp=None):
        ## first switch the axis
        B, C, W, H = input.shape
        sign = input.sign()
        input_original = input
        if (partition=='channel'):
            input = input.abs().permute(2,3,0,1).reshape((-1,C))   #(W*H*B, C)
        else:
            print('wrong partition selection!')
        
        if (B*W*H)%3 == 1:
            input = torch.cat((input, torch.zeros(2, C).cuda()), 0)
        elif (B*W*H)%3 == 2:
            input = torch.cat((input, torch.zeros(1, C).cuda()), 0)
        # truncate the exponents
        exponent = torch.floor(torch.log2(input+1e-40))
        max_possible_value = (2**max_exp) * (2-(1./2**(mantissa_bit-1)))   # max possible value that can be represented        
        input = max_possible_value * torch.ones(input.shape).cuda() * (exponent > max_exp).float().cuda() + input * ((exponent >= min_exp) * (exponent <= max_exp)).float().cuda()
        input_s = torch.cat(torch.split(input, group_size, 1), 0)
        input = input_s.reshape(-1, group_size*group_size)
        
        if (partition=='channel'):
            # using the maximum exponent
            epsilon = 1e-40
            group_exponent = torch.floor(torch.log2(input+epsilon)).max(-1)[0].detach().unsqueeze(-1)
            pow_exponent = 2**(group_exponent)
            input = input/pow_exponent
        else:
            print('wrong partition selection!')

        # another way to perform the floating-point quantization
        xhard = input
        sf = 2./(2**(mantissa_bit))
        xhard = xhard/sf
        
        if (category=='GRAD_BN') or (category=='GRAD'):
            noise = (xhard.new(xhard.shape).uniform_(-0.5, 0.5))
            xhard = xhard + noise   
        
        xhard = xhard.round()
        xhard[xhard>=(2**mantissa_bit)] = 2**mantissa_bit-1
        xhard = xhard * sf
        ###############################
        xhard = (xhard * pow_exponent).contiguous()
        
        if (partition=='channel'):
            xhard = xhard.reshape(input_s.shape)
            if (B*W*H)%3 == 1:
                xhard = torch.cat(torch.split(xhard, B*W*H + 2, 0),1)
                xhard = xhard[:-2, :]
            elif (B*W*H)%3 == 2:
                xhard = torch.cat(torch.split(xhard, B*W*H + 1, 0),1)
                xhard = xhard[:-1, :]
            else:
                xhard = torch.cat(torch.split(xhard, B*W*H, 0),1)
            xhard = xhard.reshape(W,H,B,C).permute(2,3,0,1).contiguous()    
        else:
            print('wrong partition selection!')
        
        output = xhard * sign
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        
        return grad_input, None, None, None, None, None, None, None, None, None
    
        
class GROUP_FPQuantizeGrad(InplaceFunction):
    # change to no stochastic
    @classmethod
    def forward(cls, ctx, input, exponent_bit, mantissa_bit, group_size, layer_id, epoch=-1, partition=None, max_exp=None, min_exp=None):
        
        ctx.exponent_bit = exponent_bit
        ctx.mantissa_bit = mantissa_bit
        ctx.group_size = group_size
        ctx.category = 'GRAD'
        ctx.layer_id = layer_id
        ctx.epoch = epoch
        ctx.partition = partition
        ctx.max_exp = max_exp
        ctx.min_exp = min_exp
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = GROUP_FPQuantize().apply(grad_output, ctx.exponent_bit, ctx.mantissa_bit, ctx.group_size, ctx.category, ctx.layer_id, ctx.epoch, ctx.partition, ctx.max_exp, ctx.min_exp)
        if (SAMPLE_RESULTS):
            if ("pretrain" not in ctx.layer_id):
                torch.save(grad_input, SAVED_PATH + 'output_grad_' + str(ctx.layer_id) + '.pt')
        return grad_input, None, None, None, None, None, None, None, None, None

class Detect(InplaceFunction):
    # change to no stochastic
    @classmethod
    def forward(cls, ctx, input, layer_id):
        ctx.layer_id = layer_id
        return input

    @staticmethod
    def backward(ctx, grad_output):    
        if (SAMPLE_RESULTS):
            if ("pretrain" not in ctx.layer_id):
                torch.save(grad_output, SAVED_PATH + 'input_grad_' + str(ctx.layer_id) + '.pt')
        return grad_output, None

def BFP_quantize(x, exponent_bit, mantissa_bit, group_size, category, layer_id=None, epoch=None, partition=None, max_exp=None, min_exp=None):
    return GROUP_FPQuantize().apply(x, exponent_bit, mantissa_bit, group_size, category, layer_id, epoch, partition, max_exp, min_exp)

def BFP_quantize_grad(x, exponent_bit, mantissa_bit, group_size, layer_id=None, epoch=None, partition=None, max_exp=None, min_exp=None):
    
    return GROUP_FPQuantizeGrad().apply(x, exponent_bit, mantissa_bit, group_size, layer_id, epoch, partition, max_exp, min_exp)    

def detect_grad(x, exponent_bit, mantissa_bit, group_size, layer_id=None, epoch=None, partition=None, max_exp=None, min_exp=None):   # this is used for gradient detection
    return Detect().apply(x, layer_id) 
    #return GROUP_FPQuantizeGrad().apply(x, exponent_bit, mantissa_bit, group_size, layer_id, epoch, partition, max_exp, min_exp)    


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, layer_id=None, BFP_training=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_id = layer_id
        self.group_size = 3
        self.act_exponent = 4
        self.act_mantissa = 5
        self.wgt_exponent = 4
        self.wgt_mantissa = 5
        self.grad_exponent = 4
        self.grad_mantissa = 5
        self.BFP_training = BFP_training   # this is indicate whether the BFP is used during training or not
        
        self.MAX_GRAD_EXPONENT = -5
        self.MIN_GRAD_EXPONENT = self.MAX_GRAD_EXPONENT - (2**self.grad_exponent-1)
        self.MAX_INPUT_EXPONENT = 6
        self.MIN_INPUT_EXPONENT = self.MAX_INPUT_EXPONENT - (2**self.act_exponent-1)
        self.MAX_WGT_EXPONENT = -1
        self.MIN_WGT_EXPONENT = self.MAX_WGT_EXPONENT-(2**(self.wgt_exponent)-1)
        
    def forward(self, input, epoch=-1, prec=None):
        qinput = BFP_quantize(input, exponent_bit=self.act_exponent, mantissa_bit=self.act_mantissa, group_size=self.group_size, category='ACT', layer_id=self.layer_id, epoch=epoch, partition='channel', max_exp=self.MAX_INPUT_EXPONENT, min_exp=self.MIN_INPUT_EXPONENT)
        qweight = BFP_quantize(self.weight, exponent_bit=self.wgt_exponent, mantissa_bit=self.wgt_mantissa, group_size=self.group_size, category='WGT', layer_id=self.layer_id, epoch=epoch, partition='channel', max_exp=self.MAX_WGT_EXPONENT, min_exp=self.MIN_WGT_EXPONENT) 
        
        #qinput = input
        #qweight = self.weight
        if (SAMPLE_RESULTS):
            if ("pretrain" not in self.layer_id):
                
                torch.save(qinput, SAVED_PATH + 'qinput_' + str(self.layer_id) + '.pt')
                torch.save(qweight, SAVED_PATH + 'qweight_' + str(self.layer_id) + '.pt')
        if self.bias is not None:
            qbias = self.bias
        else:
            qbias = None
        
        ############# this deal with the gradient
        if self.BFP_training:
            qinput = detect_grad(qinput, exponent_bit=self.grad_exponent, mantissa_bit=self.grad_mantissa, group_size=self.group_size, layer_id=self.layer_id, epoch=epoch, partition='channel', max_exp=self.MAX_GRAD_EXPONENT, min_exp=self.MIN_GRAD_EXPONENT)
        ###################
        output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        if (SAMPLE_RESULTS):
            if ("pretrain" not in self.layer_id):
                torch.save(output, SAVED_PATH + 'output_' + str(self.layer_id) + '.pt')        
        if self.BFP_training:
            output = BFP_quantize_grad(output, exponent_bit=self.grad_exponent, mantissa_bit=self.grad_mantissa, group_size=self.group_size, layer_id=self.layer_id, epoch=epoch, partition='channel', max_exp=self.MAX_GRAD_EXPONENT, min_exp=self.MIN_GRAD_EXPONENT)
        
        return output

class NMReLU(nn.Module):
    def __init__(self, M, N):
        super(NMReLU, self).__init__()
        self.M = M
        self.N = N
    def forward(self, input, label):    
        return nm_relu().apply(input, self.M, self.N, label)
    
    
class nm_relu(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, M, N, label):
        B,C,W,H = input.shape
        mask_o = (input>0.).float()
        #print(mask_o[0,0,:,:])
        mask = mask_o.permute(1,0,2,3).reshape(-1,M).float()   # CxBxWx(H/M) by M
        #print(label, (mask.sum(1) > N).sum().float()/ (C*B*W*(H/M)), C,B,W,H )
        dummy = mask + torch.rand(mask.shape).cuda()   # CxBxWx(H/M) by M
        thres = torch.sort(dummy, descending=True)[0][:,N]
        mask_f = (dummy.t()>=thres).t().float()   # CxBxWx(H/M) by M
        mask_f = mask_o * (mask_f.reshape(C,B,W,H).permute(1,0,2,3))    # CxBxWx(H/M) by M
        #print('1111', mask_o.sum().float()/(B*C*W*H))
        #print('2222',(mask_o-mask_f).abs().sum().float()/mask_o.sum())
        ctx.mask = mask_f
        input = input* mask_f
        return input

    @staticmethod
    def backward(ctx, grad):
        B,C,W,H = grad.shape
        #print(ctx.mask.permute(1,0,2,3).reshape(-1,8).sum(1))
        return grad * ctx.mask, None, None, None, None
    
############## the new BN layer which applies the sum of |x-mu| to estimate the variance
class Group_QBN(nn.Module):
    
    def __init__(self, num_features, exponent_bit=-1, mantissa_bit=-1, group_size=-1, group_budget=-1, dim=1, momentum=0.1, affine=True, num_chunks=128, eps=1e-5):
        super(Group_QBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.momentum = momentum
        self.dim = dim
        self.bn_group = 8
        self.bn_exponent = 3
        self.bn_mantissa = 6
        
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
            
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()
        
    def reset_params(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        
        if self.weight is not None:
            self.weight.data.fill_(1)
        if self.bias is not None:
            self.bias.data.zero_()
            

    def forward_inference(self, x):

        
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)
        mean = self.running_mean
        scale = self.running_var
        scale = 1. / ((scale + 1e-5)**(0.5))
        
        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)

        if self.weight is not None:
            qweight = self.weight
            out = out * self.weight.view(1, qweight.size(0), 1, 1)

        if self.bias is not None:
            qbias = self.bias
            out = out + qbias.view(1, qbias.size(0), 1, 1)

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
            
        return out
            
    
    
    def forward(self, x):
        if self.training:
            out, running_mean, running_var = MyFun().apply(x, self.weight, self.bias, self.running_mean, self.running_var, self.bn_group, self.bn_exponent, self.bn_mantissa)
            self.running_mean = running_mean
            self.running_var = running_var
        else:
            out = self.forward_inference(x)
            
        return out
    
    
class MyFun(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, weight, bias, running_mean, running_var, bn_group, bn_exponent, bn_mantissa):
        
        x = input     
        B, C, W, H = input.shape
        n = B*W*H

        ############# term quantize the input vector
        #input = BFP_quantize(input, exponent_bit=bn_exponent, mantissa_bit=bn_mantissa, group_size=bn_group, category='ACT', layer_id=-1, epoch=-1, partition='channel')
        ################################
        
        y = input.transpose(0, 1).contiguous().view(C,-1)  # C x (B x H x W)
        input = y.transpose(0, 1)   # (B x H x W) x C

        N, D = input.shape

        #step1: calculate mean
        mu = x.mean([0,2,3])
            
        #step2: subtract mean vector of every trainings example
        xmu = input - mu
        
        #step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        #step4: calculate variance
        var = 1./N * sq.sum(0)
        
        #step5: add eps for numerical stability, then sqrt
        sqrtvar = (var + 1e-5)**(0.5)

        #step6: invert sqrtwar 
        ivar = 1./sqrtvar    # C
            
        # save for statistics
        momentum = 0.1
        
        running_mean = (running_mean.mul_(1 - momentum).add_(mu * (momentum))).detach()
        running_var = (running_var.mul_(1 - momentum).add_(var * n/(n-1) * (momentum))).detach() 
        
        #step7: execute normalization
        xhat = xmu * ivar    # BxWxH by C
        
        ##########save for backprop
        #xhat = xhat.reshape(B,W,H,C).permute(0,3,1,2)   # B,C,W,H
        #xhat = BFP_quantize(xhat, exponent_bit=bn_exponent, mantissa_bit=bn_mantissa, group_size=bn_group, category='ACT', layer_id=-1, epoch=-1, partition='channel')
        #xhat = xhat.permute(0,2,3,1).reshape(-1, C)
        ##########
            
        #step8: Nor the two transformation steps
        gammax = weight * xhat

        #step9
        out = gammax + bias
        out = out.view(B,W,H,C).permute(0,3,1,2).contiguous()   # (BxWxHxC --> BxCxWxH)

        #store intermediate
        ctx.xhat = xhat   # BxWxH by C
        ctx.gamma = weight  # C by 1
        ctx.xmu = xmu    # C by 1
        ctx.ivar = ivar   # C by 1
        ctx.var = var   # C by 1
        ctx.bn_exponent = bn_exponent 
        ctx.bn_mantissa = bn_mantissa 
        ctx.bn_group = bn_group
        return out, running_mean, running_var

    
    @staticmethod
    def backward(ctx, grad_output, grad_running_mean, grad_running_var): 
        
        ############quantize the output gradient
        #grad_output = BFP_quantize(grad_output, exponent_bit=ctx.bn_exponent, mantissa_bit=ctx.bn_mantissa, group_size=ctx.bn_group, group_budget=-1, category='GRAD', layer_id=-1, epoch=-1, partition='channel')
        ##########
        
        input_normalized = ctx.xhat  # BxWxH by C
        gamma = ctx.gamma  # C by 1
        xmu = ctx.xmu  # C by 1
        std_inv = ctx.ivar  # C by 1
        
        grad_input = grad_gamma = grad_bias = None

        B, C, W, H = grad_output.shape
    
        grad_output = grad_output.transpose(0, 1).contiguous().view(C,-1)  # C x B x W x H
        grad_output = grad_output.transpose(0, 1)   # (B x W x H) x C
    
        N = input_normalized.size(0)
    
        input_mu = input_normalized/std_inv
        grad_input_normalized = (grad_output * gamma)   # (B x W x H) x C
    
        grad_var = (-0.5) * (grad_input_normalized * input_mu).sum(dim=0, keepdim=True) * (std_inv ** 3)   # C by 1
        grad_var = grad_var.view(-1)
    
        ##############
        grad_mean = (-1.0) * (grad_input_normalized * std_inv).sum(dim=0, keepdim=True) - 2.0 * grad_var * input_mu.mean(dim=0, keepdim=True)
        grad_mean = grad_mean.view(-1)
        grad_input = grad_input_normalized * std_inv + (2. / N) * grad_var *  input_mu + (1. / N) * grad_mean
        
        #print(((-xmu).sign().sum(0)).shape, ((1. / N) * xmu.sign()).shape)   #torch.Size([128]) torch.Size([62720, 128])
        '''
        xx = (1. / N) * xmu.sign() + (1. / N) * (1. / N) * ((-xmu).sign().sum(0))
        xx = (2 * grad_var / std_inv) * xx.float() / ((2/math.pi)**(0.5))
        grad_input = grad_input_normalized * std_inv + xx + (1. / N) * grad_mean
        '''
        
        ####
        grad_gamma = (grad_output * input_normalized).sum(dim=0)  
        grad_bias = grad_output.sum(dim=0)
        ####    
    
        grad_input = grad_input.view(B,W,H,C).permute(0,3,1,2).contiguous()
    
        return grad_input, grad_gamma, grad_bias, None, None, None, None, None
    
