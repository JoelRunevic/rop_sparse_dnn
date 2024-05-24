import detectors
import timm
import torch
import copy 

from custom_utils import * 

# Pretrained ResNet 18 model.
model = timm.create_model("resnet18_cifar100", pretrained=True)
num_classes = 100
device = get_device()
dt_string = "14_04_2024:16:38:07"


# Replacing the conv layers with sparse layers.
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
                new_sparse_layer.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    new_sparse_layer.bias.data.copy_(layer.bias.data)

                # Replace the original Conv2d layer
                setattr(model, name, new_sparse_layer)
            else:
                _replace_recursively(layer)
    
    _replace_recursively(model)

# Replacing the pretrained model conv layers with sparse conv layers.
replace_conv_with_sparse(model)

# Saving the new model.
torch.save(model.state_dict(), f'experiment_models/resnet18-cifar100-pretrained.pth')