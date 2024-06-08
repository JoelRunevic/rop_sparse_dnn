import detectors
import timm
import torch
import copy 

import torchvision
import torchvision.transforms as transforms

from custom_utils import * 

# Pretrained ResNet 18 model.
model = timm.create_model("resnet18_cifar100", pretrained=True)

device = get_device()

# Testing the model on CIFAR-100 dataset without training.
model.eval()
model.to(device)


# Getting the dataset.
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]) 


testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


def get_test_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

init_test_acc = get_test_accuracy(model, testloader)
print(f"Initial test accuracy: {init_test_acc}")


num_classes = 100

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

                layer_weight_data = layer.weight.data.reshape(-1)
                new_sparse_layer._weight.data.copy_(layer_weight_data)
                if layer.bias is not None:
                    new_sparse_layer.bias.data.copy_(layer.bias.data)

                # Replace the original Conv2d layer
                setattr(model, name, new_sparse_layer)
            else:
                _replace_recursively(layer)
    
    _replace_recursively(model)

# Replacing the pretrained model conv layers with sparse conv layers.
replace_conv_with_sparse(model)
model = model.to(device)

sparse_test_acc = get_test_accuracy(model, testloader)
print(f"Sparse test accuracy: {sparse_test_acc}")

# Saving the new model.
torch.save(model.state_dict(), f'experiment_models/resnet18-cifar100-pretrained.pth')