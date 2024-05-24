import torch 
import torchvision
import torchvision.transforms as transforms

import pickle 

from custom_utils import * 


def change_linear_and_freeze_weights(network, network_state, device):
    network.load_state_dict(network_state)

    # Changing the linear layer for CIFAR-100.
    # The manual seed is to ensure we have the same weights for both networks.
    torch.manual_seed(42)
    network.module.linear = nn.Linear(in_features=512, out_features=100, bias=True).to(device)

    # Freezing all parameters except from the linear layer.
    # Removed the weight freezing for the time being; freezing the rosko sparsity pattern instead.

    # for param in network.parameters():
    #     param.requires_grad = False 

    # for param in network.module.linear.parameters():
    #     param.requires_grad = True 


####   MODEL SETUP   ####

dt_string = "28_03_2024:15:57:14"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10

pruned_model_state = torch.load(f"experiment_models/trained_pruned_model_{dt_string}.pth")
original_model_state = torch.load(f"experiment_models/trained_initial_model_{dt_string}.pth")

pruned_network = creating_network(device, num_classes)
original_network = creating_network(device, num_classes)

change_linear_and_freeze_weights(pruned_network, pruned_model_state, device)
change_linear_and_freeze_weights(original_network, original_model_state, device)


###   SETTING UP MODEL TRAINING   ####

lr = 0.1

# Now just freezing the rosko sparsity pattern.
# orig_opt = optim.SGD(original_network.module.linear.parameters(), lr=lr,
#                       momentum=0.9, weight_decay=5e-4)

# pruned_opt = optim.SGD(pruned_network.module.linear.parameters(), lr=lr,
#                       momentum=0.9, weight_decay=5e-4)

orig_opt = optim.SGD(original_network.parameters(), lr=lr,
                       momentum=0.9, weight_decay=5e-4)

pruned_opt = optim.SGD(pruned_network.parameters(), lr=lr,
                       momentum=0.9, weight_decay=5e-4)


####   SETTING UP DATA LOADERS   ####

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


####   TRAINING THE ORIGINAL MODEL   ####

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

orig_stats = training_loop(original_network, orig_opt, trainloader, testloader, device)

####   TRAINING THE PRUNED MODEL   ####

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

pruned_stats = training_loop(pruned_network, pruned_opt, trainloader, testloader, device)


####   SAVING MODELS + STATS   ####

# Pruned Model.
torch.save(pruned_network.state_dict(), f'experiment_models/generalize_pruned_network_{dt_string}.pth')

with open(f'experiment_lists/generalize_pruned_network_stats_{dt_string}.pkl', 'wb') as f:
    pickle.dump(pruned_stats, f)

print("Finished training the pruned network.")

# Original Model.
torch.save(original_network.state_dict(), f'experiment_models/generalize_original_network_{dt_string}.pth')

with open(f'experiment_lists/generalize_original_network_stats_{dt_string}.pkl', 'wb') as f:
    pickle.dump(orig_stats, f)

print("Finished training the original network.")

get_network_sparsity(original_network)
get_network_sparsity(pruned_network)

