### Seeing if the learned rosko binary weight mask generalizes from CIFAR 100 to CIFAR 10. ###

import torch

import torchvision 
import torchvision.transforms as transforms

import copy 

import detectors # required for the timm library.
import timm 

import pickle 

from custom_utils import *

from datetime import datetime 


### Loading the data; CIFAR 10 for this experiment. ### 

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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

print("CIFAR 10 Data Loaded Successfully!")

### Training the models on CIFAR 10. ### 

device = get_device()
num_classes = 10

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y:%H:%M:%S")

# Model that has the learned rosko mask.
masked_trained_model_dict = torch.load("experiment_models/trained_pruned_model_25_05_2024:23:55:06.pth")
masked_trained_model = create_pretrained_network(device)
masked_trained_model.load_state_dict(masked_trained_model_dict)

print("Loaded masked model.")

# Unmasked model that we will train.
unmasked_new_model = create_pretrained_network(device, False)

# Changing the last layer.
unmasked_new_model.fc = torch.nn.Linear(512, num_classes)

# Saving the weights.
unmasked_new_model_initial_state_dict = unmasked_new_model.state_dict()

print("Created the unmasked new model.")

# Masked model that we will train.
masked_new_model = create_pretrained_network(device, False)
masked_new_model.fc = torch.nn.Linear(512, num_classes)

masked_new_model.load_state_dict(unmasked_new_model_initial_state_dict)

# Applying the mask. 
masked_trained_model_conv_layers = get_sparse_conv2d_layers(masked_trained_model)
masked_new_model_conv_layers = get_sparse_conv2d_layers(masked_new_model)

for i, layer in enumerate(masked_trained_model_conv_layers):
    masked_new_model_conv_layers[i]._mask = layer._mask

print("Created the masked new model with copied weight mask.")

# Masked model (with a random mask) that we will train.
masked_new_model_random = create_pretrained_network(device, False)
masked_new_model_random.fc = torch.nn.Linear(512, num_classes)

masked_new_model_random.load_state_dict(unmasked_new_model_initial_state_dict)

masked_new_model_random_conv_layers = get_sparse_conv2d_layers(masked_new_model_random)

# Applying the random mask.
for i, layer in enumerate(masked_new_model_random_conv_layers):

    num_total = len(layer._mask)
    num_nonzero = int(num_total * 0.125)
    random_mask = torch.zeros(num_total)
    random_indices = torch.randperm(num_total)
    random_mask[random_indices[:num_nonzero]] = 1

    masked_new_model_random_conv_layers[i]._mask = random_mask

print("Created the masked new model with random weight mask.")

### Training both models. ### 
lr = 0.01
patience = 15

# Training the unmasked model. #
unmasked_new_model = unmasked_new_model.to(device)
unmasked_new_model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(unmasked_new_model.parameters(), 
                      lr=lr,
                      momentum=0.9, 
                      weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
early_stopping = EarlyStopping(patience=patience, verbose=True)

unmasked_new_model_stats = training_loop(unmasked_new_model,
              optimizer,
              trainloader,
              testloader,
              device)

print("\nFinished training the unmasked new model.\n\n\n")

# Training the masked model. # 
masked_new_model = masked_new_model.to(device)
masked_new_model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(masked_new_model.parameters(), 
                      lr=lr,
                      momentum=0.9, 
                      weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
early_stopping = EarlyStopping(patience=patience, verbose=True)

masked_new_model_stats = training_loop(masked_new_model,
              optimizer,
              trainloader,
              testloader,
              device)

print("\nFinished training the masked new model.\n\n\n")

# Training the masked model with random mask. #
masked_new_model_random = masked_new_model_random.to(device)
masked_new_model_random.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(masked_new_model_random.parameters(), 
                      lr=lr,
                      momentum=0.9, 
                      weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
early_stopping = EarlyStopping(patience=patience, verbose=True)

masked_new_model_random_stats = training_loop(masked_new_model_random,
                optimizer,
                trainloader,
                testloader,
                device)

print("\nFinished training the masked new model with random mask.\n\n\n")

### Saving the models. ###
torch.save(unmasked_new_model.state_dict(), f'experiment_models/unmasked_new_model_trained_{dt_string}.pth')
torch.save(masked_new_model.state_dict(), f'experiment_models/masked_new_model_trained_{dt_string}.pth')
torch.save(masked_new_model_random.state_dict(), f'experiment_models/masked_new_model_random_trained_{dt_string}.pth')


### Saving the model statistics. ### 
with open(f'experiment_lists/unmasked_new_model_stats_{dt_string}.pkl', 'wb') as f:
    pickle.dump(unmasked_new_model_stats, f)

with open(f'experiment_lists/masked_new_model_stats_{dt_string}.pkl', 'wb') as f:
    pickle.dump(masked_new_model_stats, f)

with open(f'experiment_lists/masked_new_model_random_stats_{dt_string}.pkl', 'wb') as f:
    pickle.dump(masked_new_model_random_stats, f)


# device = get_device()

# num_classes = 10

# masked_model_trained = torch.load("experiment_models/masked_new_model_trained.pth")
# unmasked_model_trained = torch.load("experiment_models/unmasked_new_model_trained.pth")

# # Printing out the sparsity of the model.
# masked_model = create_pretrained_network(device, False)
# masked_model.fc = torch.nn.Linear(512, num_classes)
# masked_model.load_state_dict(masked_model_trained)

# unmasked_model = create_pretrained_network(device, False)
# unmasked_model.fc = torch.nn.Linear(512, num_classes)
# unmasked_model.load_state_dict(unmasked_model_trained)

# get_network_sparsity(masked_model)
# print("\n\n\n")
# get_network_sparsity(unmasked_model)

# quit()