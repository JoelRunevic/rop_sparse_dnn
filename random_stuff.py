import torch 

torch.manual_seed(128)
layer1 = torch.nn.Linear(3, 5)

torch.manual_seed(128)
layer2 = torch.nn.Linear(3, 5)

print(layer1.weight)
print("\n\n\n")
print(layer2.weight)