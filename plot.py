import pickle 
import matplotlib.pyplot as plt
import torch 

import torch.backends.cudnn as cudnn

from sparse_functions import *

dt_string = "14_04_2024:16:38:07"

with open(f'experiment_lists/training_initial_model_{dt_string}.pkl', 'rb') as f:
    initial_model_stats = pickle.load(f)

with open(f'experiment_lists/training_pruned_model_{dt_string}.pkl', 'rb') as f:
    pruned_model_stats = pickle.load(f)

# with open(f'experiment_lists/generalize_pruned_network_stats_{dt_string}.pkl', 'rb') as f:
#     initial_model_stats = pickle.load(f)

# with open(f'experiment_lists/generalize_original_network_stats_{dt_string}.pkl', 'rb') as f:
#     pruned_model_stats = pickle.load(f)


train_losses_pruned = [x["Train Loss"] for x in pruned_model_stats]
test_losses_pruned = [x["Test Loss"] for x in pruned_model_stats]
train_acc_pruned = [x["Train Accuracy"] for x in pruned_model_stats]
test_acc_pruned = [x["Test Accuracy"] for x in pruned_model_stats]
epochs_pruned = list(range(1, len(train_losses_pruned) + 1))

train_losses_initial = [y["Train Loss"] for y in initial_model_stats]
test_losses_initial = [y["Test Loss"] for y in initial_model_stats]
train_acc_initial = [y["Train Accuracy"] for y in initial_model_stats]
test_acc_initial = [y["Test Accuracy"] for y in initial_model_stats]
epochs_initial = list(range(1, len(train_losses_initial) + 1))

print(f"Epoch of lowest pruned test loss: {np.argmin(test_losses_pruned)}, with loss value: {np.min(test_losses_pruned)}")
print(f"Epoch of lowest original test loss: {np.argmin(test_losses_initial)}, with loss value: {np.min(test_losses_initial)}")
print(f"Epoch of highest pruned test accuracy: {np.argmax(test_acc_pruned)}, with accuracy value: {np.max(test_acc_pruned)}")
print(f"Epoch of highest original test accuracy: {np.argmax(test_acc_initial)}, with accuracy value: {np.max(test_acc_initial)}")

# fig, ax = plt.subplots(figsize=(10, 6))

# Pruned Model.
# plt.plot(epochs_pruned, train_losses_pruned, label='Pruned Model Train Loss')
# plt.plot(epochs_pruned, test_losses_pruned, label='Pruned Model Test Loss')

# # Initial Model.
# plt.plot(epochs_initial, train_losses_initial, label='Original Model Train Loss', linestyle='--')  # Different linestyle
# plt.plot(epochs_initial, test_losses_initial, label='Original Model Test Loss', linestyle='--')

# # Labels, title, legend
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training vs. Test Loss (Model Comparison)')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig(f"figures/ticket analysis loss {dt_string}.png")


# Pruned Model.
# plt.plot(epochs_pruned, train_acc_pruned, label='Pruned Model Train Accuracy')
# plt.plot(epochs_pruned, test_acc_pruned, label='Pruned Model Test Accuracy')

# # Initial Model.
# plt.plot(epochs_initial, train_acc_initial, label='Original Model Train Accuracy', linestyle='--')  # Different linestyle
# plt.plot(epochs_initial, test_acc_initial, label='Original Model Test Accuracy', linestyle='--')

# # Labels, title, legend
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training vs. Test Accuracy (Model Comparison)')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig(f"figures/ticket analysis accuracy {dt_string}.png")






# def creating_network(device, num_classes):
#     net = ResNet18_sparse(num_classes = num_classes)
#     net = net.to(device)
#     if device == 'cuda':
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = True 
#     return net 

# def get_network_sparsity(network):
#     for i, layer in enumerate(get_sparse_conv2d_layers(network)):
#         num_total = len(layer._mask)
#         num_nonzero = layer._mask.sum().item()
#         sparsity = 100.0 * (1 - (num_nonzero / num_total))
#         print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
#                                                     sparsity))
#         print(num_nonzero, num_total, num_nonzero/num_total)

# trained_initial_net = creating_network("cuda", 10)
# trained_pruned_net = creating_network("cuda", 10)

# trained_initial_network_state = torch.load(f"experiment_models/trained_initial_model_{dt_string}.pth")
# trained_pruned_network_state = torch.load(f"experiment_models/trained_pruned_model_{dt_string}.pth")

# trained_initial_net.load_state_dict(trained_initial_network_state)
# trained_pruned_net.load_state_dict(trained_pruned_network_state)

# get_network_sparsity(trained_initial_net)
# print("\n\n\n")
# get_network_sparsity(trained_pruned_net)

