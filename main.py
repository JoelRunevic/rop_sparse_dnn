'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from sparse_functions import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--p', default=50, type=int, help='Prune percentage')
parser.add_argument('--rop', action='store_true', help='use rop pruning')
parser.add_argument('--layer' , action='store_true', help='use layer pruning')
parser.add_argument('--filter', action ='store_true', help='prune entire filters')
parser.add_argument('--cifar100', action ='store_true', help='use cifar10')
parser.add_argument('--net', default='resnet18',  help='network')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

if not args.cifar100:
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
else:
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 100
# Model
print('==> Building model..')
print(args.net, num_classes)
if args.net in "resnet18":
    net = ResNet18_sparse(num_classes = num_classes)
elif args.net == "resnet50":
    net = ResNet50_sparse(num_classes = num_classes)
elif args.net == "vgg19":
    net = VGG_sparse('VGG19')
elif args.net == "mobilenet":
    net = MobileNetV2_sparse(num_classes = num_classes)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#Pruning
if args.p:
	print("Prune percent: {}".format(args.p))
	
prune_percentages = np.arange(10, rounddown(args.p)+10, 10)
prune_epochs = np.arange(10, rounddown(args.p)+10, 10)
#handle last pruning stage
if args.p%10 != 0:
  prune_percentages = np.append(prune_percentages, args.p)#[10, 20, 30, 40, 50, 60, 70]                          
  prune_epochs = np.append(prune_epochs, roundup(args.p))	
#prune_percentages = np.arange(10, args.p+10, 10) #[10, 20, 30, 40, 50, 60, 70]
#prune_epochs = np.arange(10, args.p+10, 10)
#prune_epochs[0] = 0 # = [10, 20, 30, 40, 50, 60, 70]
curr_prune_stage = 0
print(prune_epochs)
print(prune_percentages)


# Training
def train(epoch):
    global curr_prune_stage
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if epoch in prune_epochs:
        print('Pruning Stage {}'.format(curr_prune_stage))
        if args.layer:
            smallest_magnitude_pruning(net, prune_percentages[curr_prune_stage])
        if args.rop:
            rop_pruning(net, prune_percentages[curr_prune_stage],  multi_prune= args.net == "mobilenet")
        if args.filter:
            struct_pruning(net, prune_percentages[curr_prune_stage])
        curr_prune_stage += 1




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')
        if args.cifar100:
            cifar="CIFAR100"
        else:
            cifar="CIFAR10"
        if args.rop:
             torch.save(state, './checkpoint/ckpt_{}_{}_rop_{}.pth'.format(args.net, cifar, args.p))
        if args.layer:
             torch.save(state, './checkpoint/ckpt_{}_{}_layer_{}.pth'.format(args.net, cifar, args.p))
        if args.filter:
             torch.save(state, './checkpoint/ckpt_{}_{}_filter_{}.pth'.format(args.net, cifar, args.p))
        else:
            torch.save(state, './checkpoint/ckpt_{}_{}_no_pruning.pth'.format(args.net, cifar))
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

verbose = True
for i, layer in enumerate(get_sparse_conv2d_layers(net)):
    if verbose:
            num_total = len(layer._mask)
            num_nonzero = layer._mask.sum().item()
            print(num_nonzero, num_total)
            sparsity = 100.0 * (1 - (num_nonzero / num_total))
            print('Layer {} ({}): {}% sparse'.format(i, layer.weight.shape,
                                                     sparsity))

if args.cifar100:
    cifar="CIFAR100"
else:
    cifar="CIFAR10"
if args.rop:
     print("{} {}% ROP pruning on {}  Acc: {}%".format(args.net, args.p, cifar, best_acc))
if args.layer:
     print("{} {}% layer pruning on {} Acc: {}%".format(args.net,args.p, cifar, best_acc))
if args.filter:
    print("{} {}% filter pruning on {} Acc: {}%".format(args.net, args.p, cifar, best_acc))
else:
     print("{} on {} Acc: {}%".format(args.net, cifar, best_acc))
