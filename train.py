import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import sys

def train(args, model, device):
    # torch.manual_seed(args.seed + rank)
    data_tot, forward_tot, backward_tot = 0., 0., 0.
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss().cuda()
    total_time = time.clock()
    for epoch in range(1, args.epochs + 1):
        data_t0, forward_t1, backward_t2 = train_epoch(epoch, args, model, train_loader, optimizer, criterion, device)
        data_tot += data_t0
        forward_tot += forward_t1
        backward_t2 += backward_t2
        test_epoch(model, test_loader, device)

    print("The Total Training and Inference time: {:.4f}".format(time.clock() - total_time))
    print("The Data Loading Average: {:.10f}".format(data_tot / (50000*args.epochs)))
    print("The Forwardpass Average: {:.10f}".format(forward_tot / (50000*args.epochs)))
    print("The Backwardpass Average: {:.10f}".format(backward_tot / (50000*args.epochs)))

def train_epoch(epoch, args, model, data_loader, optimizer, criterion, device):
    model.train()
    correct = 0

    data_load_tot = 0.
    forward_tot = 0.
    backward_tot = 0.
    data_load_t0 = time.clock()

    for batch_idx, (data, target) in enumerate(data_loader):

        data_load_tot += time.clock() - data_load_t0

        data, target = data.to(device), target.to(device)
            
        optimizer.zero_grad()

        forward_t1 = time.clock()
        output = model(data)
        loss = criterion(output, target)
        forward_tot += time.clock() - forward_t1

        backward_t2 = time.clock()
        loss.backward()
        backward_tot += time.clock() - backward_t2
        optimizer.step()

        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item(), 
                correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

        data_load_t0 = time.clock()

    print("Data Load Time: {:.4f}".format(data_load_tot / batch_idx))
    print("Forwardpass Time: {:.4f}".format(forward_tot / batch_idx))
    print("Backwardpass Time: {:.4f}".format(backward_tot / batch_idx))

    return data_load_tot, forward_tot, backward_tot

def test_epoch(model, data_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = F.cross_entropy(output, target, size_average=False) # sum up batch loss
            test_loss += loss.item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
