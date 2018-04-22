import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import sys

def train(args, model, use_cuda):
    # torch.manual_seed(args.seed + rank)

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
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer, use_cuda)
        test_epoch(model, test_loader, use_cuda)


def train_epoch(epoch, args, model, data_loader, optimizer, use_cuda):
    model.train()
    correct = 0

    data_load_tot = 0.
    forward_tot = 0.
    backward_tot = 0.
    data_load_t0 = time.clock()

    for batch_idx, (data, target) in enumerate(data_loader):

        data_load_tot += time.clock() - data_load_t0

        if use_cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
            
        optimizer.zero_grad()

        forward_t1 = time.clock()
        output = model(data)
        loss = F.cross_entropy(output, target)
        forward_tot += time.clock() - forward_t1

        backward_t2 = time.clock()
        loss.backward()
        backward_tot += time.clock() - backward_t2
        optimizer.step()

        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0], 
                correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

        data_load_t0 = time.clock()

    print("Data Load Time: {:.4f}".format(data_load_tot / batch_idx))
    print("Forwardpass Time: {:.4f}".format(forward_tot / batch_idx))
    print("Backwardpass Time: {:.4f}".format(backward_tot / batch_idx))

def test_epoch(model, data_loader, use_cuda):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if use_cuda:
            data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
        else:
            data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
