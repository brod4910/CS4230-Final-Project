import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

def train(args, model, use_cuda):
    # torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
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
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
            
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))

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
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
