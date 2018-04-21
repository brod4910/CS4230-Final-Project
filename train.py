import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import DatasetLoader
import argparse
import vgg19
from tqdm import tqdm

def GetArgParser():
    parser = argparse.ArgumentParser(description='VGG19')

    parser.add_argument(
        'train_csv', 
        action="store",
        )
    parser.add_argument(
        'test_csv',
        action="store",
        )
    parser.add_argument(
        '-s',
        '--shards',
        type= int,
        default= 2,
        )

    return parser

if __name__ == '__main__':

    file = open("results.txt", "w")

    args, __ = GetArgParser().parse_known_args()

    train_dataset = DatasetLoader.DatasetLoader(args.train_csv, (224,224))

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= 64, shuffle= True,  num_workers= args.shards)

    test_dataset = DatasetLoader.DatasetLoader(args.test_csv, (224,224))

    test_dataset_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= 64, shuffle= True, num_workers= args.shards)

    use_cuda = torch.cuda.is_available()
    best_accuracy = 0
    start_epoch = 0

    net = vgg19.VGG19()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr= 10**(-2), momentum= .09, weight_decay= 5e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor= .1, patience= 5)

    def train_net(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()

        for batch_idx, data in enumerate(train_dataset_loader):
            inputs = data['image']
            targets = data['label'].type(torch.LongTensor)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs_var, targets_var = Variable(inputs), Variable(targets)

            optimizer.zero_grad()
            outputs = net(inputs_var)
            loss = criterion(outputs, targets_var)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1)% 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs_var), len(train_dataset_loader.dataset),
                100. * (batch_idx + 1) / len(train_dataset_loader), loss.data[0]))

    def evaluate(data_loader):
        net.eval()
        loss = 0
        correct = 0
        
        for raw in data_loader:
            data = raw['image']
            target = raw['label'].type(torch.LongTensor)

            data, target = Variable(data, volatile=True), Variable(target)

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            output = net(data)
            
            loss += F.cross_entropy(output, target, size_average=False).data[0]

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
        loss /= len(data_loader.dataset)
            
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

    for epoch in range(70):
        train_net(epoch)
        evaluate(test_dataset_loader)
