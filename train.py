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

def train_net(epoch, net, train_dataset_loader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(total= len(train_dataset_loader)) as t:
        for batch_idx, data in enumerate(train_dataset_loader):
            inputs = data['image']
            labels = data['label'].type(torch.LongTensor)

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('\n')
            t.update()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_net(epoch, net, test_dataset_loader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with tqdm(total= len(test_dataset_loader)) as t:
        for batch_idx, data in enumerate(test_dataset_loader):
            inputs = data['image']
            labels = data['label'].type(torch.LongTensor)

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs, labels = Variable(inputs, volatile=True), Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            
            print('\n')
            t.update()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

    args, __ = GetArgParser().parse_known_args()

    train_dataset = DatasetLoader.DatasetLoader(args.train_csv, (224,224))

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= 4, shuffle= True,  num_workers= args.shards)

    test_dataset = DatasetLoader.DatasetLoader(args.test_csv, (224,224))

    test_dataset_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= 4, shuffle= True, num_workers= args.shards)

    use_cuda = torch.cuda.is_available()
    best_accuracy = 0
    start_epoch = 0

    net = vgg19.VGG19()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr= .001, momentum= .09, weight_decay= (5* 10**(-4)), nesterov=True)

    for epoch in range(50):
        train_net(epoch, net, train_dataset_loader)
        test_net(epoch, net, test_dataset_loader)


