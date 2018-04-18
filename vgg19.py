from __future__ import print_function, division
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

class VGG19(nn.Module):

    def calculate_padding(self, padding_type, kernel):
        assert padding_type in ['same', 'valid']
        if padding_type == 'same':
            return tuple((k - 1) // 2 for k in kernel)
        return tuple(0 for __ in kernel)

    def __init__(self):
        super(VGG19, self).__init__()
        self.b0 = nn.Sequential(
            nn.conv2d(3, 64, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(64, 64, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.MaxPool2d(2, stride= 2),
            )
        self.b1 = nn.Sequential(
            nn.conv2d(64, 128, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(128, 128, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.MaxPool2d(2, stride= 2),
            )
        self.b2 = nn.Sequential(
            nn.conv2d(128, 256, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(256, 256, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(256, 256, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(256, 256, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.MaxPool2d(2, stride= 2),
            )
        self.b3 = nn.Sequential(
            nn.conv2d(256, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.MaxPool2d(2, stride= 2),
            )
        self.b4 = nn.Sequential(
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.conv2d(512, 512, 3, padding=calculate_padding('same', [3, 3])),
            nn.ReLu(True),
            nn.MaxPool2d(2, stride= 2),
            )
        self.final = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.Dropout(.5),
            nn.Linear(4096, 4096),
            nn.Dropout(.5),
            nn.Linear(4096, 4),
            )

    def forward(self, x):
        out = self.b0(x)
        out = self.b1(out)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.final(out)
        return out