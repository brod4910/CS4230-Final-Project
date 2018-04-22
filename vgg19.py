from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.b0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(64, 64, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            )
        self.b1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(128, 128, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            )
        self.b2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            )
        self.b3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            )
        self.b4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, stride= 1, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            )
        self.final = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= .5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= .5),
            nn.Linear(4096, 10),
            )

    def forward(self, x):
        out = self.b0(x)
        out = self.b1(out)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = out.view(out.size(0), -1)
        out = self.final(out)
        return F.log_softmax(out, dim=1)
