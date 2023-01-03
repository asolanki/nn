#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# define CNN class
class ConvNN(nn.Module):

    def __init__(self, num_classes):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout2d(p=0.2) 

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout2 = nn.Dropout2d(p=0.2) 

        # output size = (input_size - kernel_size + 2*padding)/stride + 1)
        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(64*7*7, 32)
#        self.fc1 = nn.Linear(14336, 32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.maxpool2(out)

        # flatten
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        return out
