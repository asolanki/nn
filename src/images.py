#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# define CNN class
def ConvNN(nn.Module):

    def __init__(self, num_classes):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.RelU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2),

        # drop out?
        self.drop_out = nn.Dropout()


        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2),

        # output size = (input_size - kernel_size + 2*padding)/stride + 1)
        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(32*7*7, 32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)
        out = self.drop_out(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.max_pool2(out)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        return out
