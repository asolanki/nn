#!/usr/bin/env python

import unittest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from neural_nets import ConvNN

import logging
logging.basicConfig(level=logging.DEBUG)


class TestConvNN(unittest.TestCase):

    def setUp(self):
        
        # define environment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define hyperparameters
        self.batch_size = 64
        self.num_classes = 10
        self.lr = 0.001
        self.epochs = 10

        # define MNIST transforms
        self.all_transforms = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # load MNIST data
        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.all_transforms)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.all_transforms)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


        print('Testing on device: {}'.format(self.device))
        print('with training size: {}'.format(len(self.train_dataset)))


    def test_setup(self):
        self.assertEqual(self.batch_size,64)

    def test_train(self):

        model = ConvNN(self.num_classes).to(self.device)
        print('model instantiated on device: {} with architecture:'.format(self.device))
        print(model)
        loss_func = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        total_step = len(self.train_loader)

        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                outputs = model(images)
                loss = loss_func(outputs, labels)

                # backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch+1, self.epochs, i+1, total_step, loss.item()))
                



if __name__ ==  '__main__':
    unittest.main(verbosity=2)