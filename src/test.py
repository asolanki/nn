#!/usr/bin/env python

import unittest
import torch
import torchvision
import torchvision.transforms as transforms

import images.py


class TestMNIST(unittest.TestCase):

    def setUp(self):
        
        # define hyperparameters
        self.batch_size = 64
        self.num_classes = 10
        self.lr = 0.001
        self.epochs = 20

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_setup(self):
        self.assertEqual(batch_size,64)

if __name__ ==  '__main__':
    unittest.main()
