#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from os.path import exists

from neural_nets import ConvNN

import logging
logging.basicConfig(level=logging.DEBUG)

model_path = "models/mnist_model.pt"
retrain = True


def train_mnist(model,batch_size,all_transforms,lr,epochs,device):


    # load MNIST data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=all_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    total_step = len(train_loader)

    print('Training on device: {}'.format(device))
    print('with training size: {}'.format(len(train_dataset)))
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
            
    torch.save(model.state_dict(), model_path)
    return model





if __name__ ==  '__main__':
        # define environment
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define hyperparameters
        batch_size = 100
        num_classes = 10
        lr = 0.001
        epochs = 10
        
        
        # define MNIST transforms
        all_transforms = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # train model
        model = ConvNN(num_classes).to(device)
        print('model instantiated on device: {} with architecture:'.format(device))
        print(model)
        
        if not exists(model_path):
            print('no model, training new model')
            model = train_mnist(model,batch_size,all_transforms,lr,epochs,device)
        else:
            print('model exists, loading model')
            model.load_state_dict(torch.load(model_path))


        # test model
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=all_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            correct=0
            total=0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print('Accuracy of the model on {} test images: {} %'.format(total, 100*correct/total))

