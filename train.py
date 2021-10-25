# load libraries
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from os import listdir
import time
import copy
import argparse

# Initiate variables with default values
arch = 'vgg16'
hidden_units = 5120
learning_rate = 0.001
epochs = 10
device = 'cpu'

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',type=str, help='Location of directory with data for image classifier to train and test')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose among 3 pretrained networks - vgg16, alexnet, and densenet121')
parser.add_argument('-H','--hidden_units',action='store',type=int, help='Select number of hidden units for 1st layer')
parser.add_argument('-l','--learning_rate',action='store',type=float, help='Choose a float number as the learning rate for the model')
parser.add_argument('-e','--epochs',action='store',type=int, help='Choose the number of epochs you want to perform gradient descent')
parser.add_argument('-s','--save_dir',action='store', type=str, help='Select name of file to save the trained model')
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available')

args = parser.parse_args()

# Select parameters entered in command line
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_model(arch='vgg16',hidden_units=5120,learning_rate=0.001):

    model =  getattr(models,arch)(pretrained=True)
    in_features = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(in_features,hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.15)),
                           ('fc2',nn.Linear(hidden_units,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.15)),
                           ('fc3',nn.Linear(512,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))

    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1,last_epoch=-1)
    
    return model, criterion, optimizer, scheduler

model, criterion, optimizer, scheduler = create_model(arch, hidden_units, learning_rate)

print('-' * 40)
print('Model has been built!')

# Directory location of images
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define transforms for sets and normalize images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dictionary holding location of data
data_dict = {'train': train_dir,
             'test': test_dir,
             'valid': valid_dir}

# Images are loaded with ImageFolder and transformations applied
image_datasets = {x: datasets.ImageFolder(data_dict[x],transform = data_transforms[x])
                  for x in ['train', 'test', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) 
               for x in ['train', 'test', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'valid']}

# Variable holding names for classes
class_names = image_datasets['train'].classes
    
def train_model(model, criterion, optimizer, scheduler, epochs=2):

    since = time.time()
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Each epoch training and validation step
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model_trained = train_model(model, criterion, optimizer, scheduler, epochs)

print('-' * 40)
print('Model has been successfully trained')
print('-' * 40)

def save_model(model_trained):

    model_trained.class_to_idx = image_datasets['train'].class_to_idx
    model_trained.cpu()
    save_dir = ''
    checkpoint = {
             'arch': arch,
             'hidden_units': hidden_units, 
             'state_dict': model_trained.state_dict(),
             'class_to_idx': model_trained.class_to_idx,
             }
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir)    
    
save_model(model_trained)
print('-' * 40)
print(model_trained)
print('Model has been successfully saved.')
print('-' * 40)