# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import time
import h5py
import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
from hdf5_dataloader import my_transforms
import torch
from torchvision import transforms, datasets, utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CBEDDataset(Dataset):
    """ CBED dataset."""

    def __init__(self, data_file, transform=None):
        self.transform = transform
        self.x_data = None
        self.y_data = None
        
        self.__read_hdf5__(data_file)
        
    def __read_hdf5__(self, data_file):
        f = h5py.File(data_file, mode='r', swmr=True)
        tmp_x, tmp_y = [], []
        for key in f.keys():
            tmp_x.append(f[key]['cbed_stack'][()])
            # indexed from 0
            tmp_y.append(int(f[key].attrs['space_group'])-1)
        #tmp_y = map(int, tmp_y)
        self.x_data = torch.from_numpy(np.array(tmp_x))
        self.y_data = torch.from_numpy(np.array(tmp_y))
        
    def __getitem__(self, index):
        _x = self.x_data[index]#.permute(1, 2, 0)
        if self.transform is not None:
            _x = self.transform(_x)
        return _x, self.y_data[index]

    def __len__(self):
        return len(self.y_data)

data_transform = transforms.Compose([
    my_transforms.TensorPower(0.25),
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    # check mean and std
    #transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 61 * 61,10000)
        self.fc2 = nn.Linear(10000,2000)
        self.fc3 = nn.Linear(2000,500)
        self.fc4 = nn.Linear(500,230)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
model = Model()
model.to(device)

# read weights
count_data = pd.read_csv("count.csv")
#weights = count_data['weight']
weights = 1.0/count_data['weight']#.replace(np.inf, 0.0)
weights[weights==np.inf] = 0.0
class_weights = torch.Tensor(weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch=1):
    model.train()
    for i in range(epoch):
        for data, target in train_loader:
            data = Variable(data).to(device)
            target = Variable(target).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
    return loss.item()

def dev():
    model.eval()
    test_loss, correct = 0, 0
    for data, target in (dev_loader):
        data = Variable(data, volatile=True).to(device)
        target = Variable(target).to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).cpu().data.numpy()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1][0]
        print(pred+1, target+1)
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(dev_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dev_loader.dataset),
        100. * correct / len(dev_loader.dataset)))

# train
data_paths = glob.glob('./train/' + '*.h5')
BATCH_SIZE = 4
for this_file in (data_paths):
    try:
        train_dataset = CBEDDataset(this_file,
                                    transform = data_transform)
    except:
        print("Warning: error handling %s, will be ignored."%this_file)
        continue
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=32)
    
    begin = time.time()
    epoch = 20
    loss = train(epoch)
    end = time.time()
    print('\nTraining file [{0:>30s} ]:\n\tTiming= {1:>5.2f} s,\tLoss= {2:>10.6f}'.format(
          this_file, end-begin, loss))

# dev
dev_paths = glob.glob('./dev/' + '*.h5')
BATCH_SIZE = 1
for this_file in (dev_paths):
    #print("Current file: %s"%this_file)
    try:
        dev_dataset = CBEDDataset(this_file,
                                  transform = data_transform)
    except:
        print("Warning: error handling %s, will be ignored."%this_file)
        continue
    else:
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=32)
    
    dev()
