# -*- coding: utf-8 -*-
"""pred_CBED.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1so8i-5nk4xpYEuXOXbHDgkIfe8b1LLav

#TODO
1. ~~~Normalize data power(0.25), and between -1 and 1:
    https://github.com/fab-jul/hdf5_dataloader~~~
2. Use prefetch to accelerate data access: https://zhuanlan.zhihu.com/p/66145913
3. Use transfer learning from Imagenet: https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad?
4. ~~~Add EDA for space imbalance~~~(had been separated as count.py)
5. Add save and restart function
6. ~~~Check if it's the right way to permute matrix~~~, and tune parameters
7. ~~~Add dropout layers~~~
8. Add parallel acceleration: distributedparallelcpu, horovod https://github.com/horovod/tutorials/tree/master/fashion_mnist
https://github.com/vlimant/mpi_learn
https://blog.csdn.net/zwqjoy/article/details/89415933
9. Imbalanced data
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
10. ~~~Add support for multiple pred~~~
11. can we really train on different hf files, instead on a total run?
"""

from __future__ import print_function, division
import os
import time
import h5py
#import shutil
#import collections
import glob
#import pylab
import pandas as pd
import numpy as np
from scipy import stats
#from tqdm import tqdm, tqdm_notebook as tn
from PIL import Image
#import matplotlib.pyplot as plt
#from skimage import io, transform
#from prefetch_generator import BackgroundGenerator, background
import torch
from torchvision import transforms, datasets, utils
import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# defined functions
from tools import my_transforms
from tools.my_model import Model
from tools.my_prefetcher import data_prefetcher
from tools.my_prefetcher import BackgroundGenerator

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
 
# %matplotlib inline

def get_weights(filename):
    count_data = pd.read_csv(filename)
    weights = 1.0/count_data['weight']
    weights[weights==np.inf] = 1000.0

    return weights

# define initial variables, constants
def init():
    global device, weights, train_paths, dev_paths, model_ckpt
    global TRAIN_BATCH_SIZE, DEV_BATCH_SIZE, OMP_NUM_THREADS, USE_ALL_GPU

    USE_ALL_GPU = False

    model_ckpt = "params.pkl"

    OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    weights = get_weights('count.csv')
    train_paths = glob.glob('./train/' + '*.h5')
    dev_paths = glob.glob('./dev/' + '*.h5')

    TRAIN_BATCH_SIZE = 32
    DEV_BATCH_SIZE = 1

class CBEDDataset(Dataset):
    """ CBED dataset."""

    def __init__(self, data_file, transform=None):
        self.transform = transform
        self.x_data = None
        self.y_data = None
        
        self.__read_hdf5(data_file)
        
    def __read_hdf5(self, data_file):
        f = h5py.File(data_file, mode='r', swmr=True)
        tmp_x, tmp_y = [], []
        for key in f.keys():
            tmp_x.append(f[key]['cbed_stack'][()])
            # indexed from 0 to avoid overflow error
            tmp_y.append(int(f[key].attrs['space_group'])-1)
        f.close()
        #tmp_y = map(int, tmp_y)
        #self.x_data = torch.from_numpy(np.array(tmp_x))
        #self.y_data = torch.from_numpy(np.array(tmp_y))
        self.x_data = torch.Tensor(tmp_x)
        self.y_data = torch.Tensor(tmp_y).long()
        
    def __getitem__(self, index):
        _x = self.x_data[index]
        if self.transform is not None:
            _x = self.transform(_x)
        return _x, self.y_data[index]

    def __len__(self):
        return len(self.y_data)

def HDF5Generator(h5_files, batch_size=1, transform=None):
    for h5_file in h5_files:
        this_dataset = CBEDDataset(data_file=h5_file,
                                   transform=transform)
        this_loader = DataLoader(dataset=this_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 #pin_memory=True,
                                 num_workers=OMP_NUM_THREADS)
        yield this_loader

class HDF5Dataset(Dataset):
    '''read HDF5 file'''

    def __init__(self, h5_files, batch_size=1, transform=None):
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.transform = transform
        self.count = 0

    def __getitem__(self, index):
        this_dataset = CBEDDataset(data_file=self.h5_files[index],
                                   transform=self.transform)
        this_loader = DataLoader(dataset=this_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=OMP_NUM_THREADS)
        return this_loader

    def __len__(self):
        return len(self.h5_files)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.count < self.__len__():
            self.count += 1
            return self.__getitem__(self.count)
        else:
            raise StopIteration()

data_transform = transforms.Compose([
                my_transforms.TensorPower(0.25),
                #transforms.ToPILImage(),
                #transforms.Resize(256),
                #transforms.CenterCrop(256),
                #transforms.ToTensor(),
                #transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
                ])

def train(epoch=0):
    model.train()
    total_loss = 0.0
    print(type(train_file_loader))
    for inputs, labels in train_file_loader:
    #prefetcher = data_prefetcher(train_loader)
    #data, target = prefetcher.next()
    #while data is not None:
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        #print("\n\n output")
        #print(outputs)
        #print("target")
        #print(labels)
        loss.backward()
        optimizer.step()
        #print(stats.describe(outputs.cpu().detach().numpy(), axis=1))

        #data, target = prefetcher.next()
        
    #end = time.time()
    #print('Training: Timing: {:.2f} s\tLoss: {:.6f}'.format(
    #      end-begin, loss.item())

    return total_loss/len(train_file_loader)

def dev():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in BackgroundGenerator(dev_loader, max_prefetch=3):
        data = Variable(data, volatile=True).to(device)
        target = Variable(target).to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).cpu().data.numpy()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1][0]
        #print(pred+1, target+1)
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(dev_loader.dataset)
    print('\nTest set:\n')
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dev_loader.dataset),
        100. * correct / len(dev_loader.dataset)))
    
# multiple predictions
def dev_multipred():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in BackgroundGenerator(dev_loader, max_prefetch=3):
        data = Variable(data, volatile=True).to(device)
        target = Variable(target).to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).cpu().data.numpy()
        # get the index of the max
        #pred = output.data.max(1, keepdim=True)[1][0]
        sorted, preds = torch.sort(output.data[0], descending=True)
        #print(indices[0:5]+1, target+1)
        #correct += pred.eq(target.data.view_as(pred)).sum()
        if target in preds[:5]:
            correct += 1

    test_loss /= len(dev_loader.dataset)
    print('\nTest set:\n')
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dev_loader.dataset),
        100. * correct / len(dev_loader.dataset)))

#================================================================
# Run starts from here
#================================================================

init()

model = Model()
if torch.cuda.device_count() > 1 and USE_ALL_GPU:
    print("We are running on {} GPUs!\n".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
model.to(device)

#load model if checkpoint exists
if os.path.isfile(model_ckpt):
    try:
        pass
        #model.load_state_dict(torch.load(model_ckpt))
    except:
        pass

# use softmax
#weights = np.exp(count_data['weight'])
class_weights = torch.Tensor(weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999))
#if torch.cuda.device_count() > 1:
#    optimizer = nn.DataParallel(optimizer)

# prefetch multiple files
# TODO: send multiple h5 files indices to CBEDDataset, and read it at once
# Then prefetch this
#train_path_dataset = HDF5Dataset(train_paths,
train_path_dataset = HDF5Generator(train_paths,
                                  batch_size=TRAIN_BATCH_SIZE,
                                  transform=data_transform)
#train_path_loader = DataLoader(dataset=train_path_dataset,
#                               batch_size=1,
#                               shuffle=True,
#                               pin_memory=True,
#                               num_workers=OMP_NUM_THREADS)
#prefetcher = data_prefetcher(train_path_dataset)
#train_file_loader = prefetcher.next()
for train_file_loader in BackgroundGenerator(train_path_dataset):#, max_prefetch=1):
    print("haha")
#train_file_loader = train_path_gen.next()
#while train_file_loader is not None:
#for this_file in (train_paths):
    #print("Current file: %s"%this_file)
    #read_start = time.time()
    #try:
    #    train_dataset = CBEDDataset(this_file,
    #                                transform = data_transform)
    #except:
    #    print("Warning: error handling %s, will be ignored."%this_file)
    #    continue
    #else:
    #    train_loader = DataLoader(dataset=train_dataset,
    #                              batch_size=TRAIN_BATCH_SIZE,
    #                              shuffle=True,
    #                              pin_memory=True,
    #                              num_workers=OMP_NUM_THREADS)

    #read_end = time.time()
    #print('Reading {} tooks {} s\n'.format(this_file, read_end-read_start))
    begin = time.time()
    for i in range(10):
        loss = train()
        print(loss),
    print('Training batch:\tTiming: {:.2f} s,\tLoss: {:.6f}'.format(
          time.time()-begin, loss))

    # save model
    #torch.save(model.state_dict(), model_ckpt)

    #train_file_loader = prefetcher.next()
    #train_file_loader = train_path_gen.next()
    print('Training total time {:.2f} s\n'.format(time.time()-begin))

# test
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
                                batch_size=DEV_BATCH_SIZE,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=OMP_NUM_THREADS)
    
    dev_multipred()
