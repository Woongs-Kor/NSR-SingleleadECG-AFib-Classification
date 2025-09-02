import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import glob
import sys
from collections import Counter

import torch
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torch.autograd import Variable


class MyDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data_list = sorted([os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.npy')])
        #self.data_list = np.load(data_path)
        self.label = np.load(label_path)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label
    
class MyDataset2(Dataset):
    def __init__(self, data_path1, data_path2, label_path):
        self.data_list1 = sorted([os.path.join(data_path1, filename) for filename in os.listdir(data_path1) if filename.endswith('.npy')])
        self.data_list2 = sorted([os.path.join(data_path2, filename) for filename in os.listdir(data_path2) if filename.endswith('.npy')])
        self.data_list = self.data_list1 + self.data_list2
        #self.data_list = np.load(data_path)
        self.label = np.load(label_path)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label
    
class MyDataset3(Dataset):
    def __init__(self, data_path1, data_path2, data_path3, label_path):
        self.data_list1 = sorted([os.path.join(data_path1, filename) for filename in os.listdir(data_path1) if filename.endswith('.npy')])
        self.data_list2 = sorted([os.path.join(data_path2, filename) for filename in os.listdir(data_path2) if filename.endswith('.npy')])
        self.data_list3 = sorted([os.path.join(data_path3, filename) for filename in os.listdir(data_path3) if filename.endswith('.npy')])
        self.data_list = self.data_list1 + self.data_list2 + self.data_list3
        self.label = np.load(label_path)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label



class MyDataset4(Dataset):
    def __init__(self, data_path, label_path):
        self.data_list = sorted([os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.npy')])
        #self.data_list = np.load(data_path)
        self.label = np.load(label_path)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label


#label 1 half downsample
class DSDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data_list = sorted([os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.npy')])
        self.label = np.load(label_path)
        self.indices = np.where(self.label == 1)[0]
        self.del_indices = np.random.choice(self.indices, size=int(len(self.indices)*0.3), replace=False)
        self.data_list = [item for i, item in enumerate(self.data_list) if i not in self.del_indices]
        self.label = np.delete(self.label, self.del_indices)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label
    
    
#label 0 downsample
class DSDataset2(Dataset):
    def __init__(self, data_path, label_path):
        self.data_list = sorted([os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.npy')])
        self.label = np.load(label_path)
        self.indices = np.where(self.label == 0)[0]
        self.del_indices = np.random.choice(self.indices, size=int(len(self.indices)*0.7), replace=False)
        self.data_list = [item for i, item in enumerate(self.data_list) if i not in self.del_indices]
        self.label = np.delete(self.label, self.del_indices)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label


class DSDataset3(Dataset):
    def __init__(self, data_path1, data_path2, label_path):
        self.data_list1 = sorted([os.path.join(data_path1, filename) for filename in os.listdir(data_path1) if filename.endswith('.npy')])
        self.data_list2 = sorted([os.path.join(data_path2, filename) for filename in os.listdir(data_path2) if filename.endswith('.npy')])
        self.data_list = self.data_list1 + self.data_list2
        self.label = np.load(label_path)
        self.indices = np.where(self.label == 0)[0]
        self.del_indices = np.random.choice(self.indices, size=int(len(self.indices)*0.7), replace=False)
        self.data_list = [item for i, item in enumerate(self.data_list) if i not in self.del_indices]
        self.label = np.delete(self.label, self.del_indices)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_idx = self.data_list[idx]
        label_idx = self.label[idx]
        data = torch.from_numpy(np.load(data_idx)).float()
        label = torch.from_numpy(np.asarray(label_idx)).long()
        
        return data, label








