import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import glob
import sys
import argparse

#gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torch.autograd import Variable
from torch.optim import AdamW


#custom
from resnet import BottleneckBlock
from resnet import BasicBlock
from resnet import ResNet1D
from CustomDataset import MyDataset

#Setup necessary functions
def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    
    #Print with fixed 4 decimal places
    np.set_printoptions(precision=4)
    
    #fix seed
    fix_seed(42)

    #GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #parser, arg, hypter parameter정리
    parser = argparse.ArgumentParser(description='Mobile ECG with hyperparameter tuning')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',help='learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    
    # Customized Dataset settings
    MobileDataset = MyDataset('data_list/srv/data_server/Mobile_ECG_Scaled(-1,1)/' ,'data_list/srv/data_server/Mobile_ECG_Label/Mobile_total_label.npy')

    #Split dataset (Train:Validation:Test = 6:2:2), stratified by label
    grouped_data = {}

    for i, label in enumerate(MobileDataset.label):
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(i)
    
    train_indices, val_indices, test_indices = [], [], []    

    for label, indices in grouped_data.items():
        split_1 = int(0.6 * len(indices))
        split_2 = int(0.8 * len(indices))
        train_indices.extend(indices[:split_1])
        val_indices.extend(indices[split_1:split_2])
        test_indices.extend(indices[split_2:])

    train_dataset = torch.utils.data.Subset(MobileDataset, train_indices)
    validation_dataset = torch.utils.data.Subset(MobileDataset, val_indices)
    test_dataset = torch.utils.data.Subset(MobileDataset, test_indices)

    # train label count
    train_label_counts = {}
    for index in train_indices:
        label = MobileDataset.label[index]
        if label not in train_label_counts:
            train_label_counts[label] = 0
        train_label_counts[label] += 1

    # validation label count
    validation_label_counts = {}
    for index in val_indices:
        label = MobileDataset.label[index]
        if label not in validation_label_counts:
            validation_label_counts[label] = 0
        validation_label_counts[label] += 1

    # test label count
    test_label_counts = {}
    for index in test_indices:
        label = MobileDataset.label[index]
        if label not in test_label_counts:
            test_label_counts[label] = 0
        test_label_counts[label] += 1

    print("Train Dataset - Label Counts:")
    print(train_label_counts)
    print("Validation Dataset - Label Counts:")
    print(validation_label_counts)
    print("Test Dataset - Label Counts:")
    print(test_label_counts)
        
    #Dataloader settings
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)

    #Model settings
    model = ResNet1D(BasicBlock, [3, 4, 6, 3], num_classes = 2)
    
    #if use multi-GPU
    # model = nn.DataParallel(model)
    
    model.to(device)
    
    #Loss, Optimizer settings
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr= args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-1)
    
    #Train, Test, Validation loop 정리
    def train_loop(model, train_dataloader, train_dataset, optimizer, criterion, device):
        model.train()
        running_loss = 0.
        corrects = 0.
    
        train_total_pred = np.array([])
        train_total_label = np.array([])
    
        #feed forward
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
        
            inputs, labels = inputs.to(device), labels.to(device)
            #backpropagation
            optimizer.zero_grad()
        
            outputs = model(inputs)
            
            loss = criterion(outputs, labels.squeeze(dim=-1).long())
            #loss = criterion(outputs, labels.long())
    
            _, preds = torch.max(outputs, 1)
        
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            corrects += torch.sum(preds == labels.squeeze(dim=-1).data)
            #corrects += torch.sum(preds == labels.data)
        
            train_total_pred = np.append(train_total_pred, preds.cpu().numpy())
            train_total_label = np.append(train_total_label, labels.cpu().numpy())        
        
        #scheduler.step()
    
        train_loss = running_loss / len(train_dataset)
        train_acc = corrects / len(train_dataset)
    
        precision, recall, f1_score, support = precision_recall_fscore_support(train_total_label, train_total_pred, average=None)
        #cm = confusion_matrix(train_total_label, train_total_pred)
        
        print("[Train set]")
        print("loss:{:.4f} acc:{:.4f} correct:{}/total:{} precision:{} recall:{} f1:{}".format(
            train_loss, train_acc, corrects, len(train_dataset), precision[1], recall[1], f1_score[1]))
        return train_acc, train_loss, precision[1], recall[1], f1_score[1]
    
    def validation_loop(model, validation_dataloader, validation_dataset, optimizer, criterion, device):
        model.eval()
        running_loss = 0.
        corrects = 0.

        validation_total_pred = np.array([])
        validation_total_label = np.array([])
    
        with torch.no_grad():
            for i,data in enumerate(validation_dataloader):
                inputs, labels = data
            
                inputs, labels = inputs.to(device), labels.to(device)
            
                outputs = model(inputs)
                
                loss = criterion(outputs, labels.squeeze(dim=-1).long())
                #loss = criterion(outputs, labels.long())
            
                _, preds = torch.max(outputs,1)
            
                running_loss += loss.item()
                corrects += torch.sum(preds == labels.squeeze(dim=-1).data)
                #corrects += torch.sum(preds == labels.data)
                
                validation_total_pred = np.append(validation_total_pred, preds.cpu().numpy())
                validation_total_label = np.append(validation_total_label, labels.cpu().numpy())
    
        validation_loss = running_loss / len(validation_dataset)
        validation_acc = corrects / len(validation_dataset)

        precision, recall, f1_score, support = precision_recall_fscore_support(validation_total_label, validation_total_pred, average=None)
        #cm = confusion_matrix(validation_total_label, validation_total_pred)
        
        print("[Validation set]")
        print("loss:{:.4f} acc:{:.4f} correct:{}/total:{} precision:{} recall:{} f1:{}".format(
            validation_loss, validation_acc, corrects, len(validation_dataset), precision[1], recall[1], f1_score[1]))
        return validation_acc, validation_loss, precision[1], recall[1], f1_score[1]

    def test_loop(model, test_dataloader, test_dataset, optimizer, criterion, device):
        model.eval()
        running_loss = 0.
        corrects = 0.
    
        test_total_pred = np.array([])
        test_total_label = np.array([])    
    
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                inputs, labels = data
            
                inputs, labels = inputs.to(device), labels.to(device)
            
                outputs = model(inputs)
                
                loss = criterion(outputs, labels.squeeze(dim=-1).long())
                #loss = criterion(outputs, labels.long())

                _, preds = torch.max(outputs,1)
            
                running_loss += loss.item()
                corrects += torch.sum(preds == labels.squeeze(dim=-1).data)
                #corrects += torch.sum(preds == labels.data)
            
                test_total_pred = np.append(test_total_pred, preds.cpu().numpy())
                test_total_label = np.append(test_total_label, labels.cpu().numpy())
                        
        test_loss = running_loss / len(test_dataset)
        test_acc = corrects / len(test_dataset)
    
        precision, recall, f1_score, support = precision_recall_fscore_support(test_total_label, test_total_pred, average=None)
        cm = confusion_matrix(test_total_label, test_total_pred)
        auc = roc_auc_score(test_total_label, test_total_pred)
        
        print("[Test set]")
        print("loss:{:.4f} acc:{:.4f} correct:{}/total:{} precision:{} recall:{} f1:{}".format(
            test_loss, test_acc, corrects, len(test_dataset), precision[1], recall[1], f1_score[1]))
        print("auc:{:.4f}".format(auc))
        print("confusion matrix")
        print(cm)
        return test_acc, test_loss, precision[1], recall[1], f1_score[1]
  
    # train,validation,test start
    for epoch in range(1, args.epochs+1):
    
        print('Epoch:{}'.format(epoch))
    
        acc_train, loss_train, precision_train, recall_train, f1_train= train_loop(model, train_dataloader, train_dataset, optimizer, criterion, device)
    
        acc_validation, loss_validation, precision_validation, recall_validation, f1_validation= validation_loop(model, validation_dataloader, validation_dataset, optimizer, criterion, device)
    
        acc_test, loss_test, precision_test, recall_test, f1_test= test_loop(model, test_dataloader, test_dataset, optimizer, criterion, device)
            
        
    
if __name__ == '__main__':
    main()   