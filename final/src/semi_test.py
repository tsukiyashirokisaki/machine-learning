import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,models, transforms
import numpy as np
import os, sys
import torch.nn.functional as F
from sklearn.model_selection import train_test_split    
from torch.autograd import Variable, Function
import torch.optim as optim
import itertools, datetime
import random
import cv2
from PIL import Image
import pandas as pd

MODEL_NAME = 'NN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trds(Dataset):
    def __init__(self,data_dir,label,transform):
        self.data_dir = data_dir
        self.label = label
        self.transform=transform
    def __getitem__(self,index):
        
        img=self.data_dir[index]
        if np.random.randint(0,2):
            img=np.expand_dims(cv2.Canny(img,170 , 300),2)[:,:,0]
#        print(img.shape)
        img  = Image.fromarray(img)
        img = self.transform(img)
        return img,torch.tensor(self.label[index],dtype=torch.long)
    def __len__(self):
        return self.label.shape[0]
class valds(Dataset):
    def __init__(self,data_dir,label,transform):
        self.data_dir = data_dir
        self.label = label
        self.transform=transform
    def __getitem__(self,index):
        img=self.data_dir[index]
        
        #img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img  = Image.fromarray(img)
        
        img = self.transform(img)
        return img,torch.tensor(self.label[index],dtype=torch.long)
    def __len__(self):
        return self.label.shape[0]
class teds(Dataset):
    def __init__(self,data_dir,transform):
        self.data_dir = data_dir
        self.transform=transform
    def __getitem__(self,index):
        
        img  = Image.fromarray(img)
        img = self.transform(img)
        return img
    def __len__(self):
        return self.data_dir.shape[0]


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),   
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(512), 
            nn.MaxPool2d(2),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
            )
        

    def forward(self, x):
        #image size (40,40)
        x = self.conv1(x) #(20,20)
        x = self.conv2(x) #(10,10)
        x = self.conv3(x) #(5,5)
        x = x.view(-1,4608)
        x = self.fc1(x)
        
        return x


# In[3]:




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            #nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  
            nn.Dropout(0.1)
        )
        

        
        
        self.fc1 = nn.Sequential(
            nn.Linear( 6272, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
            )
        

    def forward(self, x):
        #image size (40,40)
        x = self.conv1(x) #(20,20)
        
        x = x.view(-1,6272)
        x = self.fc1(x)
        
        return x
def save(model,target_testloader):
    out=[]
    confidence=[]
    m=nn.Softmax(dim=1)
    use_gpu =torch.cuda.is_available()
    print(use_gpu)
    for i, datas in enumerate(target_testloader):
        inputs, labels = datas
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputv, labelv = Variable(inputs), Variable(labels)

        outC = m(model(inputv))
        #print(outC)
        conf, predicted = torch.max(outC.data, 1)  
        out+=predicted.detach().cpu().numpy().tolist()
        confidence+=conf.detach().cpu().numpy().tolist()
        #break
    return out,confidence


if __name__ == '__main__':
    testX = np.load('data/testX.npy')
    label=np.load("data/output.npy")
    print(testX.shape)
    testX=testX[:,:,:,0]
    mean = np.array([0.5])
    std = np.array([0.5])
    trainX=[]
    trainY=[]
    for i,ele in enumerate(label[:,1]):
        if ele:
            trainX.append(testX[i])
            trainY.append(label[i,0])
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    transform_source = transforms.Compose([transforms.RandomResizedCrop(28),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomAffine(5),
                                           transforms.ToTensor(), 
                                           transforms.Normalize(mean,std)])
    transform_target = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize(mean,std)])
    x,x_,y,y_=train_test_split(trainX,trainY,test_size=0.33,random_state=0)
    train_dataset= trds(x,y, transform=transform_source)
    vali_dataset = valds(x_,y_, transform=transform_target  )
    test_dataset = valds(testX,np.zeros(len(testX)), transform_target)
    train_pic_loader = DataLoader(train_dataset, batch_size=100, shuffle=True,drop_last=True)
    vali_loader = DataLoader(vali_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    model=Resnet().cuda()
    model.load_state_dict(torch.load('model/Resnet18like.pkl'))
    ans=save(model,test_loader)[0]
    df = pd.DataFrame({'id': np.arange(0,len(ans)), 'label': ans})
    df.to_csv('submission/semi_ans.csv',index=False   )






