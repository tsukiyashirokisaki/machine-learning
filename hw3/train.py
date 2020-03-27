
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import torch.nn as nn
from PIL import Image
import sys
data_dir=sys.argv[1]
label_dir=sys.argv[2]
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)

class train_hw3(Dataset):
    def __init__(self,data_dir,label,transform=None):
        self.data_dir=data_dir
        self.label=label
        self.transform=transform
    def __getitem__(self,index):
        pic_file='{:0>5d}.jpg'.format(self.label[index][0])
        img=cv2.imread(os.path.join(self.data_dir,pic_file),cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
            img=img.reshape(-1,img.shape[1],img.shape[2])
        return torch.FloatTensor(img).unsqueeze(0),self.label[index,1]
    def __len__(self):
        return self.label.shape[0]
class test_hw3(Dataset):
    def __init__(self,data_dir,label,transform=None):
        self.data_dir=data_dir
        self.label=label
        self.transform=transform
    def __getitem__(self,index):
        pic_file='{:0>4d}.jpg'.format(self.label[index])
        img=cv2.imread(os.path.join(self.data_dir,pic_file),cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
            return img
        else:
            return torch.FloatTensor(img).unsqueeze(0)
    def __len__(self):
        return self.label.shape[0]

drop_ind=[]
for pic_file in os.listdir(data_dir):
    img=cv2.imread(os.path.join(data_dir,pic_file),cv2.IMREAD_GRAYSCALE)
    if (img==0).all():
        drop_ind.append(int(pic_file.replace(".jpg","")))
train_ind=set(np.arange(0,28888,1))-set(drop_ind)
train_ind=list(train_ind)
label=np.array(pd.read_csv(label_dir))
train_label=np.empty([len(train_ind),2],dtype=np.int)
j=0 
for i in range(28888):
    if label[i][0] in train_ind:
        train_label[j]=label[i]
        j+=1
    
BATCH_SIZE=512
train_dataset=train_hw3(data_dir,train_label)#,transform_train)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
            nn.Linear(6*6*512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 7),
            )
        

    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) #(24,24)
        x = self.conv2(x) #(12,12)
        x = self.conv3(x) #(6,6)
        x = x.view(-1, 6*6*512)
        x = self.fc1(x)
        return x
model=Net().cuda()
num_epoch=40
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
train_max=0.9
for epoch in range(num_epoch):
    model.train()
    train_loss = []
    train_acc = []
    for idx, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        predict = torch.max(output, 1)[1]
        acc = np.mean((label == predict).cpu().numpy())
        train_acc.append(acc)
        train_loss.append(loss.item())
    
    if np.mean(train_acc) > train_max:
        train_max = np.mean(train_acc)
        checkpoint_path = 'model_{}.pth'.format(epoch+1) 
        torch.save(model.state_dict(), checkpoint_path)
        print('model saved to %s' % checkpoint_path)

