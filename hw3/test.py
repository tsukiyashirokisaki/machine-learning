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
w_dir=sys.argv[2]
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
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

np.random.seed(0)    
BATCH_SIZE=256
test_dataset=test_hw3(data_dir,np.arange(7000))#,transform_test)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
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
        #image size (40,40)
        x = self.conv1(x) #(20,20)
        x = self.conv2(x) #(10,10)
        x = self.conv3(x) #(5,5)
        x = x.view(-1, 6*6*512)
        x = self.fc1(x)
        
        return x


model=Net().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()
ans=[]
a=0
for i, data in enumerate(test_loader):
    test_pred = model(data.cuda())
    ans+=np.argmax(test_pred.cpu().data.numpy(), axis=1).tolist() 
    a+=len(data)
        
for i in range(7000):
    if (test_dataset.__getitem__(i)<10).all():
        ans[i]=0    
pd.DataFrame({'id':np.arange(7000),'label':ans}).to_csv(w_dir,index=False)

