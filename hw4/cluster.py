import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import sys
from sklearn import ensemble
import pickle 
import cv2
r_path,w_path=sys.argv[1],sys.argv[2]
class Autoencoder(nn.Module):
    def __init__(self,lay_in,lay_out,linear=False,linear_in=10,linear_out=10,lay_in2=1):
        super(Autoencoder, self).__init__()
        self.linear=linear
        self.linear_in=linear_in
        self.linear_out=linear_out
        self.lay_in2=lay_in2
        # define: encoder
        self.encoder1 = nn.Sequential(nn.Conv2d(lay_in, lay_out, 3, 2, 1))
        self.encoder2 = nn.Sequential(nn.Conv2d(lay_out, lay_in2, 3, 2, 1))
        self.fc1= nn.Sequential(
            
            nn.Linear(linear_in,linear_out),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm1d(linear_out),)
                                
        self.fc2=nn.Sequential(
            
            nn.Linear(linear_out,linear_in),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm1d(linear_in),)

        # define: decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(lay_out, lay_in, 2, 2),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(lay_in2, lay_out, 2, 2),
        )


    def forward(self, x):

        encoded = self.encoder1(x)
        if self.lay_in2!=1:
            encoded = self.encoder2(encoded)
            #print(encoded.shape)
            decoded = self.decoder2(encoded)
            decoded = self.decoder1(decoded)
            
            return encoded, decoded

            
        #print(encoded.shape)
        if self.linear:
            s=encoded.shape
            code = self.fc1(encoded.view(-1,self.linear_in))
            x= self.fc2(code)
            decoded = self.decoder1(encoded.reshape(s))
            return  code, decoded 
        else:
            decoded = self.decoder1(encoded)
            return encoded,decoded

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

class dataset(Dataset):
    def __init__(self,data,train=True):
        self.data=data
    def __getitem__(self,index):
        img=self.data[index]
        
        return toTensor(img)
    def __len__(self):
        return self.data.shape[0]


trainX = np.load(r_path)
test_dataset=dataset(trainX,False)
BS=256
test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=False,num_workers=2)

def output(autoencoders,out,seed2):
    p1=np.empty([9000,out])
    for i,x in enumerate(test_dataloader):
        code=x.cuda()
        for j in range(len(autoencoders)):
            code = autoencoders[j](code)[0]
        p1[i*BS:i*BS+BS]=code.cpu().detach().numpy()
    with open('model.pkl', 'rb') as f:
        clf=pickle.load( f)
    result=clf.predict(p1)
    return result
    
def save_write(path,result):
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(path,index=False)
    

func,out,seed1,seed2=torch.load("net.pkl")
result=output(func,out,seed2)
save_write(w_path,result)

