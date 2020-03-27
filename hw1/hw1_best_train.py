import numpy as np
import sys
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
r_path1,r_path2,r_path3,w_path=sys.argv[1:]
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def ppte(data):
    h,w=data.shape
    for i in range(h):
        for j in range(w):
            if data[i,j]<0:
                if j==0 and data[i,j+1]>0:
                    data[i,j]=data[i,j+1]
                elif j==w-1 and data[i,j-1]>0:
                    data[i,j]=data[i,j-1]
                elif j==0 or j==w-1:
                    data[i,j]=0
                else:
                    pre=data[i,j-1]
                    aft=data[i,j+1]
                    if pre>0 :
                        if aft>0:
                            data[i,j]=(pre+aft)/2
                        else:
                            data[i,j]=pre
                    elif aft>0:
                        data[i,j]=aft
                    else:
                        data[i,j]=0

    return data


def modify_data(x,sf):
    feature=["AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","PM2.5","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]
    dic={}
    for i in range(18):
        dic[feature[i]]=x[:,9*i:9*(i+1)]
    #feature.append("deg_sin")
    #feature.append("deg_cos")
    feature.append("PM2.5^2")
    feature.append("PM10^2")
    feature.append("PM2.5^3")
    feature.append("WIND_DIREC_sin")
    feature.append("WIND_DIREC_cos")
    dic["WIND_DIREC_sin"]=np.sin(x[:,135:144]/180*np.pi)*(x[:,144:153])
    dic["WIND_DIREC_cos"]=np.cos(x[:,135:144]/180*np.pi)*(x[:,144:153])
    dic["PM2.5^2"]=x[:,81:90]**2
    dic["PM10^2"]=x[:,72:81]**2
    dic["PM2.5^3"]=x[:,81:90]**3
    
    #sf=feature
    for i in range(len(sf)):
        if i==0:
            out=dic[sf[i]]
        else:
            out=np.concatenate((out,dic[sf[i]]), axis=1)    
    
    return out


def preprocess(data):
    h,w=data.shape
    for i in range(h):
        for j in range(w):

            try:
                float(data[i,j])
                if np.isnan(data[i,j]):
                    data[i,j]=-1

            except:
                if data[i,j]=="NR" :#or data[i,j]==None:
                    data[i,j]=0
                elif data[i,j]==None:
                    data[i,j]=-1
                    
                else:
                    for ele in ["#","x","*"]:
                        data[i,j]=data[i,j].replace(ele,"")
    data=data.astype("float")
    return data
def pptr(data):
    h,w=data.shape
    for i in range(h):
        for j in range(w):
            if data[i,j]<0:
                if j==0:
                    pre=data[i-18,-1]
                    aft=data[i,j+1]
                elif j==23:
                    pre=data[i,j-1]
                    if i+18<w:
                        aft=data[i+18,0]
                    else:
                        aft=-1
                else:
                    pre=data[i,j-1]
                    aft=data[i,j+1]
                if pre<0:
                    if aft<0:
                        data[i,j]=0
                    else:
                        data[i,j]=aft
                else:
                    if aft<0:
                        data[i,j]=pre
                    else:
                        data[i,j]=(pre+aft)/2     
    return data

def BN(x,mean,std):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]

    return x
raw_data1 = pd.read_csv(r_path1)
raw_data2 = pd.read_csv(r_path2)
data1=raw_data1.iloc[:,2:]
data2=raw_data2.iloc[:,2:]
data1=np.array(data1)
data2=np.array(data2)
data=np.concatenate([data1,data2] , axis = 0)
data=pptr(preprocess(data))
data_r=np.empty([731*24,18])
for i in range(731):
    data_r[24*i:24*(i+1),:]=data[18*i:18*(i+1),:].T
x=np.empty([731*24-9,18*9])
y=np.empty([731*24-9,])
for i in range(731*24-9):
    x[i]=data_r[i:i+9,:].T.reshape(-1)        
    y[i]=data_r[i+9][9]

sf=["CO","O3","PM2.5","PM2.5^2","PM10","PM10^2","RAINFALL","WIND_DIREC_sin","WIND_DIREC_cos"]
out=[]
stop=[]
for i in range(1,len(y)):
    if y[i]>80:       
        out.append(i)
    elif i-1 in out and y[i]<80:
        stop.append(i)
for i in range(len(stop)):
    for j in range(9):
        out.append(stop[i]+j)
y=np.delete(y,out,axis=0)
x=np.delete(x,out,axis=0)
x=modify_data(x,sf)
mean = np.mean(x, axis = 0) 
std = np.std(x, axis = 0)
x=BN(x,mean,std)

test_raw_data = pd.read_csv(r_path3, sep=',', engine='python', header=None)
test_data=np.array(test_raw_data.iloc[1:,2:])
test_data=ppte(preprocess(test_data))
x_ = np.empty(shape = (500, 18 * 9),dtype = float)
for i in range(500):
    x_[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(-1) 
x_=BN(modify_data(x_,sf),mean,std)
x_=torch.tensor(x_).float().to(device)
x=torch.tensor(x).float().to(device)
y=torch.tensor(y.reshape([-1,1])).float().to(device)
def train(m,s,LR,act,EPOCH):
    torch.manual_seed(s)
    
    net=nn.Sequential(
        nn.Linear(x.shape[1], m),
        act,
        nn.Linear(m,m),
        act,
        nn.Linear(m,m),
        act,
        nn.Linear(m, 1),
        ).to(device)
    loss_func = nn.MSELoss()  # this is for regression mean squared loss
    
    optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    
    for t in range(EPOCH):  
        loss = loss_func(net(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return net
    
net=train(31,494,0.02,nn.Softplus(),574).to(device)  



ans=net(x_).detach().cpu().numpy().reshape(-1)
title=[]
for i in range(500):
    title.append("id_"+str(i))
pd.DataFrame({'id':title,'value':ans}).to_csv(w_path,index=False)


