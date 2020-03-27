#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pickle
import multiprocessing
import numpy as np 
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F     
import torch.nn as nn
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,load_model
from keras import initializers, regularizers, constraints, optimizers, layers, optimizers, callbacks
import sys
r_path=sys.argv[1]
w_path=sys.argv[2]

callbacks.TensorBoard(histogram_freq=0)
test=pd.read_csv(r_path)


def createdict(seq,thres):
    wordtoind=dict()
    indtoword=dict()
    count=[0]
    c=1
    for i in range(len(seq)):
        for j in range(len(seq[i])):
            if not seq[i][j] in wordtoind:
                wordtoind[seq[i][j]]=c
                indtoword[c]=seq[i][j]
                c+=1
                count.append(1)
            else:
                count[wordtoind[seq[i][j]]]+=1
    sel_ind=[]
    for i in range(len(count)):
        if count[i]>=thres:
            sel_ind.append(i)
    wti=dict()
    itw=dict()
    for i,ele in enumerate(sel_ind):
        itw[i+1]=indtoword[ele]
        wti[indtoword[ele]]=i+1
    
    
    return wti,itw
            

def tokenize(text,label=1):
    if label:
        row,l=text
    else:
        row=text
    app=row.encode('ascii', 'ignore').decode('ascii').lower()
    for ele in ["@user","url",".",",","!","?","(",")"]:
        while ele in app:
            app=app.replace(ele,"")
    app=app.split(" ")
    for ele in ['@',""]:
        while ele in app:
            app.remove(ele)
    i=0
    while i<len(app):
        lis=["#","^"]
        for ele in lis:
            if ele in app[i]:
                app.remove(app[i])
                i-=1
                break
                    
        i+=1
    if label:
        return app,l
    else:
        return app




def preprocess_bow(seq,wti,itw):
    seq_n=np.zeros([len(seq),len(wti)])
    for i in range(len(seq)):
        for j in range(len(seq[i])):
            if  seq[i][j] in wti:
                seq_n[i,wti[seq[i][j]]-1]+=1
    return seq_n


class Net(torch.nn.Module):     
    def __init__(self, inf,h1,h2,h3,out):
        super(Net, self).__init__()     
        self.fc1 = nn.Sequential(nn.Linear(inf,h1),nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(h1,h2)  ,nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(h2,h3)  ,nn.Sigmoid())
        self.out = nn.Linear(h3,out)      

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)                
        return x
def totensor(x,y):
    return torch.tensor(x,dtype=torch.float).cuda(),torch.tensor(y,dtype=torch.long).cuda()
def acc(out,y_):
    y=torch.max(F.softmax(out,dim=1), 1)[1].squeeze()
    #print(y)
    #print(y_)
    #print(y==y_)
    corr=torch.sum(y==y_,axis=0)
    return float(corr)/float(y.shape[0])



def BOW(net,wti,itw):
    seq2=[]
    for i in range(len(test)):
        seq2.append(tokenize(test.iloc[:,1][i],0))
    seq_n2=preprocess_bow(seq2,wti,itw)
    x_=torch.tensor(seq_n2,dtype=torch.float).cuda()
    y_=net(x_)
    y_=torch.max(F.softmax(y_,dim=1), 1)[1].squeeze().cpu().detach().numpy()
    return y_


wti,itw=torch.load("BOW_param.pkl")
net=torch.load("BOW_best.pkl")
ans=BOW(net,wti,itw)



def preprocess_rnn(seq,wti):
    seq_n=[]
    for i in range(len(seq)):
        app=[]
        for j in range(len(seq[i])):
            if  seq[i][j] in wti:
                app.append(wti[seq[i][j]])
        seq_n.append(app)
    return seq_n

            
test_X=test["comment"].values
for i in range(len(test_X)):
    test_X[i]=tokenize(test_X[i],label=0)
wti,itw=torch.load("RNN_param.pkl")
test_X = preprocess_rnn(test["comment"].values,wti)
test_X = pad_sequences(test_X, maxlen=50)



def RNN(model):
    ans = model.predict([test_X], batch_size=128, verbose=0)
    ans=(ans>0.5).reshape([-1]).astype("int")
    return ans


models=['0.779_lstm.h5','0.778_lstm.h5','0.777.h5','0.777_lstm.h5']
for i in range(4):
    ans+=RNN(keras.models.load_model(models[i]))
ans=(ans>=3).astype("int")
df = pd.DataFrame({'id': np.arange(0,len(ans)), 'label': ans})
df.to_csv(w_path,index=False)
