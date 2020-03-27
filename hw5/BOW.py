
from multiprocessing import Pool
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
import spacy  # For preprocessing
import pickle
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
#w2v_model.save("word2vec")
from torch.utils.data import DataLoader, Dataset
import sys
path1=sys.argv[1]
path2=sys.argv[2]
path3=sys.argv[3]
df=pd.read_csv(path1)
label=pd.read_csv(path2)
test=pd.read_csv(path3)
df["label"]=label["label"]


def train_test_split(X,Y,size):
    np.random.seed(0)
    n=len(X)
    ind=int(n*size)
    arr=np.arange(n)
    X=np.array(X)
    Y=np.array(Y)
    return X[arr[ind:]],X[arr[:ind]],Y[arr[ind:]],Y[arr[:ind]],

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


ret=[]
seq=[]
lab=[]

if __name__ == '__main__':
	p = Pool(processes=4) 
	
	for ele in np.array(df.iloc[:,1:]):
		#print(ele)
		r = p.apply_async(tokenize,(ele,))
		ret.append(r)

	p.close()
	p.join()

for r in ret:
    s,l=r.get()
    seq.append(s)
    lab.append(l)


def preprocess(seq,wti,itw):
    seq_n=np.zeros([len(seq),len(wti)])
    for i in range(len(seq)):
        for j in range(len(seq[i])):
            if  seq[i][j] in wti:
                seq_n[i,wti[seq[i][j]]-1]+=1
    return seq_n


import torch
import torch.nn.functional as F     
import torch.nn as nn

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



def pre(seq,thres=10):
    wti,itw=createdict(seq,thres=thres)
    seq_n=preprocess(seq,wti,itw)
    x_tr,x_te,y_tr,y_te = train_test_split(
    seq_n,lab, size=0.33)
    x_tr,y_tr=totensor(x_tr,y_tr)
    x_te,y_te=totensor(x_te,y_te)
    return x_tr,y_tr,x_te,y_te,wti,itw


def train(data,h1=512,h2=128,h3=64,epoch=200,seed=0):
    x_tr,y_tr,x_te,y_te=data[0],data[1],data[2],data[3]
    torch.manual_seed(seed)
    net = Net(inf=x_tr.shape[1], h1=h1,h2=h2,h3=h3,out=2).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  
    loss_func = torch.nn.CrossEntropyLoss()
    best_acc=0.
    best_epoch=0.
    for t in range(200):
        net.train()
        out = net(x_tr)     
        loss = loss_func(out, y_tr)     
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        net.eval()
        accuracy=acc(out,y_tr)
        #print("Train",loss.item(),accuracy.item())
        out = net(x_te)     
        accuracy=acc(out,y_te)
        loss = loss_func(out, y_te).item()   
        if accuracy>best_acc:
            best_acc=accuracy
            best_epoch=t
        #if t%10==0:
            #print("Test",loss,accuracy)
    return net,best_epoch,round(best_acc,4)
def save_write(net,wti,itw,acc_):
    seq2=[]
    for i in range(len(test)):
        seq2.append(tokenize(test.iloc[:,1][i],0))
    seq_n2=preprocess(seq2,wti,itw)
    x_=torch.tensor(seq_n2,dtype=torch.float).cuda()
    y_=net(x_)
    y_=torch.max(F.softmax(y_,dim=1), 1)[1].squeeze().cpu().detach().numpy()
    torch.save(net,'model/%.3f_BOW.pkl'%(acc_))
    df = pd.DataFrame({'id': np.arange(0,len(y_)), 'label': y_})
    df.to_csv('kaggle/%.3f_BOW.csv'%(acc_),index=False)

x_tr,y_tr,x_te,y_te,wti,itw=pre(seq,thres=10)
s_acc=.735
grid=[4,8,16,32,64,128,256,512,1024]
for l in range(10):
    seed=np.random.randint(0,1e9)
    for i in range(2,6):
        for j in range(1,i):
            for k in range(j):
                #x_tr,y_tr,x_te,y_te,wti,itw=pre(seq,thres=10)
                net,ep,acc_=train([x_tr,y_tr,x_te,y_te],h1=grid[i],h2=grid[j],h3=grid[k],seed=seed)
                
                if acc_>s_acc:
                    s_acc=acc_
                    net,ep,acc_=train([x_tr,y_tr,x_te,y_te],h1=grid[i],h2=grid[j],h3=grid[k],
                                      seed=seed,epoch=ep)
                    #save_write(net,wti,itw,s_acc)
                    print("\nselect!!\n",acc_,s_acc)
                print(acc_,s_acc)
                
                
        
        












