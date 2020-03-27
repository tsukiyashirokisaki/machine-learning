#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# In[2]:


def BN(x,mean,std):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    return x


# In[3]:


def train(lr,ep,gs):
    
    beta1=0.9 ; beta2=0.999 ; epsilon=10**(-8) 
    if gs:
        fold=5
        kf = KFold(n_splits=fold,random_state=143, shuffle=True)
        ret=np.zeros([2])
        for k, (train, test) in enumerate(kf.split(X, Y)):
            x,x_=X[train],X[test]
            y,y_=Y[train],Y[test]
            x_t=x.T
            w=np.zeros(x.shape[1]) ; m=np.zeros(x.shape[1]) ; v=np.zeros(x.shape[1])
            px=[]
            for i in range(ep):
                y_p=sigmoid(x.dot(w))
                L=np.sum(-(y*np.log(y_p)+(1-y)*np.log(1-y_p)))/y.shape[0]
                g=2*x_t.dot(y_p-y)
                m=beta1*m+(1-beta1)*g
                v=beta2*v+(1-beta2)*g**2
                m_hat=m/(1-beta1**(i+1))
                v_hat=v/(1-beta2**(i+1))
                w-=lr*m_hat/(np.sqrt(v_hat)+epsilon)
                y_p_=sigmoid(x_.dot(w))
                L_=np.sum(-(y_*np.log(y_p_)+(1-y_)*np.log(1-y_p_)))/y_.shape[0]
                accuracy=acc(y_p,y)
                accuracy_=acc(y_p_,y_)
                if i%100==0:
                    print(i,accuracy,accuracy_)
                
                px.append(L_)
            print(i,accuracy,accuracy_)
            i_m,l_m=min(enumerate(px),key=operator.itemgetter(1))
            ret[0]+=(i_m+1)/fold
            ret[1]+=l_m/fold
        return int(ret[0]),ret[1]
    else:
        px=[];py=[]
        x=X;y=Y
        x_t=x.T
        w=np.zeros(x.shape[1]) ; m=np.zeros(x.shape[1]) ; v=np.zeros(x.shape[1])
        for i in range(ep):  
            y_p=sigmoid(x.dot(w))
            L=np.sum(-(y*np.log(y_p)+(1-y)*np.log(1-y_p)))/y.shape[0]
            g=2*x_t.dot(y_p-y)
            m=beta1*m+(1-beta1)*g
            v=beta2*v+(1-beta2)*g**2
            m_hat=m/(1-beta1**(i+1))
            v_hat=v/(1-beta2**(i+1))
            w-=lr*m_hat/(np.sqrt(v_hat)+epsilon)
            px.append(i+1)
            py.append(L)
        return w,px,py


# In[ ]:





# In[4]:


def preprocess(X):
    for ele in X.columns:
        if " " in ele:
            X.rename({ele:ele.replace(" ","")},axis=1,inplace=True)
    X["Married"]=X["Married-AF-spouse"]+X["Married-civ-spouse"]+X["Married-spouse-absent"]
    X["UnMarried"]=X["Separated"]+X["Widowed"]+X["Divorced"]
    X["SL-gov"]=X["Local-gov"]+X["State-gov"]
    X["Self-emp"]=X["Self-emp-inc"]+X["Self-emp-not-inc"]
    X["Unemployeed"]=X["Never-worked"]+X["Without-pay"]
    X["South_America"]=X[X.columns[64]]
    for ele in [67,68,69,70,71,76,77,79,86,89,90,91,92,96,101]:
        X["South_America"]+=X[X.columns[ele]]    
    X["North_America"]=X[X.columns[65]]+X[X.columns[102]]
    X["Asia"]=X[X.columns[66]]
    for ele in [80,82,83,87,88,93,99,100,103,104]:
        X["Asia"]+=X[X.columns[ele]]    
    X["Europe"]=X[X.columns[73]]
    for ele in [74,75,78,81,72,84,85,94,95,97]:
        X["Europe"]+=X[X.columns[ele]]   
    X["Other_Country"]=X[X.columns[98]]+X[X.columns[105]]
    s_col = [X.columns[i] for i in [0,2,3,4,5,6,8,9,10,11,13,14]+list(range(15,31))+list(range(32,36))+list(range(38,59))
             +list(range(106,109))+list(range(114,115))]
    X=X[s_col]
    return X


# In[5]:


X = pd.read_csv("X_train", sep=',', engine='python', header=0)
Y=pd.read_csv("Y_train",sep=',', engine='python', header=None)


# In[6]:


X=preprocess(X)
col=X.columns
X = np.array(X).astype("float")
Y = np.array(Y).reshape(-1).astype("long")
mean = np.mean(X, axis = 0) 
std = np.std(X, axis = 0)
X=BN(X,mean,std)


# In[7]:


def acc(y1,y2):
    total = float(y1.shape[0])
    correct = float((np.round(y1) == y2).sum())
    return 100 * correct / total


# In[9]:


#x, x_, y, y_ = train_test_split(X, Y, test_size=0.2, random_state=42)
x=X
y=Y


# In[10]:


def pro(logodd):
    return np.exp(logodd)/(1+np.exp(logodd))
lr=0.1
M=250
g=[]
clf_lis=[]
p=np.sum(y)/y.shape[0]*np.ones(y.shape[0])
F=np.log(p/(1-p))
F0=F[0]
for m in range(1,M):
    r=y-p
    clf = DecisionTreeRegressor(max_depth = 4, random_state = 42)
    clf.fit(x,r)
    clf_lis.append(clf)
    j=clf.apply(x)
    d_i_1=p*(1-p)
    
    d_i={}
    n_i={}
    g_i={}
    for ele in set(j):
        d_i[ele]=0
        n_i[ele]=0
    for i in range(x.shape[0]):
        d_i[j[i]]+=d_i_1[i]
        n_i[j[i]]+=r[i]
    for ele in set(j):
        g_i[ele]=n_i[ele]/d_i[ele]
    g.append(g_i)
    for i in range(x.shape[0]):
        F[i]+=lr*g[m-1][j[i]]
    p=pro(F)


# In[77]:


#clf = DecisionTreeClassifier(max_depth = 1, random_state = 42)
#clf.fit(x,y)
#clf.predict(x)
#clf.apply(X)


# In[12]:


def evaluate(F0,clf_lis,g,lr,x_):
    F=F0*np.ones(x_.shape[0])
    for i in range(len(g)):
        ind=clf_lis[i].apply(x_)
        for j in range(x_.shape[0]):
            F[j]+=g[i][ind[j]]*lr
    p=pro(F)
    return p

print(acc(evaluate(F0,clf_lis,g,lr,x),y))


# In[ ]:




X_ = pd.read_csv("X_test", sep=',', engine='python', header=0)
X_=preprocess(X_)
X_ = BN(np.array(X_).astype("float"),mean,std)
Y_=evaluate(F0,clf_lis,g,lr,X_)
Y_=np.round(Y_)
w_path="GBDT.csv"
pd.DataFrame({'id':np.arange(1,len(Y_)+1),'label':np.round(Y_).astype("int")}).to_csv(w_path,index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




