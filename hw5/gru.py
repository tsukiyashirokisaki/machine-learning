#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np 
import pandas as pd 
import os
import re
from multiprocessing import Pool
import os
import time
import math
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,load_model
from keras import initializers, regularizers, constraints, optimizers, layers, optimizers, callbacks
import pickle


import sys

callbacks.TensorBoard(histogram_freq=0)
change_seed=np.random.randint(0,1e9)

path1=sys.argv[1]
path2=sys.argv[2]
path3=sys.argv[3]
train_df=pd.read_csv(path1)
train_df["target"]=pd.read_csv(path2)["label"]
test_df=pd.read_csv(path3)
train_df_len = train_df.shape[0]
def max_(lis):
    max_ele=0
    max_ind=0
    for i in range(len(lis)):
        if lis[i]>max_ele:
            max_ele=lis[i]
            max_ind=i
            
    return max_ind+1,max_ele
def plot(history):
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
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
def preprocess(seq,wti):
    seq_n=[]
    for i in range(len(seq)):
        app=[]
        for j in range(len(seq[i])):
            if  seq[i][j] in wti:
                app.append(wti[seq[i][j]])
        seq_n.append(app)
    return seq_n

            


# In[21]:


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



def train_test_split(df,size):
    np.random.seed(0)
    n=len(df)
    ind=int(n*size)
    arr=np.arange(n)
    return df.iloc[arr[ind:]],df.iloc[arr[:ind]]

train_df_len = train_df.shape[0]

ret=[]
seq=[]
lab=[]

if __name__ == '__main__':
	p = Pool(processes=4) 	
	for ele in np.array(train_df.iloc[:,1:]):
		#print(ele)
		r = p.apply_async(tokenize,(ele,))
		ret.append(r)
	p.close()
	p.join()
for r in ret:
    s,l=r.get()
    seq.append(s)
    lab.append(l)
train_df["comment"]=seq
train_df["target"]=lab
for i in range(len(test_df)):
    test_df["comment"][i]=tokenize(test_df["comment"][i],label=0)

train_df, val_df = train_test_split(train_df, size=0.33)
embed_size = 300 # how big is each word vector
max_features = 2500 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use
## fill up the missing values
train_X = train_df["comment"].values
val_X = val_df["comment"].values
test_X = test_df["comment"].values



thres=10
wti,itw=createdict(train_X,thres=thres)
train_X=preprocess(train_X,wti)
val_X=preprocess(val_X,wti)
test_X=preprocess(test_X,wti)


# In[25]:





train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)


# In[26]:





## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

def save_write(model,acc):
    model.save('model/%.3f.h5'%(acc))
    ans = model.predict([test_X], batch_size=2048, verbose=0)
    ans=(ans>0.5).reshape([-1]).astype("int")
    df = pd.DataFrame({'id': np.arange(0,len(ans)), 'label': ans})
    df.to_csv('kaggle/%.3f.csv'%(acc),index=False)
    file=open("parameter.txt","a")
    file.write("acc= %.3f "%(acc)+str(p)+"\n")
    file.close()

def fix_seed(s):
    np.random.seed(s)
    tf.set_random_seed(s)

def build_model(p):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)            
    fix_seed(p[-1])
    h1=p[0];h2=p[1];drop=p[2];lr=p[3]
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(CuDNNGRU(h1, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(h2, activation="relu")(x)
    x = Dropout(drop)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    adam = optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:





# In[29]:


np.random.seed(change_seed)

batch_size=512
grid=[4,8,16,32,64,128,256]
best_ep=0;best_acc=0.76
for i in range(10):
    s=np.random.randint(0,1e7)
    for i in range(1,len(grid)):
        p1=grid[i]
        for j in range(i):
            p2=grid[j]
            for drop in np.arange(0.1,0.6,0.1):
                lr=10**(-np.random.randint(2,6))
                p=[p1,p2,drop,lr,s]
                model=build_model(p)
                v=0
                hist=model.fit(train_X, train_y, batch_size=batch_size, epochs=30, 
                               validation_data=(val_X, val_y),verbose=v,shuffle=True)
                ep,acc=max_(hist.history["val_acc"])
                tf.keras.backend.clear_session()
                del model
                del hist

                if acc>best_acc:
                    best_acc=acc
                    best_ep=ep
                    print(acc,ep)
                    model=build_model(p)
                    hist=model.fit(train_X, train_y, batch_size=batch_size, 
                                   epochs=ep, validation_data=(val_X, val_y),verbose=v,shuffle=True)
                    
                    ep,acc=max_(hist.history["val_acc"])
                    print("save! acc=%.3f epoch= %d"%(acc,ep))
                    #save_write(model,acc)
                    tf.keras.backend.clear_session()
                    del model
                    del hist

