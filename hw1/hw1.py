import numpy as np
import sys
import pandas as pd
r_path=sys.argv[1]
w_path=sys.argv[2]
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
    feature.append("PM2.5^2")
    feature.append("PM10^2")
    feature.append("WIND_DIREC_sin")
    feature.append("WIND_DIREC_cos")
    dic["WIND_DIREC_sin"]=np.sin(x[:,135:144]/180*np.pi)*(x[:,144:153])
    dic["WIND_DIREC_cos"]=np.cos(x[:,135:144]/180*np.pi)*(x[:,144:153])
    dic["PM2.5^2"]=x[:,81:90]**2
    dic["PM10^2"]=x[:,72:81]**2
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

def BN(x,mean,std):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1).astype(float)
    return x
mean,std=np.load("para.npy")
w=np.load("hw1.npy")
test_raw_data = pd.read_csv(r_path, sep=',', engine='python', header=None)
test_data=np.array(test_raw_data.iloc[1:,2:])
test_data=ppte(preprocess(test_data))
x__ = np.empty(shape = (500, 18 * 9),dtype = float)
for i in range(500):
    x__[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(-1) 
sf=["CO","O3","PM2.5","PM2.5^2","PM10","PM10^2","RAINFALL","WIND_DIREC_sin","WIND_DIREC_cos"]
x__=BN(modify_data(x__,sf),mean,std)
y__=x__.dot(w)
title=[]
for i in range(500):
	title.append("id_"+str(i))
pd.DataFrame({'id':title,'value':y__}).to_csv(w_path,index=False)



