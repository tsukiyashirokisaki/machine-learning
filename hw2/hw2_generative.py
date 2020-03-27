import numpy as np
import sys
import pandas as pd
test=sys.argv[5]
out=sys.argv[6]
def BN(x,mean,std):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    return x
def preprocess(X):
    for ele in X.columns:
        if " " in ele:
            X.rename({ele:ele.replace(" ","")},axis=1,inplace=True)
    X["Married"]=X["Married-AF-spouse"]+X["Married-civ-spouse"]+X["Married-spouse-absent"]
    X["UnMarried"]=X["Separated"]+X["Widowed"]+X["Divorced"]
    X["SL-gov"]=X["Local-gov"]+X["State-gov"]
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
    s_col = [X.columns[i] for i in [0,2,3,4,5,6,8,9,10,11,13,14]+list(range(15,31))+list(range(32,36))+list(range(38,59))+list(range(106,114))]
    X=X[s_col]
    return X

def sigmoid(z):
    return np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6)


mean,std,w,b=np.load("prob.npy",allow_pickle=1)
X_ = pd.read_csv(test, sep=',', engine='python', header=0)
X_=preprocess(X_)
X_=BN(X_.values,mean,std)
Y_ = np.round(sigmoid(np.dot(w, X_.T) + b)).astype("int")
w_path=out
pd.DataFrame({'id':np.arange(1,len(X_)+1),'label':Y_}).to_csv(w_path,index=False)
