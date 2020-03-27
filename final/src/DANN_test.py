import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import datetime
import os, sys
import cv2
import torch.nn.functional as F
import pandas as pd
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

MODEL_NAME = 'NN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class train_hw3(Dataset):
    def __init__(self,data_dir,label):
        self.data_dir = data_dir
        self.label = label
    def __getitem__(self,index):
        return self.data_dir[index],self.label[index]#one_hot_vector
    def __len__(self):
        return self.label.shape[0]
class test_hw3(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
    def __getitem__(self,index):
        return self.data_dir[index],index
    def __len__(self):
        return self.data_dir.shape[0]
class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    def __init__(self, in_channel=1, hidden_dims=512):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(256, hidden_dims, 3, padding=1),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2))
            #nn.AdaptiveAvgPool2d((1,1)),
        )
        
    def forward(self, x):
        h = self.conv(x).squeeze() # (N, hidden_dims)
        return h
class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self, input_size=512, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),
            
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),

            #nn.Linear(512, 512),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),

            #nn.Linear(512, 512),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),

            #nn.Linear(512, 512),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),

            #nn.Linear(512, 512),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.5),

            nn.Linear(512, num_classes),
        )
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(1)
        #self.layer.apply(init_weights)
    
    def forward(self, h):
        
        h = h.view(h.size(0),-1)
        
        c = self.layer(h)
        return c
class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=512, num_classes=1):#imput 512
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),#512
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            #nn.Linear(512, 512),
            #nn.LeakyReLU(0.2),
            #nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
            #nn.ReLU()
            nn.Sigmoid(),
        )
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(1)
        #self.layer.apply(init_weights)
    
    def forward(self, h):
        
        h = h.view(h.size(0),-1)
        
        y = self.layer(h)
        return y

def train(EPOCH,batch_size,train_pic_loader,train_scratch_loader,seed,torch_seed):
    F = FeatureExtractor().to(DEVICE)
    C = Classifier().to(DEVICE)
    D = Discriminator().to(DEVICE)
    #F.load_state_dict(torch.load('C:/Users/stanley/Desktop/final/DANNstrong/0.85/model_C_%s_gray_%d.pkl'%(3310,161)))
    #D.load_state_dict(torch.load('C:/Users/stanley/Desktop/final/DANNstrong/0.85/model_D_%s_gray_%d.pkl'%(3310,161)))
    BCE_loss_record = []
    xe_loss_record = []
    total_loss_record=[]
    accuracyy = []
    batch_size = batch_size
    sigmoid= nn.Sigmoid()
    bce = nn.BCELoss()
    xe = nn.CrossEntropyLoss()

    F_opt = torch.optim.Adam(F.parameters())
    C_opt = torch.optim.Adam(C.parameters())
    D_opt = torch.optim.Adam(D.parameters())

    max_epoch = EPOCH
    step = 0
    n_critic = 1 # for training more k steps about Discriminator
    n_batches = 100000//batch_size
    # lamda = 0.01

    
    D_src = torch.ones(batch_size, 1).to(DEVICE) # Discriminator Label to real
    D_tgt = torch.zeros(batch_size, 1).to(DEVICE) # Discriminator Label to fake
    D_labels = torch.cat([D_src, D_tgt], dim=0)

    def get_lambda(epoch, max_epoch):
        p = epoch / max_epoch
        return 2. / (1+np.exp(-10.*p)) - 1.

    mnist_set = iter(train_scratch_loader)
    def sample_mnist(step, n_batches):
        global mnist_set
        if step % n_batches == 0:
            mnist_set = iter(train_scratch_loader)
        return mnist_set.next()
    F.train()
    C.train()
    D.train()
    good_model = 1
    best_correct = 163
    
    

    for epoch in range(1, max_epoch+1):
        F.train()
        C.train()
        D.train()
        print(epoch," Start!")
        bce_loss = 0
        xe_loss =0
        total_loss = 0
        for idx, (src_images, labels) in enumerate(train_pic_loader):
            tgt_images, _ = sample_mnist(step, n_batches)
            # Training Discriminator
            src, labels, tgt = src_images.to(DEVICE), labels.to(DEVICE), tgt_images.to(DEVICE)

            x = torch.cat([src, tgt], dim=0)
            h = F(x)
            y = D(h.detach())

            Ld = bce(y, D_labels)#*0.0001
            D.zero_grad()
            Ld.backward()
            D_opt.step()


            c = C(h[:batch_size])
            y = D(h)
            Lc = xe(c, labels)#*0.0001
            Ld = bce(y, D_labels)#*0.0001
            xe_loss+=Lc
            bce_loss+=Ld
            lamda = 0.1*get_lambda(epoch, max_epoch)#origin0.1
            Ltot = (Lc -lamda*Ld)#*0.01
            total_loss+=Ltot
        
        
            F.zero_grad()
            C.zero_grad()
            D.zero_grad()
        
            Ltot.backward()
        
            C_opt.step()
            F_opt.step()
            step+=1
        
        print("BCE_loss:",bce_loss.item())
        print("XE_loss:",xe_loss.item())
        print("total_loss:",total_loss.item())
        print('seed:',seed)
        print('torch_seed',torch_seed)
        BCE_loss_record.append(bce_loss.item())
        xe_loss_record.append(xe_loss.item())
        total_loss_record.append(total_loss.item())
        torch.save(F.state_dict(),'model/model_F_%s_gray_%d.pkl'%(epoch,86))
        torch.save(C.state_dict(),'model/model_C_%s_gray_%d.pkl'%(epoch,86))
        torch.save(D.state_dict(),'model/model_D_%s_gray_%d.pkl'%(epoch,86))


def predict(model_num,test_loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    F = FeatureExtractor()
    C = Classifier()
    F.load_state_dict(torch.load('model/model_F_%s_gray_%d.pkl'%(model_num,86),map_location=device))
    C.load_state_dict(torch.load('model/model_C_%s_gray_%d.pkl'%(model_num,86),map_location=device))
    F=F.to(device)
    C=C.to(device)
    F.eval()
    C.eval()
    prediction = []
    index_stack =[]
    with torch.no_grad():
        for idx, (tgt, index) in enumerate(test_loader):
            tgt  = tgt.to(DEVICE)
            c = C(F(tgt))
            _, preds = torch.max(c, 1)
            for i in preds.tolist():
                prediction.append(i)
            #print(index)
            for i in index.tolist():
                index_stack.append(i)
    prediction = np.array(prediction)#.reshape(1,-1).squeeze()
    #print(prediction)
    index_stack = np.array(index_stack)#.reshape(1,-1).squeeze()
    combine = np.column_stack((index_stack,prediction))
    final= pd.DataFrame(data=combine,    # values
        #index=data[1:,0],    # 1st column as index
        columns=['id','label'])  # 1st row as the column names
    return final
if __name__ == '__main__':
    #seedd = 123
    #np.random.seed(0)
    st0 = np.random.get_state()[1][0]
    st1 = 0
    #torch.initial_seed()

    print('Today seed:',st0)
    print('torch.initial_seed',st1)
    np.random.seed(st0)
    torch.manual_seed(st1)
    torch.cuda.manual_seed_all(st1)
    #trainX = np.load(r'D:\ML\final\original_data\trainX_gray.npy')
    trainX_color = np.load('data/trainX.npy')
    print(trainX_color.shape)

    trainX_gray = np.zeros((5000,32,32,1))
    print(trainX_gray.shape)
    for i in range(0,len(trainX_gray)):
        trainX_gray[i]=np.expand_dims(cv2.cvtColor(trainX_color[i],cv2.COLOR_BGR2GRAY),2)
    trainX_gray = trainX_gray.astype(np.uint8)
    print(trainX_gray.shape)
    print(trainX_gray[0])
    #np.save('D:/ML/final/code/DANN/pptImage/trainX_gray.npy',trainX_gray)




    #_canny=np.load(r'D:\ML\final\original_data\trainX_gray.npy')
    #trainX_canny = np.load('D:/ML/final/code/DANN/pptImage/trainX_gray.npy')
    trainX = np.copy(trainX_gray)
    for i in range(0,len(trainX)):
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i],170 , 300),2)
    print(trainX.shape)

    trainY = np.load('data/trainY.npy')
    testX = np.load('data/testX.npy')
    print('gray shape',trainX.shape)


    #print(trainX[0].shape)
    #canny = cv2.Canny(trainX[0], 60, 150)
    #print(canny.shape)
    for i in range(2000):
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i],170 , 300),2)
    for i in range(2000,2500):#cat
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i], 170, 300),2)
    for i in range(2500,3500):
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i], 170, 300),2)
    for i in range(3500,4000):#dog
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i], 151, 200),2)
    for i in range(4000,4500):#dophin
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i], 170, 300),2)
    for i in range(4500,5000):#spider
        trainX[i]=np.expand_dims(cv2.Canny(trainX[i], 151, 200),2)
    print('canny shape',trainX.shape)
    print(trainX[2000].shape)

    #testX = np.stack((testX,)*3, axis=3).reshape(100000,28,28,3)
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255.
    trainX  = (trainX-0.5)/np.std(trainX)
    testX = np.transpose(testX, (0, 3, 1, 2)) / 255.
    testX  = (testX-0.5)/np.std(testX)
    trainX = torch.Tensor(trainX)
    testX = torch.Tensor(testX)
    
    train_dataset = train_hw3(trainX,trainY)
    pad = nn.ZeroPad2d(2)
    testX = pad(testX)
    test_dataset = test_hw3(testX)
    train_pic_loader = DataLoader(train_dataset, batch_size=100, shuffle=True,drop_last=True)
    train_scratch_loader = DataLoader(test_dataset, batch_size=100, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    #train(10000,100,train_pic_loader,train_scratch_loader,st0,st1)
    print("><")
    final = predict(288,test_loader)
    final.to_csv('submission/DANN_edge_288.csv',index=False)
    
    print("done")
