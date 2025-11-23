#Import the dependencies

from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Data splitting into train and test samples
x,y = make_blobs(n_samples = 1000,n_features = 2,centers = 4,random_state = 0)
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify = y,test_size = 0.2,random_state = 0)

#convert data into pytorch tensors for processing in gpu's
x_train,x_test,y_train,y_test = map(torch.tensor,(x_train,x_test,y_train,y_test))
x_train = x_train.float()
x_test = x_test.float()
y_train = y_train.long()
y_test = y_test.long()

#tailor the neural network model from scratch
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2,3),
            nn.Sigmoid(),
            nn.Linear(3,5),
            nn.Sigmoid(),
            nn.Linear(5,2),
            nn.Sigmoid(),
            nn.Linear(2,4)
        )
    def forward(self,x):
        return self.net(x)

fnn_model = Model()

#set or select loss function and optimizer algorithm
loss_fn = F.cross_entropy
opt = optim.SGD(fnn_model.parameters(),lr = 0.02)

#evaluation funtion
def accuracy(y_preds,y_train):
    preds = torch.argmax(y_preds,dim = 1)
    return (preds == y_train).float().mean().item()

#set the tensors to cuda along with model
# dev = torch.device("cuda:0")

# x_train = x_train.to(dev)
# x_test = x_test.to(dev)
# y_train = y_train.to(dev)
# y_test = y_test.to(dev)

# fnn_model = fnn_model.to(dev)

#training function
def fit(X,Y,Model,loss_fn,opt,epochs):
    loss_arr = []
    accuracy_arr = []
    for i in range(epochs):
        preds = Model(X)
        loss = loss_fn(preds,Y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_arr.append(loss.item())
        accuracy_arr.append(accuracy(preds,Y))
    plt.plot(loss_arr,'r-')
    plt.plot(accuracy_arr,'g-')
    plt.show()

#training
fit(x_train,y_train,fnn_model,loss_fn,opt,10000)