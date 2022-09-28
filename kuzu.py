# kuzu.py
# ZZEN9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.flat = nn.Flatten()
        self.lin = nn.Linear(28*28, 10)
        
    def forward(self, x):
        out = self.flat(x)
        out = F.log_softmax(self.lin(out), dim=1)
        return out 

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        out = self.flat(x)
        out = F.tanh(self.fc1(out))
        out = F.log_softmax(self.fc2(out), dim=1)
        return out

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5)
        self.conv2 = nn.Conv2d(24, 48, 5)
        
        self.fc1   = nn.Linear(48*4*4, 120)
        self.fc2   = nn.Linear(120, 10)


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        
        out = F.relu(self.conv2(out))   
        out = F.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        
        out = F.log_softmax(self.fc2(out), dim=1)
        return out
