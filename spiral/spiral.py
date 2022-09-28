# spiral.py
# ZZEN9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, 1)

    def forward(self, input):      
        x = input[:,0]
        y = input[:,1]
        
        r = torch.sqrt(x**2 + y**2)
        r = r.reshape(-1,1) 
        
        a = torch.atan2(y,x).reshape(-1,1)
        a = a .reshape(-1,1)
        input = torch.cat((r,a) ,1)
        
        self.hid1 = torch.tanh(self.fc1(input))
        out = torch.sigmoid(self.fc2(self.hid1))

        return out

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid)
        self.fc3 = nn.Linear(num_hid, 1)
   
    def forward(self, input):
        self.hid1 = torch.tanh(self.fc1(input))
        self.hid2 = torch.tanh(self.fc2(self.hid1))
        out = torch.sigmoid(self.fc3(self.hid2))
        return out

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): 
        net.eval()        
        output = net(grid)
        if layer == 1:
            pred = (net.hid1[:, node]>=0).float()
        elif layer == 2:
            pred = (net.hid2[:, node]>=0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
