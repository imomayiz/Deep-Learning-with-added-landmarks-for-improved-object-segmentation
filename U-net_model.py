
# THE LOSS FUNCTION IS NOT YET WORKING
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
# THE LOSS FUNCTION IS NOT YET WORKING
# Define the network
# This is not the "real" U-net but a simplification 

def conv3x3(input_features, output_features):    
    return nn.Conv2d(
        input_features,
        output_features,
        kernel_size=3,
        padding=1)

def upconv2x2(input_features, output_features):
        return nn.ConvTranspose2d(
            input_features,
            output_features,
            kernel_size=2,
            stride=2)

def conv1x1(input_features, output_features):
    return nn.Conv2d(
        input_features,
        output_features,
        kernel_size=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        FoFL = 32 # Features of first layer
        self.conv1 = conv3x3(1,FoFL)
        self.conv2 = conv3x3(FoFL,FoFL)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = conv3x3(FoFL,FoFL*2)
        self.conv4 = conv3x3(FoFL*2,FoFL*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = conv3x3(FoFL*2,FoFL*4)
        self.conv6 = conv3x3(FoFL*4,FoFL*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = conv3x3(FoFL*4,FoFL*8)
        self.conv8 = conv3x3(FoFL*8,FoFL*8)
        self.upconv1 = upconv2x2(FoFL*8,FoFL*4)
        
        self.conv9 = conv3x3(FoFL*4,FoFL*4)
        self.conv10 = conv3x3(FoFL*4,FoFL*4)
        self.upconv2 = upconv2x2(FoFL*4,FoFL*2)
        
        self.conv11 = conv3x3(FoFL*2,FoFL*2)
        self.conv12 = conv3x3(FoFL*2,FoFL*2)
        self.upconv3 = upconv2x2(FoFL*2,FoFL)
        
        self.conv13 = conv3x3(FoFL,FoFL)
        self.conv14 = conv3x3(FoFL,FoFL)
        self.convlast = conv1x1(FoFL,1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.upconv1(x)
        
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.upconv2(x)
        
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.upconv3(x)
        
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = self.convlast(x)

        return x


# Load the data created by Slices.py
data = np.zeros((20, 256, 256))
for i in range(20):
    data[i,:,:] = np.loadtxt('./fat/data%d.csv' % (i+80), delimiter=',')
truth = np.zeros((20, 256, 256))
for i in range(20):
    truth[i,:,:] = np.loadtxt('./liver/data%d.csv' % (i+80), delimiter=',')

# Define the network, loss function and optimizer
# THE LOSS FUNCTION IS NOT YET WORKING
net = Net()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# attempt to train the network with the non-working loss function
x = torch.from_numpy(data)
x = x.unsqueeze(1)
x = x.unsqueeze(1)
y = torch.from_numpy(truth)

for t in range(20):
    optimizer.zero_grad()
    y_pred = net(x[t,:,:].float())
    loss = criterion(y_pred.squeeze(0).squeeze(0),y[t,:,:].float())
    print(t, loss.item())
    loss.backward()
    optimizer.step()

print('Finished Training')

