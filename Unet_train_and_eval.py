# DEFINE AND TRAIN THE MODEL, THE MODEL CURRENTLY NEEDS THE PICTURES TO BE OF SIZE 256*256 
# THE IMAGES ARE MILDLY CROPPED TO REDUCE THE COMPUTIONAL TIME
# THE IMAGES SHOULD BE LOCATED INSIDE A FOLDER INSIDE THE FOLDERS x_train, x_eval, y_trainand y_eval 
# (eg put the data of the liver in ./x_train/liver etc.)

btch_siz = 5 # !!!! BATCH SIZE CURRENTLY HAS TO BE A DIVISOR OF n_train !!!!
n_train = 1000 # !!!! THE NUMBER OF SAMPLES IN THE TRAINING FOLDER CURRENTLY HAS TO BE MANUALLY SPECIFIED !!!!
n_eval = 824 # !!!! THE NUMBER OF SAMPLES IN THE EVALUATION FOLDER CURRENTLY HAS TO BE MANUALLY SPECIFIED !!!!

import os
import torch
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2)

        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)

        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)

        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)

        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels = 64,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, image):

        x1 = self.down_conv_1(image)
        x = self.max_pool_2x2(x1)
        x2 = self.down_conv_2(x)
        x = self.max_pool_2x2(x2)
        x3 = self.down_conv_3(x)
        x = self.max_pool_2x2(x3)
        x4 = self.down_conv_4(x)
        x = self.max_pool_2x2(x4)
        x = self.down_conv_5(x)


        x = self.up_trans_1(x)
        x = self.up_conv_1(torch.cat([x, x4], 1))
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x3], 1))
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x2], 1))
        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))
        x = self.out(x)
        return x

# define the model
model = UNet()
if(torch.cuda.is_available()):
    model.to(device='cuda')
dir_checkpoint = 'checkpoints/'
if(not os.path.exists(dir_checkpoint)):
    os.mkdir(dir_checkpoint)

# choose what loss function to use
loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = torch.nn.KLDivLoss(reduction='sum')
#loss_fn = torch.nn..BCEWithLogitsLoss(reduction='sum')

# choose optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

# load data
datasetx = torchvision.datasets.ImageFolder(root='x_train', transform=transforms.ToTensor())
loaded_datax = DataLoader(datasetx, batch_size=btch_siz, shuffle=False,)
datasety = torchvision.datasets.ImageFolder(root='y_train', transform=transforms.ToTensor())
y_train = torch.empty((n_train, 1, 192, 192))
for idx, plane in enumerate(datasety):
    y_train[idx,0,:,:] = plane[0][0,32:224,32:224]  

# train model
epochs = 50
model.train()
for epoch in range(epochs):
    idx = 0
    epoch_loss = 0
    for batch in loaded_datax:
        print('hej')
        x_train = batch[0][:,0,32:224,32:224].unsqueeze(1)
        y_tr = y_train[(idx*btch_siz):((idx+1)*btch_siz)]
        if(torch.cuda.is_available()):
            x_train = x_train.to(device='cuda', dtype=torch.float32)
            y_tr = y_tr.to(device='cuda', dtype=torch.float32)
        y_pred = model(x_train)
        loss = loss_fn(y_pred,y_tr)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        idx = idx + 1
   
    # save the state of the model after each epoch and print the cambined loss function over the epoch
    torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
    print(f'Created checkpoint {epoch + 1}')
    print(f'The loss of the epoch was: {epoch_loss}')

model_eval = UNet()
model_eval.load_state_dict(torch.load('CP_epoch50.pth'))
datasetx = torchvision.datasets.ImageFolder(root='x_eval', transform=transforms.ToTensor())
datasety = torchvision.datasets.ImageFolder(root='y_eval', transform=transforms.ToTensor())
dice = 0
model_eval.eval()

for i in range(n_eval):
    y = datasety[i][0][0,32:224,32:224].numpy()
    x = datasetx[i][0][0,32:224,32:224].unsqueeze(0).unsqueeze(0)
    yp = model_eval(x).squeeze(0).squeeze(0).detach().numpy()
    dice += sum(sum( (y==1) & (yp>0.5) ))*2/( sum(sum(y)) +sum(sum(yp>0.5)) )
print(dice/n_eval)

# Plot the ten first 
# for k in range(10):
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     y_pred = model_eval(datasetx[k][0][0,32:224,32:224].unsqueeze(0).unsqueeze(0) )
#     ax1.imshow(y_pred.squeeze(0).squeeze(0).detach().numpy()>0.5,cmap='gray')

#     ax2.imshow(y_train[k].squeeze(0).squeeze(0).detach().numpy(),cmap='gray')
#     fig.suptitle('Model vs truth')
#     plt.show()




