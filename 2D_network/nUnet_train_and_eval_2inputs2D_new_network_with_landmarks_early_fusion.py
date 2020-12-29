import os
import torch
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
import random
import math
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

import vtk
from vtk.util.numpy_support import vtk_to_numpy

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # self.max_pool_2x2_3d = nn.MaxPool3d(kernel_size=(2,1,2), stride=(2,1,2))
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
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
            out_channels=2,
            kernel_size=1
        )        
        self.soft = nn.LogSoftmax(dim = 1)

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
        x = self.soft(x)
        return x

# define the model
model = UNet()
model.to(device='cuda')
dir_checkpoint = 'checkpoints_2d/'
if(not os.path.exists(dir_checkpoint)):
    os.mkdir(dir_checkpoint)

# choose what loss function to use
#loss_fn = torch.nn.NLLLoss()
loss_fn = torch.nn.NLLLoss(weight=torch.tensor([1., 1.2], device='cuda'))
#loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = torch.nn.KLDivLoss(reduction='sum')
#loss_fn = torch.nn..BCEWithLogitsLoss(reduction='sum')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

maps_list = [None]*50
for k in range(1,51):
    file = open(r"theirLandmarks/LMs_"+ str(k) + ".txt")
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    X1Z1 = file.readline()
    X2Z2 = file.readline()
    X1Z1 = X1Z1.split()
    X2Z2 = X2Z2.split()
    
    X1 = int(float(X1Z1[0]))
    Z1 = int(float(X1Z1[2]))
    X2 = int(float(X2Z2[0]))
    Z2 = int(float(X2Z2[2]))
    file.close()
    
    distance_map = np.zeros((256,256))
    distance_map1 = np.zeros((256,256))
    distance_map2 = np.zeros((256,256))

    for i in range(256):
         for j in range(256):
             distance_map1[i,j] = abs(i-X1) + abs(j-Z1)
             distance_map2[i,j] = abs(i-X2) +abs(j-Z2)
    distance_map1 = distance_map1 + 1
    distance_map1 = 1/distance_map1

    distance_map2 = distance_map2 + 1
    distance_map2 = 1/distance_map2

    distance_map = distance_map1 + distance_map2
    distance_map = distance_map/distance_map[X1,Z1]
    maps_list[k-1] = distance_map
    maps_list = np.asarray(maps_list)

#Corresponding image number to each distance map in maps_list
nlandmark = np.array([17,18,22,26,51,53,56,61,62,75,77,86,117,124,158,159,167,179,204,205,235,241,242,253,268,280,281,288,291,297,304,316,318,321,327,346,347,348,354,357,358,379,395,403,406,424,429,433,473,487])

def load_vtk(file_number):
    """Loads .vtk file into tensor given file number"""
    file_number = str(file_number).zfill(3)
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName("../../Project course/500%s_fat_content.vtk" %(file_number))
    reader.Update()
    data_fat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    x_range, y_range, z_range = reader.GetOutput().GetDimensions()
    data_fat = data_fat.reshape(x_range,y_range,z_range)

    reader.SetFileName("../../Project course/500%s_wat_content.vtk" %(file_number))
    reader.Update()
    data_wat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_wat = data_wat.reshape(x_range,y_range,z_range)

    reader.SetFileName("../../Project course/binary_liver500%s.vtk" %(file_number))
    reader.Update()
    data_liv = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_liv = data_liv.reshape(x_range,y_range,z_range)
    
    data = np.zeros(3*x_range*y_range*z_range).reshape(3,x_range,y_range,z_range)
    data[0] = data_fat
    data[1] = data_wat
    data[2] = data_liv


    grid = torch.from_numpy(data)
    return grid.to(dtype=torch.float32)

body_array = []
for i in range(500):
    if(os.path.exists("../../Project course/500%s_fat_content.vtk" %(str(i).zfill(3)))):
        body_array.append(load_vtk(i))

# print(torch.sum(torch.sum(body_array[0][2,:,:,:],dim=0),dim=1)>0)

# miss classification in the .vtk file
body_array[1][2,157,240,109] = 0

body_array_new = []

for i in range(len(body_array)):
    my_list = torch.sum(torch.sum(body_array[i][2,:,:,:],dim=0),dim=1)>0
    indices = [j for j, x in enumerate(my_list) if x == True]
    # print(str(min(indices)) + ' ' +str(max(indices)))
    body_array_new.append(body_array[i][:,:,min(indices)-2:max(indices)+3,:])

# print(body_array_new[0].size())

# print(list(chunks(list(range(len(body_array_new))),3)))

smart_list = []

for i in range(len(body_array_new)):
    for j in range(body_array_new[i].size()[2]-4):
        smart_list.append([i,j])

# print(smart_list)
batch_size = 4

x_range,y_range,z_range = 256,252,256

x_train = torch.zeros(batch_size,3,x_range,z_range).to(device='cuda')
y_train = torch.zeros(batch_size,x_range,z_range).to(device='cuda')
# print(x_train.size())

train_list = smart_list[:int(len(smart_list)*0.85)]
eval_list = smart_list[int(len(smart_list)*0.85):]

print(eval_list)
# save evaluation index pairs for future evaluation
f = open("eval_list.txt", "w")
f.write(str(eval_list))
f.close()

# define optimizer
learn_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)

# train model
for epoch in range(201):
    print('Epoch ' + str(epoch))

    # print(batch_list)
    epoch_loss = 0
    if epoch % 20 == 0:
        optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
        learn_rate/=2
        print('Learn rate halved')

    random.shuffle(train_list)
    batch_list = list(chunks(train_list,batch_size))
    
    for batch in batch_list:
        # print(batch)
        # x_train = grid_body[:,batch:batch+5,:].unsqueeze(0).unsqueeze(0).to(device='cuda')
        for i in range(len(batch)):
            # print(batch[i][0])
            # print(body_array_new.size())
            x_train[i,0:2,:,:] = body_array_new[batch[i][0]][0:2,:,batch[i][1]+2,:]
            x_train[i,2,:,:] = torch.from_numpy(np.asarray(maps_list[batch[i][0]]))
            y_train[i] = body_array_new[batch[i][0]][2,:,batch[i][1]+2,:]
        # print(y_train.size())
        # print(x_train.size())
        y_pred = model(x_train)
        # loss = loss_fn(y_pred.squeeze(3),grid_binary[:,batch+1,:].unsqueeze(0).unsqueeze(0).to(device='cuda'))
        loss = loss_fn(y_pred,y_train.to(dtype=torch.long))
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # plt.imsave(f'images/img{epoch}.png',y_pred.detach().cpu().numpy()[0,0,:,:],cmap='gray')
  
    print('Loss ' + str(epoch_loss))

    if (epoch % 10 == 0):
        # save the state of the model after each epoch and print the cambined loss function over the epoch
        torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')

# eval model
        dice = 0
        model.eval()
        x_eval = torch.zeros(1,3,x_range,z_range).to(device='cuda')
        for sample in eval_list:
            x_eval[0,0:2,:,:] = body_array_new[sample[0]][0:2,:,sample[1]+2,:]
            x_eval[0,2,:,:] = torch.from_numpy(np.asarray(maps_list[sample[0]]))
            y_eval = body_array_new[sample[0]][2,:,sample[1]+2,:].to(device='cuda')
            y_pred = model(x_eval).squeeze(0)[1].exp()
            if( sum(sum(y_eval))+sum(sum(y_pred>0.5)) == 0 ):
                dice += 1
            else:
                dice += sum(sum( (y_eval==1) & (y_pred>0.5) ))*2/( sum(sum(y_eval)) +sum(sum(y_pred>0.5)) )
        print('dice ' + str(dice.item()/len(eval_list)))
        model.train()
