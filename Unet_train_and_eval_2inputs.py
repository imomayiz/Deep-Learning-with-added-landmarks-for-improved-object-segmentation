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

def double_conv_3d(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3,padding=(1,0,1)),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=3,padding=(1,0,1)),
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
        self.down_conv_1 = double_conv_3d(2, 64)
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
        x1 = self.down_conv_1(image).squeeze(3)
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
dir_checkpoint = 'checkpoints_3d/'
if(not os.path.exists(dir_checkpoint)):
    os.mkdir(dir_checkpoint)

# choose what loss function to use
loss_fn = torch.nn.NLLLoss(weight=torch.tensor([1., 2.], device='cuda'))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

# body_array contains all .vtk-files
body_array = []
for i in range(500):
    if(os.path.exists("../../Project course/500%s_fat_content.vtk" %(str(i).zfill(3)))):
        body_array.append(load_vtk(i))

# miss classification in the .vtk file
body_array[1][2,157,240,109] = 0

# body_array_new contains all .vtk-files but is croppen to include
# only slices in y-direction that have liver + 2 slices of margin
body_array_new = []
for i in range(len(body_array)):
    my_list = torch.sum(torch.sum(body_array[i][2,:,:,:],dim=0),dim=1)>0
    indices = [j for j, x in enumerate(my_list) if x == True]
    body_array_new.append(body_array[i][:,:,min(indices)-2:max(indices)+3,:])

# smart_list contains index pairs of bodies and slices
smart_list = []
for i in range(len(body_array_new)):
    for j in range(body_array_new[i].size()[2]-4):
        smart_list.append([i,j])
random.shuffle(smart_list)

# size of scans
x_range,y_range,z_range = 256,252,256

# prepare torch tensors and assign index pairs for
# training and evaluation
batch_size=4
x_train = torch.zeros(batch_size,2,x_range,5,z_range).to(device='cuda')
y_train = torch.zeros(batch_size,x_range,z_range).to(device='cuda')
train_share = 0.9
train_list = smart_list[:int(len(smart_list)*train_share)]
eval_list = smart_list[int(len(smart_list)*train_share):]

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
    epoch_loss = 0

    # update optimizer learning rate every 20 epoch
    if epoch % 20 == 0:
        optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
        learn_rate/=2
        print('Learn rate halved')

    random.shuffle(train_list)
    batch_list = list(chunks(train_list,batch_size))

    for batch in batch_list:
        for i in range(len(batch)):
            x_train[i] = body_array_new[batch[i][0]][0:2,:,batch[i][1]:batch[i][1]+5,:]
            y_train[i] = body_array_new[batch[i][0]][2,:,batch[i][1]+2,:]
        y_pred = model(x_train)
        loss = loss_fn(y_pred,y_train.to(dtype=torch.long))
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss ' + str(epoch_loss))

    # save model and print dice score every 10 epoch
    if (epoch % 10 == 0):
        torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        dice = 0
        model.eval()
        for sample in eval_list:
            x_eval = body_array_new[sample[0]][0:2,:,sample[1]:sample[1]+5,:].unsqueeze(0).to(device='cuda')
            y_eval = body_array_new[sample[0]][2,:,sample[1]+2,:].to(device='cuda')
            y_pred = model(x_eval).squeeze(0)[1].exp()
            if( sum(sum(y_eval))+sum(sum(y_pred>0.5)) == 0 ):
                dice += 1
            else:
                dice += sum(sum( (y_eval==1) & (y_pred>0.5) ))*2/( sum(sum(y_eval)) +sum(sum(y_pred>0.5)) )
        print('dice ' + str(dice.item()/len(eval_list)))
        model.train()











