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

        # self.max_pool_2x2_3d = nn.MaxPool3d(kernel_size=(2,1,2), stride=(2,1,2))
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv_3d(1, 64)
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
        return x

# define the model
model = UNet()
# model.to(device='cuda')
dir_checkpoint = 'checkpoints/'
if(not os.path.exists(dir_checkpoint)):
    os.mkdir(dir_checkpoint)

# choose what loss function to use
loss_fn = torch.nn.BCEWithLogitsLoss()
#loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = torch.nn.KLDivLoss(reduction='sum')
#loss_fn = torch.nn..BCEWithLogitsLoss(reduction='sum')

# choose optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# load data
# btch_siz = 10
# val_percent = 0.15
# dataset = torchvision.datasets.ImageFolder(root='../data', transform=transforms.ToTensor())
# n_val = int(len(dataset) * val_percent)
# n_train = len(dataset) - n_val
# train, val = random_split(dataset, [n_train, n_val])
# loaded_train = DataLoader(train, batch_size=btch_siz, shuffle=True,)
# loaded_eval = DataLoader(val, batch_size=1, shuffle=False,)

# load vtk fat data
filename = "../data/500017_fat_content.vtk"
reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName(filename)
reader.Update()
data = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
x_range, y_range, z_range = reader.GetOutput().GetDimensions()
data = data.reshape(x_range,y_range,z_range)
grid_body = torch.from_numpy(data)

# load vtk binary data
filename = "../data/binary_liver500017.vtk"
reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName(filename)
reader.Update()
data = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
x_range, y_range, z_range = reader.GetOutput().GetDimensions()
data = data.reshape(x_range,y_range,z_range)
grid_binary = torch.from_numpy(data)
grid_binary = grid_binary.type(torch.DoubleTensor)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_vtk(file_number):
    file_number = str(file_number).zfill(3)
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName("../data/500%s_fat_content.vtk" %(file_number))
    reader.Update()
    data_fat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    x_range, y_range, z_range = reader.GetOutput().GetDimensions()
    data_fat = data_fat.reshape(x_range,y_range,z_range)

    reader.SetFileName("../data/500%s_fat_content.vtk" %(file_number))
    reader.Update()
    data_wat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_wat = data_wat.reshape(x_range,y_range,z_range)

    reader.SetFileName("../data/binary_liver500%s.vtk" %(file_number))
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
    if(os.path.exists("../data/500%s_fat_content.vtk" %(str(i).zfill(3)))):
        body_array.append(load_vtk(i))

# print(torch.sum(torch.sum(body_array[0][2,:,:,:],dim=0),dim=1)>0)

body_array_new = []

for i in range(len(body_array)):
    my_list = torch.sum(torch.sum(body_array[i][2,:,:,:],dim=0),dim=1)>0
    indices = [j for j, x in enumerate(my_list) if x == True]
    body_array_new.append(body_array[i][:,:,min(indices)-2:max(indices)+3,:])

# print(body_array_new[0].size())

# print(list(chunks(list(range(len(body_array_new))),3)))

smart_list = []

for i in range(len(body_array_new)):
    for j in range(body_array_new[i].size()[2]-4):
        smart_list.append([i,j])

batch_size = 3

x_train = torch.zeros(batch_size,1,x_range,5,z_range)
y_train = torch.zeros(batch_size,1,x_range,z_range)
print(x_train.size())

# train model
for epoch in range(1000):
    random.shuffle(smart_list)
    batch_list = list(chunks(smart_list,batch_size))
    print(epoch)
    epoch_loss = 0
    for batch in batch_list:
        print(batch)
        # x_train = grid_body[:,batch:batch+5,:].unsqueeze(0).unsqueeze(0).to(device='cuda')
        for i in range(batch_size):
            x_train[i] = body_array_new[batch[i][0]][0,:,batch[i][1]:batch[i][1]+5,:].unsqueeze(0)
            y_train[i] = body_array_new[batch[i][0]][2,:,batch[i][1]+2,:].unsqueeze(0)
        print(y_train.size())
        print(x_train.size())
        y_pred = model(x_train)
        # loss = loss_fn(y_pred.squeeze(3),grid_binary[:,batch+1,:].unsqueeze(0).unsqueeze(0).to(device='cuda'))
        loss = loss_fn(y_pred,y_train)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.imsave(f'images/img{epoch}.png',y_pred.detach().cpu().numpy()[0,0,:,:],cmap='gray')
    print(epoch_loss)


'''

# train model
epochs = 40
model.train()
for epoch in range(epochs):
    idx = 0
    epoch_loss = 0
    for batch in loaded_train:
        x_train = batch[0][:,0:2,32:224,32:224]
        y_tr = batch[0][:,2,32:224,32:224].unsqueeze(1)
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

model.to(device='cpu')
model.eval()
dice = 0
for batch in loaded_eval:
    x_eval = batch[0][:,0:2,32:224,32:224]
    y_eval = batch[0][:,2,32:224,32:224]
    yp = model(x_eval).squeeze(0).squeeze(0).detach().numpy()
    y_eval = y_eval.squeeze(0).detach().numpy()
    dice += sum(sum( (y_eval==1) & (yp>0.5) ))*2/( sum(sum(y_eval)) +sum(sum(yp>0.5)) )
print(dice/n_val)

'''