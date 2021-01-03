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
import time
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from ast import literal_eval

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
            out_channels=5,
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
        #x = self.soft(x)
        return x

# define the model
model = UNet()
model.to(device='cuda')
dir_checkpoint = 'checkpoints_3d/'
if(not os.path.exists(dir_checkpoint)):
    os.mkdir(dir_checkpoint)

# choose what loss function to use
#loss_fn = torch.nn.NLLLoss(weight=torch.tensor([1., 1., 1.], device='cuda'))
loss_fn = torch.nn.BCEWithLogitsLoss()


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

    reader.SetFileName("../../Project course/binary_kidneys500%s.vtk" %(file_number))
    reader.Update()
    data_kid = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_kid = data_kid.reshape(x_range,y_range,z_range)

    reader.SetFileName("../../Project course/binary_spleen500%s.vtk" %(file_number))
    reader.Update()
    data_spl = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_spl = data_spl.reshape(x_range,y_range,z_range)

    reader.SetFileName("../../Project course/binary_panc500%s.vtk" %(file_number))
    reader.Update()
    data_pan = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_pan = data_pan.reshape(x_range,y_range,z_range)

    reader.SetFileName("../../Project course/binary_bladder500%s.vtk" %(file_number))
    reader.Update()
    data_bla = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_bla = data_bla.reshape(x_range,y_range,z_range)

    data = np.zeros(7*x_range*y_range*z_range).reshape(7,x_range,y_range,z_range)
    data[0] = data_fat
    data[1] = data_wat
    data[2] = data_liv 
    data[3] = data_kid
    data[4] = data_spl
    data[5] = data_pan
    data[6] = data_bla
    #print(sum(sum(sum(data[6]))))
    # print(data_liv.shape)
    # print(data_kid.shape)
    # print("----------")
    # print(sum(sum(sum((data_liv == 1) & (data_kid == 1)))))
    # print(sum(sum(sum(data_kid == 1))))
    # print(sum(sum(sum(data_liv == 1))))

    grid = torch.from_numpy(data)
    return grid.to(dtype=torch.float32)

# body_array contains all .vtk-files
body_array = []
range_array = []
for i in range(500):
    if(os.path.exists("../../Project course/500%s_fat_content.vtk" %(str(i).zfill(3)))):
        body_array.append(load_vtk(i))
        f = open("../../Project course/landmarks/LMs_500%s.txt" %(str(i).zfill(3)), 'r')
        landmarks = [[x for x in line.split()] for line in f]
        range_array.append([int(float(landmarks[8][1])),int(float(landmarks[6][1])+20)])

#for i in range(50):
#    print(body_array[i][2,:,:,:].max())
#print(body_array[0][6].sum())
# miss classification in the .vtk file
body_array[1][2,157,240,109] = 0

# body_array_new contains all .vtk-files but is croppen to include
# only slices in y-direction that have liver + 2 slices of margin
body_array_new = []
for i in range(len(body_array)):
    my_list = torch.sum(body_array[i][2:7,:,:,:],dim=(0,1,3))>0
    #print(my_list)
    indices = [j for j, x in enumerate(my_list) if x == True]
    #print('---')
    #print(min(indices))
    #print(max(indices))
    # body_array_new.append(body_array[i][:,:,min(indices)-2:max(indices)+3,:])
    body_array_new.append(body_array[i][:,:,range_array[i][0]-2:range_array[i][1]+3,:])
    # body_array_new.append(body_array[i][:,:,100-2:140+3,:])

training_bodies = body_array_new[:40]
testing_bodies = body_array_new[40:]

# smart_list contains index pairs of bodies and slices
smart_list = []
for i in range(len(training_bodies)):
    for j in range(training_bodies[i].size()[2]-4):
        smart_list.append([i,j])
random.shuffle(smart_list)


smart_list_eval = []
for i in range(len(testing_bodies)):
    for j in range(testing_bodies[i].size()[2]-4):
        smart_list_eval.append([i,j])

# size of scans
x_range,y_range,z_range = 256,252,256

# prepare torch tensors and assign index pairs for
# training and evaluation
batch_size=4
x_train = torch.zeros(batch_size,2,x_range,5,z_range).to(device='cuda')
y_train = torch.zeros(batch_size,5,x_range,z_range).to(device='cuda')
train_share = 0.9
# train_list = smart_list[:int(len(smart_list)*train_share)]
train_list = smart_list
# eval_list = smart_list[int(len(smart_list)*train_share):]
eval_list = smart_list_eval

# train_list = literal_eval(open("train_list.txt").read())
# eval_list = literal_eval(open("eval_list.txt").read())

# print(eval_list)

# save evaluation index pairs for future evaluation
# f = open("eval_list.txt", "w")
# f.write(str(eval_list))
# f.close()

# f = open("train_list.txt", "w")
# f.write(str(train_list))
# f.close()

# define optimizer
learn_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)

# define time
total_time = 0

counter = 0
output = nn.ReLU()
# train model
print("\n\t\t\t\t\t\t\tLiver\tKidn\tSpleen\tPanc\tBlad")
for epoch in range(201):

    #start timer
    start_time = time.perf_counter()

    print('\nEpoch\t' + str(epoch), end='')
    epoch_loss = 0

    random.shuffle(train_list)
    batch_list = list(chunks(train_list,batch_size))

    for batch in batch_list:
        # print(batch)
        for i in range(len(batch)):
            x_train[i] = body_array_new[batch[i][0]][0:2,:,batch[i][1]:batch[i][1]+5,:]
            y_train[i] = body_array_new[batch[i][0]][2:7,:,batch[i][1]+2,:]
        # print(y_train[:,4,:,:].sum())
        i, j, h, w = transforms.RandomCrop.get_params(y_train, output_size=(128, 128))
        x_train_transformed = torchvision.transforms.functional.crop(x_train.permute(0, 1, 3, 2, 4),i,j,h,w)
        y_train_transformed = torchvision.transforms.functional.crop(y_train,i,j,h,w)
        
        if (counter < 10):
            plt.imsave('transforms/x_before%s.png'%(str(counter)),x_train_transformed[0,0,2,:,:].cpu().numpy())
            plt.imsave('transforms/y_before%s.png'%(str(counter)),y_train_transformed[0,4,:,:].cpu().numpy())
        
        angle, translations, scale, shear = torchvision.transforms.RandomAffine.get_params(degrees=(0,0),scale_ranges=(1,1),shears=(0,0),translate=(0,0),img_size=(256,256))
        # angle, translations, scale, shear = torchvision.transforms.RandomAffine.get_params(degrees=(-3,3),scale_ranges=(1,1),shears=(-3,3,-3,3),translate=(0,0),img_size=(208,208))
        x_train_transformed[:,0,:,:,:] = torchvision.transforms.functional.affine(x_train_transformed[:,0,:,:,:], angle, translations, scale, shear)
        x_train_transformed[:,1,:,:,:] = torchvision.transforms.functional.affine(x_train_transformed[:,1,:,:,:], angle, translations, scale, shear)
        x_train_transformed = x_train_transformed.permute(0, 1, 3, 2, 4)
        #x_train_transformed = torchvision.transforms.functional.affine(x_train_transformed, angle, translations, scale, shear).permute(0, 1, 3, 2, 4)
        y_train_transformed = torchvision.transforms.functional.affine(y_train_transformed, angle, translations, scale, shear)
        
        if (counter < 10):
            plt.imsave('transforms/x_after%s.png'%(str(counter)),x_train_transformed[0,0,:,2,:].cpu().numpy())
            plt.imsave('transforms/y_after%s.png'%(str(counter)),y_train_transformed[0,4,:,:].cpu().numpy())
            counter += 1

        y_pred = model(x_train_transformed)
        loss = loss_fn(y_pred,y_train_transformed)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # update time (not including dice score)
    total_time += time.perf_counter()-start_time

    print('\tLoss ' + "{:.4f}".format(epoch_loss), end='')
    print('\tTime ' + "{:.2f}".format(total_time), end='')

    # save model and print dice score every 10 epoch
    if (epoch % 1 == 0):
        torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        dice_num = torch.zeros(5, device='cuda')
        dice_den = torch.zeros(5, device='cuda')
        model.eval()
        for sample in eval_list:
            x_eval = testing_bodies[sample[0]][0:2,:,sample[1]:sample[1]+5,:].unsqueeze(0).to(device='cuda')
            y_eval = testing_bodies[sample[0]][2:7,:,sample[1]+2,:].to(device='cuda')
            y_pred = output(model(x_eval).squeeze(0))

            dice_num += ( (y_eval==1) & (y_pred>0.5) ).sum(dim=(1, 2))*2
            dice_den +=  y_eval.sum(dim=(1, 2)) +(y_pred>0.5).sum(dim=(1, 2))
            #print((y_eval.sum(dim=(1, 2)) +(y_pred>0.5).sum(dim=(1, 2)) ))
        print('\tDice ',end='')
        for i in range(5):
            if (dice_den[i] == 0): 
                print('\t' + "Undef", end='')
            else: 
                print('\t' + "{:.4f}".format(dice_num[i].item()/dice_den[i].item()), end='')


        model.train()

    # update optimizer learning rate every 20 epoch
    if epoch % 20 == 0:
        if (epoch > 0):
            optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
            learn_rate/=2
            print('\tLR 1/' + "{:.0f}".format(1/learn_rate), end='')
        else:
            print('\tLR 1/' + "{:.0f}".format(1/learn_rate), end='')

