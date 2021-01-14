from unet import Unet
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms
import numpy as np
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import random
from tqdm import tqdm
import os
import argparse




dir_checkpoint = 'checkpoints/'

def unstack(tensor):
    """
    input is stack of 2D slices from one body
    input shape is [3,256,n_slices,256]
    outputs a new stack of tensors s.t. shape is [n_slices,3,256,256]
    """
    new_tensor = []
    for i in range(tensor.shape[2]):
        new_tensor.append(tensor[:,:,i,:])
    return torch.stack(new_tensor)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_vtk(file_number,vtk_dir="vtk_files",organ="liver"):
    """Loads .vtk file into tensor given file number"""
    file_number = str(file_number).zfill(3)
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(vtk_dir+"/500%s_fat_content.vtk" %(file_number))
    reader.Update()
    data_fat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    x_range, y_range, z_range = reader.GetOutput().GetDimensions()
    data_fat = data_fat.reshape(x_range,y_range,z_range)

    reader.SetFileName(vtk_dir+"/500%s_wat_content.vtk" %(file_number))
    reader.Update()
    data_wat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_wat = data_wat.reshape(x_range,y_range,z_range)

    reader.SetFileName(vtk_dir+"/binary_"+organ+"500%s.vtk" %(file_number))
    reader.Update()
    data_liv = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_liv = data_liv.reshape(x_range,y_range,z_range)

    data = np.zeros(3*x_range*y_range*z_range).reshape(3,x_range,y_range,z_range)
    data[0] = data_fat
    data[1] = data_wat
    data[2] = data_liv 
    
    grid = torch.from_numpy(data)
    return grid.to(dtype=torch.float32)

def load_data(vtk_dir="vtk_files",organ='liver'):
    print('_______________Loading the data_______________')
    # body_array contains all .vtk-files
    body_array = []
    for i in range(500):
        if(os.path.exists(vtk_dir+"/500%s_fat_content.vtk" %(str(i).zfill(3)))):
            body_array.append(load_vtk(i,vtk_dir,organ))
    #correct the groundtruth for body n#2
    body_array[1][2,157,240,109] = 0        
    ## body_array_new contains only slices with liver 
    body_array_new = []
    for i in range(len(body_array)):
        my_list = torch.sum(torch.sum(body_array[i][2,:,:,:],dim=0),dim=1)>0
        indices = [j for j, x in enumerate(my_list) if x == True]
        if len(indices)!=0:
            body_array_new.append(body_array[i][:,:,min(indices):max(indices),:])
    train_bodies = body_array_new[:40]
    test_bodies = body_array_new[40:]
    #compute distance maps
    maps_list = [None]*50
    for k in range(1,51):
        file = open(r"theirLandmarks/LMs_"+ str(k) + ".txt")
        lines = file.readlines()
        file.close()
        X1,_,Z1 = map(int,map(float,lines[6].split()))
        X2,_,Z2 = map(int,map(float,lines[7].split()))
        
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
        distance_map = distance_map/distance_map[X1,Z1] #normalization
        maps_list[k-1] = distance_map
        maps_list = np.asarray(maps_list)
    return(train_bodies, test_bodies, maps_list)


def train_net(net,body_array_new,maps_list,epochs,device,lr,batch_size=10):

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    
    #save [body_idx,slice_idx]
    smart_list = []
    for i in range(len(body_array_new)):
        for j in range(body_array_new[i].size()[2]):
            smart_list.append([i,j])


    x_range,y_range,z_range = 256,252,256
    
    
    #train_list = smart_list[:int(len(smart_list)*0.85)]
    train_list = smart_list
    val_list = smart_list[int(len(smart_list)*0.85):]
    random.shuffle(train_list)
    batch_list = list(chunks(train_list,batch_size))
    
    print('_________________Training_____________________')
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        n_train = 0
        with tqdm(total=len(batch_list), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in batch_list:
                x_train = torch.zeros(len(batch),3,x_range,z_range).to(device='cuda')
                y_train = torch.zeros(len(batch),1,x_range,z_range).to(device='cuda')
                for i in range(len(batch)):
                    x_train[i,0:2,:,:] = body_array_new[batch[i][0]][0:2,:,batch[i][1],:]
                    x_train[i,2,:,:] = torch.from_numpy(np.asarray(maps_list[batch[i][0]]))
                    y_train[i] = body_array_new[batch[i][0]][2:,:,batch[i][1],:]
                    n_train+=1
                y_pred = model(x_train)
                loss = loss_fn(y_pred,y_train.to(dtype=torch.float32))
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
              
        torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        print(f'Checkpoint {epoch + 1}')
 
        print(f'Average epoch loss: {epoch_loss/n_train}')
        
        print("_____________Evaluating on validation set__________________")
        model.eval()
        dice=0
        x_eval = torch.zeros(1,3,x_range,z_range).to(device='cuda')
        for sample in val_list:
            x_eval[0,0:2,:,:] = body_array_new[sample[0]][0:2,:,sample[1],:]
            x_eval[0,2,:,:] = torch.from_numpy(np.asarray(maps_list[sample[0]]))
            y_eval = body_array_new[sample[0]][2:,:,sample[1],:].to(device='cuda')
            with torch.no_grad():
                y_pred = model(x_eval)
                   
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred>0.5).float()
            inter = torch.dot(y_eval.view(-1),y_pred.view(-1))
            union = torch.sum(y_pred) + torch.sum(y_eval)
            if union==0:
                dice+=1
            else:
                dice += 2*inter/union
        print('dice ' + str(dice.item()/len(val_list)))
  
  
        
def eval_net(model,body_array_new, maps_list, device):
    model.eval()
    
    #save [body_idx,slice_idx]
    smart_list = []
    for i in range(len(body_array_new)):
        for j in range(body_array_new[i].size()[2]):
            smart_list.append([i,j])
            
    x_range,y_range,z_range = 256,252,256
    x_eval = torch.zeros(1,3,x_range,z_range).to(device='cuda')

    #store sum of intersections/unions for all slices per body
    inter_array = np.zeros(len(body_array_new))
    union_array = np.zeros(len(body_array_new))
    dice_n = 0
    for body,slc in smart_list:
        x_eval[0,0:2,:,:] = body_array_new[body][0:2,:,slc,:]
        x_eval[0,2,:,:] = torch.from_numpy(np.asarray(maps_list[body]))
        y_eval = body_array_new[body][2:,:,slc,:].to(device='cuda')
        with torch.no_grad():
            y_pred = model(x_eval)
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
        inter = torch.dot(y_eval.view(-1),y_pred.view(-1))
        union = torch.sum(y_pred) + torch.sum(y_eval)
        inter_array[body] += inter.float()
        union_array[body] += union.float()
        
        #compute normal dice
        if union==0:
            dice_n += 1
        else:
            dice_n += 2*inter.float()/union.float()
        
    #compute micro-average dice for each body          
    dice_array = np.zeros(len(body_array_new))
    for i in range(len(body_array_new)):
        if union_array[i]==0:
            dice_array[i] = 1    
        else:
            dice_array[i] = 2*inter_array[i]/union_array[i]
    print("Micro-average dice scores for each_body: ")
    print(*[(i+1,round(dice,4)) for i,dice in enumerate(dice_array)],sep=" ")
    print(f"Average dice score over {len(dice_array)} bodies: {round(sum(dice_array)/len(dice_array),4)}")         
    print(f'Average dice over {len(smart_list)} samples: {round(dice_n.item()/len(smart_list),4)}')
    


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batchsize', type=int, nargs='?', default=10,
                        help='Batch size')
    parser.add_argument('--lr', type=float, nargs='?', default=0.00001,
                        help='Learning rate')
    parser.add_argument('--organ', type=str, default='liver',
                        help='Name of the organ to be segmented')
    parser.add_argument('--load', type=str, default='CP_epoch20.pth',
                        help='Load model')
    parser.add_argument('--test', type=str, default=None,
                        help='Load test dataset')
    parser.add_argument('--no-train',dest='train', action='store_false', help='Skip the training phase')
    parser.set_defaults(train=True)
    return parser.parse_args()



if __name__=='__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = Unet(n_channels=3, n_classes=1)
    model.to(device=device)
    

    train_data, test_data, maps = load_data(organ=args.organ)        

    if args.train:
        train_net(net=model, body_array_new=train_data, maps_list=maps, epochs=args.epochs, lr=args.lr, device=device, batch_size=args.batchsize)
    
        
    model_eval = Unet(n_channels=3, n_classes=1)
    model_eval.load_state_dict(torch.load(dir_checkpoint+args.load))
    model_eval.to(device=device)
    eval_net(model=model_eval,body_array_new=test_data, maps_list=maps, device=device)
    
   
