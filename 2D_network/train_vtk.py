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


def load_vtk(file_number):
    """Loads .vtk file into tensor given file number"""
    file_number = str(file_number).zfill(3)
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName("vtk_files/500%s_fat_content.vtk" %(file_number))
    reader.Update()
    data_fat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    x_range, y_range, z_range = reader.GetOutput().GetDimensions()
    data_fat = data_fat.reshape(x_range,y_range,z_range)

    reader.SetFileName("vtk_files/500%s_wat_content.vtk" %(file_number))
    reader.Update()
    data_wat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_wat = data_wat.reshape(x_range,y_range,z_range)

    reader.SetFileName("vtk_files/binary_liver500%s.vtk" %(file_number))
    reader.Update()
    data_liv = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    data_liv = data_liv.reshape(x_range,y_range,z_range)

    data = np.zeros(3*x_range*y_range*z_range).reshape(3,x_range,y_range,z_range)
    data[0] = data_fat
    data[1] = data_wat
    data[2] = data_liv 
    
    grid = torch.from_numpy(data)
    return grid.to(dtype=torch.float32)


def train_net(net,epochs,device,lr,batch_size=10):

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    
    print('_______________Loading the data_______________')
    # body_array contains all .vtk-files
    body_array = []
    for i in range(500):
        if(os.path.exists("../../Project course/500%s_fat_content.vtk" %(str(i).zfill(3)))):
            body_array.append(load_vtk(i))
            
    # body_array_new contains all .vtk-files but is cropped to include
    # only slices in y-direction that have liver + 2 slices of margin
    body_array_new = []
    for i in range(len(body_array)):
        my_list = torch.sum(torch.sum(body_array[i][2,:,:,:],dim=0),dim=1)>0
        indices = [j for j, x in enumerate(my_list) if x == True]
        body_array_new.append(body_array[i][:,:,min(indices)-2:max(indices)+3,:])
        
    smart_list = []
    for i in range(len(body_array_new)):
        for j in range(body_array_new[i].size()[2]-4):
            smart_list.append([i,j])


    x_range,y_range,z_range = 256,252,256
    x_train = torch.zeros(batch_size,2,x_range,z_range).to(device='cuda')
    y_train = torch.zeros(batch_size,1,x_range,z_range).to(device='cuda')
    
    train_list = smart_list[:int(len(smart_list)*0.85)]
    eval_list = smart_list[int(len(smart_list)*0.85):]
    random.shuffle(train_list)
    batch_list = list(chunks(train_list,batch_size))
    
    print('_________________Training_____________________')
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        n_train = 0
        with tqdm(total=len(batch_list), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in batch_list:
                for i in range(len(batch)):
                    x_train[i,0:2,:,:] = body_array_new[batch[i][0]][0:2,:,batch[i][1]+2,:]
                    #x_train[i,2,:,:] = torch.from_numpy(np.asarray(maps_list[batch[i][0]]))
                    y_train[i] = body_array_new[batch[i][0]][2:,:,batch[i][1]+2,:]
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
        x_eval = torch.zeros(1,2,x_range,z_range).to(device='cuda')
        for sample in eval_list:
            x_eval[0,0:2,:,:] = body_array_new[sample[0]][0:2,:,sample[1]+2,:]
            #x_eval[0,2,:,:] = torch.from_numpy(np.asarray(maps_list[sample[0]]))
            y_eval = body_array_new[sample[0]][2:,:,sample[1]+2,:].to(device='cuda')
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
        print('dice ' + str(dice.item()/len(eval_list)))
  
  
        
def eval_net(net,pred,mask,device):
    net.eval()
    dice = 0
    
    imgs, masks = batch[:,0:2,64:192,64:192], torch.unsqueeze(batch[:,2,64:192,64:192],1) #(n_s,2,256,256) and (n_s,1,256,256) respectively
   
    
    imgs = imgs.to(device=device, dtype=torch.float32)
    masks = masks.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask_preds = net(imgs.cuda())
        
    #compute dice for the whole body
    
    inter_body = torch.FloatTensor(1).cuda().zero_()
    union_body = torch.FloatTensor(1).cuda().zero_()
    for i in range(masks.shape[0]):
        #for each slice compute inter and union
        #final dice score of the whole body is 2*sum(inter)/sum(union)
        mask_pred = mask_preds[i,:,:,:]
        mask = masks[i,:,:,:]
        pred = torch.sigmoid(mask_pred)
        pred = (pred > 0.5).float()
        
        inter = torch.dot(mask.cuda().view(-1),pred.view(-1))
        union = torch.sum(pred) + torch.sum(mask.cuda())
        inter_body += inter
        union_body += union   
    if union_body!=0:
        dice += 2*inter_body.item()/union_body.item() 
    else:
        dice+=1    
    pbar.update()
    print(f'dice: {dice/n_val}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batchsize', type=int, nargs='?', default=20,
                        help='Batch size')
    parser.add_argument('--lr', type=float, nargs='?', default=0.00001,
                        help='Learning rate')
    parser.add_argument('--load', type=str, default='CP_epoch20.pth',
                        help='Load model')
    parser.add_argument('--test', type=str, default=None,
                        help='Load test dataset')
    parser.add_argument('--train',type=str, default='true', help='Train the model')
    return parser.parse_args()



if __name__=='__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = Unet(n_channels=2, n_classes=1)
    model.to(device=device)
    

            

    if args.train=='true':
        train_net(net=model, epochs=args.epochs, lr=args.lr, device=device, batch_size=args.batchsize)
    
        
    #model_eval = Unet(n_channels=2, n_classes=1)
    #model_eval.load_state_dict(torch.load(dir_checkpoint+args.load))
    #model_eval.to(device=device)
    #eval_net(model_eval,test,device)
    
   
