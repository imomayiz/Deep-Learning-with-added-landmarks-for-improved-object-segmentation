from unet import Unet

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms
import numpy as np
import math

from tqdm import tqdm
import os
import argparse

from dataset import *

from dice_loss import dice_coeff


dir_img = 'imgs/'
dir_mask = 'masks/'
dir_checkpoint = 'checkpoints/'


def load_data(dir_img='data/'+dir_img, dir_mask='data/'+dir_mask, bs=1, train_ratio=0.8,tsfm='all',degree=40,rcrop_size=140,scale=128):
    '''
    load data, prepare batches, apply data augmentation
    apply function to dir_mask and dir_imgs simultaneously!!
    outputs for default data_folder : x_train,y_train and x_val,y_val loaders 
    '''
    #transformations for data augmentation
    crop = Crop()
    rotation = Rotate(degree)
    rdcrop = RandomCrop(rcrop_size)
    resize = Rescale(scale)
    tsfm_raw = transforms.Compose([crop,resize,ToTensor()])    
    tsfm1 = transforms.Compose([crop,rotation,resize,ToTensor()])
    tsfm2 = transforms.Compose([crop,rdcrop,resize,ToTensor()])
    
    # load data
    if tsfm=='none':
        data = LiverDataset(dir_img,dir_mask,transform=ToTensor())
        
    elif tsfm=='crop':
        data = LiverDataset(dir_img,dir_mask,transform=tsfm_raw)
        
    elif tsfm=='rot':
        raw_data = LiverDataset(dir_img,dir_mask,transform=tsfm_raw)
        data_rotated = LiverDataset(dir_img,dir_mask,transform=tsfm1)
        data = ConcatDataset([raw_data,data_rotated])
        
    elif tsfm=='rdcrop':
        raw_data = LiverDataset(dir_img,dir_mask,transform=tsfm_raw)
        data_cropped = LiverDataset(dir_img,dir_mask,transform=tsfm2)
        data = ConcatDataset([raw_data,data_cropped])

    elif tsfm=='all':
        raw_data = LiverDataset(dir_img,dir_mask,transform=tsfm_raw)
        data_cropped = LiverDataset(dir_img,dir_mask,transform=tsfm2)
        data_rotated = LiverDataset(dir_img,dir_mask,transform=tsfm1)
        data = ConcatDataset([raw_data,data_rotated,data_cropped])
        
    if train_ratio!=1:
    #split train and val
        n_train = int(len(data)*train_ratio) 
        n_val = len(data) - n_train
        train, val = random_split(data, [n_train,n_val])
        train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=8,pin_memory=True)
        val_loader = DataLoader(val, batch_size=bs, shuffle=False,num_workers=8,pin_memory=True,drop_last=True) #drop last batch if not a divisor of len(val), purpose: make same size batches to get mean dice scores fairly comparable
    else:
        train_loader = DataLoader(data,batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = None
    return train_loader, val_loader



def train_net(net,train_loader, val_loader, epochs,device,lr,batchsize):

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss()

    global_step = 0
    n_train = len(train_loader)*batchsize
    avg_dice_val = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                X = torch.squeeze(batch['img'],1)
                #print(X.shape)
                y = torch.squeeze(batch['mask'],1)
                X.to(device=device, dtype=torch.float32)
                y.to(device=device, dtype=torch.float32)
        	#forward pass
                y_pred = net(X.cuda())
               
               
        	#computing the loss 
                loss = loss_function(y_pred,y.cuda())
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                #backpropagation
                optimizer.zero_grad()
                loss.backward()
            	#nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(X.shape[0])
        n_val, dice_val = eval_net(net,loader=val_loader,device=device)
        avg_dice_val += dice_val 
                
        torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        print(f'Checkpoint {epoch + 1}')
 
        print(f'Average epoch loss: {epoch_loss/n_train}')
    print(f'Average validation dice: {avg_dice_val/epochs} over {n_val} samples')
         
        
def eval_net(net,loader,device):
    n_val = len(loader)
    net.eval()
    dice = 0
    with tqdm(total=n_val, desc='Evaluation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            img, mask = batch['img'][0], batch['mask'][0]
            img = img.to(device=device, dtype=torch.float32) #(1,3,256,256)
            mask = mask.to(device=device, dtype=torch.float32) #(1,1,256,256)
            #print(img.shape)
            #print(mask.shape)

            with torch.no_grad():
                mask_pred = net(img.cuda())
            #print(mask_pred.shape) (1,1,256,256)
   
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            if torch.sum(mask)==0:
                n_val-=1
            else:
                dice += dice_coeff(pred, mask.cuda()).item()
            pbar.update()
          
    print(n_val)
    print(f'Dice: {dice/n_val}')
    return (n_val,dice/n_val)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batchsize', type=int, nargs='?', default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, nargs='?', default=0.0001,
                        help='Learning rate')
    parser.add_argument('--load', type=str, default='CP_epoch5.pth',
                        help='Load model')
    parser.add_argument('--test', type=str, default=None,
                        help='Load test dataset')
    parser.add_argument('--train',type=str, default='true', help='Train the model')
    
    parser.add_argument('--tsfm',type=str, default='all', help='Transformations to apply')
    parser.add_argument('--degree',type=int, default=40, help='Degree of rotation in data augmentation')
    parser.add_argument('--scale',type=int, default=128, help='Scale of input image')
    parser.add_argument('--rcrop',type=int, default=140, help='Size after random cropping in data augmentation')
    return parser.parse_args()



if __name__=='__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = Unet(n_channels=3, n_classes=1)
    model.to(device=device)

    train_loader, val_loader = load_data(bs=args.batchsize, tsfm=args.tsfm, degree=args.degree, scale=args.scale, rcrop_size=args.rcrop)


    if args.train=='true':
        train_net(net=model, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, lr=args.lr, device=device, batchsize=args.batchsize)
    if args.test != None:
        test,_ = load_data(dir_img=args.test+dir_img, dir_mask=args.test+dir_mask,train_ratio=1,tsfm='none')
        
    model_eval = Unet(n_channels=3, n_classes=1)
    model_eval.load_state_dict(torch.load(dir_checkpoint+args.load))
    model_eval.to(device=device)
    eval_net(model_eval,test,device=device)
    
   
