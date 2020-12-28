from unet_mtl import Unet
from torchvision.utils import save_image
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

from dataset_mtl import *

from dice_loss import dice_coeff


dir_img = 'data_mtl/imgs/'
dir_mask_1 = 'data_mtl/liver/'
dir_mask_2 = 'data_mtl/kidneys/'
dir_mask_3 = 'data_mtl/panc/'
dir_mask_4 = 'data_mtl/spleen/'
dir_mask_5 = 'data_mtl/bladder/'

dir_checkpoint = 'checkpoints/'


def save_img(img_tensor,name):
    img_tensor = torch.squeeze(img_tensor)
    save_image(img_tensor,f'./predictions/{name}.png')
    
def load_data(dir_img=dir_img, dir_mask_1=dir_mask_1, dir_mask_2=dir_mask_2,dir_mask_3=dir_mask_3, dir_mask_4=dir_mask_4, 
dir_mask_5=dir_mask_5, bs=1, train_ratio=0.8):
    '''
    load data, prepare batches, apply data augmentation
    apply function to dir_mask and dir_imgs simultaneously!!
    outputs for default data_folder : x_train,y_train and x_val,y_val loaders 
    '''
    #transformations for data augmentation
    rotation = Rotate(40)
    crop = RandomCrop(180)
    to_tensor = ToTensor()
    tsfm1 = transforms.Compose([rotation,to_tensor])
    tsfm2 = transforms.Compose([crop,to_tensor])
    # load data
    raw_data = LiverDataset(dir_img,dir_mask_1,dir_mask_2,dir_mask_3,dir_mask_4,dir_mask_5,transform=to_tensor)
    #data_cropped = LiverDataset(dir_img,dir_mask,transform=tsfm2)
    #data_rotated = LiverDataset(dir_img,dir_mask,transform=tsfm1)
    #data = ConcatDataset([raw_data,data_cropped,data_rotated])
    
    #split train and val
    n_train = int(len(raw_data)*train_ratio) 
    n_val = len(raw_data) - n_train
    train, val = random_split(raw_data, [n_train,n_val])	

    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False,num_workers=8,pin_memory=True,drop_last=True) #drop last batch if not a divisor of len(val), purpose: make same size batches to get mean dice scores fairly comparable

    return train_loader, val_loader



def train_net(net,train_loader,val_loader,epochs,device,lr):

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss_function = nn.BCEWithLogitsLoss()
    global_step = 0
    n_train = len(train_loader)
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        loss_dict = dict()
        loss_dict['liver'] = 0
        loss_dict['kidneys'] = 0
        loss_dict['panc'] = 0
        loss_dict['spleen'] = 0
        loss_dict['bladder'] = 0
        loss_dict['total'] = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                X = batch['img'][0]
                y_1 = batch['mask_1'][0]
                y_2 = batch['mask_2'][0]
                y_3 = batch['mask_3'][0]
                y_4 = batch['mask_4'][0]
                y_5 = batch['mask_5'][0]

                X.to(device=device, dtype=torch.float32)
                y_1.to(device=device, dtype=torch.float32)
                y_2.to(device=device, dtype=torch.float32)
                y_3.to(device=device, dtype=torch.float32)
                y_4.to(device=device, dtype=torch.float32)
                y_5.to(device=device, dtype=torch.float32)

        	#forward pass
                y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = net(X.cuda())
        	#computing the loss 
                loss_1 = loss_function(y_pred_1,y_1.cuda())
                loss_2 = loss_function(y_pred_2,y_2.cuda())
                loss_3 = loss_function(y_pred_3,y_3.cuda())
                loss_4 = loss_function(y_pred_4,y_4.cuda())
                loss_5 = loss_function(y_pred_5,y_5.cuda())

                loss_total = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
                
                loss_dict['liver'] = loss_1.item()
                loss_dict['kidneys'] = loss_2.item()
                loss_dict['panc'] = loss_3.item()
                loss_dict['spleen'] = loss_4.item()
                loss_dict['bladder'] = loss_5.item()
                loss_dict['total'] = loss_total.item() #should be the same as loss_total.item() !!tocheck
                #print(loss_dict)
                epoch_loss += loss_total.item()
                pbar.set_postfix(**{'loss (batch)': loss_total.item()})
                #backpropagation
                optimizer.zero_grad()
                loss_total.backward()
            	#nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(X.shape[0])
                global_step += 1
                
        torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        print(f'Checkpoint {epoch + 1}')
        print({k:v/n_train for k,v in loss_dict.items()})
        print(f'Average epoch loss: {epoch_loss/n_train}')
        
def eval_net(net,val_loader,device,save_imgs=False):
    n_val = len(val_loader)
    n_val_1,n_val_2,n_val_3,n_val_4,n_val_5 = n_val,n_val,n_val,n_val,n_val
    net.eval()
    dice_1, dice_2, dice_3, dice_4, dice_5 = 0,0,0,0,0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(val_loader):
            img, mask_1, mask_2, mask_3, mask_4, mask_5 = batch['img'][0], batch['mask_1'][0], batch['mask_2'][0],batch['mask_3'][0], batch['mask_4'][0], batch['mask_5'][0]
            img = img.to(device=device, dtype=torch.float32)
            #print(img)
            mask_1 = mask_1.to(device=device, dtype=torch.float32)
            mask_2 = mask_2.to(device=device, dtype=torch.float32)
            mask_3 = mask_3.to(device=device, dtype=torch.float32)
            mask_4 = mask_4.to(device=device, dtype=torch.float32)
            mask_5 = mask_5.to(device=device, dtype=torch.float32)
            
            
            #if torch.sum(mask_1) == 0:
                #n_val_1 = n_val_1 - 1
                #dice_1 -= 1
            #if torch.sum(mask_2) == 0:
             #   n_val_2 = n_val_2 - 1
              #  dice_2 -= 1
            #if torch.sum(mask_3) == 0:
             #   n_val_3 = n_val_3 - 1
              #  dice_3 -= 1
#            if torch.sum(mask_4) == 0:
 #               n_val_4 = n_val_4 - 1
  #              dice_4 -= 1
   #         if torch.sum(mask_5) == 0:
    #            n_val_5 = n_val_5 - 1
     #           dice_5 -= 1
            with torch.no_grad():
                mask_pred_1, mask_pred_2, mask_pred_3, mask_pred_4, mask_pred_5 = net(img.cuda())
            #print(dice_1)
            pred_1 = torch.sigmoid(mask_pred_1)
            pred_2 = torch.sigmoid(mask_pred_2)
            pred_3 = torch.sigmoid(mask_pred_3)
            pred_4 = torch.sigmoid(mask_pred_4)
            pred_5 = torch.sigmoid(mask_pred_5)
        
            pred_1 = (pred_1 > 0.5).float()
            pred_2 = (pred_2 > 0.5).float()
            pred_3 = (pred_3 > 0.5).float()
            pred_4 = (pred_4 > 0.5).float()
            pred_5 = (pred_5 > 0.5).float()
            if torch.sum(mask_1)!=0:
                dice_1 += dice_coeff(pred_1, mask_1.cuda()).item()
            else:
                n_val_1-=1
            if torch.sum(mask_2)!=0:
                dice_2 += dice_coeff(pred_2, mask_2.cuda()).item()
            else:
                n_val_2-=1
            if torch.sum(mask_3)!=0:
                dice_3 += dice_coeff(pred_3, mask_3.cuda()).item()
            else:
                n_val_3 -= 1
            if torch.sum(mask_4)!=0:
                dice_4 += dice_coeff(pred_4, mask_4.cuda()).item()
            else:
                n_val_4 -=1
            if torch.sum(mask_5)!=0:
                dice_5 += dice_coeff(pred_5, mask_5.cuda()).item()
            else:
                n_val_5 -=1

            pbar.update()
            
            if save_imgs:
            #save predictions
                save_img(mask_pred_1,'liver'+str(i))
                save_img(mask_pred_2,'kidneys'+str(i))
                save_img(mask_pred_3,'panc'+str(i))
                save_img(mask_pred_4,'spleen'+str(i))
                save_img(mask_pred_5,'bladder'+str(i))

    if save_imgs:
        print('Predicted masks saved in ./predictions/ !')
    print(n_val_1)
                    
    print(f'dice_liver: {dice_1/n_val_1}')
    print(f'dice_kidneys: {dice_2/n_val_2}')
    print(f'dice_panc: {dice_3/n_val_3}')
    print(f'dice_spleen: {dice_4/n_val_4}')
    print(f'dice_bladder: {dice_5/n_val_5}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batchsize', type=int, nargs='?', default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, nargs='?', default=0.0001,
                        help='Learning rate')
    parser.add_argument('--data', type=str, default='./data/',
                        help='Load training dataset')
    parser.add_argument('--val', type=str, default=None,
                        help='Load validation dataset')
    parser.add_argument('--save_imgs', dest='save', action='store_true',
                        help='Save predicted images')
    parser.set_defaults(save=False)
    parser.add_argument('--no-train', dest='train', action='store_false',
                        help='Save predicted images')
    parser.set_defaults(train=True)

    return parser.parse_args()



if __name__=='__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = Unet(n_channels=3, n_classes=1)
    model.to(device=device)
    
    train, val = load_data(dir_img, dir_mask_1, dir_mask_2,dir_mask_3,dir_mask_4,dir_mask_5, bs=args.batchsize)

    if args.train:
        train_net(net=model, train_loader = train, val_loader = val, epochs=args.epochs, lr=args.lr, device=device)
    
    model_eval = Unet(n_channels=3, n_classes=1)
    model_eval.load_state_dict(torch.load(dir_checkpoint+'CP_epoch5.pth'))
    model_eval.to(device=device)
    eval_net(model_eval,val,device=device,save_imgs=args.save)
    
   
