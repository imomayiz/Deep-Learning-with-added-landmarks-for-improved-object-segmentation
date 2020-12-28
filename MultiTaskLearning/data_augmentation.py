import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import math
import torch.nn.functional as F


def rot_x(angle,ptx,pty):
    return math.cos(angle)*ptx + math.sin(angle)*pty

def rot_y(angle,ptx,pty):
    return -math.sin(angle)*ptx + math.cos(angle)*pty


def rotate(img,degree):
    """takes opened PIL img asinput, returns a transformed img"""
    x, y = img.size
    angle = math.radians(degree)
    xextremes = np.array([rot_x(angle,0,0),rot_x(angle,0,y-1),rot_x(angle,x-1,0),rot_x(angle,x-1,y-1)])
    yextremes = np.array([rot_y(angle,0,0),rot_y(angle,0,y-1),rot_y(angle,x-1,0),rot_y(angle,x-1,y-1)])
    mnx = min(xextremes)
    mxx = max(xextremes)
    mny = min(yextremes)
    mxy = max(yextremes)
    T = np.array([[math.cos(angle),math.sin(angle),-mnx],[-math.sin(angle),math.cos(angle),-mny],[0,0,1]])
    Tinv = np.linalg.inv(T);
    Tinvtuple = Tinv.flatten()[:6]
    im_transformed = img.transform((int(round(mxx-mnx)),int(round((mxy-mny)))),Image.AFFINE,Tinvtuple,resample=Image.BILINEAR)
    return im_transformed

def crop(img):
  h,w = img.size
  left = w/6
  top = h / 4
  right = w-30
  bottom = h - 30
  img = img.crop((left,top,right,bottom))
  return img

def rcrop(img, output_size):
    h, w = img.size
    new_h, new_w = output_size
    bottom = np.random.randint(0,h-new_h)
    top = bottom + new_h
    left = np.random.randint(0,w-new_w)
    right = left + new_w
    img = img.crop((left,bottom,right,top))
    return img


def rescale(img,output_size):
    h, w = img.size
    h_o, w_o = output_size
    resize = transforms.Resize((h_o,w_o))
    img = resize(img)
    return img


#img_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)


if __name__=='__main__':
    for file in os.listdir('data_mtl/imgs'):
       img = Image.open('data_mtl/imgs/'+file)
       img = crop(img)
       img = rescale(img,(128,128))
       #img = rcrop(img,(128,128))
       img.save('transformed_data/imgs/'+file)

    for file in os.listdir('data_mtl/liver'):
       img = Image.open('data_mtl/liver/'+file)
       img = crop(img)
       img = rescale(img,(128,128))
       #img = rcrop(img,(128,128))
       img.save('transformed_data/liver/'+file)

    for file in os.listdir('data_mtl/kidneys'):
       img = Image.open('data_mtl/kidneys/'+file)
       img = crop(img)
       img = rescale(img,(128,128))
       #img = rcrop(img,(128,128))
       img.save('transformed_data/kidneys/'+file)

    for file in os.listdir('data_mtl/panc'):
       img = Image.open('data_mtl/panc/'+file)
       img = crop(img)
       img = rescale(img,(128,128))
       #img = rcrop(img,(128,128))
       img.save('transformed_data/panc/'+file)

    for file in os.listdir('data_mtl/spleen'):
       img = Image.open('data_mtl/spleen/'+file)
       img = crop(img)
       img = rescale(img,(128,128))
       #img = rcrop(img,(128,128))
       img.save('transformed_data/spleen/'+file)

    for file in os.listdir('data_mtl/bladder'):
       img = Image.open('data_mtl/bladder/'+file)
       img = crop(img)
       img = rescale(img,(128,128))
       #img = rcrop(img,(128,128))
       img.save('transformed_data/bladder/'+file)


