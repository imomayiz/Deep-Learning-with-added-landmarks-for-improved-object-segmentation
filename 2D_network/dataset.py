import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import math


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



class LiverDataset(Dataset):
    """
    should include: len and getitem
    getitem returns a dict with img and corresponding mask
    transform should be a callable object
    if called, transform gets image and mask
    """
    def __init__(self,dir_img,dir_mask,transform=None):
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.transform = transform
        
        self.imgids = [os.path.splitext(file)[0] for file in os.listdir(dir_img) if not file.startswith('.')]
        self.maskids = [os.path.splitext(file)[0] for file in os.listdir(dir_mask) if not file.startswith('.')]
        
    def __len__(self):
        assert len(self.imgids) == len(self.maskids), 'dir_img and dir_mask do not contain the same number of files.'
        return len(self.imgids)
        
    def __getitem__(self,i):
        idx = self.imgids[i]
        img_file = glob(self.dir_img + idx + '.*')
        mask_file = glob(self.dir_mask + idx + '.*')    
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
            
        sample = {'img':img,'mask':mask}
        if self.transform:
            sample = self.transform(sample)
        return(sample)
        
        
##implement some transformations

class Crop(object):
    def __call__(self,sample):
        img, mask = sample['img'], sample['mask']
        img = crop(img)
        mask = crop(mask)
        return {'img':img,'mask':mask}
         
class Rotate(object):
    def __init__(self,degree):
        assert 0<degree<180, 'Degree must be between 0 and 180'
        self.degree = degree
        
    def __call__(self,sample):
        img, mask = sample['img'], sample['mask']
        img_transformed = rotate(img,self.degree)
        mask_transformed = rotate(mask,self.degree)
        return {'img':img_transformed,'mask':mask_transformed}
        

class RandomCrop(object):
    """Crop randomly the image and the mask in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        h, w = img.size
        new_h, new_w = self.output_size
        bottom = np.random.randint(0,h-new_h)
        top = bottom + new_h
        left = np.random.randint(0,w-new_w)
        right = left + new_w
        img = img.crop((left,bottom,right,top))
        mask = mask.crop((left,bottom,right,top))
        return {'img': img, 'mask': mask}

class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self,sample):
        img, mask = sample['img'], sample['mask']
        h, w = img.size
        h_o, w_o = self.output_size
        resize = transforms.Resize((h_o,w_o))
        img = resize(img)
        mask = resize(mask)
        return {'img': img, 'mask': mask}

class ToTensor(object):
    """Convert PIL images in sample to Tensors."""

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        img_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
        mask_to_tensor = transforms.ToTensor()(mask).unsqueeze_(0)
        return {'img': img_to_tensor,
                'mask': mask_to_tensor}
        

