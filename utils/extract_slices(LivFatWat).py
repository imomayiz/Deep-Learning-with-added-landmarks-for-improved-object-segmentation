#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import matplotlib
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy
import os
from PIL import Image

def saveimages(tmp):
    filenumb = str(tmp).zfill(3)
    reader = vtkStructuredPointsReader()
    reader.SetFileName("Project course/500%s_fat_content.vtk" %(filenumb) )
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    x,y,z = reader.GetOutput().GetDimensions()
    array_fat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
#     min_fat = array_fat.min()
#     max_fat = array_fat.max()
#     print('min_fat: %f' %(min_fat))
#     print('max_fat: %f' %(max_fat))
    min_fat = 0
    max_fat = 1.2
    
    reader.SetFileName("Project course/500%s_wat_content.vtk" %(filenumb) )
    reader.Update()
    array_wat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
#     min_wat = array_wat.min()
#     max_wat = array_wat.max()
#     print('min_wat: %f' %(min_wat))
#     print('max_wat: %f' %(max_wat))
    min_wat = 0
    max_wat = 1
    
    reader.SetFileName("Project course/binary_liver500%s.vtk" %(filenumb) )
    reader.Update()
    array_liver = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
  
    tmp = np.array(range(x))
    planeIndex = np.array([])
    for j in range(z):
        planeIndex = np.append(planeIndex,tmp+j*x*y)
        
#     if(not os.path.exists('./fat')):
#         os.mkdir('./fat')
#     if(not os.path.exists('./liver')):
#         os.mkdir('./liver')
#     if(not os.path.exists('./water')):
#         os.mkdir('./water')
    if(not os.path.exists('./data')):
        os.mkdir('./data')
    
    for n in range(50, 150):
        plane_fat = np.zeros(x*z)
        plane_wat = np.zeros(x*z)
        plane_liver = np.zeros(x*z)
        plane_all = np.zeros(x*z*3).reshape(x, z, 3)
        for i in range(x*z):
            plane_fat[i] = array_fat[int(planeIndex[i])+n*x]
            plane_wat[i] = array_wat[int(planeIndex[i])+n*x]
            plane_liver[i] = array_liver[int(planeIndex[i])+n*x]
        plane_fat = plane_fat.reshape(x, z)
        plane_wat = plane_wat.reshape(x, z)
        plane_liver = plane_liver.reshape(x, z)
        if(plane_liver.max() > 0):
            plane_all[:,:,0] = plane_fat
            plane_all[:,:,1] = plane_wat
            plane_all[:,:,2] = plane_liver
            matplotlib.image.imsave('./data/%sdata%d.png' % (filenumb,n), plane_all,)
#             matplotlib.image.imsave('./fat/%sdata%d.png' % (filenumb,n), plane_fat, cmap="gray", vmin=min_fat, vmax=max_fat)
#             matplotlib.image.imsave('./water/%sdata%d.png' % (filenumb,n), plane_wat, cmap="gray", vmin=min_wat, vmax=max_wat)
#             matplotlib.image.imsave('./liver/%sdata%d.png' % (filenumb,n), plane_liver, cmap="gray")

for i in range(500):
    if ( os.path.exists("Project course/500%s_fat_content.vtk" %(str(i).zfill(3)) ) ):
        saveimages(i)

