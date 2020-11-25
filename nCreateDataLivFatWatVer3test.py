import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

#landmarks
maps_list = [None]*50lsave
for k in range(1,51):
	file = open(r"ourLandmarks/landmark"+ str(k) + ".txt")
	X1 = int(file.readline())
	Y1 = int(file.readline())
	X2 = int(file.readline())
	Y2 = int(file.readline())
	file.close()

	distance_map = np.zeros((256,256))
	distance_map1 = np.zeros((256,256))
	distance_map2 = np.zeros((256,256))
	#Calculate manhattan distance from each landmark coordinate to every element.
	for i in range(256):
		for j in range(256):
			distance_map1[i,j] = abs(i-X1) + abs(j-Y1)
			distance_map2[i,j] = abs(i-X2) + abs(j-Y2)

	distance_map1 = distance_map1 + 1
	distance_map1 = 1/distance_map1

	distance_map2 = distance_map2 + 1
	distance_map2 = 1/distance_map2

	distance_map = distance_map1 + distance_map2
	distance_map = distance_map/distance_map[X1,Y1]
	print(distance_map[X1,Y1])
	print(distance_map[X2,Y2])
	maps_list[k-1] = distance_map




# Corresponding image number to each distance map in maps_list
nlandmark = np.array([17,18,22,26,51,53,56,61,62,75,77,86,117,124,158,159,167,179,204,205,235,241,242,253,268,280,281,288,291,297,304,316,318,321,327,346,347,348,354,357,358,379,395,403,406,424,429,433,473,487])

def saveimages(fn):
    filenumb = str(fn).zfill(3)
    reader = vtkStructuredPointsReader()
    reader.SetFileName("../../Project course/500%s_fat_content.vtk" %(filenumb) )
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
    
    reader.SetFileName("../../Project course/500%s_wat_content.vtk" %(filenumb) )
    reader.Update()
    array_wat = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
#     min_wat = array_wat.min()
#     max_wat = array_wat.max()
#     print('min_wat: %f' %(min_wat))
#     print('max_wat: %f' %(max_wat))
    min_wat = 0
    max_wat = 1
    
    reader.SetFileName("../../Project course/binary_liver500%s.vtk" %(filenumb) )
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
        plane_all = np.zeros(x*z*4).reshape(x, z, 4)
        for i in range(x*z):
            plane_fat[i] = array_fat[int(planeIndex[i])+n*x]
            plane_wat[i] = array_wat[int(planeIndex[i])+n*x]
            plane_liver[i] = array_liver[int(planeIndex[i])+n*x]
        plane_fat = plane_fat.reshape(x, z)
        plane_wat = plane_wat.reshape(x, z)
        plane_liver = plane_liver.reshape(x, z)
        if(plane_liver.max() > 0):
            plane_all[:,:,0] = (plane_fat+0.2)/1.4
            plane_all[:,:,1] = plane_wat
            plane_all[:,:,2] = np.asarray(maps_list[(np.where(nlandmark == fn))[0][0]])
            plane_all[:,:,3] = plane_liver
            #np.save('./data/%sdata%d' % (filenumb,n), plane_all)
            matplotlib.image.imsave('./data/images/%sdata%d.png' % (filenumb,n), plane_all)
            #matplotlib.image.imsave('./data/data/%sdata%d.png' % (filenumb,n), plane_all)
#             matplotlib.image.imsave('./fat/%sdata%d.png' % (filenumb,n), plane_fat, cmap="gray", vmin=min_fat, vmax=max_fat)
#             matplotlib.image.imsave('./water/%sdata%d.png' % (filenumb,n), plane_wat, cmap="gray", vmin=min_wat, vmax=max_wat)
#             matplotlib.image.imsave('./liver/%sdata%d.png' % (filenumb,n), plane_liver, cmap="gray")
    
for i in range(500):
    if ( os.path.exists("../../Project course/500%s_fat_content.vtk" %(str(i).zfill(3)) ) ):
        saveimages(i)
