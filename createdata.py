import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy
import os

def saveimages(tmp):
    filenumb = str(tmp).zfill(3)
    reader = vtkStructuredPointsReader()
    reader.SetFileName("Project course/500%s_fat_content.vtk" %(filenumb) )
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data1 = reader.GetOutput()
    x,y,z = data1.GetDimensions()

    scalars1 = data1.GetPointData().GetScalars()
    array1 = vtk_to_numpy(scalars1)
    mini = array1.min()
    maxi = array1.max()
    tmp = np.array(range(x))
    planeIndex = np.array([])

    if(not os.path.exists('./fat')):
        os.mkdir('./fat')

    for j in range(z):
        planeIndex = np.append(planeIndex,tmp+j*x*y)
    for n in range(80, 100):
        plane = np.zeros(x*z)
        for i in range(x*z):
            plane[i] = array1[int(planeIndex[i])+n*x]
        plane = plane.reshape(x, z)
        matplotlib.image.imsave('./fat/%sdata%d.png' % (filenumb,n), plane, cmap="gray", vmin=mini, vmax=maxi)
        #np.savetxt('./fat/data%d.csv' % (n), plane, delimiter=',')
        #plt.imshow(plane, cmap="gray", alpha=0.7)
        #plt.axis('off')
        #plt.show()


    reader.SetFileName("Project course/binary_liver500%s.vtk" %(filenumb) )
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data2 = reader.GetOutput()
    x,y,z   = data2.GetDimensions()

    scalars2 = data2.GetPointData().GetScalars()
    array2 = vtk_to_numpy(scalars2)

    tmp = np.array(range(x))
    planeIndex = np.array([])

    if(not os.path.exists('./liver')):
        os.mkdir('./liver')

    for j in range(z):
        planeIndex = np.append(planeIndex,tmp+j*x*y)
    for n in range(80, 100):
        plane = np.zeros(x*z)
        for i in range(x*z):
            plane[i] = array2[int(planeIndex[i])+n*x]
        plane = plane.reshape(x, z)
        matplotlib.image.imsave('./liver/%sdata%d.png' % (filenumb,n), plane, cmap="gray")
        #np.savetxt('./liver/data%d.csv' % (n), plane, delimiter=',')
        #plt.imshow(plane, cmap="gray", alpha=0.7)
        #plt.axis('off')
        #plt.show()


for i in range(500):
    if ( os.path.exists("Project course/500%s_fat_content.vtk" %(str(i).zfill(3)) ) ):
        saveimages(i)


