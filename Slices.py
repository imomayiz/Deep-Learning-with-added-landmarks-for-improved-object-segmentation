import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy
import os

# Plot the data
reader = vtkStructuredPointsReader()
reader.SetFileName("500017_fat_content.vtk")
#eader.SetFileName("binary_liver500017.vtk")
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()

data = reader.GetOutput()
scalars = data.GetPointData().GetScalars()
array = vtk_to_numpy(scalars)
x,y,z   = data.GetDimensions()
xs = []
ys = []
zs = []

for i in range(0, x*y*z)[0::50]:
    if (array[i]>0.5): 
        xs.append(data.GetPoint(i)[0])
        ys.append(data.GetPoint(i)[1])
        zs.append(data.GetPoint(i)[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
lims = data.GetPoint(x*y*z-1)
ax.set_xlim3d(0, lims[0])
ax.set_ylim3d(0, lims[1])
ax.set_zlim3d(0, lims[2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.view_init(90, 90)
plt.show()

# Save 20 slices of data and true segmentation of liver
reader = vtkStructuredPointsReader()
reader.SetFileName("500017_fat_content.vtk")
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data2 = reader.GetOutput()
x,y,z   = data2.GetDimensions()

scalars2 = data2.GetPointData().GetScalars()
array2 = vtk_to_numpy(scalars2)

tmp = np.array(range(x))
planeIndex = np.array([])

if(not os.path.exists('./fat')):
    os.mkdir('./fat')

for j in range(z):
    planeIndex = np.append(planeIndex,tmp+j*x*y)
for n in range(80, 100):
    plane = np.zeros(x*z)
    for i in range(x*z):
        plane[i] = array2[int(planeIndex[i])+n*x]
    plane = plane.reshape(x, z)
    np.savetxt('./fat/data%d.csv' % (n), plane, delimiter=',')    
    
reader = vtkStructuredPointsReader()
reader.SetFileName("binary_liver500017.vtk")
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
    np.savetxt('./liver/data%d.csv' % (n), plane, delimiter=',')

# Plot one slice with masked liver from true segmentation
plane1 = np.loadtxt('./fat/data%d.csv' % (90), delimiter=',')
plane2 = np.loadtxt('./liver/data%d.csv' % (90), delimiter=',')
plane2 = np.ma.masked_where(plane2 < 0.9, plane2)
im1 = plt.imshow(plane1, cmap="gray")
im2 = plt.imshow(plane2, cmap="brg", alpha=0.7)
plt.axis('off')
plt.show()

