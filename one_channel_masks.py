import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import cv2
import glob

def one_channel(img):
   
 
    if(not os.path.exists('./liver3')):
        os.mkdir('./liver3')

    ii = cv2.imread(img)
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./liver3/"+img.split('/')[-1], gray_image)
i=0
for name in glob.glob("./code/Unet/data_jpg/masks/*.jpg"):
    one_channel(name)
    i+=1
print("number of images: %i" %(i))

