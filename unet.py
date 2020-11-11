import torch
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def double_conv(in_c, out_c):
	conv = nn.Sequential(
		nn.Conv2d(in_c, out_c, kernel_size=3),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_c, out_c, kernel_size=3),
		nn.ReLU(inplace=True),
	)
	return conv

def crop_img(tensor, target_tensor):
	target_size = target_tensor.size()[2]
	tensor_size = tensor.size()[2]
	delta = tensor_size - target_size
	delta = delta // 2
	return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()

		self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.down_conv_1 = double_conv(1, 64)
		self.down_conv_2 = double_conv(64, 128)
		self.down_conv_3 = double_conv(128, 256)
		self.down_conv_4 = double_conv(256, 512)
		self.down_conv_5 = double_conv(512, 1024)

		self.up_trans_1 = nn.ConvTranspose2d(
			in_channels=1024,
			out_channels=512,
			kernel_size=2,
			stride=2)

		self.up_conv_1 = double_conv(1024, 512)

		self.up_trans_2 = nn.ConvTranspose2d(
			in_channels=512,
			out_channels=256,
			kernel_size=2,
			stride=2)

		self.up_conv_2 = double_conv(512, 256)

		self.up_trans_3 = nn.ConvTranspose2d(
			in_channels=256,
			out_channels=128,
			kernel_size=2,
			stride=2)

		self.up_conv_3 = double_conv(256, 128)

		self.up_trans_4 = nn.ConvTranspose2d(
			in_channels=128,
			out_channels=64,
			kernel_size=2,
			stride=2)

		self.up_conv_4 = double_conv(128, 64)

		self.out = nn.Conv2d(
			in_channels = 64,
			out_channels=1,
			kernel_size=1
		)

	def forward(self, image):
		# bs, c, h, w
		# encoder
		x1 = self.down_conv_1(image) #
		x2 = self.max_pool_2x2(x1)
		x3 = self.down_conv_2(x2) #
		x4 = self.max_pool_2x2(x3)
		x5 = self.down_conv_3(x4) #
		x6 = self.max_pool_2x2(x5)
		x7 = self.down_conv_4(x6) #
		x8 = self.max_pool_2x2(x7)
		x9 = self.down_conv_5(x8)

		# decoder
		x = self.up_trans_1(x9)
		y = crop_img(x7, x)
		x = self.up_conv_1(torch.cat([x, y], 1))

		x = self.up_trans_2(x)
		y = crop_img(x5, x)
		x = self.up_conv_2(torch.cat([x, y], 1))

		x = self.up_trans_3(x)
		y = crop_img(x3, x)
		x = self.up_conv_3(torch.cat([x, y], 1))

		x = self.up_trans_4(x)
		y = crop_img(x1, x)
		x = self.up_conv_4(torch.cat([x, y], 1))

		x = self.out(x)
		return x
		
if __name__ == "__main__":

	# define machine learning functions
	model = UNet()
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

	# load data
	dataset = torchvision.datasets.ImageFolder(root='data', transform=transforms.ToTensor())
	x_train = dataset[235][0][0,:,:]
	x_train = x_train.unsqueeze(0).unsqueeze(0)
	matplotlib.image.imsave('./train.jpg',dataset[235][0][0,:,:],cmap='gray')
	matplotlib.image.imsave('./mask.jpg',dataset[235+251][0][0,:,:],cmap='gray')	

	# train model
	for t in range(100):
		y_pred = model(x_train)
		y_train = crop_img(dataset[251+235][0][0,:,:].unsqueeze(0).unsqueeze(0),y_pred)

		loss = loss_fn(y_pred,y_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss)

		matplotlib.image.imsave(f'./outputTrain{t}.jpg',y_pred.detach().numpy()[0,0,:,:],cmap='gray')








