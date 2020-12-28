import torch
import torch.nn as nn
import torch.nn.functional as F



## Unet parts

class DoubleConv(nn.Module):
    """[Conv2D + BatchNorm2D + Relu]*2"""
    def __init__(self,C_in,C_out,C_mid=None):
        super().__init__()
        if C_mid==None:
            C_mid = C_out
            self.double_conv = nn.Sequential(
                nn.Conv2d(C_in,C_mid, kernel_size=3,padding=1),
                nn.BatchNorm2d(C_mid),
		nn.ReLU(inplace=True),
		nn.Conv2d(C_mid,C_out, kernel_size=3,padding=1),
		nn.BatchNorm2d(C_out),
		nn.ReLU(inplace=True)
		)
    def forward(self,x):
        return(self.double_conv(x))

class Down(nn.Module):
    """maxpool + Doubleconv"""
    def __init__(self,C_in,C_out):
        super().__init__()
        self.down = nn.Sequential(
        nn.MaxPool2d(kernel_size=2),
	DoubleConv(C_in,C_out)
	)
    def forward(self,x):
        return(self.down(x))


class Up(nn.Module):
    """upconv + Doubleconv([cropped_features,x]) """
    def __init__(self,C_in,C_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(C_in, C_in//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(C_in,C_out)
    def forward(self,x,x2):
        x1 = self.up(x)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        in_conv = torch.cat([x2,x1],dim=1)
        return self.conv(in_conv)


class Out(nn.Module):
    def __init__(self,C_in,C_out):
        super().__init__()
        self.out = nn.Conv2d(C_in,C_out,kernel_size=1)
        self.soft = nn.LogSoftmax(dim=1)
    def forward(self,x):
        output = self.out(x)
        return output

				
## Combine Unet parts and feed forward

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes #useful attributes
	## layers of contracting path
        self.double_conv = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)
        ##layers of expansive path
        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)	
	##last layer
        self.out = Out(64,n_classes)

    def forward(self,x):
        x1 = self.double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        #skip-connections
        x = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        #output segmentation map
        output = self.out(x)
        return output
								 
				

				
						

