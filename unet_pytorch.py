from tokenize import Double
import torch
import torch.nn as nn]
import torch.nn.functional as F
import torchvision.transforms as T

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, drop_rate=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if not drop_rate:
            self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=drop_rate),
                nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = nn.double_conv(x)
        return x

class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.max_pool(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.up_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_classes=4, img_height=256, img_width=256, img_channels=1):
        super().__init__()
        # Double Convolutions
        self.double_conv1 = DoubleConv(img_channels, 64, drop_rate=0.1)
        self.double_conv2 = DoubleConv(64, 128, drop_rate=0.1)
        self.double_conv3 = DoubleConv(128, 256, drop_rate=0.2)
        self.double_conv4 = DoubleConv(256, 512, drop_rate=0.2)
        self.double_conv5 = DoubleConv(512, 1024, drop_rate=0.3)
        self.double_conv6 = DoubleConv(1024, 512, drop_rate=0.2)
        self.double_conv7 = DoubleConv(512, 256, drop_rate=0.2)
        self.double_conv8 = DoubleConv(256, 128, drop_rate=0.1)
        self.double_conv9 = DoubleConv(128, 64, drop_rate=0.1)

        # Max Pooling
        self.max_pool = MaxPool()

        # Up Convolutions
        self.up_conv1 = UpConv(1024)
        self.up_conv2 = UpConv(512)
        self.up_conv3 = UpConv(256)
        self.up_conv4 = UpConv(128)

        # Last Convolution
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size = 1)


    def forward(self, input):
        # Contraction path
        x1 = self.double_conv1(input)

        x2 = self.max_pool(x1)
        x2 = self.double_conv2(x2)

        x3 = self.max_pool(x2)
        x3 = self.double_conv3(x3)

        x4 = self.max_pool(x3)
        x4 = self.double_conv4(x4)

        x5 = self.max_pool(x4)
        x5 = self.double_conv5(x5)

        # Expansive path
        x6 = self.up_conv1(x5)
        center_crop = T.CenterCrop(x6.shape[2:])
        x6 = torch.cat([x6, center_crop(x4)])
        x6 = self.double_conv6(x6)

        x7 = self.up_conv1(x6)
        center_crop = T.CenterCrop(x7.shape[2:])
        x7 = torch.cat([x7, center_crop(x3)])
        x7 = self.double_conv7(x7)

        x8 = self.up_conv1(x7)
        center_crop = T.CenterCrop(x8.shape[2:])
        x8 = torch.cat([x8, center_crop(x2)])
        x8 = self.double_conv8(x8)

        x9 = self.up_conv1(x8)
        center_crop = T.CenterCrop(x9.shape[2:])
        x9 = torch.cat([x9, center_crop(x1)])
        x9 = self.double_conv9(x9)
        
        out = self.last_conv(x9)

        return out



        
        