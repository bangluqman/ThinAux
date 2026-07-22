import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)
        self.c4 = ConvBlock(256, 512)
        self.bridge = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.d4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.pool(x1))
        x3 = self.c3(self.pool(x2))
        x4 = self.c4(self.pool(x3))
        x = self.bridge(self.pool(x4))
        x = self.d4(torch.cat([self.up4(x), x4], 1))
        x = self.d3(torch.cat([self.up3(x), x3], 1))
        x = self.d2(torch.cat([self.up2(x), x2], 1))
        x = self.d1(torch.cat([self.up1(x), x1], 1))
        return self.out(x)
