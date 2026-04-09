"""Unet sample model"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    The buidling block of the convulitional layers for the layers of a UNet
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward method of ConvBlock

        params:
                x: tensor
        returns
                output of a conv_block
        """
        return self.conv_block(x)


class UNet(nn.Module):
    """
    Main model class that is composed of the conv blocks.
    """

    def __init__(self, n_slices=3, n_classes=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder1 = ConvBlock(in_channels=n_slices, out_channels=features[0])
        self.encoder2 = ConvBlock(in_channels=features[0], out_channels=features[1])
        self.encoder3 = ConvBlock(in_channels=features[1], out_channels=features[2])
        self.encoder4 = ConvBlock(in_channels=features[2], out_channels=features[3])

        self.bottleneck = ConvBlock(
            in_channels=features[3], out_channels=features[3] * 2
        )

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)
        self.dec4 = ConvBlock(in_channels=features[3] * 2, out_channels=features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])

        self.head = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        """
        The forward pass method of the Unet
        """
        s1 = self.encoder1(x)
        s2 = self.encoder2(self.pool(s1))
        s3 = self.encoder3(self.pool(s2))
        s4 = self.encoder4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.head(d1)
