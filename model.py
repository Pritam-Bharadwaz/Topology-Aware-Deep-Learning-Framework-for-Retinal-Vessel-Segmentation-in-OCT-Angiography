"""
==========================================================
Topology-Aware Skeleton U-Net
Project:
Topology-Aware Deep Learning Framework for
Retinal Vessel Segmentation in OCT Angiography

Dataset : OCTA-500
Framework : PyTorch
==========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Double Convolution Block
# ==========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.conv(x)


# ==========================================================
# Encoder Block
# ==========================================================

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Down, self).__init__()

        self.maxpool = nn.MaxPool2d(2)

        self.double_conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x):

        x = self.maxpool(x)

        x = self.double_conv(x)

        return x


# ==========================================================
# Decoder Block
# ==========================================================

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Up, self).__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2
            ]
        )

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x


# ==========================================================
# Output Layer
# ==========================================================

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        return self.conv(x)
     

# ==========================================================
# Skeleton-Aware U-Net
# ==========================================================

class SkeletonAwareUNet(nn.Module):

    def __init__(self):

        super(SkeletonAwareUNet, self).__init__()

           # Encoder
        self.inc = DoubleConv(1, 32)

        self.down1 = Down(32, 64)

        self.down2 = Down(64, 128)

        self.down3 = Down(128, 256)

        self.down4 = Down(256, 512)

        # Dropout at Bottleneck
        self.dropout = nn.Dropout2d(0.5)

        # Decoder
        self.up1 = Up(512 + 256, 256)

        self.up2 = Up(256 + 128, 128)

        self.up3 = Up(128 + 64, 64)

        self.up4 = Up(64 + 32, 32)

        # Output
        self.outc = OutConv(32, 1)

        self.activation = nn.Sigmoid()

    def forward(self, x):

        # ---------------- Encoder ---------------- #

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x5 = self.dropout(x5)

        # ---------------- Decoder ---------------- #

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        # ---------------- Output ---------------- #

        output = self.outc(x)

        output = self.activation(output)

        return output
    

# ==========================================================
# Test Model
# ==========================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing Skeleton-Aware U-Net")
    print("=" * 50)

    # Create a random OCTA image
    x = torch.randn(1, 1, 304, 304)

    # Create model
    model = SkeletonAwareUNet()

    # Forward pass
    y = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)

    # Count trainable parameters
    total_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable Parameters: {total_params:,}")

    print("=" * 50)
    print("Model Test Completed Successfully")
    print("=" * 50)