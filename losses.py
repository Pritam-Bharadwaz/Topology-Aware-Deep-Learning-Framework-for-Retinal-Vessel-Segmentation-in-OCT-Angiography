"""
==========================================================
Loss Functions for Skeleton-Aware U-Net
==========================================================
"""

import torch
import torch.nn as nn


# ==========================================================
# Dice Loss
# ==========================================================

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target):

        smooth = 1e-6

        prediction = prediction.view(-1)
        target = target.view(-1)

        intersection = (prediction * target).sum()

        dice = (
            (2 * intersection + smooth)
            /
            (prediction.sum() + target.sum() + smooth)
        )

        return 1 - dice


# ==========================================================
# Skeleton Loss
# ==========================================================

class SkeletonLoss(nn.Module):

    def __init__(self):
        super(SkeletonLoss, self).__init__()

        self.l1 = nn.L1Loss()

    def forward(self, prediction, skeleton):

        return self.l1(prediction, skeleton)


# ==========================================================
# Combined Loss
# ==========================================================

class TotalLoss(nn.Module):

    def __init__(self):

        super(TotalLoss, self).__init__()

        self.bce = nn.BCELoss()

        self.dice = DiceLoss()

        self.skeleton = SkeletonLoss()

    def forward(
        self,
        prediction,
        target,
        skeleton
    ):

        bce_loss = self.bce(
            prediction,
            target
        )

        dice_loss = self.dice(
            prediction,
            target
        )

        skeleton_loss = self.skeleton(
            prediction,
            skeleton
        )

        total = (
            0.4 * bce_loss
            +
            0.4 * dice_loss
            +
            0.2 * skeleton_loss
        )

        return total



# ==========================================================
# Test Loss
# ==========================================================

if __name__ == "__main__":

    prediction = torch.rand(
        2,
        1,
        304,
        304
    )

    target = torch.rand(
        2,
        1,
        304,
        304
    )

    skeleton = torch.rand(
        2,
        1,
        304,
        304
    )

    criterion = TotalLoss()

    loss = criterion(
        prediction,
        target,
        skeleton
    )

    print("=" * 50)
    print("Total Loss :", loss.item())
    print("=" * 50)