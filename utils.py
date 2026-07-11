"""
==========================================================
Utility Functions
==========================================================
"""

import os
import random
import torch
import numpy as np


# ==========================================================
# Set Random Seed
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# Save Model
# ==========================================================

def save_checkpoint(model, optimizer, epoch, loss, filename):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "loss": loss

    }

    torch.save(checkpoint, filename)

    print(f"Model saved -> {filename}")


# ==========================================================
# Dice Score
# ==========================================================

def dice_score(prediction, target):

    smooth = 1e-6

    prediction = (prediction > 0.5).float()

    prediction = prediction.view(-1)

    target = target.view(-1)

    intersection = (prediction * target).sum()

    score = (

        (2 * intersection + smooth)

        /

        (prediction.sum() + target.sum() + smooth)

    )

    return score.item()



# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    set_seed()

    pred = torch.rand(1,1,304,304)

    gt = torch.rand(1,1,304,304)

    score = dice_score(pred, gt)

    print("="*40)

    print("Dice Score :", score)

    print("="*40)