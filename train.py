"""
==========================================================
Training Script
==========================================================
"""

import torch

from torch.utils.data import DataLoader

from torch.optim import Adam

from tqdm import tqdm

from dataset import OCTADataset

from model import SkeletonAwareUNet

from losses import TotalLoss

from utils import save_checkpoint

from config import (
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    TRAIN_SKELETON_DIR,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    VAL_SKELETON_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE
)


# ==========================================================
# Dataset
# ==========================================================

train_dataset = OCTADataset(
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    TRAIN_SKELETON_DIR
)

val_dataset = OCTADataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    VAL_SKELETON_DIR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))


# ==========================================================
# Model
# ==========================================================

model = SkeletonAwareUNet().to(DEVICE)

criterion = TotalLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Model Loaded Successfully")



# ==========================================================
# Training Function
# ==========================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels, skeletons in progress_bar:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        skeletons = skeletons.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
            skeletons
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    return running_loss / len(train_loader)


# ==========================================================
# Validation Function
# ==========================================================

def validate():

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, labels, skeletons in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            skeletons = skeletons.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
                skeletons
            )

            running_loss += loss.item()

    return running_loss / len(val_loader)



# ==========================================================
# Main
# ==========================================================

best_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    train_loss = train_one_epoch()

    val_loss = validate()

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")

    if val_loss < best_loss:

        best_loss = val_loss

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_loss,
            "Checkpoints/best_model.pth"
        )

        print("Best model saved!")