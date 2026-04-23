"""The base training file for the 2D UNet"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import sys

sys.path.append("/Users/nickjenkins/CS7642_prostate")
from models import UNet
import os
import glob
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Selecting device in heirarchy of speed
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class MRIDataset(Dataset):
    """
    MRIDataset class extends the Dataset class from Pytorch to load the
    MRI files into a tensor to begin training
    """

    def __init__(self, file_list):
        """Takes a list of .pt file paths instead of a directory."""
        self.slices: list = []
        self.cache: dict = {}
        for file_idx, file in enumerate(file_list):
            data = torch.load(file)
            self.cache[file_idx] = data
            for slice_idx in range(data["t2"].shape[-1]):
                mask = data["lesion_t2"][..., slice_idx].float()
                if (mask > 2).any():
                    self.slices.append((file_idx, slice_idx))

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Isolates the image and the mask based on the keys saved in the .pt file
        Downsamples to a 128x128 image due to variability in file size and speed considerations

        params:
                idx: int - the index of the file that is being processed
        returns:
                image: torch.tensor - the 128x128 image converted to torch tensors
                mask: torch.tensor - the 128x128 mask converted to torch tensors
        """
        file_idx, slice_idx = self.slices[idx]
        data = self.cache[file_idx]
        image = data["t2"][..., slice_idx].float().unsqueeze(0)
        mask = (data["lesion_t2"][..., slice_idx].float() > 2).float().unsqueeze(0)
        image = TF.resize(image, [128, 128])
        mask = TF.resize(mask, [128, 128], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def __len__(self):
        return len(self.slices)


def dice_score(preds, targets):
    """
    Dice on binary predictions from raw logits.

    params:
            preds: torch.tensor - the predictions from the model on the logits
            targets: torch.tensor - the ground truth labels
    returns:
            dice: float - the calcualted dice measure
    """
    binary_preds = (preds > 0.0).float()
    intersection = (binary_preds * targets).sum(dim=(1, 2, 3))
    union = binary_preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    has_lesion = targets.sum(dim=(1, 2, 3)) > 0
    if has_lesion.sum() == 0:
        return float("nan")
    dice = (2 * intersection[has_lesion] + 1e-6) / (union[has_lesion] + 1e-6)
    return dice.mean().item()


def dice_loss(preds: torch.tensor, targets: torch.tensor, eps: float = 1e-6) -> float:
    """
    Dice loss metric based on Dice Soreson metric pushed through the
    SoftMax function

    params:
            preds: torch.tensor - predictions from the model
            targets: torch.tensor - ground truth labels
            eps: float - prevent divide by 0 errors
    returns:
            dice: float - the calculated probabilities from the dice score

    """
    probs = torch.sigmoid(preds)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def precision_score(preds: torch.tensor, targets: torch.tensor) -> float:
    """
    Calculates the precision score (TP / (TP + FN)) for pixel mask. Used in printing
    during training to assess performance in addition to the Dice metric

    params:
            preds: torch.tensor - predictions from the model
            targets: torch.tensor - ground truth labels
    returns:
            precision: float - the calculated precision score
    """
    has_lesion = targets.sum(dim=(1, 2, 3)) > 0
    if has_lesion.sum() == 0:
        return float("nan")
    binary_preds = (preds > 0).float()
    tp = (binary_preds * targets).sum(dim=(1, 2, 3))
    predicted_positive = binary_preds.sum(dim=(1, 2, 3))
    # 1e-6 used to prevent divide by 0 errors
    precision = tp[has_lesion] / (predicted_positive[has_lesion] + 1e-6)
    return precision.mean().item()


def recall_score(preds: torch.tensor, targets: torch.tensor)-> float:
    """
    Calculates the recall score based on the parameters. Used for monitoring performance
    and not passed into loss functions.

    params:
            preds: torch.tensor - the predictions from the model
            targets: torch.tesnor - the ground truth labels
    returns:
            recall: float - the recall score of the parameters
    """
    has_lesion = targets.sum(dim=(1, 2, 3)) > 0
    if has_lesion.sum() == 0:
        return float("nan")

    binary_preds = (preds > 0).float()
    tp = (binary_preds * targets).sum(dim=(1, 2, 3))
    actual_positive = targets.sum(dim=(1, 2, 3))
    # 1e-6 used to prevent divide by 0 errors
    recall = tp[has_lesion] / (actual_positive[has_lesion] + 1e-6)
    return recall.mean().item()


def reg_unet():
    """
    The main training function for the 2D UNet. Saves the model to the current
    working directory to load for other functions. 

    params: 
            None
    returns:
            train_dice_list, val_dice_list, train_losses, val_losses - lists containing their respective measures

    """
    all_files:list = glob.glob(
        os.path.join("../output", "*.pt")
    )  # Necessary for file imports. Could be improved
    # Uncomment for debug runs
    # all_files = all_files[100:700]
    split_idx:int = int(0.7 * len(all_files))
    train_files:list = all_files[:split_idx]
    val_files:list = all_files[split_idx:]
    print(f"Train patients: {len(train_files)} | Val patients: {len(val_files)}")
    train_dice_list:list = []
    val_dice_list:list = []
    val_losses:list = []
    train_losses:list = []

    start:float = time.time()
    train_dataset:MRIDataset = MRIDataset(train_files)
    val_dataset:MRIDataset = MRIDataset(val_files)
    print(
        f"Datasets loaded in {time.time() - start:.1f}s | "
        f"Train slices: {len(train_dataset)} | Val slices: {len(val_dataset)}"
    )

    train_loader:DataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader:DataLoader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    model:UNet = UNet(n_slices=1, n_classes=1).to(device)

    #### The optimizer section. Opportunity for tuning and improving performance #####
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99),
    )
    focal = FocalLoss(alpha=0.75, gamma=2.0)

    for epoch in range(30):
        ############### Training ################
        model.train()
        train_loss, train_dice_total, train_correct, train_total = 0.0, 0.0, 0, 0
        train_prec, train_recall = 0.0, 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)

            loss = focal.forward(preds, masks) + dice_loss(preds, masks)            
            loss.backward()
            optimizer.step()
            train_dice_score = dice_score(preds, masks)
            train_loss += loss.item()
            train_dice_total += train_dice_score
            train_prec += precision_score(preds, masks)
            train_recall += recall_score(preds, masks)
            binary_preds = (preds > 0.0).float()
            train_correct += (binary_preds == masks).sum().item()
            train_total += masks.numel()

        ######### Validation Section ############
        model.eval()
        val_loss, val_dice_total, val_correct, val_total = 0.0, 0.0, 0, 0
        val_recall, val_precision = 0.0, 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = focal.forward(outputs, masks) + dice_loss(outputs, masks)
                val_dice_score = dice_score(outputs, masks)
                val_dice_total += val_dice_score
                val_recall += recall_score(outputs, masks)
                val_precision += precision_score(outputs, masks)
                binary_preds = (outputs > 0.0).float()
                val_correct += (binary_preds == masks).sum().item()
                val_total += masks.numel()

        n_train = len(train_loader)
        n_val = len(val_loader)

        train_dice_list.append(train_dice_total / n_train)
        val_dice_list.append(val_dice_total / n_val)
        train_losses.append(train_correct / train_total)
        val_losses.append(val_correct / val_total)
        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss/n_train:.4f} | Train Dice: {train_dice_total/n_train:.4f} | Train Acc: {train_correct/train_total:.4f} | \n"
            f"Train Precision: {train_prec/n_train:.4f} | Train Recall: {train_recall/n_train} |\n"
            f"Val Loss: {val_loss/n_val:.4f} | Val Dice: {val_dice_total/n_val:.4f} | Val Acc: {val_correct/val_total:.4f}|\n"
            f"Val Precision: {val_precision/n_val:.4f} | Val Recall: {val_recall/n_val:.4f}\n"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_dice": train_dice_list,
                "val_dice": val_dice_list,
            },
            "unet.pth",
        )
    return train_dice_list, val_dice_list, train_losses, val_losses


class FocalLoss:
    def __init__(self, alpha: float = 0.75, gamma=0.25):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        probs = torch.sigmoid(preds)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce_loss).mean()


def plot():
    """
    Calls the reg_unet() function to train a model. The return values from the
    reg_unet function are plotted to show loss, accuracies, or dice score for the 
    training and validation set.

    params:
            None
    returns:
            None - Shows the figure and saves it to the working directory
    """
    train_list, val_list, train_losses, val_losses = reg_unet()

    plt.plot(train_list, label="Train Dice")
    plt.plot(val_list, label="Val Dice")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Scoring Metric")
    plt.title("Training versus Validation Dice")
    plt.legend()
    plt.savefig("train_vs_val_dice.png")
    plt.show()

    plt.plot(train_losses, label="Train Accuracy")
    plt.plot(val_losses, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training versus Validation Accuracy")
    plt.legend()
    plt.savefig("train_vs_val_accuracy.png")
    plt.show()


def prob_map(idx:int=5) -> None:
    """
    Plots a probability heat map based on the passed in index paramter.
    Loads the saved model from the reg_unet() function instead of training a 
    new model to plot different cases.

    params:
            idx: int - the index of the file to be plotted
    returns:
            None - plots and saves the figure
    """
    all_files = sorted(glob.glob(os.path.join("../output", "*.pt")))
    split_idx = int(0.7 * len(all_files))
    val_files = all_files[split_idx:]

    val_dataset = MRIDataset(val_files)

    checkpoint = torch.load("unet.pth", map_location=device)
    model = UNet(n_slices=1, n_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image, mask = val_dataset[idx]

    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
        prob = torch.sigmoid(pred).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image.squeeze().numpy(), cmap="gray")
    axes[0].set_title("T2 Input")

    axes[1].imshow(mask.squeeze().numpy(), cmap="hot")
    axes[1].set_title("Ground Truth Mask")

    im = axes[2].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted Probability")
    plt.colorbar(im, ax=axes[2])

    plt.savefig("probability_heatmap.png", dpi=150)
    plt.show()


if __name__ == "__main__":
  
    # reg_unet()
    # plot()
    prob_map()
    # visualize_feature_maps()
