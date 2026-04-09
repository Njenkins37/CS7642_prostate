import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import torchvision.transforms.functional as TF
import sys
sys.path.append('/Users/nickjenkins/CS7642_prostate')
from models import UNet
import os
import glob
import time

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )


class MRIDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.slices = []
        self.cache = []

        for file_idx, file in enumerate(self.files):
            data = torch.load(file)
            self.cache.append(data)
            for slice_idx in range(data["t2"].shape[-1]):
                self.slices.append((file_idx, slice_idx))

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slices[idx]
        data = self.cache[file_idx]
        image = data['t2'][..., slice_idx].float().unsqueeze(0)
        mask = (data['lesion_t2'][..., slice_idx].float() > 2).float().unsqueeze(0)
    

        image = TF.resize(image, [128, 128])
        mask = TF.resize(mask, [128, 128], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask
    
    def __len__(self):
        return len(self.slices)


def reg_unet():
    start = time.time()
    dataset = MRIDataset("../output")
    end = time.time()
    print(f"Dataset takes {end - start} seconds to load")

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)


    model = UNet(n_slices=1, n_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    start = time.time()
    for epoch in range(10):
        
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()


        end = time.time()
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Time: {round((end - start) / 60, 2)} Minutes")
        


if __name__ == "__main__":
    reg_unet()
    # data = torch.load("../output/10000.pt")
    # print(data['t2'].shape)
    # print(data['lesion_t2'].shape)