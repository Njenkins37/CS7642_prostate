from loader import IMAGES_ROOT, LABELS_ROOT, load_picai_case
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("tensor_saving.log"),
        logging.StreamHandler()
    ]
)
class PicaiDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.label = labels
        self.transform = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class DatasetUtil:
    def __init__(self):
        self.image = IMAGES_ROOT
        self.label = LABELS_ROOT
        self.case_ids = self.collect_images_ids()
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        return load_picai_case(case_id=case_id, image_root=self.image, label_root=self.label)

    def collect_images_ids(self) -> list:
        """
        Collects images ids to allow the getitem to load images using the load_picai_case

        returns: list - all image ids to allow the get method to isolate by image index
        """

        case_id: list = []
        for case_dir in self.image.iterdir():
            if case_dir.is_dir():
                try:
                    case_id.append(int(case_dir.name))
                except ValueError as e:
                    print(f"Error with {case_id}: Unable to cast case to int value")


        return case_id
    
    def prepocess_and_save(self, output_dir="CS7642_prostate/output"):
        """
        Loads the images and converts the images into a torch tensor and saves it to the 
        parameterized output directors

        params: output_dir - default is CS7642_prostate/output. This path is attached to the home directory
        returns: None. Saves the tensors as a .pt
        """
        start = time.time()
        output_dir = Path.home()/output_dir
        output_dir.mkdir(exist_ok=True)

        for id in self.case_ids:
            out_path = output_dir/f"{id}.pt"

            if out_path.exists():
                continue
            
            try:
                case_data = load_picai_case(case_id=id, image_root=self.image, label_root=self.label)
                tensors = {k: torch.tensor(v) for k, v in case_data.items()}
                torch.save(tensors, out_path)
                logging.info(f"{id} tensor saved to {out_path}")
            except FileNotFoundError as e:
                logging.error(f"{id} not found and not saved")
        end = time.time()
        print(f"Tensor conversion takes {(end - start)/60} minutes")


if __name__ == "__main__":
    # util = DatasetUtil()
    # util.prepocess_and_save()



    data = torch.load("output/10005.pt")

    print(data["lesion_t2"][data["lesion_t2"] > 0.0])

    grade = data["lesion_t2"].int().max().item()
    y = 1.0 if grade >= 2 else 0.0