import glob
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torch
from torch import Tensor
from jaxtyping import Float32, UInt8
from lightning import LightningDataModule
from torchvision.transforms.v2 import ToDtype, ToImage, Compose, Transform

class TorchvisionCPUClothingDataset(Dataset):
    def __init__(self, root_dir: str | Path, augmentations: Transform | None) -> None:
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.img_files = glob.glob(str(Path(root_dir).resolve() / "images" / "*.jpg"))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Float32[Tensor, "3 h w"]:
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = Compose([ToImage(), ToDtype(torch.float32)])(image)
        if self.augmentations:
            image = self.augmentations(image)
        return image
    
class TorchvisionGPUClothingDataset(TorchvisionCPUClothingDataset):
    def __init__(self, root_dir: str | Path, augmentations: Transform | None, device: torch.device) -> None:
        super().__init__(root_dir, augmentations)
        self.device = device

    def __getitem__(self, idx: int) -> Float32[Tensor, "3 h w"]:
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = Compose([ToImage(), ToDtype(torch.float32)])(image).to(device=self.device)
        if self.augmentations:
            image = self.augmentations(image)
        return image
    
if __name__ == "__main__":
    from torchvision.transforms.v2 import (
        RandomHorizontalFlip, 
        RandomResizedCrop, 
        Normalize
    )

    augmentations = Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
        RandomHorizontalFlip(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cpu_dataset = TorchvisionCPUClothingDataset(
        root_dir="clothing-dataset", 
        augmentations=augmentations
    )

    gpu_dataset = TorchvisionGPUClothingDataset(
        root_dir="clothing-dataset", 
        augmentations=augmentations,
        device=torch.device("mps")
    )

    # Test the datasets
    cpu_image = cpu_dataset[0]
    gpu_image = gpu_dataset[0]
    print(f"CPU image shape: {cpu_image.shape}")
    print(f"GPU image shape: {gpu_image.shape}")