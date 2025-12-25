"""
Dataset management module for Image Retrieval System
Supports downloading and processing CIFAR-10 and other datasets
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from config import DATA_DIR, IMAGES_DIR, DATASET_NAME, MAX_IMAGES


class ImageDataset:
    """Manages image dataset for retrieval system"""
    
    def __init__(self, dataset_name: str = DATASET_NAME, max_images: int = MAX_IMAGES):
        self.dataset_name = dataset_name
        self.max_images = max_images
        self.images_dir = IMAGES_DIR / dataset_name
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        
    def download_and_prepare(self) -> None:
        """Download dataset and save as individual images"""
        if self.dataset_name == "cifar10":
            self._prepare_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
    def _prepare_cifar10(self) -> None:
        """Download and prepare CIFAR-10 dataset"""
        print("Downloading CIFAR-10 dataset...")
        
        # Download using torchvision
        train_dataset = datasets.CIFAR10(
            root=str(DATA_DIR / "raw"),
            train=True,
            download=True
        )
        test_dataset = datasets.CIFAR10(
            root=str(DATA_DIR / "raw"),
            train=False,
            download=True
        )
        
        self.class_names = train_dataset.classes
        
        # Combine train and test
        all_images = list(train_dataset) + list(test_dataset)
        
        # Limit to max_images
        all_images = all_images[:self.max_images]
        
        print(f"Saving {len(all_images)} images...")
        
        for idx, (image, label) in enumerate(tqdm(all_images)):
            # Save image
            class_dir = self.images_dir / self.class_names[label]
            class_dir.mkdir(exist_ok=True)
            
            image_path = class_dir / f"{idx:06d}.png"
            image.save(image_path)
            
            self.image_paths.append(image_path)
            self.labels.append(label)
            
        # Save metadata
        self._save_metadata()
        print(f"Dataset prepared: {len(self.image_paths)} images in {len(self.class_names)} classes")
        
    def _save_metadata(self) -> None:
        """Save dataset metadata"""
        metadata = {
            "image_paths": [str(p) for p in self.image_paths],
            "labels": self.labels,
            "class_names": self.class_names
        }
        np.save(self.images_dir / "metadata.npy", metadata, allow_pickle=True)
        
    def load_metadata(self) -> bool:
        """Load existing dataset metadata"""
        metadata_path = self.images_dir / "metadata.npy"
        if not metadata_path.exists():
            return False
            
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.image_paths = [Path(p) for p in metadata["image_paths"]]
        self.labels = metadata["labels"]
        self.class_names = metadata["class_names"]
        return True
        
    def get_image(self, idx: int) -> Image.Image:
        """Get image by index"""
        return Image.open(self.image_paths[idx]).convert("RGB")
    
    def get_image_paths(self) -> List[Path]:
        """Get all image paths"""
        return self.image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)


def load_images_from_directory(directory: Path) -> List[Path]:
    """Load all images from a directory recursively"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(directory.rglob(f"*{ext}"))
        image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        
    return sorted(image_paths)


if __name__ == "__main__":
    # Test dataset download
    dataset = ImageDataset("cifar10", max_images=1000)
    if not dataset.load_metadata():
        dataset.download_and_prepare()
    print(f"Loaded {len(dataset)} images")
