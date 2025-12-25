"""
Script to download and prepare dataset for image retrieval
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import ImageDataset


def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cifar10",
        choices=["cifar10"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=60000,
        help="Maximum number of images to prepare"
    )
    args = parser.parse_args()
    
    print(f"Preparing {args.dataset} dataset...")
    print(f"Max images: {args.max_images}")
    
    dataset = ImageDataset(args.dataset, max_images=args.max_images)
    
    # Check if already prepared
    if dataset.load_metadata():
        print(f"Dataset already prepared: {len(dataset)} images")
        response = input("Re-download? (y/N): ")
        if response.lower() != 'y':
            return
            
    # Download and prepare
    dataset.download_and_prepare()
    print(f"Dataset preparation complete: {len(dataset)} images")


if __name__ == "__main__":
    main()
