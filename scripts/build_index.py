"""
Script to build FAISS index for image retrieval
"""
import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import INDEX_DIR
from dataset import ImageDataset
from retrieval_engine import RetrievalEngine


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for image retrieval")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=60000,
        help="Maximum number of images to index"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="Flat",
        choices=["Flat", "IVF", "HNSW"],
        help="FAISS index type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(INDEX_DIR),
        help="Output directory for index"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building Image Retrieval Index")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Max images: {args.max_images}")
    print(f"Index type: {args.index_type}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = ImageDataset(args.dataset, max_images=args.max_images)
    if not dataset.load_metadata():
        print("Dataset not found. Downloading...")
        dataset.download_and_prepare()
    print(f"Loaded {len(dataset)} images from {len(dataset.class_names)} classes")
    
    # Build index
    print("\n[2/3] Building index...")
    start_time = time.time()
    
    engine = RetrievalEngine()
    engine.build_index(
        image_paths=dataset.get_image_paths(),
        labels=dataset.labels,
        class_names=dataset.class_names,
        index_type=args.index_type
    )
    
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds")
    
    # Save index
    print("\n[3/3] Saving index...")
    output_dir = Path(args.output_dir)
    engine.save(output_dir)
    
    print("\n" + "=" * 60)
    print("Index building complete!")
    print(f"Total images indexed: {len(dataset)}")
    print(f"Build time: {build_time:.2f}s")
    print(f"Index saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
