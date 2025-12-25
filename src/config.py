"""
Global configuration for Image Retrieval System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
IMAGES_DIR = DATA_DIR / "images"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
DEVICE = "cuda"

# FAISS configuration
FAISS_INDEX_TYPE = "IVF100,Flat"  # IVF for faster search
FAISS_METRIC = "L2"  # L2 distance (we use normalized vectors, equivalent to cosine)
FAISS_NPROBE = 10  # Number of clusters to search

# Retrieval configuration
DEFAULT_TOP_K = 10
BATCH_SIZE = 64

# Dataset configuration
DATASET_NAME = "cifar10"  # Options: cifar10, imagenet_subset
MAX_IMAGES = 50000  # Maximum images to index
