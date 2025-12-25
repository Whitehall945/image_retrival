"""
Feature extraction module using CLIP model
Extracts visual embeddings for image retrieval
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from config import MODEL_NAME, EMBEDDING_DIM, DEVICE, BATCH_SIZE


class FeatureExtractor:
    """CLIP-based feature extractor for images"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading CLIP model: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Get embedding dimension from model
        self.embedding_dim = self.model.config.projection_dim
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    @torch.no_grad()
    def extract_single(self, image: Union[Image.Image, Path, str]) -> np.ndarray:
        """Extract feature from a single image"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, images: List[Union[Image.Image, Path, str]], 
                     batch_size: int = BATCH_SIZE,
                     show_progress: bool = True) -> np.ndarray:
        """Extract features from a batch of images"""
        all_features = []
        
        # Convert paths to PIL images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)
        
        # Process in batches
        iterator = range(0, len(pil_images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")
            
        for i in iterator:
            batch = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_image_features(**inputs)
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
            
        return np.vstack(all_features)
    
    @torch.no_grad()
    def extract_text(self, text: str) -> np.ndarray:
        """Extract feature from text query (for text-to-image search)"""
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Create a test image
    test_image = Image.new("RGB", (224, 224), color="red")
    features = extractor.extract_single(test_image)
    print(f"Feature shape: {features.shape}")
    print(f"Feature norm: {np.linalg.norm(features):.4f}")
