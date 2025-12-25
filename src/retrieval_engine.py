"""
Image Retrieval Engine
Combines feature extraction and FAISS indexing for image search
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
import json

from config import INDEX_DIR, DEFAULT_TOP_K
from feature_extractor import FeatureExtractor
from indexer import FAISSIndexer


class RetrievalEngine:
    """Main retrieval engine combining feature extraction and indexing"""
    
    def __init__(self, index_path: Optional[Path] = None):
        self.feature_extractor = FeatureExtractor()
        self.indexer = FAISSIndexer(embedding_dim=self.feature_extractor.embedding_dim)
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        
        if index_path and Path(index_path).exists():
            self.load(index_path)
            
    def build_index(self, image_paths: List[Path], 
                   labels: Optional[List[int]] = None,
                   class_names: Optional[List[str]] = None,
                   index_type: str = "Flat") -> None:
        """
        Build index from a list of image paths
        
        Args:
            image_paths: List of paths to images
            labels: Optional list of labels for each image
            class_names: Optional list of class names
            index_type: FAISS index type ("Flat", "IVF", "HNSW")
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels or [-1] * len(image_paths)
        self.class_names = class_names or []
        
        print(f"Building index for {len(image_paths)} images...")
        
        # Extract features
        features = self.feature_extractor.extract_batch(
            self.image_paths, 
            show_progress=True
        )
        
        # Create and populate index
        self.indexer.create_index(index_type)
        
        if index_type.startswith("IVF"):
            self.indexer.train(features)
            
        self.indexer.add(features)
        
        print(f"Index built successfully with {len(self.image_paths)} images")
        
    def search(self, query: Union[Image.Image, Path, str, np.ndarray], 
               k: int = DEFAULT_TOP_K) -> List[dict]:
        """
        Search for similar images
        
        Args:
            query: Query image (PIL Image, path, or feature vector)
            k: Number of results to return
            
        Returns:
            List of result dictionaries with keys: path, distance, label, class_name, rank
        """
        # Extract query features
        if isinstance(query, np.ndarray):
            query_features = query
        else:
            query_features = self.feature_extractor.extract_single(query)
            
        # Search
        distances, indices = self.indexer.search(query_features.reshape(1, -1), k)
        
        # Format results
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:  # Invalid index
                continue
                
            result = {
                "rank": rank + 1,
                "path": str(self.image_paths[idx]),
                "distance": float(dist),
                "label": self.labels[idx] if idx < len(self.labels) else -1,
                "class_name": self.class_names[self.labels[idx]] if (
                    self.class_names and self.labels[idx] >= 0 and 
                    self.labels[idx] < len(self.class_names)
                ) else "unknown"
            }
            results.append(result)
            
        return results
    
    def search_by_text(self, text: str, k: int = DEFAULT_TOP_K) -> List[dict]:
        """Search for images by text description (CLIP text-to-image)"""
        text_features = self.feature_extractor.extract_text(text)
        return self.search(text_features, k)
    
    def save(self, save_dir: Optional[Path] = None) -> None:
        """Save index and metadata"""
        save_dir = Path(save_dir) if save_dir else INDEX_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.indexer.save(save_dir / "faiss_index.bin")
        
        # Save metadata
        metadata = {
            "image_paths": [str(p) for p in self.image_paths],
            "labels": self.labels,
            "class_names": self.class_names
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Engine saved to {save_dir}")
        
    def load(self, load_dir: Optional[Path] = None) -> None:
        """Load index and metadata"""
        load_dir = Path(load_dir) if load_dir else INDEX_DIR
        
        # Load FAISS index
        self.indexer.load(load_dir / "faiss_index.bin")
        
        # Load metadata
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.image_paths = [Path(p) for p in metadata["image_paths"]]
            self.labels = metadata.get("labels", [])
            self.class_names = metadata.get("class_names", [])
            
        print(f"Engine loaded: {len(self.image_paths)} images indexed")


if __name__ == "__main__":
    # Test retrieval engine
    engine = RetrievalEngine()
    
    # Test with dummy data
    from dataset import ImageDataset
    
    dataset = ImageDataset("cifar10", max_images=100)
    if dataset.load_metadata():
        print(f"Found existing dataset with {len(dataset)} images")
        engine.build_index(
            dataset.get_image_paths(),
            dataset.labels,
            dataset.class_names
        )
        
        # Test search
        query_img = dataset.get_image(0)
        results = engine.search(query_img, k=5)
        print("Search results:")
        for r in results:
            print(f"  Rank {r['rank']}: {r['class_name']} (dist={r['distance']:.4f})")
