"""
FAISS-based vector indexing module for efficient similarity search
Supports GPU acceleration for fast retrieval
"""
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from config import INDEX_DIR, EMBEDDING_DIM, FAISS_NPROBE


class FAISSIndexer:
    """FAISS-based vector indexer with GPU support"""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.num_vectors = 0
        
        if self.use_gpu:
            print(f"FAISS will use {faiss.get_num_gpus()} GPU(s)")
        else:
            print("FAISS running on CPU")
            
    def create_index(self, index_type: str = "Flat") -> None:
        """
        Create a new FAISS index
        
        Args:
            index_type: Type of index
                - "Flat": Exact search (brute force, best accuracy)
                - "IVF": Inverted file index (faster, approximate)
                - "HNSW": Hierarchical NSW (very fast, approximate)
        """
        if index_type == "Flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.is_trained = True
        elif index_type.startswith("IVF"):
            # IVF index for faster search
            # Format: IVF{nlist},Flat
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.is_trained = False
        elif index_type == "HNSW":
            # HNSW for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.is_trained = True
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        print(f"Created {index_type} index with dimension {self.embedding_dim}")
        
    def train(self, vectors: np.ndarray) -> None:
        """Train the index (required for IVF-type indices)"""
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
            
        if not self.is_trained:
            print(f"Training index with {len(vectors)} vectors...")
            self.index.train(vectors.astype(np.float32))
            self.is_trained = True
            print("Index trained successfully")
            
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index"""
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        if not self.is_trained:
            raise RuntimeError("Index not trained. Call train() first.")
            
        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.num_vectors += len(vectors)
        print(f"Added {len(vectors)} vectors. Total: {self.num_vectors}")
        
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors
        
        Args:
            query_vectors: Query vectors, shape (n_queries, embedding_dim)
            k: Number of nearest neighbors to return
            
        Returns:
            distances: Distance to neighbors, shape (n_queries, k)
            indices: Indices of neighbors, shape (n_queries, k)
        """
        if self.index is None:
            raise RuntimeError("Index not created or loaded.")
            
        # Ensure 2D array
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
            
        query_vectors = query_vectors.astype(np.float32)
        
        # Set nprobe for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = FAISS_NPROBE
            
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save index to disk"""
        if self.index is None:
            raise RuntimeError("No index to save")
            
        if path is None:
            path = INDEX_DIR / "faiss_index.bin"
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU before saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path))
        else:
            faiss.write_index(self.index, str(path))
            
        print(f"Index saved to {path}")
        return path
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load index from disk"""
        if path is None:
            path = INDEX_DIR / "faiss_index.bin"
            
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
            
        self.index = faiss.read_index(str(path))
        self.num_vectors = self.index.ntotal
        self.is_trained = True
        
        # Move to GPU if available
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        print(f"Index loaded from {path}. Contains {self.num_vectors} vectors")
        
    def to_gpu(self) -> None:
        """Move index to GPU"""
        if self.index is not None and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.use_gpu = True
            print("Index moved to GPU")


if __name__ == "__main__":
    # Test indexer
    indexer = FAISSIndexer(embedding_dim=512)
    indexer.create_index("Flat")
    
    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.randn(1000, 512).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    indexer.add(vectors)
    
    # Search
    query = vectors[0:1]
    distances, indices = indexer.search(query, k=5)
    print(f"Query result: indices={indices[0]}, distances={distances[0]}")
