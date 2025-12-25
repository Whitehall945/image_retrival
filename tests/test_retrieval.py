"""
Unit tests for image retrieval system
"""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestFeatureExtractor:
    """Tests for feature extraction module"""
    
    @pytest.fixture
    def extractor(self):
        from feature_extractor import FeatureExtractor
        return FeatureExtractor()
    
    def test_extract_single(self, extractor):
        """Test single image feature extraction"""
        # Create test image
        img = Image.new("RGB", (224, 224), color="blue")
        features = extractor.extract_single(img)
        
        assert features.shape == (512,)
        assert np.isclose(np.linalg.norm(features), 1.0, atol=1e-5)
        
    def test_extract_batch(self, extractor):
        """Test batch feature extraction"""
        images = [Image.new("RGB", (224, 224), color=c) for c in ["red", "green", "blue"]]
        features = extractor.extract_batch(images, show_progress=False)
        
        assert features.shape == (3, 512)
        for f in features:
            assert np.isclose(np.linalg.norm(f), 1.0, atol=1e-5)
            
    def test_extract_text(self, extractor):
        """Test text feature extraction"""
        text_features = extractor.extract_text("a photo of a cat")
        
        assert text_features.shape == (512,)
        assert np.isclose(np.linalg.norm(text_features), 1.0, atol=1e-5)


class TestFAISSIndexer:
    """Tests for FAISS indexer module"""
    
    @pytest.fixture
    def indexer(self):
        from indexer import FAISSIndexer
        idx = FAISSIndexer(embedding_dim=512, use_gpu=False)
        idx.create_index("Flat")
        return idx
    
    def test_add_and_search(self, indexer):
        """Test adding vectors and searching"""
        np.random.seed(42)
        vectors = np.random.randn(100, 512).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        indexer.add(vectors)
        
        query = vectors[0:1]
        distances, indices = indexer.search(query, k=5)
        
        assert indices.shape == (1, 5)
        assert indices[0, 0] == 0  # First result should be the query itself
        assert distances[0, 0] < 1e-5  # Distance to itself should be ~0
        
    def test_save_load(self, indexer, tmp_path):
        """Test saving and loading index"""
        np.random.seed(42)
        vectors = np.random.randn(50, 512).astype(np.float32)
        indexer.add(vectors)
        
        # Save
        save_path = tmp_path / "test_index.bin"
        indexer.save(save_path)
        
        # Load in new indexer
        from indexer import FAISSIndexer
        new_indexer = FAISSIndexer(embedding_dim=512, use_gpu=False)
        new_indexer.load(save_path)
        
        assert new_indexer.num_vectors == 50


class TestRetrievalEngine:
    """Tests for retrieval engine"""
    
    def test_search_returns_results(self, tmp_path):
        """Test that search returns properly formatted results"""
        from retrieval_engine import RetrievalEngine
        
        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        
        image_paths = []
        for i in range(10):
            img = Image.new("RGB", (32, 32), color=(i * 25, i * 25, i * 25))
            path = img_dir / f"img_{i}.png"
            img.save(path)
            image_paths.append(path)
            
        # Build index
        engine = RetrievalEngine()
        engine.build_index(image_paths, index_type="Flat")
        
        # Search
        query = Image.open(image_paths[0])
        results = engine.search(query, k=5)
        
        assert len(results) == 5
        assert results[0]["rank"] == 1
        assert "path" in results[0]
        assert "distance" in results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
