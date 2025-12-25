"""
Evaluation script for image retrieval system
Computes metrics: mAP@K, Precision@K, Recall@K
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import INDEX_DIR
from dataset import ImageDataset
from retrieval_engine import RetrievalEngine


def precision_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
    """Compute Precision@K"""
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / k


def recall_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
    """Compute Recall@K"""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(relevant)


def average_precision(relevant: List[int], retrieved: List[int], k: int = None) -> float:
    """Compute Average Precision"""
    if len(relevant) == 0:
        return 0.0
        
    if k is not None:
        retrieved = retrieved[:k]
        
    relevant_set = set(relevant)
    precisions = []
    hits = 0
    
    for i, r in enumerate(retrieved):
        if r in relevant_set:
            hits += 1
            precisions.append(hits / (i + 1))
            
    if len(precisions) == 0:
        return 0.0
    return sum(precisions) / min(len(relevant), len(retrieved))


def mean_average_precision(all_relevant: List[List[int]], 
                           all_retrieved: List[List[int]], 
                           k: int = None) -> float:
    """Compute Mean Average Precision"""
    aps = [average_precision(rel, ret, k) for rel, ret in zip(all_relevant, all_retrieved)]
    return np.mean(aps)


def evaluate(engine: RetrievalEngine, 
             dataset: ImageDataset, 
             num_queries: int = 100,
             k_values: List[int] = [1, 5, 10, 20]) -> Dict:
    """
    Evaluate retrieval performance
    
    Args:
        engine: Retrieval engine
        dataset: Image dataset
        num_queries: Number of query images
        k_values: K values for metrics
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating with {num_queries} queries...")
    
    # Build label to indices mapping
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        label_to_indices[label].append(idx)
    
    # Random sample query indices
    np.random.seed(42)
    query_indices = np.random.choice(len(dataset), size=min(num_queries, len(dataset)), replace=False)
    
    all_retrieved = []
    all_relevant = []
    query_times = []
    
    for query_idx in tqdm(query_indices, desc="Querying"):
        query_label = dataset.labels[query_idx]
        query_image = dataset.get_image(query_idx)
        
        # Get relevant images (same class, excluding query)
        relevant = [i for i in label_to_indices[query_label] if i != query_idx]
        
        # Search
        start_time = time.time()
        results = engine.search(query_image, k=max(k_values))
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        # Extract retrieved indices
        retrieved = []
        for r in results:
            path = Path(r["path"])
            # Find index by path
            for idx, p in enumerate(dataset.image_paths):
                if p == path:
                    retrieved.append(idx)
                    break
                    
        all_retrieved.append(retrieved)
        all_relevant.append(relevant)
    
    # Compute metrics
    metrics = {}
    
    for k in k_values:
        metrics[f"mAP@{k}"] = mean_average_precision(all_relevant, all_retrieved, k)
        metrics[f"P@{k}"] = np.mean([precision_at_k(rel, ret, k) 
                                     for rel, ret in zip(all_relevant, all_retrieved)])
        metrics[f"R@{k}"] = np.mean([recall_at_k(rel, ret, k) 
                                     for rel, ret in zip(all_relevant, all_retrieved)])
    
    metrics["avg_query_time_ms"] = np.mean(query_times) * 1000
    metrics["queries_per_second"] = 1 / np.mean(query_times)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate image retrieval system")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(INDEX_DIR),
        help="Index directory"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=500,
        help="Number of query images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Image Retrieval System Evaluation")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = ImageDataset(args.dataset)
    if not dataset.load_metadata():
        print("Error: Dataset not found. Run download_dataset.py first.")
        return
    print(f"Loaded {len(dataset)} images")
    
    # Load engine
    print("\n[2/3] Loading retrieval engine...")
    engine = RetrievalEngine()
    engine.load(Path(args.index_dir))
    
    # Evaluate
    print("\n[3/3] Running evaluation...")
    k_values = [1, 5, 10, 20, 50]
    metrics = evaluate(engine, dataset, num_queries=args.num_queries, k_values=k_values)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print("\n📊 Retrieval Performance:")
    for k in k_values:
        print(f"  K={k:2d}:  mAP={metrics[f'mAP@{k}']:.4f}  "
              f"P={metrics[f'P@{k}']:.4f}  R={metrics[f'R@{k}']:.4f}")
    
    print(f"\n⏱️ Speed Performance:")
    print(f"  Average query time: {metrics['avg_query_time_ms']:.2f} ms")
    print(f"  Throughput: {metrics['queries_per_second']:.2f} queries/sec")
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
