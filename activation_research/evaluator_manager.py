from typing import List, Dict, Any, Optional, Callable
import torch
from loguru import logger
from .metric_evaluator import MetricEvaluator


class EvaluatorManager:
    """
    Manager class that accumulates model outputs during evaluation and computes metrics at the end of an epoch.
    
    This class can be used to:
    1. Accumulate z1 and z2 embeddings during evaluation
    2. Store additional metadata (hashkeys, labels, etc.)
    3. Compute metrics using registered evaluators at the end of an epoch
    4. Clear accumulated data for the next epoch
    """
    
    def __init__(self):
        """Initialize the evaluator manager."""
        self.accumulated_z1: List[torch.Tensor] = []
        self.accumulated_z2: List[torch.Tensor] = []
        self.accumulated_hashkeys: List[str] = []
        self.accumulated_labels: List[torch.Tensor] = []
        self.evaluators: List[MetricEvaluator] = []
        
    def register_evaluator(self, evaluator: MetricEvaluator):
        """
        Register a metric evaluator to be computed at the end of each epoch.
        
        Args:
            evaluator: A MetricEvaluator instance
        """
        self.evaluators.append(evaluator)
        logger.info(f"Registered evaluator: {type(evaluator).__name__}")
    
    def accumulate_batch(self, 
                        z1: torch.Tensor, 
                        z2: torch.Tensor, 
                        hashkeys: Optional[List[str]] = None,
                        labels: Optional[torch.Tensor] = None):
        """
        Accumulate embeddings and metadata from a batch.
        
        Args:
            z1: First set of embeddings (B, D)
            z2: Second set of embeddings (B, D)
            hashkeys: Optional list of hashkeys for the batch
            labels: Optional labels for the batch
        """
        # Move to CPU to save GPU memory
        self.accumulated_z1.append(z1.cpu())
        self.accumulated_z2.append(z2.cpu())
        
        if hashkeys is not None:
            self.accumulated_hashkeys.extend(hashkeys)
            
        if labels is not None:
            self.accumulated_labels.append(labels.cpu())
    
    def get_accumulated_embeddings(self) -> Dict[str, Any]:
        """
        Get all accumulated embeddings as a list of dictionaries.
        
        Returns:
            List of dictionaries containing z1, z2, and optional metadata
        """
        if not self.accumulated_z1:
            return []
        
        # Concatenate all accumulated tensors
        all_z1 = torch.cat(self.accumulated_z1, dim=0)
        all_z2 = torch.cat(self.accumulated_z2, dim=0)
        
        records = []
        for i in range(len(all_z1)):
            record = {
                'z1': all_z1[i],
                'z2': all_z2[i]
            }
            
            # Add hashkey if available
            if i < len(self.accumulated_hashkeys):
                record['hashkey'] = self.accumulated_hashkeys[i]
            
            # Add label if available
            if self.accumulated_labels:
                all_labels = torch.cat(self.accumulated_labels, dim=0)
                if i < len(all_labels):
                    record['halu'] = all_labels[i].item()
            
            records.append(record)
        
        return records
    
    def compute_metrics(self, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
        """
        Compute metrics using all registered evaluators.
        
        Args:
            model: Optional model (some evaluators might need it)
            
        Returns:
            Dictionary containing metrics from all evaluators
        """
        if not self.accumulated_z1:
            logger.warning("No accumulated data to compute metrics on")
            return {}
        
        all_metrics = {}
        records = self.get_accumulated_embeddings()
        
        logger.info(f"Computing metrics on {len(records)} accumulated samples")
        
        for evaluator in self.evaluators:
            try:
                # For evaluators that work with accumulated embeddings directly
                if hasattr(evaluator, 'compute_from_embeddings'):
                    metrics = evaluator.compute_from_embeddings(records)
                else:
                    # For evaluators that need a data loader, we'll need to create a mock one
                    # This is a simplified approach - in practice you might want to create
                    # a proper dataset/dataloader from the accumulated data
                    logger.warning(f"Evaluator {type(evaluator).__name__} requires data_loader interface")
                    continue
                
                # Prefix metrics with evaluator name to avoid conflicts
                evaluator_name = type(evaluator).__name__
                prefixed_metrics = {f"{evaluator_name}_{k}": v for k, v in metrics.items()}
                all_metrics.update(prefixed_metrics)
                
            except Exception as e:
                logger.error(f"Error computing metrics for {type(evaluator).__name__}: {e}")
        
        return all_metrics
    
    def clear(self):
        """Clear all accumulated data."""
        self.accumulated_z1.clear()
        self.accumulated_z2.clear()
        self.accumulated_hashkeys.clear()
        self.accumulated_labels.clear()
        logger.debug("Cleared accumulated evaluation data")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about accumulated data.
        
        Returns:
            Dictionary with counts of accumulated data
        """
        return {
            'num_samples': len(self.accumulated_z1) * (self.accumulated_z1[0].size(0) if self.accumulated_z1 else 0),
            'num_batches': len(self.accumulated_z1),
            'num_hashkeys': len(self.accumulated_hashkeys),
            'num_labels': sum(len(labels) for labels in self.accumulated_labels),
            'num_evaluators': len(self.evaluators)
        }


class EmbeddingDataset:
    """
    Simple dataset wrapper for accumulated embeddings to work with existing evaluators.
    """
    
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        return self.records[idx]


class MockDataLoader:
    """
    Mock data loader for accumulated embeddings to work with existing evaluators.
    """
    
    def __init__(self, records: List[Dict[str, Any]], batch_size: int = 32):
        self.dataset = EmbeddingDataset(records)
        self.batch_size = batch_size
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = []
            for j in range(i, min(i + self.batch_size, len(self.dataset))):
                batch.append(self.dataset[j])
            yield batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class CosineSimEvaluator:
    """
    Simple evaluator that computes cosine similarity statistics from accumulated embeddings.
    This is a concrete example of how to create evaluators that work with the EvaluatorManager.
    """

    def compute_from_embeddings(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute cosine similarity statistics from accumulated embeddings.

        Args:
            embeddings: List of embedding dictionaries with z1 and z2

        Returns:
            Dictionary containing cosine similarity statistics
        """
        if not embeddings:
            return {}

        import torch.nn.functional as F

        z1_list = [record['z1'] for record in embeddings]
        z2_list = [record['z2'] for record in embeddings]

        z1_tensor = torch.stack(z1_list)
        z2_tensor = torch.stack(z2_list)

        # Compute cosine similarities
        cosine_sims = F.cosine_similarity(z1_tensor, z2_tensor, dim=1)

        stats = {
            'cosine_sim_mean': cosine_sims.mean().item(),
            'cosine_sim_std': cosine_sims.std().item(),
            'cosine_sim_min': cosine_sims.min().item(),
            'cosine_sim_max': cosine_sims.max().item(),
            'num_samples': len(embeddings)
        }

        # If labels are available, compute per-class statistics
        if embeddings and 'halu' in embeddings[0]:
            labels = torch.tensor([record['halu'] for record in embeddings])

            # Stats for non-hallucinated (label 0)
            non_halu_mask = labels == 0
            if non_halu_mask.any():
                non_halu_sims = cosine_sims[non_halu_mask]
                stats.update({
                    'cosine_sim_non_halu_mean': non_halu_sims.mean().item(),
                    'cosine_sim_non_halu_std': non_halu_sims.std().item(),
                    'cosine_sim_non_halu_count': non_halu_mask.sum().item()
                })

            # Stats for hallucinated (label 1)
            halu_mask = labels == 1
            if halu_mask.any():
                halu_sims = cosine_sims[halu_mask]
                stats.update({
                    'cosine_sim_halu_mean': halu_sims.mean().item(),
                    'cosine_sim_halu_std': halu_sims.std().item(),
                    'cosine_sim_halu_count': halu_mask.sum().item()
                })

        return stats
