"""
Example usage of EvaluatorManager for accumulating embeddings during evaluation.
"""

import torch
from torch.utils.data import DataLoader
from .evaluator_manager import EvaluatorManager
from .metric_evaluator import HallucinationEvaluator
from .evaluation import evaluate


def example_training_loop_with_evaluator_manager():
    """
    Example showing how to integrate EvaluatorManager into a training loop.
    """
    # Mock setup (replace with your actual components)
    model = None  # Your trained model
    train_loader = None  # Your training data loader
    eval_loader = None  # Your evaluation data loader
    loss_fn = None  # Your loss function
    activation_parser_df = None  # Your activation parser dataframe
    
    # Create evaluator manager
    evaluator_manager = EvaluatorManager()
    
    # Create and register hallucination evaluator
    hallucination_evaluator = HallucinationEvaluator(
        activation_parser_df=activation_parser_df,
        train_data_loader=train_loader,
        layers=None,
        batch_size=64,
        sub_batch_size=32,
        device='cuda',
        outlier_class=1
    )
    
    # First, compute baseline embeddings (do this once)
    print("Computing baseline embeddings...")
    baseline_stats = hallucination_evaluator.compute(train_loader, model)
    print(f"Baseline computed: {baseline_stats}")
    
    # Register the evaluator
    evaluator_manager.register_evaluator(hallucination_evaluator)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase (your training code here)
        model.train()
        # ... training code ...
        
        # Evaluation phase with accumulation
        print("Evaluating...")
        model.eval()
        
        # Clear previous epoch's accumulated data
        evaluator_manager.clear()
        
        # Run evaluation with accumulation
        avg_loss, avg_intra_cos, avg_intra_inter_margin = evaluate(
            model=model,
            test_dataloader=eval_loader,
            batch_size=64,
            loss_fn=loss_fn,
            device='cuda',
            sub_batch_size=32,
            use_labels=True,
            evaluator_manager=evaluator_manager  # This enables accumulation
        )
        
        print(
            f"Evaluation metrics - Loss: {avg_loss:.4f}, "
            f"Intra Cos: {avg_intra_cos:.4f}, "
            f"Intra/Inter Margin: {avg_intra_inter_margin:.4f}"
        )
        
        # Get stats about accumulated data
        stats = evaluator_manager.get_stats()
        print(f"Accumulated {stats['num_samples']} samples in {stats['num_batches']} batches")
        
        # Compute metrics using accumulated embeddings
        if stats['num_samples'] > 0:
            print("Computing hallucination detection metrics...")
            metrics = evaluator_manager.compute_metrics(model)
            
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value}")
        
        # Optional: Save metrics to wandb, tensorboard, etc.
        # wandb.log(metrics, step=epoch)


def example_manual_accumulation():
    """
    Example showing manual accumulation of embeddings.
    """
    # Create evaluator manager
    evaluator_manager = EvaluatorManager()
    
    # Simulate some embeddings
    batch_size = 4
    embedding_dim = 128
    num_views = 4
    
    for batch_idx in range(3):  # Simulate 3 batches
        # Generate mock embeddings
        z_views = torch.randn(batch_size, num_views, embedding_dim)
        
        # Generate mock hashkeys and labels
        hashkeys = [f"hash_{batch_idx}_{i}" for i in range(batch_size)]
        labels = torch.randint(0, 2, (batch_size,))
        
        # Accumulate
        evaluator_manager.accumulate_batch(z_views, hashkeys, labels)
        print(f"Accumulated batch {batch_idx + 1}")
    
    # Get accumulated embeddings
    records = evaluator_manager.get_accumulated_embeddings()
    print(f"Total accumulated records: {len(records)}")
    
    # Print first record as example
    if records:
        first_record = records[0]
        print(f"First record keys: {list(first_record.keys())}")
        print(f"z_views shape: {first_record['z_views'].shape}")
        if 'hashkey' in first_record:
            print(f"hashkey: {first_record['hashkey']}")
        if 'halu' in first_record:
            print(f"halu label: {first_record['halu']}")
    
    # Clear accumulated data
    evaluator_manager.clear()
    print("Cleared accumulated data")


def example_custom_evaluator():
    """
    Example showing how to create a custom evaluator that works with accumulated embeddings.
    """
    from .metric_evaluator import MetricEvaluator
    import torch.nn.functional as F
    
    class CosineSimEvaluator(MetricEvaluator):
        """Custom evaluator that computes cosine similarity statistics."""
        
        def __init__(self):
            pass
        
        def compute(self, data_loader, model):
            # This would be the standard interface for data loader
            raise NotImplementedError("Use compute_from_embeddings for this evaluator")
        
        def compute_from_embeddings(self, embeddings):
            """Compute cosine similarity statistics from accumulated embeddings."""
            if not embeddings:
                return {}
            
            z_views = torch.stack([record['z_views'] for record in embeddings])
            z_views = F.normalize(z_views, dim=-1)
            _, k, _ = z_views.shape
            tri = torch.triu_indices(k, k, offset=1)
            sim = torch.matmul(z_views, z_views.transpose(1, 2))
            cosine_sims = sim[:, tri[0], tri[1]].mean(dim=1)
            
            stats = {
                'cosine_sim_mean': cosine_sims.mean().item(),
                'cosine_sim_std': cosine_sims.std().item(),
                'cosine_sim_min': cosine_sims.min().item(),
                'cosine_sim_max': cosine_sims.max().item(),
            }
            
            # If labels are available, compute per-class statistics
            if 'halu' in embeddings[0]:
                labels = torch.tensor([record['halu'] for record in embeddings])
                
                # Stats for non-hallucinated (label 0)
                non_halu_mask = labels == 0
                if non_halu_mask.any():
                    non_halu_sims = cosine_sims[non_halu_mask]
                    stats['cosine_sim_non_halu_mean'] = non_halu_sims.mean().item()
                    stats['cosine_sim_non_halu_std'] = non_halu_sims.std().item()
                
                # Stats for hallucinated (label 1)
                halu_mask = labels == 1
                if halu_mask.any():
                    halu_sims = cosine_sims[halu_mask]
                    stats['cosine_sim_halu_mean'] = halu_sims.mean().item()
                    stats['cosine_sim_halu_std'] = halu_sims.std().item()
            
            return stats
    
    # Example usage
    evaluator_manager = EvaluatorManager()
    cosine_evaluator = CosineSimEvaluator()
    evaluator_manager.register_evaluator(cosine_evaluator)
    
    # Simulate some data
    z_views = torch.randn(10, 4, 64)
    labels = torch.randint(0, 2, (10,))

    evaluator_manager.accumulate_batch(z_views, labels=labels)
    
    # Compute metrics
    metrics = evaluator_manager.compute_metrics()
    print("Custom evaluator metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    print("=== Manual Accumulation Example ===")
    example_manual_accumulation()
    
    print("\n=== Custom Evaluator Example ===")
    example_custom_evaluator()
    
    print("\n=== Training Loop Example ===")
    print("(This would require actual model and data - see function for details)")
    # example_training_loop_with_evaluator_manager()
