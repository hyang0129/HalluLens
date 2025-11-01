import torch
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from .evaluation import inference_embeddings
from .metrics import mahalanobis_ood_stats


def set_logging_level(level: str = "INFO"):
    """
    Set the logging level for loguru logger.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
               Use "WARNING" or higher to suppress info messages during data loading
    """
    logger.remove()  # Remove default handler
    logger.add(lambda msg: print(msg, end=""), level=level)


def suppress_dataloader_logs():
    """
    Convenience function to suppress info-level logs that are noisy during data loading.
    Sets logging level to WARNING to reduce verbosity.
    """
    set_logging_level("WARNING")


class HallucinationEvaluator:
    """
    A reusable class for evaluating hallucination detection using activation embeddings.
    
    This class encapsulates the workflow of:
    1. Computing baseline embeddings from training data
    2. Computing test embeddings from evaluation data  
    3. Assigning hallucination labels using activation parser data
    4. Computing Mahalanobis OOD statistics for detection performance
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 activation_parser_df: Any,
                 layers: Optional[List[int]] = None,
                 batch_size: int = 64,
                 sub_batch_size: int = 32,
                 device: str = 'cuda',
                 num_workers: int = 4,
                 persistent_workers: bool = False):
        """
        Initialize the hallucination evaluator.
        
        Args:
            model: The trained model to evaluate
            activation_parser_df: DataFrame containing prompt_hash and halu columns
            layers: List of layer indices to analyze (if None, uses default z1/z2)
            batch_size: Batch size for inference
            sub_batch_size: Sub-batch size for processing
            device: Device to run inference on
            num_workers: Number of workers for data loading
            persistent_workers: Whether to keep workers persistent
        """
        self.model = model
        self.activation_parser_df = activation_parser_df
        self.layers = layers
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        
        # Cache for computed embeddings
        self._baseline_embeddings = None
        self._test_embeddings = None
        
    def compute_baseline_embeddings(self, train_dataset) -> List[Dict[str, Any]]:
        """
        Compute baseline embeddings from training dataset.
        
        Args:
            train_dataset: Training dataset
            
        Returns:
            List of embedding dictionaries
        """
        logger.info("Computing baseline embeddings from training data...")
        
        self._baseline_embeddings = inference_embeddings(
            self.model,
            train_dataset,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
            device=self.device,
            num_workers=self.num_workers,
            layers=self.layers,
            persistent_workers=self.persistent_workers
        )
        
        logger.info(f"Computed {len(self._baseline_embeddings)} baseline embeddings")
        return self._baseline_embeddings
    
    def compute_test_embeddings(self, eval_dataset) -> List[Dict[str, Any]]:
        """
        Compute test embeddings from evaluation dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            List of embedding dictionaries
        """
        logger.info("Computing test embeddings from evaluation data...")
        
        self._test_embeddings = inference_embeddings(
            self.model,
            eval_dataset,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
            device=self.device,
            num_workers=self.num_workers,
            layers=self.layers,
            persistent_workers=self.persistent_workers
        )
        
        logger.info(f"Computed {len(self._test_embeddings)} test embeddings")
        return self._test_embeddings
    
    def assign_hallucination_labels(self, 
                                   embeddings: Optional[List[Dict[str, Any]]] = None,
                                   max_rows: int = 10000000) -> List[Dict[str, Any]]:
        """
        Assign hallucination labels to embeddings using activation parser data.
        
        Args:
            embeddings: List of embedding dictionaries (uses cached test embeddings if None)
            max_rows: Maximum number of rows to consider from activation parser df
            
        Returns:
            List of embedding dictionaries with 'halu' labels added
        """
        if embeddings is None:
            if self._test_embeddings is None:
                raise ValueError("No test embeddings available. Call compute_test_embeddings first.")
            embeddings = self._test_embeddings
            
        logger.info("Assigning hallucination labels...")
        
        # Limit dataframe size for performance
        df = self.activation_parser_df.head(max_rows)
        
        labeled_embeddings = []
        for i, record in enumerate(embeddings):
            hashkey = record['hashkey']
            ishalu = df[df['prompt_hash'] == hashkey]['halu']
            
            if len(ishalu) != 1:
                logger.warning(f"Expected exactly 1 match for hashkey {hashkey}, found {len(ishalu)}. Skipping.")
                continue
                
            # Create a copy of the record and add the label
            labeled_record = record.copy()
            labeled_record['halu'] = ishalu.values[0]
            labeled_embeddings.append(labeled_record)
        
        logger.info(f"Successfully labeled {len(labeled_embeddings)} embeddings")
        return labeled_embeddings
    
    def evaluate_detection(self, 
                          train_dataset,
                          eval_dataset,
                          outlier_class: int = 1,
                          max_rows: int = 10000000) -> Dict[str, float]:
        """
        Complete evaluation pipeline for hallucination detection.
        
        Args:
            train_dataset: Training dataset for baseline embeddings
            eval_dataset: Evaluation dataset for test embeddings
            outlier_class: Which class to treat as outlier (0 or 1)
            max_rows: Maximum rows to consider from activation parser df
            
        Returns:
            Dictionary containing Mahalanobis OOD statistics
        """
        logger.info("Starting hallucination detection evaluation...")
        
        # Compute baseline embeddings
        baseline_embeddings = self.compute_baseline_embeddings(train_dataset)
        
        # Compute test embeddings
        test_embeddings = self.compute_test_embeddings(eval_dataset)
        
        # Assign hallucination labels
        labeled_test_embeddings = self.assign_hallucination_labels(
            test_embeddings, max_rows=max_rows
        )
        
        # Compute Mahalanobis OOD statistics
        logger.info("Computing Mahalanobis OOD statistics...")
        stats = mahalanobis_ood_stats(
            baseline_embeddings, 
            labeled_test_embeddings,
            outlier_class=outlier_class
        )
        
        logger.info("Evaluation complete!")
        logger.info(f"Results: {stats}")
        
        return stats
    
    def get_cached_embeddings(self) -> tuple:
        """
        Get cached baseline and test embeddings.
        
        Returns:
            Tuple of (baseline_embeddings, test_embeddings)
        """
        return self._baseline_embeddings, self._test_embeddings
    
    def clear_cache(self):
        """Clear cached embeddings to free memory."""
        self._baseline_embeddings = None
        self._test_embeddings = None
        logger.info("Cleared embedding cache")
