from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
from loguru import logger
from .evaluation import inference_embeddings
from .metrics import mahalanobis_ood_stats, cosine_similarity_ood_stats, knn_ood_stats


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


class MetricEvaluator(ABC):
    """
    Abstract base class for metric evaluators that can be integrated into training loops.

    This class provides a standard interface for evaluators that compute metrics
    on a model and dataset during training.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the metric evaluator with relevant parameters.

        Args:
            **kwargs: Evaluator-specific parameters
        """
        pass

    @abstractmethod
    def compute(self, data_loader, model) -> Dict[str, Any]:
        """
        Compute metrics using the provided data loader and model.

        Args:
            data_loader: DataLoader containing the evaluation data
            model: The model to evaluate

        Returns:
            Dictionary containing computed metrics
        """
        pass

    def compute_from_embeddings(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optional method to compute metrics directly from accumulated embeddings.

        Args:
            embeddings: List of embedding dictionaries with z1, z2, and optional metadata

        Returns:
            Dictionary containing computed metrics
        """
        raise NotImplementedError("This evaluator does not support computing from embeddings directly")


class HallucinationEvaluator(MetricEvaluator):
    """
    A metric evaluator for hallucination detection using activation embeddings.

    This class encapsulates the workflow of:
    1. Computing baseline embeddings from training data
    2. Computing test embeddings from evaluation data
    3. Assigning hallucination labels using activation parser data
    4. Computing Mahalanobis OOD statistics for detection performance
    """

    def __init__(self,
                 activation_parser_df: Any,
                 train_data_loader,
                 layers: Optional[List[int]] = None,
                 batch_size: int = 64,
                 sub_batch_size: int = 32,
                 device: str = 'cuda',
                 num_workers: int = 4,
                 persistent_workers: bool = False,
                 outlier_class: int = 1,
                 max_rows: int = 10000000,
                 metric: Union[str, Callable[..., Dict[str, Any]]] = "mahalanobis",
                 metric_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the hallucination evaluator.

        Args:
            activation_parser_df: DataFrame containing prompt_hash and halu columns
            train_data_loader: DataLoader for training data (used for baseline embeddings)
            layers: List of layer indices to analyze (if None, uses default z1/z2)
            batch_size: Batch size for inference
            sub_batch_size: Sub-batch size for processing
            device: Device to run inference on
            num_workers: Number of workers for data loading
            persistent_workers: Whether to keep workers persistent
            outlier_class: Which class to treat as outlier (0 or 1)
            max_rows: Maximum number of rows to consider from activation parser df
            metric: Which OOD metric/stats function to use. Either a string name
                ('mahalanobis', 'cosine', 'knn') or a callable.
            metric_kwargs: Optional kwargs passed to the metric callable.
                For KNN, common keys are {'k': int, 'metric': str}.
        """
        self.activation_parser_df = activation_parser_df
        self.train_data_loader = train_data_loader
        self.layers = layers
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.outlier_class = outlier_class
        self.max_rows = max_rows

        self.metric_name, self.metric_fn = self._resolve_metric(metric)
        self.metric_kwargs = metric_kwargs or {}

        # Cache for computed embeddings
        self._baseline_embeddings = None

    @staticmethod
    def _resolve_metric(metric: Union[str, Callable[..., Dict[str, Any]]]):
        if callable(metric):
            return getattr(metric, "__name__", "custom_metric"), metric

        metric_key = str(metric).strip().lower()
        metric_map = {
            "mahalanobis": mahalanobis_ood_stats,
            "mds": mahalanobis_ood_stats,
            "cosine": cosine_similarity_ood_stats,
            "cosine_similarity": cosine_similarity_ood_stats,
            "knn": knn_ood_stats,
            "knn_halu": knn_ood_stats,
            "knn_hallucination": knn_ood_stats,
        }
        if metric_key not in metric_map:
            raise ValueError(
                f"Unknown metric '{metric}'. Expected one of {sorted(metric_map.keys())} or a callable."
            )

        return metric_key, metric_map[metric_key]

    def _compute_stats(
        self,
        baseline_embeddings: List[Dict[str, Any]],
        labeled_test_embeddings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the configured metric function with best-effort compatibility."""

        # Most metrics in activation_research.metrics accept (train_records, test_records, outlier_class=...).
        # For custom callables, we try a couple of common signatures.
        try:
            return self.metric_fn(
                baseline_embeddings,
                labeled_test_embeddings,
                outlier_class=self.outlier_class,
                **self.metric_kwargs,
            )
        except TypeError:
            # Fallback: metric doesn't accept outlier_class (or kwargs).
            return self.metric_fn(baseline_embeddings, labeled_test_embeddings, **self.metric_kwargs)

    def compute(self, data_loader, model) -> Dict[str, Any]:
        """
        Compute hallucination detection metrics using the provided data loader and model.

        Args:
            data_loader: DataLoader containing the evaluation data
            model: The model to evaluate

        Returns:
            Dictionary containing Mahalanobis OOD statistics
        """
        logger.info("Starting hallucination detection evaluation...")

        # Compute baseline embeddings from training data
        baseline_embeddings = self._compute_baseline_embeddings(model)

        # Compute test embeddings from evaluation data
        test_embeddings = self._compute_test_embeddings(data_loader, model)

        # Assign hallucination labels
        labeled_test_embeddings = self._assign_hallucination_labels(test_embeddings)

        logger.info(f"Computing OOD stats via metric='{self.metric_name}'...")
        stats = self._compute_stats(baseline_embeddings, labeled_test_embeddings)

        logger.info("Evaluation complete!")
        logger.info(f"Results: {stats}")

        return stats

    def compute_from_embeddings(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute hallucination detection metrics directly from accumulated embeddings.

        Args:
            embeddings: List of embedding dictionaries with z1, z2, and halu labels

        Returns:
            Dictionary containing Mahalanobis OOD statistics
        """
        logger.info("Computing hallucination detection metrics from accumulated embeddings...")

        # Compute baseline embeddings from training data if not cached
        if self._baseline_embeddings is None:
            logger.warning("No cached baseline embeddings. Computing from training data...")
            # This requires a model, which we don't have in this context
            # In practice, baseline embeddings should be computed once and cached
            raise ValueError("Baseline embeddings not available. Call compute() with training data first.")

        # Assign hallucination labels to the embeddings
        labeled_embeddings = self._assign_hallucination_labels(embeddings)

        logger.info(f"Computing OOD stats via metric='{self.metric_name}'...")
        stats = self._compute_stats(self._baseline_embeddings, labeled_embeddings)

        logger.info("Evaluation complete!")
        logger.info(f"Results: {stats}")

        return stats

    def _compute_baseline_embeddings(self, model) -> List[Dict[str, Any]]:
        """
        Compute baseline embeddings from training dataset.

        Args:
            model: The model to use for inference

        Returns:
            List of embedding dictionaries
        """
        if self._baseline_embeddings is not None:
            return self._baseline_embeddings

        logger.info("Computing baseline embeddings from training data...")

        self._baseline_embeddings = inference_embeddings(
            model,
            self.train_data_loader.dataset,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
            device=self.device,
            num_workers=self.num_workers,
            layers=self.layers,
            persistent_workers=self.persistent_workers
        )

        logger.info(f"Computed {len(self._baseline_embeddings)} baseline embeddings")
        return self._baseline_embeddings

    def _compute_test_embeddings(self, data_loader, model) -> List[Dict[str, Any]]:
        """
        Compute test embeddings from evaluation dataset.

        Args:
            data_loader: DataLoader containing evaluation data
            model: The model to use for inference

        Returns:
            List of embedding dictionaries
        """
        logger.info("Computing test embeddings from evaluation data...")

        test_embeddings = inference_embeddings(
            model,
            data_loader.dataset,
            batch_size=self.batch_size,
            sub_batch_size=self.sub_batch_size,
            device=self.device,
            num_workers=self.num_workers,
            layers=self.layers,
            persistent_workers=self.persistent_workers
        )

        logger.info(f"Computed {len(test_embeddings)} test embeddings")
        return test_embeddings
    
    def _assign_hallucination_labels(self, embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign hallucination labels to embeddings using activation parser data.

        Args:
            embeddings: List of embedding dictionaries

        Returns:
            List of embedding dictionaries with 'halu' labels added
        """
        logger.info("Assigning hallucination labels...")

        # Limit dataframe size for performance
        df = self.activation_parser_df.head(self.max_rows)

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

    def get_cached_baseline_embeddings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached baseline embeddings.

        Returns:
            Cached baseline embeddings or None if not computed yet
        """
        return self._baseline_embeddings

    def clear_cache(self):
        """Clear cached embeddings to free memory."""
        self._baseline_embeddings = None
        logger.info("Cleared embedding cache")


class KNNHallucinationEvaluator(HallucinationEvaluator):
    """Convenience evaluator for KNN-based hallucination detection.

    This is equivalent to ``HallucinationEvaluator(metric='knn', ...)`` with
    explicit KNN arguments.
    """

    def __init__(
        self,
        activation_parser_df: Any,
        train_data_loader,
        *,
        layers: Optional[List[int]] = None,
        batch_size: int = 64,
        sub_batch_size: int = 32,
        device: str = 'cuda',
        num_workers: int = 4,
        persistent_workers: bool = False,
        outlier_class: int = 1,
        max_rows: int = 10000000,
        k: int = 5,
        distance_metric: str = "euclidean",
        metric_kwargs: Optional[Dict[str, Any]] = None,
    ):
        resolved_metric_kwargs = {
            "k": int(k),
            "metric": str(distance_metric),
        }
        if metric_kwargs:
            resolved_metric_kwargs.update(metric_kwargs)

        super().__init__(
            activation_parser_df=activation_parser_df,
            train_data_loader=train_data_loader,
            layers=layers,
            batch_size=batch_size,
            sub_batch_size=sub_batch_size,
            device=device,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            outlier_class=outlier_class,
            max_rows=max_rows,
            metric="knn",
            metric_kwargs=resolved_metric_kwargs,
        )


class MultiMetricHallucinationEvaluator(HallucinationEvaluator):
    """Hallucination evaluator that computes a list of OOD metrics in one pass.

    This evaluator reuses cached baseline/test embeddings and executes multiple
    metric functions over the same labeled records.

    Metric specification format:
    - ``"knn"`` or ``"cosine"`` or ``"mahalanobis"`` (string metric key)
    - callable metric function
    - dict spec with keys:
        - ``metric`` (required): str key or callable
        - ``kwargs`` (optional): dict kwargs for this metric
        - ``name`` (optional): name/prefix for collision handling
        - ``prefix`` (optional): explicit prefix for output keys
    """

    def __init__(
        self,
        activation_parser_df: Any,
        train_data_loader,
        *,
        metrics: List[Union[str, Callable[..., Dict[str, Any]], Dict[str, Any]]],
        layers: Optional[List[int]] = None,
        batch_size: int = 64,
        sub_batch_size: int = 32,
        device: str = 'cuda',
        num_workers: int = 4,
        persistent_workers: bool = False,
        outlier_class: int = 1,
        max_rows: int = 10000000,
        default_metric_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not metrics:
            raise ValueError("metrics must contain at least one metric specification")

        # Parent init provides embedding computation/caching + label assignment.
        super().__init__(
            activation_parser_df=activation_parser_df,
            train_data_loader=train_data_loader,
            layers=layers,
            batch_size=batch_size,
            sub_batch_size=sub_batch_size,
            device=device,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            outlier_class=outlier_class,
            max_rows=max_rows,
            metric="mahalanobis",
            metric_kwargs=None,
        )

        self.default_metric_kwargs = default_metric_kwargs or {}
        self.metric_specs = [self._normalize_metric_spec(spec) for spec in metrics]

    def _normalize_metric_spec(
        self, spec: Union[str, Callable[..., Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(spec, dict):
            if "metric" not in spec:
                raise ValueError("Metric dict spec must include a 'metric' key")
            metric_obj = spec["metric"]
            metric_name, metric_fn = self._resolve_metric(metric_obj)
            per_metric_kwargs = dict(self.default_metric_kwargs)
            per_metric_kwargs.update(spec.get("kwargs", {}) or {})

            return {
                "name": str(spec.get("name") or metric_name),
                "metric_fn": metric_fn,
                "metric_kwargs": per_metric_kwargs,
                "prefix": spec.get("prefix"),
            }

        metric_name, metric_fn = self._resolve_metric(spec)
        return {
            "name": str(metric_name),
            "metric_fn": metric_fn,
            "metric_kwargs": dict(self.default_metric_kwargs),
            "prefix": None,
        }

    def _compute_stats_for_metric(
        self,
        *,
        metric_fn: Callable[..., Dict[str, Any]],
        metric_kwargs: Dict[str, Any],
        baseline_embeddings: List[Dict[str, Any]],
        labeled_test_embeddings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            return metric_fn(
                baseline_embeddings,
                labeled_test_embeddings,
                outlier_class=self.outlier_class,
                **metric_kwargs,
            )
        except TypeError:
            return metric_fn(
                baseline_embeddings,
                labeled_test_embeddings,
                **metric_kwargs,
            )

    @staticmethod
    def _prefix_stats_keys(stats: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {f"{prefix}_{k}": v for k, v in stats.items()}

    def _merge_metric_stats(self, aggregate: Dict[str, Any], name: str, prefix: Optional[str], stats: Dict[str, Any]) -> None:
        if not stats:
            return

        explicit_prefix = str(prefix).strip() if prefix is not None else None
        if explicit_prefix:
            aggregate.update(self._prefix_stats_keys(stats, explicit_prefix))
            return

        collisions = set(aggregate.keys()).intersection(stats.keys())
        if collisions:
            aggregate.update(self._prefix_stats_keys(stats, str(name)))
            return

        aggregate.update(stats)

    def _compute_all_metrics(
        self,
        baseline_embeddings: List[Dict[str, Any]],
        labeled_test_embeddings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for metric_spec in self.metric_specs:
            metric_name = metric_spec["name"]
            metric_fn = metric_spec["metric_fn"]
            metric_kwargs = metric_spec["metric_kwargs"]
            metric_prefix = metric_spec["prefix"]

            logger.info(f"Computing OOD stats via metric='{metric_name}'...")
            metric_stats = self._compute_stats_for_metric(
                metric_fn=metric_fn,
                metric_kwargs=metric_kwargs,
                baseline_embeddings=baseline_embeddings,
                labeled_test_embeddings=labeled_test_embeddings,
            )
            self._merge_metric_stats(out, str(metric_name), metric_prefix, metric_stats)

        return out

    def compute(self, data_loader, model) -> Dict[str, Any]:
        logger.info("Starting multi-metric hallucination detection evaluation...")

        baseline_embeddings = self._compute_baseline_embeddings(model)
        test_embeddings = self._compute_test_embeddings(data_loader, model)
        labeled_test_embeddings = self._assign_hallucination_labels(test_embeddings)

        stats = self._compute_all_metrics(baseline_embeddings, labeled_test_embeddings)

        logger.info("Evaluation complete!")
        logger.info(f"Results: {stats}")
        return stats

    def compute_from_embeddings(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Computing multi-metric hallucination stats from accumulated embeddings...")

        if self._baseline_embeddings is None:
            raise ValueError("Baseline embeddings not available. Call compute() first or set cache manually.")

        labeled_embeddings = self._assign_hallucination_labels(embeddings)
        stats = self._compute_all_metrics(self._baseline_embeddings, labeled_embeddings)

        logger.info("Evaluation complete!")
        logger.info(f"Results: {stats}")
        return stats


# Example of how to create other metric evaluators:
#
# class AccuracyEvaluator(MetricEvaluator):
#     """Example metric evaluator for computing accuracy."""
#
#     def __init__(self, num_classes: int = 2):
#         self.num_classes = num_classes
#
#     def compute(self, data_loader, model) -> Dict[str, Any]:
#         model.eval()
#         correct = 0
#         total = 0
#
#         with torch.no_grad():
#             for batch in data_loader:
#                 outputs = model(batch['input'])
#                 predictions = outputs.argmax(dim=-1)
#                 correct += (predictions == batch['labels']).sum().item()
#                 total += batch['labels'].size(0)
#
#         accuracy = correct / total if total > 0 else 0.0
#         return {'accuracy': accuracy, 'correct': correct, 'total': total}
