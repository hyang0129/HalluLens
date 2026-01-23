"""Tests for loading dummy activations from JSON/NPY and Zarr formats."""
import os
import time
import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from activation_logging.activations_logger import ActivationsLogger, JsonActivationsLogger
from activation_logging.activation_parser import ActivationDataset


JSON_OUTPUT = "json_dummy_data"
ZARR_OUTPUT = "zarr_dummy_data/activations.zarr"


def passthrough_collate(batch):
    return batch


class JsonFullDataset(Dataset):
    def __init__(self, logger: JsonActivationsLogger, keys: list[str]):
        self.logger = logger
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        entry = self.logger.get_entry_by_key(self.keys[idx])
        activations = entry.get("all_layers_activations")
        if activations is None:
            return None
        return activations[0] if len(activations) > 0 else None


class ZarrPartialDataset(Dataset):
    def __init__(self, logger: ActivationsLogger, keys: list[str], layer_idx: int = 0):
        self.logger = logger
        self.keys = keys
        self.layer_idx = layer_idx

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        return self.logger.get_layer_activation(
            self.keys[idx],
            self.layer_idx,
            sequence_mode="response",
        )


class TestDummyActivationLoading(unittest.TestCase):
    def setUp(self):
        self.json_exists = os.path.exists(JSON_OUTPUT)
        self.zarr_exists = os.path.exists(ZARR_OUTPUT)

    def test_json_loading(self):
        if not self.json_exists:
            self.skipTest("JSON dummy data not found")

        logger = JsonActivationsLogger(output_dir=JSON_OUTPUT, read_only=True, verbose=False)
        keys = logger.list_entries()
        self.assertGreaterEqual(len(keys), 100)

        entry = logger.get_entry_by_key(keys[0])
        activations = entry.get("all_layers_activations")
        self.assertIsNotNone(activations)
        self.assertEqual(len(activations), 32)

        first_layer = activations[0]
        self.assertIsNotNone(first_layer)
        self.assertEqual(tuple(first_layer.shape), (1, 128, 4096))
        self.assertEqual(first_layer.dtype, torch.float16)

    def test_zarr_loading(self):
        if not self.zarr_exists:
            self.skipTest("Zarr dummy data not found")

        logger = ActivationsLogger(lmdb_path=ZARR_OUTPUT, read_only=True, target_layers="all", sequence_mode="all")
        keys = logger.list_entries()
        self.assertGreaterEqual(len(keys), 100)

        entry = logger.get_entry_by_key(keys[0])
        activations = entry.get("all_layers_activations")
        self.assertIsNotNone(activations)
        self.assertEqual(len(activations), 32)

        first_layer = activations[0]
        self.assertIsNotNone(first_layer)
        self.assertEqual(tuple(first_layer.shape), (128, 4096))
        self.assertEqual(first_layer.dtype, torch.float16)

        layer_act = logger.get_layer_activation(keys[0], 0, sequence_mode="response")
        self.assertIsNotNone(layer_act)
        self.assertEqual(tuple(layer_act.shape), (1, 64, 4096))
        self.assertEqual(layer_act.dtype, torch.float16)

    def test_dataloader_json_vs_zarr(self):
        if not self.json_exists or not self.zarr_exists:
            self.skipTest("Dummy data not found")

        json_logger = JsonActivationsLogger(output_dir=JSON_OUTPUT, read_only=True, verbose=False)
        json_keys = json_logger.list_entries()

        zarr_logger = ActivationsLogger(lmdb_path=ZARR_OUTPUT, read_only=True, target_layers="all", sequence_mode="all")
        zarr_keys = zarr_logger.list_entries()

        sample_count = min(50, len(json_keys), len(zarr_keys))
        if sample_count < 8:
            self.skipTest("Not enough samples for DataLoader test")

        df = pd.DataFrame(
            {
                "prompt_hash": json_keys[:sample_count],
                "halu": [0] * sample_count,
                "split": ["train"] * sample_count,
            }
        )

        json_dataset = JsonFullDataset(json_logger, json_keys[:sample_count])
        zarr_dataset = ZarrPartialDataset(zarr_logger, zarr_keys[:sample_count], layer_idx=0)

        json_sample = json_dataset[0]
        self.assertIsNotNone(json_sample)

        zarr_sample = zarr_dataset[0]
        self.assertIsNotNone(zarr_sample)

        timings = {}

        max_batches = 50
        warmup_batches = 5

        for name, dataset in ("json", json_dataset), ("zarr", zarr_dataset):
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=4,
                collate_fn=passthrough_collate,
                drop_last=True,
                persistent_workers=True,
            )
            for i, _batch in enumerate(loader):
                if i >= warmup_batches:
                    break

            start = time.perf_counter()
            for i, _batch in enumerate(loader):
                if i >= max_batches:
                    break
            elapsed = time.perf_counter() - start
            timings[name] = elapsed
            self.assertLess(elapsed, 120.0)

        print(
            f"DataLoader timing ({max_batches} batches, 4 workers): "
            f"JSON(full)={timings['json']:.3f}s, Zarr(partial)={timings['zarr']:.3f}s"
        )


if __name__ == "__main__":
    unittest.main()
