import os
import tempfile

import pytest
import torch


def test_eval_auto_sub_batch_selection_matches_worker_target_example():
    from activation_research.trainer import _select_eval_sub_batch_size

    sub_batch_size = _select_eval_sub_batch_size(
        full_batch_size=512,
        expected_full_batches=17,
        num_workers=30,
        target_worker_multiplier=2.0,
        min_sub_batch_size=1,
    )
    assert sub_batch_size == 128


def test_eval_explicit_sub_batch_size_override_is_used():
    from activation_research.trainer import ContrastiveTrainer, ContrastiveTrainerConfig

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            _ = idx
            return {
                "views_activations": torch.randn(4, 4, 8),
                "view_indices": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "halu": torch.tensor(0.0),
                "hashkey": "hk",
            }

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32, 8))

        def forward(self, x):
            return self.net(x)

    cfg = ContrastiveTrainerConfig(
        max_epochs=1,
        batch_size=512,
        num_workers=30,
        eval_sub_batch_size=128,
        device="cpu",
    )
    trainer = ContrastiveTrainer(DummyEncoder(), config=cfg)
    resolved = trainer._resolve_eval_loader_batch_size(DummyDataset())
    assert resolved == 128


@pytest.mark.parametrize("num_views", [2, 4, 8])
def test_contrastive_trainer_smoke_cpu_checkpoint_and_resume(num_views):
    from activation_research.trainer import ContrastiveTrainer, ContrastiveTrainerConfig

    class DummyContrastiveDataset(torch.utils.data.Dataset):
        def __init__(self, n: int, *, seq_len: int, dim: int, num_views: int):
            self.n = int(n)
            self.seq_len = int(seq_len)
            self.dim = int(dim)
            self.num_views = int(num_views)

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            _ = idx
            return {
                "views_activations": torch.randn(self.num_views, self.seq_len, self.dim),
                "view_indices": torch.arange(self.num_views, dtype=torch.long),
                "halu": torch.tensor(0.0),
                "hashkey": f"hk_{idx}",
            }

    class DummyEncoder(torch.nn.Module):
        def __init__(self, seq_len: int, dim: int, out_dim: int = 16):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(seq_len * dim, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    train_ds = DummyContrastiveDataset(16, seq_len=4, dim=8, num_views=num_views)
    val_ds = DummyContrastiveDataset(8, seq_len=4, dim=8, num_views=num_views)

    with tempfile.TemporaryDirectory() as tmp:
        model = DummyEncoder(seq_len=4, dim=8)
        cfg = ContrastiveTrainerConfig(
            max_epochs=1,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            checkpoint_dir=tmp,
            save_every=1,
            use_labels=True,
            num_views=num_views,
        )
        trainer = ContrastiveTrainer(model, config=cfg)
        trainer.fit(train_ds, val_ds)

        last_path = os.path.join(tmp, "contrastive_last.pt")
        assert os.path.exists(last_path)

        # Resume for one more epoch (sanity check load path + optimizer state).
        model2 = DummyEncoder(seq_len=4, dim=8)
        cfg2 = ContrastiveTrainerConfig(
            max_epochs=2,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            checkpoint_dir=tmp,
            save_every=1,
            resume_from=last_path,
            use_labels=True,
            num_views=num_views,
        )
        trainer2 = ContrastiveTrainer(model2, config=cfg2)
        trainer2.fit(train_ds, val_ds)


@pytest.mark.parametrize("num_views", [2, 4, 8])
def test_layer_aware_contrastive_trainer_smoke_cpu_checkpoint_and_resume(num_views):
    from activation_research.trainer import LayerAwareContrastiveTrainer, LayerAwareContrastiveTrainerConfig

    class DummyContrastiveDataset(torch.utils.data.Dataset):
        def __init__(self, n: int, *, seq_len: int, dim: int, num_layers: int, num_views: int):
            self.n = int(n)
            self.seq_len = int(seq_len)
            self.dim = int(dim)
            self.num_layers = int(num_layers)
            self.num_views = int(num_views)

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            start = idx % self.num_layers
            view_indices = torch.tensor(
                [(start + offset) % self.num_layers for offset in range(self.num_views)],
                dtype=torch.long,
            )
            return {
                "views_activations": torch.randn(self.num_views, self.seq_len, self.dim),
                "view_indices": view_indices,
                "halu": torch.tensor(0.0),
                "hashkey": f"hk_{idx}",
            }

    class DummyLayerAwareEncoder(torch.nn.Module):
        def __init__(self, seq_len: int, dim: int, out_dim: int = 16, num_layers: int = 8, layer_dim: int = 4):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(seq_len * dim, out_dim),
            )
            self.layer_emb = torch.nn.Embedding(int(num_layers), int(layer_dim))
            self.layer_proj = torch.nn.Linear(int(layer_dim), int(out_dim))

        def forward(self, x, *, layer_idx=None, **kwargs):
            _ = kwargs
            z = self.net(x)
            if layer_idx is None:
                return z
            if isinstance(layer_idx, int):
                layer_idx = torch.full((z.shape[0],), layer_idx, dtype=torch.long, device=z.device)
            elif isinstance(layer_idx, torch.Tensor):
                layer_idx = layer_idx.to(device=z.device, dtype=torch.long)
                if layer_idx.dim() == 0:
                    layer_idx = layer_idx.view(1).expand(z.shape[0])
                elif layer_idx.dim() > 1:
                    layer_idx = layer_idx.view(-1)
            return z + self.layer_proj(self.layer_emb(layer_idx))

    train_ds = DummyContrastiveDataset(16, seq_len=4, dim=8, num_layers=16, num_views=num_views)
    val_ds = DummyContrastiveDataset(8, seq_len=4, dim=8, num_layers=16, num_views=num_views)

    with tempfile.TemporaryDirectory() as tmp:
        model = DummyLayerAwareEncoder(seq_len=4, dim=8, num_layers=16)
        cfg = LayerAwareContrastiveTrainerConfig(
            max_epochs=1,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            checkpoint_dir=tmp,
            save_every=1,
            use_labels=True,
            num_views=num_views,
        )
        trainer = LayerAwareContrastiveTrainer(model, config=cfg)
        trainer.fit(train_ds, val_ds)

        last_path = os.path.join(tmp, "layer_aware_contrastive_last.pt")
        assert os.path.exists(last_path)

        model2 = DummyLayerAwareEncoder(seq_len=4, dim=8, num_layers=16)
        cfg2 = LayerAwareContrastiveTrainerConfig(
            max_epochs=2,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            checkpoint_dir=tmp,
            save_every=1,
            resume_from=last_path,
            use_labels=True,
            num_views=num_views,
        )
        trainer2 = LayerAwareContrastiveTrainer(model2, config=cfg2)
        trainer2.fit(train_ds, val_ds)


@pytest.mark.parametrize("num_views", [2, 4, 8])
def test_contrastive_collate_kview_shapes(num_views):
    from activation_research.training import _contrastive_collate_kview

    batch = [
        {
            "views_activations": torch.randn(num_views, 5, 7),
            "view_indices": torch.arange(num_views, dtype=torch.long),
            "halu": torch.tensor(0.0),
            "hashkey": "hk_0",
            "input_length": 5,
        },
        {
            "views_activations": torch.randn(num_views, 5, 7),
            "view_indices": torch.arange(num_views, dtype=torch.long),
            "halu": torch.tensor(1.0),
            "hashkey": "hk_1",
            "input_length": 5,
        },
    ]

    out = _contrastive_collate_kview(batch)
    assert tuple(out["views_activations"].shape) == (2, num_views, 5, 7)
    assert tuple(out["view_indices"].shape) == (2, num_views)
    assert tuple(out["halu"].shape) == (2,)
