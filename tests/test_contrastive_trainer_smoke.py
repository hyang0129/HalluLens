import os
import tempfile

import torch


def test_contrastive_trainer_smoke_cpu_checkpoint_and_resume():
    from activation_research.trainer import ContrastiveTrainer, ContrastiveTrainerConfig

    class DummyContrastiveDataset(torch.utils.data.Dataset):
        def __init__(self, n: int, *, seq_len: int, dim: int):
            self.n = int(n)
            self.seq_len = int(seq_len)
            self.dim = int(dim)

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            _ = idx
            return {
                "layer1_activations": torch.randn(1, self.seq_len, self.dim),
                "layer2_activations": torch.randn(1, self.seq_len, self.dim),
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

    train_ds = DummyContrastiveDataset(16, seq_len=4, dim=8)
    val_ds = DummyContrastiveDataset(8, seq_len=4, dim=8)

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
        )
        trainer2 = ContrastiveTrainer(model2, config=cfg2)
        trainer2.fit(train_ds, val_ds)


def test_layer_aware_contrastive_trainer_smoke_cpu_checkpoint_and_resume():
    from activation_research.trainer import LayerAwareContrastiveTrainer, LayerAwareContrastiveTrainerConfig

    class DummyContrastiveDataset(torch.utils.data.Dataset):
        def __init__(self, n: int, *, seq_len: int, dim: int, num_layers: int):
            self.n = int(n)
            self.seq_len = int(seq_len)
            self.dim = int(dim)
            self.num_layers = int(num_layers)

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            layer1_idx = idx % self.num_layers
            layer2_idx = (idx + 1) % self.num_layers
            return {
                "layer1_activations": torch.randn(1, self.seq_len, self.dim),
                "layer2_activations": torch.randn(1, self.seq_len, self.dim),
                "layer1_idx": int(layer1_idx),
                "layer2_idx": int(layer2_idx),
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

    train_ds = DummyContrastiveDataset(16, seq_len=4, dim=8, num_layers=8)
    val_ds = DummyContrastiveDataset(8, seq_len=4, dim=8, num_layers=8)

    with tempfile.TemporaryDirectory() as tmp:
        model = DummyLayerAwareEncoder(seq_len=4, dim=8, num_layers=8)
        cfg = LayerAwareContrastiveTrainerConfig(
            max_epochs=1,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            checkpoint_dir=tmp,
            save_every=1,
            use_labels=True,
        )
        trainer = LayerAwareContrastiveTrainer(model, config=cfg)
        trainer.fit(train_ds, val_ds)

        last_path = os.path.join(tmp, "layer_aware_contrastive_last.pt")
        assert os.path.exists(last_path)

        model2 = DummyLayerAwareEncoder(seq_len=4, dim=8, num_layers=8)
        cfg2 = LayerAwareContrastiveTrainerConfig(
            max_epochs=2,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            checkpoint_dir=tmp,
            save_every=1,
            resume_from=last_path,
            use_labels=True,
        )
        trainer2 = LayerAwareContrastiveTrainer(model2, config=cfg2)
        trainer2.fit(train_ds, val_ds)
