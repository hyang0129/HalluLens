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
