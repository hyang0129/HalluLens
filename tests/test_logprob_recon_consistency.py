from __future__ import annotations

import pandas as pd
import torch


class _TinyContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.rows = [
            torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
            torch.tensor([[0.0, 1.0], [0.1, 0.9]]),
            torch.tensor([[-1.0, 0.0], [-0.9, -0.1]]),
            torch.tensor([[0.0, -1.0], [-0.1, -0.9]]),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # (views=2, sequence=1, hidden=2)
        return {
            "views_activations": self.rows[index].unsqueeze(1),
            "halu": torch.tensor(index % 2),
            "hashkey": f"row-{index}",
        }


class _TinyReconModel(torch.nn.Module):
    recon_lambda = 0.0

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)

    def forward_with_recon(self, x, **_kwargs):
        return self.projection(x.mean(dim=1)), None


class _FrozenEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        hashkey, value, label = self.rows[index]
        return {
            "views_activations": torch.tensor([[[value, 0.0]]]),
            "halu": torch.tensor(label),
            "hashkey": hashkey,
        }


class _MeanEmbedding(torch.nn.Module):
    def forward(self, x, **_kwargs):
        return x.mean(dim=1)


def test_logprob_recon_restores_best_validation_weights(monkeypatch, tmp_path):
    from activation_research import training

    validation_losses = iter([0.1, 1.0])

    def fake_evaluate(*_args, **_kwargs):
        return next(validation_losses), 0.0, 0.0

    monkeypatch.setattr(training, "evaluate", fake_evaluate)

    step_states = []
    original_step = torch.optim.Adam.step

    def recording_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        parameter = optimizer.param_groups[0]["params"][0]
        step_states.append(parameter.detach().cpu().clone())
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)

    model = _TinyReconModel()
    dataset = _TinyContrastiveDataset()
    training.train_contrastive_logprob_recon(
        model,
        train_dataset=dataset,
        test_dataset=dataset,
        epochs=2,
        batch_size=4,
        sub_batch_size=4,
        lr=1e-2,
        device="cpu",
        num_workers=0,
        checkpoint_dir=tmp_path,
        persistent_workers=False,
        use_labels=False,
        use_infinite_index_stream=False,
        select_on_val=True,
    )

    assert len(step_states) == 2
    assert not torch.equal(step_states[0], step_states[1])
    assert torch.equal(model.projection.weight.detach().cpu(), step_states[0])


def test_logprob_recon_restores_maximum_validation_score(monkeypatch, tmp_path):
    from activation_research import training

    monkeypatch.setattr(
        training,
        "evaluate",
        lambda *_args, **_kwargs: (0.5, 0.0, 0.0),
    )
    validation_scores = iter([0.9, 0.2])

    step_states = []
    original_step = torch.optim.Adam.step

    def recording_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        parameter = optimizer.param_groups[0]["params"][0]
        step_states.append(parameter.detach().cpu().clone())
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)

    model = _TinyReconModel()
    dataset = _TinyContrastiveDataset()
    summary = training.train_contrastive_logprob_recon(
        model,
        train_dataset=dataset,
        test_dataset=dataset,
        epochs=2,
        batch_size=4,
        sub_batch_size=4,
        lr=1e-2,
        device="cpu",
        num_workers=0,
        checkpoint_dir=tmp_path,
        persistent_workers=False,
        use_labels=False,
        use_infinite_index_stream=False,
        select_on_val=True,
        validation_score_fn=lambda _model: next(validation_scores),
        validation_score_name="validation_knn_auroc",
    )

    assert len(step_states) == 2
    assert torch.equal(model.projection.weight.detach().cpu(), step_states[0])
    assert summary == {
        "checkpoint_selection": "maximum_validation_knn_auroc",
        "best_validation_score": 0.9,
        "best_validation_loss": None,
        "best_validation_epoch": 1,
    }
    checkpoint = torch.load(
        tmp_path / "contrastive_last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["validation_score_name"] == "validation_knn_auroc"
    assert checkpoint["validation_score"] == 0.2
    assert checkpoint["best_validation_score"] == 0.9


def test_validation_knn_scorer_uses_held_out_labels_and_fixed_bank():
    from scripts.run_experiment import _make_validation_knn_scorer

    train_rows = [
        ("train-t0", -0.1, 0),
        ("train-t1", 0.0, 0),
        ("train-t2", 0.1, 0),
        ("train-h0", 5.0, 1),
        ("train-h1", 10.0, 1),
    ]
    val_rows = [
        ("val-t0", -0.05, 0),
        ("val-t1", 0.05, 0),
        ("val-h0", 7.5, 1),
        ("val-h1", 15.0, 1),
    ]
    all_rows = train_rows + val_rows
    labels = pd.DataFrame(
        {
            "prompt_hash": [row[0] for row in all_rows],
            "halu": [row[2] for row in all_rows],
        }
    )
    scorer = _make_validation_knn_scorer(
        activation_parser_df=labels,
        train_dataset=_FrozenEmbeddingDataset(train_rows),
        val_dataset=_FrozenEmbeddingDataset(val_rows),
        device="cpu",
        num_workers=0,
        eval_batch_size=2,
        sub_batch_size=1,
        outlier_class=1,
        k=1,
        max_train_size=100,
        sample_seed=0,
    )

    assert scorer(_MeanEmbedding()) == 1.0
