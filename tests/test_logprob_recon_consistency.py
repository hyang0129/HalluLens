from __future__ import annotations

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
