"""Generate P1 imbalance-sweep experiment configs (#140).

One config per (dataset, training-prevalence). Each runs the 3 arms
(act_vit, act_vit_cw, contrastive_logprob_recon_b5) at fixed total N with the
target positive (halu) prevalence, 3 seeds. Fixed-N picked from train-pool
availability so the extreme prevalences (5/90%) are reachable:
  hotpotqa pool ~45k (62% halu) -> N=15000 ; triviaqa pool ~9.9k (23% halu) -> N=2200
"""
import json
from pathlib import Path

_CFG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "experiments"
_METHODS = ["act_vit", "act_vit_cw", "contrastive_logprob_recon_b5"]
_PREVS = [5, 10, 20, 35, 50, 70, 90]
_DATASETS = {"hotpotqa": ("hotpotqa_memmap", 15000), "triviaqa": ("triviaqa_memmap", 2200)}


def main():
    n = 0
    for short, (ds, total_n) in _DATASETS.items():
        for p in _PREVS:
            name = f"imbalance_{short}_prev{p:02d}"
            cfg = {
                "device": "auto",
                "num_workers": 4,
                "persistent_workers": True,
                "output_dir": "runs",
                "methods": _METHODS,
                "split_seed": 42,
                "experiment_name": name,
                "dataset": ds,
                "training_seeds": [0, 1, 2],
                "split_seeds": [42, 1, 2],
                "train_prevalence": round(p / 100.0, 2),
                "train_total_n": total_n,
            }
            path = _CFG_DIR / f"{name}.json"
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            n += 1
            print(f"wrote {path.name}  prev={p}%  N={total_n}")
    print(f"Done — {n} configs")


if __name__ == "__main__":
    main()
