#!/usr/bin/env python
"""Re-run ONLY the KNN eval of a trained contrastive_actvit checkpoint across a
sweep of fixed k values (calibrate_k off), to check k-sensitivity without
retraining. Mirrors run_contrastive_actvit's eval block exactly (same parsers,
combined train+test df for labels, flip/outlier_class convention).

NOTE: loads contrastive_last.pt, which is the LAST-epoch checkpoint. The fair
run's reported number is on the best-val weights (restored in-memory, not
persisted). Compare sweep@<calibrated k> against the reported number to gauge
how close last-epoch is to best-val.
"""
import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from activation_research.act_vit import ContrastiveACTViT
from activation_research.contrastive_actvit_dataset import ContrastiveACTViTDataset
from activation_research.memmap_activation_parser import MemmapActivationParser
from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator


def parse_layer_range(spec):
    if spec is None:
        return None
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ks", default="50,100,200,500,1000")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()

    dcfg = json.load(open(a.dataset))
    mcfg = json.load(open(a.method))
    icr = dcfg["icr_capture"]
    mp = mcfg["model_params"]
    ecfg = mcfg["evaluation"]
    data_cfg = mcfg["data"]
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    train_parser = MemmapActivationParser(
        icr["train_dir"], random_seed=a.split_seed, split_strategy="three_way"
    )
    test_parser = MemmapActivationParser(
        icr["test_dir"], random_seed=a.split_seed, split_strategy="none"
    )
    train_idx = train_parser.df[train_parser.df["split"] == "train"]["sample_index"].values
    test_idx = test_parser.df["sample_index"].values

    def hmap(p):
        return dict(zip(p.df["sample_index"].astype(int), p.df["prompt_hash"].astype(str)))

    rel = parse_layer_range(data_cfg["relevant_layers"]) if data_cfg.get("relevant_layers") else None
    ds_kw = dict(relevant_layers=rel, view_aug="raw")
    train_eval = ContrastiveACTViTDataset(
        icr["train_dir"], train_idx, num_views=1, seed=a.seed, hashkey_map=hmap(train_parser), **ds_kw
    )
    test_eval = ContrastiveACTViTDataset(
        icr["test_dir"], test_idx, num_views=1, seed=a.seed, hashkey_map=hmap(test_parser), **ds_kw
    )

    model = ContrastiveACTViT(
        n_layers=train_eval.n_layers, n_tokens=train_eval.n_tokens, input_dim=dcfg["input_dim"],
        final_dim=mp.get("final_dim", 256), recon_seq_len=mp.get("recon_seq_len", 64),
        recon_hidden_dim=mp.get("recon_hidden_dim", 256), recon_lambda=mp.get("recon_lambda", 1.0),
        logprob_var_threshold=mp.get("logprob_var_threshold", 1e-4),
        L_p=mp.get("L_p", 8), N_p=mp.get("N_p", 100),
        patch_h=mp.get("patch_h", 2), patch_w=mp.get("patch_w", 10),
        d_adapter=mp.get("d_adapter", 256), d_model=mp.get("d_model", 256),
        num_heads=mp.get("num_heads", 8), depth=mp.get("depth", 4),
        mlp_ratio=mp.get("mlp_ratio", 4.0), dropout=mp.get("dropout", 0.1),
        normalize_output=mp.get("normalize_output", False),
    )
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    model.load_state_dict(sd)
    model.to(dev).eval()
    ckpt_epoch = ck.get("epoch", "?") if isinstance(ck, dict) else "?"

    ev_bs = ecfg.get("eval_batch_size", 64)
    combined = pd.concat([train_parser.df, test_parser.df], ignore_index=True)
    flip = bool(ecfg.get("flip_auroc", False))
    oc = 0 if flip else dcfg.get("outlier_class", 1)
    metric = ecfg["knn_params"].get("metric", "euclidean")
    max_train = ecfg["knn_params"].get("max_train_size", 200000)

    print(f"# checkpoint={a.checkpoint} (saved epoch={ckpt_epoch}, last-epoch weights)")
    print(f"# normalize_output={mp.get('normalize_output', False)} flip_auroc={flip} outlier_class={oc}")
    print(f"{'k':>6}  {'auroc':>8}")
    for k in [int(x) for x in a.ks.split(",")]:
        knn = {"k": k, "metric": metric, "calibrate_k": False,
               "max_train_size": max_train, "sample_seed": a.seed}
        train_loader = DataLoader(train_eval, batch_size=ev_bs, shuffle=False, num_workers=a.num_workers)
        eval_loader = DataLoader(test_eval, batch_size=ev_bs, shuffle=False, num_workers=a.num_workers)
        evaluator = MultiMetricHallucinationEvaluator(
            activation_parser_df=combined, train_data_loader=train_loader,
            metrics=[{"metric": "knn", "kwargs": knn, "train_selection": "all"}],
            batch_size=ev_bs, sub_batch_size=ecfg.get("sub_batch_size", 64), device=str(dev),
            num_workers=a.num_workers, persistent_workers=False, outlier_class=oc,
        )
        stats = evaluator.compute(eval_loader, model)
        print(f"{k:>6}  {stats.get('knn_auroc', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
