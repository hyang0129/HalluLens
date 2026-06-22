#!/usr/bin/env python3
"""Merge two icr_capture (inference_capture_v1) dirs into one.

The arrays are RAW memmaps (no .npy header) stored row-major along the sample
axis, so concatenating dir A then dir B is a pure byte-append per file. Dir A's
rows occupy merged positions [0, nA); dir B's occupy [nA, nA+nB). meta.jsonl
sample_index for B is shifted by nA (A's config n_samples = its array row count).

prompt_activations.npy is skipped by default: it is NOT read by
MemmapContrastiveDataset / ACTViTDataset / MemmapActivationParser, and it is
~6 TB. Drop it from --skip to merge it too.

Usage:
  python scripts/merge_icr_captures.py --a <dirA> --b <dirB> --out <dir>
"""
import argparse, json, shutil, sys
from pathlib import Path

SKIP_DEFAULT = ["prompt_activations.npy"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first dir (rows go first, e.g. _0-50000)")
    p.add_argument("--b", required=True, help="second/delta dir (rows appended)")
    p.add_argument("--out", required=True)
    p.add_argument("--skip", nargs="*", default=SKIP_DEFAULT)
    args = p.parse_args()
    A, B, OUT = Path(args.a), Path(args.b), Path(args.out)
    skip = set(args.skip)
    for d in (A, B):
        if not (d / "config.json").exists():
            sys.exit(f"ERROR: {d} missing config.json")
    OUT.mkdir(parents=True, exist_ok=True)

    cfgA = json.load(open(A / "config.json"))
    cfgB = json.load(open(B / "config.json"))
    nA, nB = int(cfgA["n_samples"]), int(cfgB["n_samples"])
    print(f"[merge] A={A.name} nA={nA}  B={B.name} nB={nB}  merged_n={nA+nB}", flush=True)

    names = sorted({q.name for q in A.glob('*.npy')} | {q.name for q in B.glob('*.npy')})
    for name in names:
        if name in skip:
            print(f"[merge] SKIP  {name}", flush=True); continue
        fa, fb = A / name, B / name
        if not fa.exists() or not fb.exists():
            sys.exit(f"ERROR: {name} present in only one dir (A={fa.exists()} B={fb.exists()})")
        szA, szB = fa.stat().st_size, fb.stat().st_size
        if szA % nA or szB % nB:
            sys.exit(f"ERROR: {name} size not divisible by n (szA={szA}/nA={nA}, szB={szB}/nB={nB})")
        rowA, rowB = szA // nA, szB // nB
        if rowA != rowB:
            sys.exit(f"ERROR: {name} per-row bytes differ A={rowA} B={rowB} (incompatible capture geometry)")
        out = OUT / name
        print(f"[merge] CONCAT {name}: {szA}+{szB} -> {szA+szB} bytes ({rowA} B/row)", flush=True)
        with open(out, "wb") as w:
            for src in (fa, fb):
                with open(src, "rb") as r:
                    shutil.copyfileobj(r, w, length=1 << 26)  # 64 MiB chunks

    # meta.jsonl: A verbatim, then B with sample_index += nA
    print("[merge] meta.jsonl (B sample_index += %d)" % nA, flush=True)
    with open(OUT / "meta.jsonl", "w") as w:
        with open(A / "meta.jsonl") as r:
            for line in r:
                if line.strip(): w.write(line if line.endswith("\n") else line + "\n")
        nb_meta = 0
        with open(B / "meta.jsonl") as r:
            for line in r:
                line = line.strip()
                if not line: continue
                d = json.loads(line); d["sample_index"] = int(d["sample_index"]) + nA
                w.write(json.dumps(d) + "\n"); nb_meta += 1

    # generation.jsonl: concat verbatim (not read by training) if present
    if (A / "generation.jsonl").exists() and (B / "generation.jsonl").exists():
        with open(OUT / "generation.jsonl", "w") as w:
            for d in (A, B):
                with open(d / "generation.jsonl") as r:
                    shutil.copyfileobj(r, w)

    cfgM = dict(cfgA)
    cfgM["n_samples"] = nA + nB
    cfgM["_merged_from"] = [A.name, B.name]
    cfgM["_skipped_arrays"] = sorted(skip)
    json.dump(cfgM, open(OUT / "config.json", "w"), indent=2)
    print(f"[merge] DONE -> {OUT}  (config n_samples={nA+nB})", flush=True)

if __name__ == "__main__":
    main()
