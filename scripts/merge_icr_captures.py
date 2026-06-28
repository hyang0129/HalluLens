#!/usr/bin/env python3
"""Merge two icr_capture (inference_capture_v1) dirs into one.

Arrays are RAW memmaps (no .npy header), row-major along the sample axis. The
capture PRE-ALLOCATES config["n_samples"] rows but only writes the samples that
succeed (failed/filtered samples are dropped, leaving an untouched tail); the
valid row count is therefore the meta.jsonl line count (n_written), NOT config
n_samples.

We append only each dir's WRITTEN prefix: A's first nA_w rows then B's first
nB_w rows. Appending the full pre-allocated file (the old bug) inserts A's
untouched tail rows between A and B, shifting every B sample out of alignment
with its label -- silently wrecking supervised probes (act_vit -> ~chance AUROC)
while contrastive/kNN readouts mask it.

A occupies merged rows [0, nA_w); B occupies [nA_w, nA_w+nB_w). meta.jsonl
sample_index for B is shifted by nA_w.

prompt_activations.npy is skipped by default: not read by MemmapContrastiveDataset
/ ACTViTDataset / MemmapActivationParser, and ~6 TB. Drop from --skip to merge it.

Usage:
  python scripts/merge_icr_captures.py --a <dirA> --b <dirB> --out <dir>
"""
import argparse, json, shutil, sys
from pathlib import Path

SKIP_DEFAULT = ["prompt_activations.npy"]
CHUNK = 1 << 26  # 64 MiB

def _meta_count(d):
    with open(d / "meta.jsonl") as r:
        return sum(1 for line in r if line.strip())

def _copy_prefix(src, w, nbytes):
    remaining = nbytes
    with open(src, "rb") as r:
        while remaining > 0:
            chunk = r.read(min(CHUNK, remaining))
            if not chunk:
                break
            w.write(chunk)
            remaining -= len(chunk)
    return nbytes - remaining

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
    nA, nB = int(cfgA["n_samples"]), int(cfgB["n_samples"])      # allocated rows
    nA_w, nB_w = _meta_count(A), _meta_count(B)                  # written rows
    if nA_w > nA or nB_w > nB:
        sys.exit(f"ERROR: meta exceeds alloc (A {nA_w}/{nA}, B {nB_w}/{nB})")
    print(f"[merge] A={A.name} nA_w={nA_w}/{nA}  B={B.name} nB_w={nB_w}/{nB}  merged_n={nA_w+nB_w}", flush=True)

    names = sorted({q.name for q in A.glob("*.npy")} | {q.name for q in B.glob("*.npy")})
    for name in names:
        if name in skip:
            print(f"[merge] SKIP  {name}", flush=True); continue
        fa, fb = A / name, B / name
        if not fa.exists() or not fb.exists():
            sys.exit(f"ERROR: {name} present in only one dir (A={fa.exists()} B={fb.exists()})")
        szA, szB = fa.stat().st_size, fb.stat().st_size
        if szA % nA or szB % nB:
            sys.exit(f"ERROR: {name} size not divisible by alloc n (szA={szA}/nA={nA}, szB={szB}/nB={nB})")
        rowA, rowB = szA // nA, szB // nB
        if rowA != rowB:
            sys.exit(f"ERROR: {name} per-row bytes differ A={rowA} B={rowB}")
        wbytesA, wbytesB = nA_w * rowA, nB_w * rowB
        out = OUT / name
        print(f"[merge] CONCAT {name}: {wbytesA}+{wbytesB} -> {wbytesA+wbytesB} bytes ({rowA} B/row, written only)", flush=True)
        with open(out, "wb") as w:
            ga = _copy_prefix(fa, w, wbytesA)
            gb = _copy_prefix(fb, w, wbytesB)
        if ga != wbytesA or gb != wbytesB:
            sys.exit(f"ERROR: {name} short read (A {ga}/{wbytesA}, B {gb}/{wbytesB})")

    print(f"[merge] meta.jsonl (B sample_index += {nA_w})", flush=True)
    with open(OUT / "meta.jsonl", "w") as w:
        with open(A / "meta.jsonl") as r:
            for line in r:
                if line.strip(): w.write(line if line.endswith("\n") else line + "\n")
        with open(B / "meta.jsonl") as r:
            for line in r:
                line = line.strip()
                if not line: continue
                d = json.loads(line); d["sample_index"] = int(d["sample_index"]) + nA_w
                w.write(json.dumps(d) + "\n")

    if (A / "generation.jsonl").exists() and (B / "generation.jsonl").exists():
        with open(OUT / "generation.jsonl", "w") as w:
            for d in (A, B):
                with open(d / "generation.jsonl") as r:
                    shutil.copyfileobj(r, w)

    cfgM = dict(cfgA)
    cfgM["n_samples"] = nA_w + nB_w
    cfgM["_merged_from"] = [A.name, B.name]
    cfgM["_merged_written_rows"] = [nA_w, nB_w]
    cfgM["_skipped_arrays"] = sorted(skip)
    json.dump(cfgM, open(OUT / "config.json", "w"), indent=2)
    print(f"[merge] DONE -> {OUT}  (config n_samples={nA_w+nB_w})", flush=True)

if __name__ == "__main__":
    main()
