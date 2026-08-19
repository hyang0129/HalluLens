#!/usr/bin/env python3
"""Merge two or more icr_capture (inference_capture_v1) dirs into one.

Arrays are RAW memmaps (no .npy header), row-major along the sample axis. The
capture PRE-ALLOCATES config["n_samples"] rows but only writes the samples that
succeed (failed/filtered samples are dropped, leaving an untouched tail); the
valid row count is therefore the meta.jsonl line count (n_written), NOT config
n_samples.

We append only each dir's WRITTEN prefix, in --inputs order: input[0]'s first
n0_w rows, then input[1]'s first n1_w rows, ..., then input[k]'s first nk_w
rows. Appending the full pre-allocated file (the old bug) inserts an earlier
input's untouched tail rows before the next input's rows, shifting every
downstream sample out of alignment with its label -- silently wrecking
supervised probes (act_vit -> ~chance AUROC) while contrastive/kNN readouts
mask it.

input[0] occupies merged rows [0, n0_w); input[1] occupies [n0_w, n0_w+n1_w);
input[i] occupies [sum(n_w[:i]), sum(n_w[:i+1])). Each input's meta.jsonl
sample_index is shifted by the cumulative written count of all prior inputs.

prompt_activations.npy is skipped by default: not read by MemmapContrastiveDataset
/ ACTViTDataset / MemmapActivationParser, and ~6 TB. Drop from --skip to merge it.

Usage:
  # N-way (2 or more), ordered:
  python scripts/merge_icr_captures.py --inputs <dir0> <dir1> [<dir2> ...] --out <dir>

  # Pairwise (kept for backward compat; --a/--b == --inputs A B):
  python scripts/merge_icr_captures.py --a <dirA> --b <dirB> --out <dir>
"""
import argparse, json, shutil, sys
from pathlib import Path

SKIP_DEFAULT = ["icr_scores.npy", "prompt_activations.npy", "prompt_token_ids.npy"]
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
    p.add_argument("--a", help="first dir (rows go first, e.g. _0-50000). "
                                "Shorthand for --inputs A B; mutually exclusive with --inputs.")
    p.add_argument("--b", help="second/delta dir (rows appended). Paired with --a.")
    p.add_argument("--inputs", nargs="+", metavar="DIR",
                    help="Ordered list of 2+ input dirs to concatenate (written prefix only). "
                         "Mutually exclusive with --a/--b.")
    p.add_argument("--out", required=True)
    p.add_argument("--skip", nargs="*", default=SKIP_DEFAULT)
    args = p.parse_args()

    if args.inputs:
        if args.a or args.b:
            sys.exit("ERROR: pass either --inputs or --a/--b, not both")
        input_dirs = [Path(x) for x in args.inputs]
    else:
        if not args.a or not args.b:
            sys.exit("ERROR: provide --inputs DIR DIR [DIR ...] or both --a and --b")
        input_dirs = [Path(args.a), Path(args.b)]

    if len(input_dirs) < 2:
        sys.exit(f"ERROR: need at least 2 input dirs to merge, got {len(input_dirs)}")

    OUT = Path(args.out)
    skip = set(args.skip)
    for d in input_dirs:
        if not (d / "config.json").exists():
            sys.exit(f"ERROR: {d} missing config.json")
    OUT.mkdir(parents=True, exist_ok=True)

    configs = [json.load(open(d / "config.json")) for d in input_dirs]
    n_alloc = [int(c["n_samples"]) for c in configs]     # allocated rows, per input
    n_written = [_meta_count(d) for d in input_dirs]      # written rows, per input
    for d, nw, na in zip(input_dirs, n_written, n_alloc):
        if nw > na:
            sys.exit(f"ERROR: meta exceeds alloc for {d} ({nw}/{na})")

    # Config sanity check across ALL inputs: same model, same chat_template
    # convention. Mixing these would silently merge incompatible captures
    # (different model activations, or templated + raw prompts) into one
    # dataset with no way to tell them apart downstream.
    base_name, base_cfg = input_dirs[0].name, configs[0]
    for d, cfg in zip(input_dirs[1:], configs[1:]):
        if cfg.get("model_name") != base_cfg.get("model_name"):
            sys.exit(
                f"ERROR: model_name mismatch: {base_name}={base_cfg.get('model_name')!r} "
                f"vs {d.name}={cfg.get('model_name')!r}"
            )
        if bool(cfg.get("chat_template", False)) != bool(base_cfg.get("chat_template", False)):
            sys.exit(
                f"ERROR: chat_template mismatch: {base_name}="
                f"{bool(base_cfg.get('chat_template', False))} vs {d.name}="
                f"{bool(cfg.get('chat_template', False))} — chat-templated and legacy "
                f"captures must not be merged together."
            )

    total_written = sum(n_written)
    print(
        f"[merge] inputs={[d.name for d in input_dirs]}  n_written={n_written}  "
        f"merged_n={total_written}",
        flush=True,
    )

    names = sorted(set().union(*({q.name for q in d.glob("*.npy")} for d in input_dirs)))
    for name in names:
        if name in skip:
            print(f"[merge] SKIP  {name}", flush=True); continue
        files = [d / name for d in input_dirs]
        missing = [d.name for d, f in zip(input_dirs, files) if not f.exists()]
        if missing:
            sys.exit(f"ERROR: {name} missing in {missing} (present elsewhere)")

        # Array shape/dtype consistency: these are headerless raw memmaps, so
        # the only signal available is per-row byte width (size / alloc_n);
        # a shape or dtype mismatch between inputs shows up as a differing
        # row width here.
        row_bytes = []
        for f, na in zip(files, n_alloc):
            sz = f.stat().st_size
            if sz % na:
                sys.exit(f"ERROR: {name} size not divisible by alloc n for {f} (sz={sz}/n={na})")
            row_bytes.append(sz // na)
        if len(set(row_bytes)) > 1:
            sys.exit(
                f"ERROR: {name} per-row bytes differ across inputs: "
                f"{dict(zip((d.name for d in input_dirs), row_bytes))}"
            )
        row_width = row_bytes[0]

        out = OUT / name
        total_bytes = sum(nw * row_width for nw in n_written)
        print(
            f"[merge] CONCAT {name}: {len(input_dirs)}-way -> {total_bytes} bytes "
            f"({row_width} B/row, written only)",
            flush=True,
        )
        with open(out, "wb") as w:
            for f, nw in zip(files, n_written):
                wbytes = nw * row_width
                got = _copy_prefix(f, w, wbytes)
                if got != wbytes:
                    sys.exit(f"ERROR: {name} short read from {f} ({got}/{wbytes})")

    print(f"[merge] meta.jsonl (sample_index shifted by cumulative prior written counts)", flush=True)
    with open(OUT / "meta.jsonl", "w") as w:
        cumulative = 0
        for d, nw in zip(input_dirs, n_written):
            with open(d / "meta.jsonl") as r:
                for line in r:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["sample_index"] = int(rec["sample_index"]) + cumulative
                    w.write(json.dumps(rec) + "\n")
            cumulative += nw

    gen_paths = [d / "generation.jsonl" for d in input_dirs]
    if all(gp.exists() for gp in gen_paths):
        with open(OUT / "generation.jsonl", "w") as w:
            for gp in gen_paths:
                with open(gp) as r:
                    shutil.copyfileobj(r, w)

    cfgM = dict(configs[0])
    cfgM["n_samples"] = total_written
    cfgM["_merged_from"] = [d.name for d in input_dirs]
    cfgM["_merged_written_rows"] = n_written
    cfgM["_skipped_arrays"] = sorted(skip)
    json.dump(cfgM, open(OUT / "config.json", "w"), indent=2)
    print(f"[merge] DONE -> {OUT}  (config n_samples={total_written})", flush=True)

if __name__ == "__main__":
    main()
