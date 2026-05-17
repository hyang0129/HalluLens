# Vendored ICR Probe reference (`icr_score.py`)

`icr_score_reference.py` is a verbatim copy of `src/icr_score.py` from
[XavierZhang2002/ICR_Probe](https://github.com/XavierZhang2002/ICR_Probe)
at `main` as of 2026-05-16.

## Why vendored

The numerical-equivalence gate for Issue #72's capture rewrite asserts that
our stream-stitched response-to-response attention matches upstream's
`_pre_process_attn` output to fp16 tolerance on a real generation. We need
the reference in-tree so the gate is reproducible and survives upstream
repo changes.

Used **only** by `tests/test_capture_equivalence.py` (and any future
ablation that wants to cross-check against the published baseline).
Production code does not import from this directory.

## License

Apache 2.0. See `LICENSE` in this directory for the full text. Original
copyright belongs to Xavier Zhang and contributors.

## Citation

Zhang et al. *ICR Probe: Tracking Hidden State Dynamics for Reliable
Hallucination Detection in LLMs.* ACL 2025.
[arXiv:2507.16488](https://arxiv.org/abs/2507.16488) · [ACL Anthology](https://aclanthology.org/2025.acl-long.880/)
