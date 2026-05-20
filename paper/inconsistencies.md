# Outline Inconsistencies — Working Log

Issues found between `outline.md` and the per-section outlines (`02_related_work.md`, `03_method.md`, `04_experimental_setup.md`). Work through these top-to-bottom; mark each resolved with a decision + date.

---

## High severity

### I-1 SEP-SE: cut in `outline.md`, active baseline in section outlines

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `outline.md` Open Question 2 (resolved): "SEP-SE **cut 2026-05-20**"
- `outline.md` §2.2 summary: SEP warrants only "one-sentence footnote"; "not run as a standalone experiment"
- `02_related_work.md` §2.2: SEP-SE is "the closest single-pass white-box competitor"; full bullet devoted to it; "In §4 we implement this exactly"
- `04_experimental_setup.md` §4.3: SEP-SE explicitly listed as a sampling-based baseline with config path and implementation pointer

**Decision needed:** Is SEP-SE in the paper as a live baseline, or cut?

**Resolution:** SEP-SE **removed** as a baseline (2026-05-20). ACT-ViT is the closest competitor, not SEP-SE. `02_related_work.md` §2.2 SEP bullet updated to remove the "closest competitor" claim; §2.2 closing summary updated. SEP-SE entry removed from `04_experimental_setup.md` §4.3 and its §4 materialized reference entry deleted. `outline.md` Open Question 2 ("SEP-SE cut") is now consistent with section outlines.

---

### I-2 ICR Probe absent from `outline.md` §4 baselines list

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `outline.md` §4 activation-space probes list: "single-layer linear probe, SAPLMA, LLMsKnow, ACT-ViT" — ICR Probe not mentioned
- `04_experimental_setup.md` §4.3: ICR Probe is a named baseline with config path (`configs/methods/icr_probe.json`), methodology footnote, and Phase-2 rollout discussion

**Decision needed:** Add ICR Probe to the `outline.md` §4 baselines list, or is it intentionally omitted from the summary?

**Resolution:** ICR Probe **cut as a §4 baseline** due to reproducibility failure (undocumented labeling pipeline, non-functional training code, ~0.675 vs. paper's 0.7982 AUROC). Removed from `04_experimental_setup.md` §4.3 and its materialized references. Open Question 4 (Phase-2 status) deleted from §4 open questions. `02_related_work.md` §2.1 ICR Probe bullet updated with scope-decision note: cite in §2.1, flag in §9 limitations, do not reproduce. `outline.md` §4 list was already correct in omitting it.

---

### I-3 Bib status of `khosla2020supcon`, `poole2019variational`, `wang2021understanding`

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `outline.md` "Bibliography state": "Most recent additions: `khosla2020supcon`, `poole2019variational`, `wang2021understanding`, `tishby2015deep`, `hjelm2019deepinfomax` — human-approved 2026-05-19" → implies all five are in `references.bib`
- `03_method.md` pending-approval section: all three of `khosla2020supcon`, `poole2019variational`, `wang2021understanding` still marked `[PENDING-APPROVAL]` with "NOT in `.bib`" warning; section header states the §3.2 MI argument cannot be drafted cleanly until they land

**Decision needed:** Are these three in `references.bib` or not? Verify against the actual `.bib` file and update whichever document is wrong.

**Resolution:** All five (`khosla2020supcon`, `poole2019variational`, `wang2021understanding`, `tishby2015deep`, `hjelm2019deepinfomax`) confirmed present in `references.bib` via grep. `outline.md` was correct. `03_method.md` bib-policy header, Open Question 1, and all five pending-approval entries updated to reflect confirmed status.

---

### I-4 Bib status of `zhang2025icr`, `suresh2025clap`, `barshalom2025actvit`

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `02_related_work.md` "Pending-approval candidates": all three listed as NOT in `.bib`, pending human approval
- `04_experimental_setup.md` "References — materialized for §4": header says "Every entry below is **already in `paper/references.bib`** as of 2026-05-19" and lists `zhang2025icr` and `barshalom2025actvit` under it
- `outline.md` "Bibliography state": no mention of these three being added

**Decision needed:** Verify the actual `.bib` file; update all three documents to agree on which are in and which are pending.

**Resolution:** All three confirmed present in `references.bib` via grep. `04_experimental_setup.md` was correct. `02_related_work.md` pending-approval entries for all three updated to "CONFIRMED IN `.bib` 2026-05-20". `zhang2025icr` entry also notes the I-2 scope decision (cite-only, not a §4 baseline).

---

## Medium severity

### I-5 LLMsKnow year: "2023" in `outline.md`, likely "2024" in `04_experimental_setup.md`

**Status:** OPEN

**Conflict:**
- `outline.md` §4: "LLMsKnow (Slobodkin et al. **2023**)"
- `04_experimental_setup.md` §4.3: cites it as "[UNKNOWN — verify]"; most likely candidate is "Slobodkin et al., NeurIPS **2024** / arXiv:2410.02707"

The arXiv ID 2410.02707 (October 2024) makes 2023 implausible.

**Decision needed:** Confirm the year and venue; update `outline.md` §4 accordingly.

**Resolution:**

---

### I-6 ECE placement: supplementary in `outline.md`, main table in `04_experimental_setup.md`

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `outline.md` §4 Metrics: "ECE + FPR@95 + bootstrap 95% CIs in **supplementary tables**"
- `04_experimental_setup.md` §4.5: "ECE numbers in the **main table** where the column fits" — only FPR@95 is supplementary-only

**Decision needed:** Does ECE go in the main table or supplementary? Update `outline.md` §4 Metrics bullet to match.

**Resolution:** Decision superseded 2026-05-20 — AUROC ± std (across 5 seeds) is the **only main-paper metric**; AUPRC, ECE, FPR@95 all move to supplementary. Both `outline.md` (§4 Metrics + §5.1 + §5.4) and `04_experimental_setup.md` §4.5 updated to reflect this. `\resultCI` macro reference also removed from §4.5.

---

## Low severity

### I-7 `outline.md` §4 missing §4.4 Training Procedure subsection

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `outline.md` §4 has no mention of a dedicated Training Procedure section
- `04_experimental_setup.md` adds §4.4 as a full subsection covering per-cell protocol, activation caching, layer band, optimizer hyperparameters, and the transfer-evaluation procedure

**Decision needed:** Add a "Training procedure" bullet to the `outline.md` §4 summary, or accept the section outline as the canonical expansion and leave the top-level outline as a high-level summary.

**Resolution:** Training procedure bullet added to `outline.md` §4. Prose constraint added to `04_experimental_setup.md` §4.4: one short paragraph in the final paper, no inline hyperparameters — forward readers to GitHub for full implementation details.

---

### I-8 `outline.md` §2.1 summary omits `belinkov2019analysis`

**Status:** RESOLVED 2026-05-20

**Conflict:**
- `outline.md` §2 summary names six references for §2.1; Belinkov & Glass (2019) is not among them
- `02_related_work.md` §2.1 has a full bullet for Belinkov & Glass as the "survey-level framing" anchor citation

**Decision needed:** Add Belinkov to the `outline.md` §2.1 reference list, or accept that the top-level summary intentionally omits survey citations.

**Resolution:** A general note added to the `outline.md` preamble stating that per-section outlines are the authoritative reference lists and may contain citations not enumerated in the top-level summary. Absence from this outline is not an inconsistency. No change needed to either reference list.
